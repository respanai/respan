"""Regression coverage for exported Claude tool invocation identity."""

import json
import logging

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.semconv_ai import SpanAttributes

from respan_instrumentation_claude_agent_sdk import _processor
from respan_instrumentation_claude_agent_sdk._constants import (
    CLAUDE_AGENT_SDK_TOOL_CALL_ID_ATTR,
)
from respan_sdk.constants.llm_logging import LOG_TYPE_AGENT, LOG_TYPE_CHAT, LOG_TYPE_TOOL
from respan_sdk.constants.span_attributes import RESPAN_LOG_TYPE
from respan_tracing.exporters.respan import (
    _build_otlp_payload,
    _prepare_spans_for_export,
)


_TOOL_CALLS_ATTR = f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls"


def _call(arguments, *, call_id="toolu_grep", name="Grep"):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


@pytest.mark.parametrize("call_id", ["toolu_grep", ""])
def test_equivalent_json_arguments_merge_without_warning(call_id, caplog):
    first = _call(
        '{"pattern":"TODO","options":{"paths":["src","tests"],"case":true}}',
        call_id=call_id,
    )
    reordered = _call(
        '{ "options": { "case": true, "paths": ["src", "tests"] }, '
        '"pattern": "TODO" }',
        call_id=call_id,
    )

    with caplog.at_level(logging.WARNING, logger=_processor.__name__):
        merged = _processor._merge_tool_calls([first], [reordered, first])

    assert merged == [first]
    assert not caplog.records


def test_distinct_invocation_ids_with_identical_arguments_stay_separate():
    first = _call('{"pattern":"TODO"}', call_id="toolu_first")
    second = _call('{"pattern":"TODO"}', call_id="toolu_second")

    assert _processor._merge_tool_calls([first, second], [first]) == [first, second]


def test_synthetic_102_observations_merge_to_76_invocations(caplog):
    # Mirrors the reported duplicate counts with synthetic content; this is not
    # a replay of the original trace or its tool inputs.
    model_calls = [
        _call(
            json.dumps({"pattern": f"TODO-{index}", "path": "src"}),
            call_id=f"toolu_{index}",
        )
        for index in range(76)
    ]
    repeated_tool_calls = [
        _call(
            json.dumps({"path": "src", "pattern": f"TODO-{index}"}, separators=(",", ":")),
            call_id=f"toolu_{index}",
        )
        for index in range(26)
    ]

    with caplog.at_level(logging.WARNING, logger=_processor.__name__):
        merged = _processor._merge_tool_calls(model_calls, repeated_tool_calls)

    assert len(model_calls) + len(repeated_tool_calls) == 102
    assert merged == model_calls
    assert len({call["id"] for call in merged}) == 76
    assert not caplog.records


@pytest.mark.parametrize(
    "conflicting_call",
    [
        _call('{"pattern":"FIXME"}'),
        _call('{"pattern":"TODO"}', name="Search"),
        _call("unparseable arguments"),
    ],
    ids=["arguments", "name", "malformed-arguments"],
)
def test_conflicting_invocation_keeps_first_call_and_warns(conflicting_call, caplog):
    first = _call('{"pattern":"TODO"}')

    with caplog.at_level(logging.WARNING, logger=_processor.__name__):
        merged = _processor._merge_tool_calls([first], [conflicting_call])

    assert merged == [first]
    warnings = [record.getMessage() for record in caplog.records]
    assert any("toolu_grep" in message and "conflict" in message.lower() for message in warnings)


@pytest.mark.parametrize(
    "partial_function",
    [
        None,
        {},
        {"name": "Grep"},
        {"name": "Grep", "arguments": None},
        {"name": "Grep", "arguments": ""},
        {"arguments": '{"pattern":"TODO"}'},
        {"name": "", "arguments": '{"pattern":"TODO"}'},
    ],
)
@pytest.mark.parametrize("partial_first", [True, False])
def test_partial_observation_merges_with_complete_invocation(
    partial_function, partial_first, caplog
):
    complete = _call('{"pattern":"TODO"}')
    partial = {"id": "toolu_grep", "type": "function"}
    if partial_function is not None:
        partial["function"] = partial_function
    observations = [partial, complete] if partial_first else [complete, partial]

    with caplog.at_level(logging.WARNING, logger=_processor.__name__):
        merged = _processor._merge_tool_calls(observations)

    assert merged == [complete]
    assert not caplog.records


@pytest.mark.parametrize(
    ("first_arguments", "second_arguments"),
    [
        ("{broken: one", "{broken: two"),
        ("first raw value", "second raw value"),
        ("null", '"null"'),
        ("null", "not JSON"),
        ('{"value":true}', '{"value":1}'),
        ('{"paths":["src","tests"]}', '{"paths":["tests","src"]}'),
    ],
)
def test_calls_without_ids_keep_distinct_arguments(first_arguments, second_arguments):
    first = _call(first_arguments, call_id="")
    second = _call(second_arguments, call_id="")

    assert _processor._merge_tool_calls([first, second, first]) == [first, second]


@pytest.mark.parametrize(("raw_value", "canonical_value"), [("", ""), ([], "[]")])
def test_repeated_normalization_keeps_empty_tool_input_and_result(
    raw_value, canonical_value
):
    exporter = InMemorySpanExporter()
    provider = TracerProvider(shutdown_on_exit=False)
    provider.add_span_processor(_processor.ClaudeAgentSDKSpanProcessor())
    provider.add_span_processor(_processor.ClaudeAgentSDKSpanProcessor())
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    try:
        tracer = provider.get_tracer("opentelemetry.instrumentation.claude_agent_sdk")
        with tracer.start_as_current_span(
            "execute_tool Grep",
            attributes={
                "gen_ai.operation.name": "execute_tool",
                "gen_ai.tool.name": "Grep",
                "gen_ai.tool.call.arguments": raw_value,
                "gen_ai.tool.call.result": raw_value,
                CLAUDE_AGENT_SDK_TOOL_CALL_ID_ATTR: "toolu_grep",
            },
        ):
            pass
        span = exporter.get_finished_spans()[0]
    finally:
        provider.shutdown()

    assert span.attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT] == canonical_value
    assert span.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] == canonical_value
    normalized = dict(span.attributes)
    _processor.enrich_claude_agent_sdk_span(span)
    assert dict(span.attributes) == normalized


@pytest.mark.parametrize("processor_count", [1, 2])
@pytest.mark.parametrize("tool_span_name", ["execute_tool Grep", "Grep"])
def test_real_export_preserves_tool_identity_result_and_parent_calls(
    processor_count, tool_span_name, caplog
):
    exporter = InMemorySpanExporter()
    provider = TracerProvider(shutdown_on_exit=False)
    for _ in range(processor_count):
        provider.add_span_processor(_processor.ClaudeAgentSDKSpanProcessor())
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("opentelemetry.instrumentation.claude_agent_sdk")
    arguments = {"path": "src", "pattern": "TODO"}
    result = {"content": [{"type": "text", "text": "src/main.py:7: TODO"}]}
    input_messages = [{"role": "user", "content": "Find TODO comments in src"}]
    output_messages = [
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "toolu_grep", "name": "Grep", "input": arguments},
                {"type": "text", "text": "Found one TODO comment."},
            ],
        }
    ]
    try:
        with tracer.start_as_current_span(
            "invoke_agent code_search",
            attributes={
                "gen_ai.operation.name": "invoke_agent",
                "gen_ai.agent.name": "code_search",
                "gen_ai.input.messages": json.dumps(input_messages),
                "gen_ai.output.messages": json.dumps(output_messages),
                "gen_ai.tool.definitions": '[{"name":"Grep"}]',
                "gen_ai.response.model": "claude-sonnet-4-5",
                "gen_ai.usage.input_tokens": 30,
                "gen_ai.usage.output_tokens": 7,
                SpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS: 10,
                SpanAttributes.LLM_USAGE_CACHE_CREATION_INPUT_TOKENS: 4,
            },
        ):
            with tracer.start_as_current_span(
                tool_span_name,
                attributes={
                    "gen_ai.operation.name": "execute_tool",
                    "gen_ai.tool.name": "Grep",
                    CLAUDE_AGENT_SDK_TOOL_CALL_ID_ATTR: "toolu_grep",
                    # Same invocation as the model output, with changed JSON
                    # formatting and key order as produced by the SDK tool path.
                    "gen_ai.tool.call.arguments": '{"pattern":"TODO","path":"src"}',
                    "gen_ai.tool.call.result": json.dumps(result),
                },
            ):
                pass
        finished = exporter.get_finished_spans()
    finally:
        provider.shutdown()

    assert len(finished) == 2
    tool, agent = finished
    assert tool.attributes[RESPAN_LOG_TYPE] == LOG_TYPE_TOOL
    assert tool.attributes[SpanAttributes.TRACELOOP_ENTITY_NAME] == "Grep"
    assert tool.attributes[SpanAttributes.TRACELOOP_ENTITY_PATH] == "Grep"
    assert tool.attributes[CLAUDE_AGENT_SDK_TOOL_CALL_ID_ATTR] == "toolu_grep"
    assert json.loads(tool.attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT]) == arguments
    assert json.loads(tool.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]) == result
    assert {key for key in tool.attributes if key.startswith("gen_ai.tool.")} == {
        CLAUDE_AGENT_SDK_TOOL_CALL_ID_ATTR
    }
    assert agent.attributes[RESPAN_LOG_TYPE] == LOG_TYPE_AGENT
    assert json.loads(agent.attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT]) == input_messages
    assert json.loads(agent.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]) == output_messages
    assert agent.attributes[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] == 16
    assert agent.attributes[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS] == 7
    assert agent.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] == 23
    assert agent.attributes[SpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS] == 10
    assert agent.attributes[SpanAttributes.LLM_USAGE_CACHE_CREATION_INPUT_TOKENS] == 4
    assert json.loads(agent.attributes[_TOOL_CALLS_ATTR]) == [_call(json.dumps(arguments))]

    # Explicit re-normalization must preserve all canonical fields after the
    # upstream role and content helper attributes have already been removed.
    for span in finished:
        normalized = dict(span.attributes)
        _processor.enrich_claude_agent_sdk_span(span)
        _processor.enrich_claude_agent_sdk_span(span)
        assert dict(span.attributes) == normalized

    prepared = _prepare_spans_for_export(finished)
    assert len(prepared) == 3
    exported_tool, exported_agent, exported_chat = prepared
    assert exported_chat.attributes[RESPAN_LOG_TYPE] == LOG_TYPE_CHAT
    assert json.loads(exported_chat.attributes[_TOOL_CALLS_ATTR]) == [
        _call(json.dumps(arguments))
    ]
    assert exported_chat.attributes[SpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS] == 10
    assert exported_chat.attributes[SpanAttributes.LLM_USAGE_CACHE_CREATION_INPUT_TOKENS] == 4
    assert exported_tool.parent == exported_agent.get_span_context()
    assert exported_chat.parent == exported_agent.get_span_context()

    wire_spans = [
        span
        for resource in _build_otlp_payload(prepared)["resourceSpans"]
        for scope in resource["scopeSpans"]
        for span in scope["spans"]
    ]
    wire_tool, wire_agent, wire_chat = wire_spans
    wire_tool_attrs = {item["key"]: item["value"] for item in wire_tool["attributes"]}
    wire_chat_attrs = {item["key"]: item["value"] for item in wire_chat["attributes"]}
    invocation_id = wire_tool_attrs[CLAUDE_AGENT_SDK_TOOL_CALL_ID_ATTR]["stringValue"]
    assert invocation_id == "toolu_grep"
    assert json.loads(wire_chat_attrs[_TOOL_CALLS_ATTR]["stringValue"])[0]["id"] == invocation_id
    assert json.loads(
        wire_tool_attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]["stringValue"]
    ) == result
    assert wire_tool["traceId"] == wire_agent["traceId"] == wire_chat["traceId"]
    assert wire_tool["parentSpanId"] == wire_chat["parentSpanId"] == wire_agent["spanId"]
    assert not [record for record in caplog.records if record.name == _processor.__name__]
