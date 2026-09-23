"""Contract tests for current OpenAI Agents SDK span translation."""

from __future__ import annotations

import json
from collections.abc import Sequence
from types import SimpleNamespace

import pytest
from agents import Agent, RunConfig, Runner, function_tool
from agents.items import ModelResponse
from agents.models.interface import Model
from agents.tracing import (
    SpanError,
    agent_span,
    function_span,
    generation_span,
    guardrail_span,
    handoff_span,
    set_trace_processors,
    task_span,
    trace,
    turn_span,
)
from agents.usage import Usage
from openai.types.responses import (
    Response,
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputText,
)
from openai.types.responses.response_completed_event import ResponseCompletedEvent
from openai.types.responses.response_created_event import ResponseCreatedEvent
from openai.types.responses.response_text_delta_event import ResponseTextDeltaEvent
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.trace.status import StatusCode
from respan_instrumentation_openai_agents import _otel_emitter
from respan_instrumentation_openai_agents._instrumentation import (
    _RespanTracingProcessor,
    _wrap_stream_method,
)
from respan_instrumentation_openai_agents._serialization import (
    MAX_ATTRIBUTE_BYTES,
    error_status_code,
    flatten_metadata_attributes,
    json_string,
)
from respan_sdk.constants.span_attributes import RESPAN_LOG_TYPE, RESPAN_METADATA

_BANNED_ALIASES = {
    "completion_tokens",
    "has_tool_calls",
    "model",
    "parallel_tool_calls",
    "prompt_tokens",
    "respan.span.handoffs",
    "respan.span.tool_calls",
    "respan.span.tools",
    "span_tools",
    "tool_calls",
    "tools",
    "total_request_tokens",
}


def _make_span_item(*, error=None) -> SimpleNamespace:
    return SimpleNamespace(
        trace_id="trace_123",
        span_id="span_456",
        parent_id=None,
        started_at=None,
        ended_at=None,
        error=error,
        trace_metadata={"tenant": "example"},
    )


def _capture_spans(monkeypatch):
    captured = []
    monkeypatch.setattr(_otel_emitter, "inject_span", captured.append)
    return captured


def test_emit_response_uses_complete_chat_contract(monkeypatch):
    captured = _capture_spans(monkeypatch)
    response = SimpleNamespace(
        model="gpt-4o",
        output=[
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "lookup_weather",
                "arguments": '{"city":"NYC"}',
            }
        ],
        tools=[
            {
                "type": "function",
                "name": "lookup_weather",
                "description": "Look up the weather.",
                "parameters": {"type": "object"},
            }
        ],
        usage=SimpleNamespace(input_tokens=12, output_tokens=4, total_tokens=16),
    )
    span_data = SimpleNamespace(
        response=response,
        input="What is the weather in NYC?",
        usage=None,
    )

    _otel_emitter.emit_response(
        _make_span_item(),
        span_data,
        extra_metadata={"example_run_id": "marker"},
        is_streaming=True,
    )

    span = captured[0]
    attrs = span.attributes
    assert attrs[RESPAN_LOG_TYPE] == "chat"
    assert attrs[SpanAttributes.LLM_REQUEST_TYPE] == "chat"
    assert attrs[SpanAttributes.LLM_SYSTEM] == "openai"
    assert attrs[SpanAttributes.LLM_REQUEST_MODEL] == "gpt-4o"
    assert attrs[SpanAttributes.LLM_IS_STREAMING] is True
    assert attrs["gen_ai.usage.input_tokens"] == 12
    assert attrs["gen_ai.usage.output_tokens"] == 4
    assert attrs[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] == 12
    assert attrs[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS] == 4
    assert attrs[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] == 16
    assert json.loads(attrs[RESPAN_METADATA]) == {
        "example_run_id": "marker",
        "tenant": "example",
    }
    assert attrs[f"{RESPAN_METADATA}.example_run_id"] == "marker"
    assert attrs[f"{RESPAN_METADATA}.tenant"] == "example"
    assert json.loads(attrs[f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls"]) == [
        {
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "lookup_weather",
                "arguments": '{"city":"NYC"}',
            },
        }
    ]
    assert json.loads(attrs[SpanAttributes.LLM_REQUEST_FUNCTIONS]) == [
        {
            "type": "function",
            "function": {
                "name": "lookup_weather",
                "description": "Look up the weather.",
                "parameters": {"type": "object"},
            },
        }
    ]
    assert _BANNED_ALIASES.isdisjoint(attrs)
    assert SpanAttributes.TRACELOOP_SPAN_KIND not in attrs


def test_emit_generation_handles_chat_completion_tool_shape(monkeypatch):
    captured = _capture_spans(monkeypatch)
    span_data = SimpleNamespace(
        input=[{"role": "user", "content": "Use the tool"}],
        output=[
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {
                            "name": "search_docs",
                            "arguments": '{"query":"otel"}',
                        },
                    }
                ],
            }
        ],
        model="gpt-4o",
        model_config={"stream": False},
        usage={"input_tokens": 8, "output_tokens": 2, "total_tokens": 10},
    )

    _otel_emitter.emit_generation(
        _make_span_item(),
        span_data,
        agent_context={"tools": ["search_docs"]},
    )

    attrs = captured[0].attributes
    tool_calls = json.loads(attrs[f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls"])
    assert tool_calls[0]["id"] == "call_2"
    assert tool_calls[0]["function"]["name"] == "search_docs"
    assert attrs[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"] == ""
    assert json.loads(attrs[SpanAttributes.LLM_REQUEST_FUNCTIONS]) == [
        {"type": "function", "function": {"name": "search_docs"}}
    ]
    assert attrs[SpanAttributes.LLM_IS_STREAMING] is False
    assert _BANNED_ALIASES.isdisjoint(attrs)


def test_tool_agent_handoff_and_guardrail_are_structured(monkeypatch):
    captured = _capture_spans(monkeypatch)
    item = _make_span_item()

    _otel_emitter.emit_function(
        item,
        SimpleNamespace(name="weather", input='{"city":"Tokyo"}', output="sunny"),
    )
    _otel_emitter.emit_agent(
        item,
        SimpleNamespace(
            name="Triage",
            tools=["weather"],
            handoffs=["Spanish"],
            output_type="Answer",
        ),
    )
    _otel_emitter.emit_handoff(
        item, SimpleNamespace(from_agent="Triage", to_agent="Spanish")
    )
    _otel_emitter.emit_guardrail(
        item, SimpleNamespace(name="safe_input", triggered=True)
    )

    tool, agent, handoff, guardrail = captured
    assert json.loads(tool.attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT]) == {
        "arguments": {"city": "Tokyo"},
        "name": "weather",
    }
    assert (
        json.loads(tool.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]) == "sunny"
    )
    assert json.loads(agent.attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT]) == {
        "handoffs": ["Spanish"],
        "name": "Triage",
        "output_type": "Answer",
        "tools": ["weather"],
    }
    assert json.loads(handoff.attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT]) == {
        "from_agent": "Triage"
    }
    assert json.loads(handoff.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]) == {
        "to_agent": "Spanish"
    }
    assert json.loads(guardrail.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]) == {
        "triggered": True
    }
    for span in captured:
        assert _BANNED_ALIASES.isdisjoint(span.attributes)
        assert SpanAttributes.TRACELOOP_SPAN_KIND not in span.attributes


def test_error_status_is_precise_bounded_and_redacted(monkeypatch):
    captured = _capture_spans(monkeypatch)
    error = {
        "message": "401 denied for Bearer secret-token-value",
        "data": {"status_code": 401, "api_key": "sk-super-secret-value"},
    }

    _otel_emitter.emit_agent(
        _make_span_item(error=error),
        SimpleNamespace(name="Agent", tools=[], handoffs=[], output_type=None),
    )

    span = captured[0]
    assert span.status.status_code is StatusCode.ERROR
    assert "401" in (span.status.description or "")
    assert "secret-token-value" not in (span.status.description or "")
    assert error_status_code(error) == 401
    assert error_status_code({"message": "application exploded"}) == 500
    assert error_status_code("cache status code: 404") == 500
    assert error_status_code({"message": "cache status code: 404"}) == 500
    assert json.loads(span.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]) == {
        "error": {"message": "401 denied for [REDACTED]", "status_code": 401}
    }


def test_json_serializer_is_valid_bounded_and_cycle_safe():
    cycle: dict[str, object] = {}
    cycle["self"] = cycle
    payload = {
        "api_key": "sk-super-secret-value",
        "authorization": "Bearer token-value",
        "cycle": cycle,
        "payload": "x" * 100_000,
    }
    encoded = json_string(payload)
    decoded = json.loads(encoded)
    assert len(encoded.encode("utf-8")) <= MAX_ATTRIBUTE_BYTES
    assert "sk-super-secret-value" not in encoded
    assert "token-value" not in encoded
    assert decoded["api_key"] == "[REDACTED]"
    assert decoded["cycle"]["self"] == "<cycle>"


def test_json_serializer_bounds_iteration_keys_tokens_and_non_finite_values():
    class BoundedSequence(Sequence):
        def __len__(self):
            raise AssertionError("serializer must not ask for the full length")

        def __getitem__(self, index):
            if not isinstance(index, int) or index > 50:
                raise AssertionError("serializer consumed beyond one-item lookahead")
            return index

    class HostileKey:
        def __str__(self):
            raise AssertionError("serializer must not stringify arbitrary keys")

    encoded = json_string(
        {
            "items": BoundedSequence(),
            "token": "plain-token-value",
            "session_token": "session-token-value",
            "not_a_number": float("nan"),
            HostileKey(): "safe",
        }
    )
    decoded = json.loads(encoded)

    assert decoded["token"] == "[REDACTED]"
    assert decoded["session_token"] == "[REDACTED]"
    assert decoded["not_a_number"] is None
    assert len(decoded["items"]) == 51
    assert decoded["items"][-1] == "<truncated:more items>"
    assert decoded["<HostileKey>"] == "safe"


def test_flattened_metadata_is_bounded_redacted_and_scalar_safe():
    class HostileKey:
        def __str__(self):
            raise AssertionError("metadata keys must not invoke arbitrary strings")

    flattened = flatten_metadata_attributes(
        {
            "example_run_id": "exact-marker",
            "api_key": "sk-super-secret-value",
            "nested": {"session_token": "plain-session-secret"},
            "long": "🧪" * 10_000,
            "not_a_number": float("nan"),
            HostileKey(): "safe",
            "missing": None,
        }
    )

    assert flattened["example_run_id"] == "exact-marker"
    assert flattened["api_key"] == "[REDACTED]"
    assert json.loads(flattened["nested"])["session_token"] == "[REDACTED]"
    assert len(flattened["long"].encode("utf-8")) <= MAX_ATTRIBUTE_BYTES
    assert flattened["not_a_number"] == "null"
    assert flattened["<HostileKey>"] == "safe"
    assert "missing" not in flattened
    assert all(len(key) <= 128 for key in flattened)


def test_flattened_metadata_limits_field_count():
    flattened = flatten_metadata_attributes(
        {f"key-{index}": index for index in range(75)}
    )

    assert len(flattened) == 50
    assert flattened["key-0"] == "0"
    assert "key-50" not in flattened


def test_scalar_attributes_are_bounded_and_redacted(monkeypatch):
    captured = _capture_spans(monkeypatch)
    secret = "Bearer raw-agent-secret-token"

    _otel_emitter.emit_agent(
        _make_span_item(),
        SimpleNamespace(name=secret, tools=[secret], handoffs=[], output_type=None),
    )
    _otel_emitter.emit_handoff(
        _make_span_item(),
        SimpleNamespace(from_agent=secret, to_agent=f"target-{secret}"),
    )
    _otel_emitter.emit_response(
        _make_span_item(),
        SimpleNamespace(
            input=[{"role": "user", "content": f"authorization={secret}"}],
            response=SimpleNamespace(model=secret, output=[], tools=[], usage=None),
            usage=None,
        ),
    )

    encoded = json.dumps([span.attributes for span in captured], default=str)
    assert "raw-agent-secret-token" not in encoded
    assert "[REDACTED]" in encoded


def test_real_sdk_primitives_keep_tree_metadata_and_agent_tools(monkeypatch):
    captured = _capture_spans(monkeypatch)
    processor = _RespanTracingProcessor(metadata={"example_run_id": "real-marker"})
    set_trace_processors([processor])

    with (
        trace(
            "contract",
            group_id="group-1",
            metadata={"scenario": "real-framework"},
        ),
        task_span("run"),
        agent_span(
            "Triage",
            tools=["weather"],
            handoffs=["Spanish"],
            output_type="Answer",
        ),
        turn_span(1, "Triage"),
    ):
        with generation_span(
            input=[{"role": "user", "content": "Weather?"}],
            output=[{"role": "assistant", "content": "Sunny"}],
            model="gpt-4o-mini",
            usage={"input_tokens": 4, "output_tokens": 1},
        ) as generation:
            pass
        with function_span("weather", input='{"city":"Tokyo"}', output="sunny"):
            pass
        with handoff_span("Triage", "Spanish"):
            pass
        with guardrail_span("safe_input", triggered=False):
            pass

    assert len(captured) == 8
    span_ids = {span.get_span_context().span_id for span in captured}
    assert len(span_ids) == len(captured)
    roots = [span for span in captured if span.parent is None]
    assert len(roots) == 1
    assert roots[0].start_time < roots[0].end_time
    for span in captured:
        metadata = json.loads(span.attributes[RESPAN_METADATA])
        assert metadata["example_run_id"] == "real-marker"
        assert span.attributes[f"{RESPAN_METADATA}.example_run_id"] == "real-marker"
        if span is not roots[0]:
            assert span.parent is not None
            assert span.parent.span_id in span_ids
    chat = next(span for span in captured if span.attributes[RESPAN_LOG_TYPE] == "chat")
    assert json.loads(chat.attributes[SpanAttributes.LLM_REQUEST_FUNCTIONS]) == [
        {"type": "function", "function": {"name": "weather"}}
    ]

    processor.on_span_end(generation)
    assert len(captured) == 8


def test_real_sdk_error_exports_otel_error(monkeypatch):
    captured = _capture_spans(monkeypatch)
    processor = _RespanTracingProcessor()
    set_trace_processors([processor])

    with (
        trace("failure"),
        generation_span(
            input=[{"role": "user", "content": "fail"}], model="gpt-4o-mini"
        ) as span,
    ):
        span.set_error(
            SpanError(
                message="Provider request failed",
                data={"status_code": 503, "error": "temporarily unavailable"},
            )
        )

    failed = next(
        span for span in captured if span.attributes[RESPAN_LOG_TYPE] == "chat"
    )
    assert failed.status.status_code is StatusCode.ERROR
    assert failed.status.description == "Provider request failed"


@pytest.mark.asyncio
async def test_real_runner_offline_tool_flow_exports_one_canonical_tree(monkeypatch):
    captured = _capture_spans(monkeypatch)

    class OfflineToolModel(Model):
        def __init__(self):
            self.calls = 0

        async def get_response(
            self,
            system_instructions,
            input,
            model_settings,
            tools,
            output_schema,
            handoffs,
            tracing,
            *,
            previous_response_id,
            conversation_id,
            prompt,
        ):
            del (
                system_instructions,
                model_settings,
                tools,
                output_schema,
                handoffs,
                tracing,
                previous_response_id,
                conversation_id,
                prompt,
            )
            self.calls += 1
            if self.calls == 1:
                output = [
                    ResponseFunctionToolCall(
                        arguments='{"city":"Paris"}',
                        call_id="call_weather",
                        name="weather",
                        status="completed",
                        type="function_call",
                    )
                ]
            else:
                output = [
                    ResponseOutputMessage(
                        id="message_weather",
                        content=[
                            ResponseOutputText(
                                annotations=[],
                                logprobs=[],
                                text="Sunny in Paris.",
                                type="output_text",
                            )
                        ],
                        role="assistant",
                        status="completed",
                        type="message",
                    )
                ]
            output_payload = [item.model_dump(mode="json") for item in output]
            usage = {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5}
            with generation_span(
                input=input,
                output=output_payload,
                model="offline-model",
                usage=usage,
            ):
                return ModelResponse(
                    output=output,
                    usage=Usage(
                        requests=1,
                        input_tokens=3,
                        output_tokens=2,
                        total_tokens=5,
                    ),
                    response_id=f"response_{self.calls}",
                )

        async def stream_response(self, *_args, **_kwargs):
            if False:
                yield None

    @function_tool
    def weather(city: str) -> str:
        """Return deterministic weather."""
        return f"Sunny in {city}."

    processor = _RespanTracingProcessor(metadata={"example_run_id": "runner-marker"})
    set_trace_processors([processor])
    result = await Runner.run(
        Agent(name="Offline Agent", model=OfflineToolModel(), tools=[weather]),
        "What is the weather?",
        run_config=RunConfig(workflow_name="offline-runner"),
    )

    assert result.final_output == "Sunny in Paris."
    assert len(captured) == 8
    span_ids = {span.get_span_context().span_id for span in captured}
    roots = [span for span in captured if span.parent is None]
    assert len(roots) == 1
    assert roots[0].start_time < roots[0].end_time
    for span in captured:
        assert json.loads(span.attributes[RESPAN_METADATA])["example_run_id"] == (
            "runner-marker"
        )
        if span.parent is not None:
            assert span.parent.span_id in span_ids

    chats = [span for span in captured if span.attributes[RESPAN_LOG_TYPE] == "chat"]
    assert len(chats) == 2
    first_calls = json.loads(
        chats[0].attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls"]
    )
    assert first_calls[0]["function"] == {
        "arguments": '{"city":"Paris"}',
        "name": "weather",
    }
    assert f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls" not in chats[1].attributes
    tool = next(span for span in captured if span.attributes[RESPAN_LOG_TYPE] == "tool")
    assert json.loads(tool.attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT]) == {
        "arguments": {"city": "Paris"},
        "name": "weather",
    }


@pytest.mark.asyncio
async def test_real_runner_offline_stream_closes_source_and_marks_chat(monkeypatch):
    captured = _capture_spans(monkeypatch)

    def response(output, status, usage=None):
        return Response(
            id="response_stream",
            created_at=1,
            error=None,
            incomplete_details=None,
            instructions=None,
            metadata={},
            model="offline-model",
            object="response",
            output=output,
            parallel_tool_calls=True,
            status=status,
            temperature=1,
            tool_choice="auto",
            tools=[],
            top_p=1,
            usage=usage,
        )

    class OfflineStreamingModel(Model):
        def __init__(self):
            self.source_closed = False

        async def get_response(self, *_args, **_kwargs):
            raise AssertionError("streaming Runner must use stream_response")

        async def stream_response(
            self,
            system_instructions,
            input,
            model_settings,
            tools,
            output_schema,
            handoffs,
            tracing,
            *,
            previous_response_id,
            conversation_id,
            prompt,
        ):
            del (
                system_instructions,
                model_settings,
                tools,
                output_schema,
                handoffs,
                tracing,
                previous_response_id,
                conversation_id,
                prompt,
            )
            message = ResponseOutputMessage(
                id="message_stream",
                content=[
                    ResponseOutputText(
                        annotations=[],
                        logprobs=[],
                        text="Hello stream.",
                        type="output_text",
                    )
                ],
                role="assistant",
                status="completed",
                type="message",
            )
            with generation_span(
                input=input,
                output=[message.model_dump(mode="json")],
                model="offline-model",
                model_config={"stream": True},
                usage={"input_tokens": 2, "output_tokens": 2, "total_tokens": 4},
            ):
                try:
                    yield ResponseCreatedEvent(
                        response=response([], "in_progress"),
                        sequence_number=0,
                        type="response.created",
                    )
                    yield ResponseTextDeltaEvent(
                        content_index=0,
                        delta="Hello stream.",
                        item_id="message_stream",
                        logprobs=[],
                        output_index=0,
                        sequence_number=1,
                        type="response.output_text.delta",
                    )
                    yield ResponseCompletedEvent(
                        response=response(
                            [message],
                            "completed",
                            {
                                "input_tokens": 2,
                                "input_tokens_details": {
                                    "cache_write_tokens": 0,
                                    "cached_tokens": 0,
                                },
                                "output_tokens": 2,
                                "output_tokens_details": {"reasoning_tokens": 0},
                                "total_tokens": 4,
                            },
                        ),
                        sequence_number=2,
                        type="response.completed",
                    )
                finally:
                    self.source_closed = True

    OfflineStreamingModel.stream_response = _wrap_stream_method(
        OfflineStreamingModel.stream_response
    )
    model = OfflineStreamingModel()
    processor = _RespanTracingProcessor(metadata={"example_run_id": "stream-marker"})
    set_trace_processors([processor])
    result = Runner.run_streamed(
        Agent(name="Streaming Agent", model=model),
        "Stream hello.",
        run_config=RunConfig(workflow_name="offline-stream-runner"),
    )

    async for _event in result.stream_events():
        pass

    assert result.final_output == "Hello stream."
    assert model.source_closed is True
    chats = [span for span in captured if span.attributes[RESPAN_LOG_TYPE] == "chat"]
    assert len(chats) == 1
    assert chats[0].attributes[SpanAttributes.LLM_IS_STREAMING] is True
    assert chats[0].attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"] == (
        "Hello stream."
    )


@pytest.mark.parametrize(
    ("factory_name", "kwargs", "log_type", "expected_input", "expected_output"),
    [
        (
            "mcp_tools_span",
            {"server": "weather", "result": ["forecast"]},
            "tool",
            {"name": "weather.list_tools", "arguments": {"server": "weather"}},
            ["forecast"],
        ),
        (
            "transcription_span",
            {"input": "YXVkaW8=", "input_format": "pcm", "output": "Hello"},
            "transcription",
            {"data": "YXVkaW8=", "format": "pcm"},
            "Hello",
        ),
        (
            "speech_span",
            {"input": "Hello", "output": "YXVkaW8=", "output_format": "pcm"},
            "speech",
            "Hello",
            {"data": "YXVkaW8=", "format": "pcm"},
        ),
        ("speech_group_span", {"input": "Hello"}, "task", "Hello", None),
    ],
)
def test_current_sdk_mcp_and_voice_spans_preserve_content(
    monkeypatch, factory_name, kwargs, log_type, expected_input, expected_output
):
    import agents.tracing as sdk_tracing

    captured = _capture_spans(monkeypatch)
    set_trace_processors([_RespanTracingProcessor()])
    with trace("native-span-coverage"):  # noqa: SIM117 - assert explicit SDK nesting
        with getattr(sdk_tracing, factory_name)(**kwargs):
            pass
    child, root = captured
    assert child.parent.span_id == root.get_span_context().span_id
    assert child.attributes[RESPAN_LOG_TYPE] == log_type
    assert (
        json.loads(child.attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT])
        == expected_input
    )
    if expected_output is not None:
        assert (
            json.loads(child.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT])
            == expected_output
        )
    assert not _BANNED_ALIASES.intersection(child.attributes)


def test_structural_rollups_do_not_double_count_model_usage(monkeypatch):
    captured = _capture_spans(monkeypatch)
    set_trace_processors([_RespanTracingProcessor()])
    with trace("usage-rollup"):  # noqa: SIM117 - assert explicit SDK nesting
        with task_span("run") as task:
            task.span_data.usage = {
                "input_tokens": 10,
                "output_tokens": 2,
                "total_tokens": 12,
            }
            task.span_data.metadata = {"checkpoint": "saved"}
            with turn_span(1, "Assistant") as turn:
                turn.span_data.usage = {"input_tokens": 10, "output_tokens": 2}
                with generation_span(
                    model="gpt-4o-mini", usage={"input_tokens": 10, "output_tokens": 2}
                ):
                    pass
    assert sum(s.attributes.get("gen_ai.usage.input_tokens", 0) for s in captured) == 10
    task_attrs = next(
        s.attributes
        for s in captured
        if s.attributes.get(SpanAttributes.TRACELOOP_ENTITY_NAME) == "run"
    )
    metadata = json.loads(task_attrs[RESPAN_METADATA])
    assert metadata["checkpoint"] == "saved"
    assert metadata["openai_agents.usage"]["total_tokens"] == 12


@pytest.mark.parametrize(
    "tool_type, action",
    [
        ("shell_call", {"commands": ["pwd"]}),
        (
            "apply_patch_call",
            {"type": "update_file", "path": "hello.txt", "diff": "+hello"},
        ),
        ("image_generation_call", {"action": "generate"}),
    ],
)
def test_responses_hosted_tools_are_calls_not_assistant_text(
    monkeypatch, tool_type, action
):
    captured = _capture_spans(monkeypatch)
    _otel_emitter.emit_response(
        _make_span_item(),
        SimpleNamespace(
            input=[{"role": "user", "content": "Perform the action"}],
            usage=None,
            response=SimpleNamespace(
                model="gpt-4o",
                usage=None,
                tools=[],
                output=[
                    {"type": tool_type, "id": "hosted_call", "action": action},
                    {"type": "reasoning", "summary": []},
                ],
            ),
        ),
    )
    attrs = captured[0].attributes
    calls = json.loads(attrs[f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls"])
    assert calls[0]["function"]["name"] == tool_type
    assert json.loads(calls[0]["function"]["arguments"]) == action
    assert attrs[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"] == ""


def test_streamed_chat_generation_unwraps_native_response_envelope(monkeypatch):
    from agents.tracing.span_data import GenerationSpanData

    captured = _capture_spans(monkeypatch)
    response = Response(
        id="response_stream",
        object="response",
        created_at=1,
        model="gpt-4o-mini",
        parallel_tool_calls=True,
        tool_choice="auto",
        tools=[],
        output=[
            ResponseFunctionToolCall(
                type="function_call",
                call_id="call_forecast",
                name="forecast",
                arguments='{"city":"Paris"}',
            ),
            ResponseOutputMessage(
                id="message",
                type="message",
                role="assistant",
                status="completed",
                content=[
                    ResponseOutputText(
                        type="output_text",
                        text="Sunny in Paris.",
                        annotations=[],
                        logprobs=[],
                    )
                ],
            ),
        ],
    )
    _otel_emitter.emit_generation(
        _make_span_item(),
        GenerationSpanData(
            input=[{"role": "user", "content": "Weather?"}],
            output=[response.model_dump()],
            model="gpt-4o-mini",
        ),
        is_streaming=True,
    )
    attrs = captured[0].attributes
    assert attrs[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"] == "Sunny in Paris."
    assert (
        json.loads(attrs[f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls"])[0][
            "function"
        ]["name"]
        == "forecast"
    )
    assert attrs[SpanAttributes.LLM_IS_STREAMING] is True


@pytest.mark.parametrize("control", ["environment", "context"])
def test_content_opt_out_removes_native_mcp_and_audio_payloads(monkeypatch, control):
    from agents.tracing import mcp_tools_span, speech_span
    from opentelemetry import context as otel_context
    from respan_tracing.constants.context_constants import ENABLE_CONTENT_TRACING_KEY

    captured = _capture_spans(monkeypatch)
    set_trace_processors([_RespanTracingProcessor()])
    token = None
    if control == "environment":
        monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")
    else:
        token = otel_context.attach(
            otel_context.set_value(ENABLE_CONTENT_TRACING_KEY, False)
        )
    try:
        with trace("privacy"):
            with mcp_tools_span(server="weather", result=["PRIVATE_CONTENT"]):
                pass
            with speech_span(
                input="PRIVATE_CONTENT", output="PRIVATE_CONTENT", model="tts-1"
            ):
                pass
    finally:
        if token is not None:
            otel_context.detach(token)
    assert len(captured) == 3
    for span in captured:
        assert "PRIVATE_CONTENT" not in json.dumps(dict(span.attributes))
        assert SpanAttributes.TRACELOOP_ENTITY_INPUT not in span.attributes
        assert SpanAttributes.TRACELOOP_ENTITY_OUTPUT not in span.attributes


@pytest.mark.parametrize(
    "style, expected", [("semantic", "transcribe"), ("legacy", "transcription")]
)
def test_audio_model_does_not_change_semantic_operation(monkeypatch, style, expected):
    from agents.tracing.span_data import TranscriptionSpanData
    from respan_tracing.exporters.respan import _span_to_otlp_json

    captured = _capture_spans(monkeypatch)
    monkeypatch.setenv("RESPAN_SPAN_NAME_STYLE", style)
    _otel_emitter.emit_audio(
        _make_span_item(),
        TranscriptionSpanData(input="YXVkaW8=", output="Hello", model="whisper-1"),
    )
    payload = _span_to_otlp_json(captured[0])
    assert payload["name"] == expected
    assert captured[0].attributes[RESPAN_LOG_TYPE] == "transcription"
    assert captured[0].attributes[SpanAttributes.LLM_REQUEST_MODEL] == "whisper-1"
    assert not any(
        a["key"].startswith("respan.internal.") for a in payload["attributes"]
    )


def test_hosted_tool_history_preserves_native_item_id():
    from respan_instrumentation_openai_agents._utils import _format_input_messages

    messages = _format_input_messages(
        [
            {
                "type": "web_search_call",
                "id": "ws_123",
                "action": {"type": "search", "query": "weather"},
            },
            {"type": "web_search_call_output", "call_id": "ws_123", "output": "Sunny"},
        ]
    )
    assert messages[0]["tool_calls"][0]["id"] == "ws_123"
    assert messages[1]["tool_call_id"] == "ws_123"


@pytest.mark.parametrize(
    "payload, expected_input, expected_output",
    [
        (
            {"type": "file_search_call", "id": "fs", "queries": ["hello"]},
            {"queries": ["hello"]},
            None,
        ),
        (
            {
                "type": "code_interpreter_call",
                "id": "ci",
                "code": "print(42)",
                "container_id": "c",
                "outputs": [{"type": "logs", "logs": "42"}],
            },
            {"code": "print(42)", "container_id": "c"},
            [{"type": "logs", "logs": "42"}],
        ),
        (
            {
                "type": "image_generation_call",
                "id": "img",
                "revised_prompt": "cat",
                "result": "BASE64",
            },
            {"revised_prompt": "cat"},
            "BASE64",
        ),
    ],
)
def test_hosted_call_payloads_retain_native_arguments_and_results(
    payload, expected_input, expected_output
):
    call = _otel_emitter._extract_tool_calls([payload])[0]
    assert json.loads(call["function"]["arguments"]) == expected_input
    if expected_output is not None:
        assert call["output"] == expected_output
