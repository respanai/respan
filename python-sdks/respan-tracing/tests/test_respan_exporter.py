from types import SimpleNamespace
from unittest.mock import Mock

from opentelemetry.semconv_ai import SpanAttributes

from respan_tracing.exporters.respan import _prepare_spans_for_export


def _make_span(
    *,
    name: str,
    span_id: int,
    trace_id: int = 1,
    parent: SimpleNamespace | None = None,
    attributes: dict | None = None,
    scope_name: str = "test-scope",
) -> Mock:
    span = Mock()
    span.name = name
    span.parent = parent
    span._parent = parent
    span.attributes = attributes or {}
    span.instrumentation_scope = SimpleNamespace(name=scope_name, version="1.0.0")
    span.get_span_context.return_value = SimpleNamespace(
        trace_id=trace_id,
        span_id=span_id,
    )
    return span


def test_prepare_spans_for_export_merges_pydantic_chat_wrapper_into_openai_chat():
    agent_span = _make_span(
        name="invoke_agent agent",
        span_id=1001,
        attributes={"respan.entity.log_type": "agent"},
        scope_name="pydantic-ai",
    )
    agent_context = agent_span.get_span_context.return_value

    wrapper_full_request = [
        {
            "role": "user",
            "parts": [
                {
                    "type": "text",
                    "content": "use add to compute 1 + 2",
                }
            ],
        }
    ]
    wrapper_full_response = [
        {
            "role": "assistant",
            "parts": [
                {
                    "type": "tool_call",
                    "id": "tool-call-1",
                    "name": "add",
                    "arguments": {"a": 1, "b": 2},
                }
            ],
        }
    ]
    wrapper_span = _make_span(
        name="chat gpt-4o",
        span_id=1002,
        parent=agent_context,
        attributes={
            "respan.entity.log_type": "chat",
            "respan.entity.log_method": "tracing_integration",
            "model": "gpt-4o",
            "prompt_tokens": 11,
            "completion_tokens": 7,
            "total_request_tokens": 18,
            "full_request": wrapper_full_request,
            "full_response": wrapper_full_response,
        },
        scope_name="pydantic-ai",
    )
    wrapper_context = wrapper_span.get_span_context.return_value

    openai_chat_span = _make_span(
        name="openai.chat",
        span_id=1003,
        parent=wrapper_context,
        attributes={
            "gen_ai.system": "openai",
            SpanAttributes.TRACELOOP_ENTITY_PATH: "calc.agent.openai_chat",
        },
        scope_name="opentelemetry.instrumentation.openai",
    )

    prepared_spans = _prepare_spans_for_export(
        spans=[agent_span, wrapper_span, openai_chat_span]
    )

    assert [span.name for span in prepared_spans] == [
        "invoke_agent agent",
        "openai.chat",
    ]

    merged_chat_span = prepared_spans[1]
    assert merged_chat_span.parent.span_id == agent_context.span_id
    assert merged_chat_span.attributes["full_request"] == wrapper_full_request
    assert merged_chat_span.attributes["full_response"] == wrapper_full_response
    assert merged_chat_span.attributes["tool_calls"] == [
        {
            "type": "tool_call",
            "id": "tool-call-1",
            "name": "add",
            "arguments": {"a": 1, "b": 2},
        }
    ]
    assert merged_chat_span.attributes["has_tool_calls"] is True
    assert merged_chat_span.attributes["span_tools"] == ["add"]
    assert merged_chat_span.attributes["respan.entity.log_type"] == "chat"
    assert merged_chat_span.attributes["respan.entity.log_method"] == "tracing_integration"
    assert merged_chat_span.attributes["gen_ai.system"] == "openai"


def test_prepare_spans_for_export_keeps_wrapper_without_openai_chat_child():
    agent_span = _make_span(
        name="invoke_agent agent",
        span_id=2001,
        attributes={"respan.entity.log_type": "agent"},
        scope_name="pydantic-ai",
    )
    agent_context = agent_span.get_span_context.return_value

    wrapper_span = _make_span(
        name="chat gpt-4o",
        span_id=2002,
        parent=agent_context,
        attributes={
            "respan.entity.log_type": "chat",
            "full_request": [{"role": "user", "parts": [{"type": "text", "content": "hi"}]}],
            "full_response": [{"role": "assistant", "parts": [{"type": "text", "content": "hello"}]}],
        },
        scope_name="pydantic-ai",
    )
    wrapper_context = wrapper_span.get_span_context.return_value

    unrelated_child_span = _make_span(
        name="http.request",
        span_id=2003,
        parent=wrapper_context,
        attributes={"http.method": "POST"},
        scope_name="opentelemetry.instrumentation.requests",
    )

    prepared_spans = _prepare_spans_for_export(
        spans=[agent_span, wrapper_span, unrelated_child_span]
    )

    assert [span.name for span in prepared_spans] == [
        "invoke_agent agent",
        "chat gpt-4o",
        "http.request",
    ]
