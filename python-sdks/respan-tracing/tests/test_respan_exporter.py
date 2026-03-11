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


def test_prepare_spans_drops_openai_chat_child_keeps_wrapper():
    """The pydantic-ai chat wrapper has clean extracted attributes.
    The openai.chat child has many raw gen_ai.* properties.
    We drop the child and keep the wrapper."""

    agent_span = _make_span(
        name="invoke_agent agent",
        span_id=1001,
        attributes={"respan.entity.log_type": "agent"},
        scope_name="pydantic-ai",
    )
    agent_context = agent_span.get_span_context.return_value

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
            "full_request": [
                {
                    "role": "user",
                    "parts": [{"type": "text", "content": "compute 1 + 2"}],
                }
            ],
            "full_response": [
                {
                    "role": "assistant",
                    "parts": [
                        {
                            "type": "tool_call",
                            "id": "tc-1",
                            "name": "add",
                            "arguments": {"a": 1, "b": 2},
                        }
                    ],
                }
            ],
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
            "gen_ai.request.model": "gpt-4o",
            "gen_ai.usage.input_tokens": 11,
            "gen_ai.usage.output_tokens": 7,
            "gen_ai.input.messages": "[...]",
            "gen_ai.output.messages": "[...]",
            SpanAttributes.TRACELOOP_ENTITY_PATH: "calc.agent.openai_chat",
        },
        scope_name="opentelemetry.instrumentation.openai",
    )

    prepared = _prepare_spans_for_export(
        spans=[agent_span, wrapper_span, openai_chat_span]
    )

    assert [s.name for s in prepared] == [
        "invoke_agent agent",
        "chat gpt-4o",
    ]

    kept_chat = prepared[1]
    assert kept_chat.attributes["respan.entity.log_type"] == "chat"
    assert kept_chat.attributes["model"] == "gpt-4o"
    assert kept_chat.attributes["prompt_tokens"] == 11
    assert kept_chat.attributes["full_request"] is not None
    assert kept_chat.attributes["full_response"] is not None


def test_prepare_spans_reparents_openai_chat_grandchildren():
    """Children of a dropped openai.chat span are reparented to the wrapper."""

    wrapper_span = _make_span(
        name="chat gpt-4o",
        span_id=2001,
        attributes={
            "respan.entity.log_type": "chat",
            "full_request": [{"role": "user", "parts": [{"type": "text", "content": "hi"}]}],
            "full_response": [{"role": "assistant", "parts": [{"type": "text", "content": "hello"}]}],
        },
        scope_name="pydantic-ai",
    )
    wrapper_context = wrapper_span.get_span_context.return_value

    openai_chat_span = _make_span(
        name="openai.chat",
        span_id=2002,
        parent=wrapper_context,
        attributes={
            "gen_ai.system": "openai",
            SpanAttributes.TRACELOOP_ENTITY_PATH: "agent.openai_chat",
        },
        scope_name="opentelemetry.instrumentation.openai",
    )
    openai_context = openai_chat_span.get_span_context.return_value

    http_child = _make_span(
        name="http.request",
        span_id=2003,
        parent=openai_context,
        attributes={"http.method": "POST"},
        scope_name="opentelemetry.instrumentation.requests",
    )

    prepared = _prepare_spans_for_export(
        spans=[wrapper_span, openai_chat_span, http_child]
    )

    assert [s.name for s in prepared] == ["chat gpt-4o", "http.request"]
    reparented = prepared[1]
    assert reparented.parent.span_id == wrapper_context.span_id


def test_prepare_spans_keeps_wrapper_without_openai_child():
    """Wrapper without an openai.chat child is kept as-is."""

    wrapper_span = _make_span(
        name="chat anthropic",
        span_id=3001,
        attributes={
            "respan.entity.log_type": "chat",
            "full_request": [{"role": "user", "parts": [{"type": "text", "content": "hi"}]}],
            "full_response": [{"role": "assistant", "parts": [{"type": "text", "content": "hello"}]}],
        },
        scope_name="pydantic-ai",
    )

    prepared = _prepare_spans_for_export(spans=[wrapper_span])

    assert [s.name for s in prepared] == ["chat anthropic"]
