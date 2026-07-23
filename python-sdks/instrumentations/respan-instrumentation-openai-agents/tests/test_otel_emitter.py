"""Unit tests for OpenAI Agents OTEL emitter contract attrs."""

import json
from types import SimpleNamespace

from opentelemetry.semconv_ai import SpanAttributes

from respan_sdk.constants.span_attributes import RESPAN_LOG_TYPE

from respan_instrumentation_openai_agents import _otel_emitter


_BANNED_ALIASES = {
    "respan.span.tool_calls",
    "respan.span.tools",
    "respan.span.handoffs",
    "tool_calls",
    "tools",
    "span_tools",
    "has_tool_calls",
    "parallel_tool_calls",
}


def _make_span_item() -> SimpleNamespace:
    return SimpleNamespace(
        trace_id="trace_123",
        span_id="span_456",
        parent_id=None,
        started_at=None,
        ended_at=None,
        error=None,
    )


def _capture_attrs(monkeypatch):
    captured = {}

    def _fake_build_readable_span(**kwargs):
        captured.update(kwargs)
        return kwargs

    monkeypatch.setattr(_otel_emitter, "build_readable_span", _fake_build_readable_span)
    monkeypatch.setattr(_otel_emitter, "inject_span", lambda span: None)
    return captured


def test_emit_response_uses_chat_contract_tool_attrs(monkeypatch):
    captured = _capture_attrs(monkeypatch)
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
        usage=SimpleNamespace(input_tokens=12, output_tokens=4),
    )
    span_data = SimpleNamespace(response=response, input="What is the weather in NYC?")

    _otel_emitter.emit_response(_make_span_item(), span_data)

    attrs = captured["attributes"]
    assert attrs[RESPAN_LOG_TYPE] == "chat"
    assert attrs[SpanAttributes.LLM_REQUEST_TYPE] == "chat"
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
    assert attrs[f"{SpanAttributes.LLM_COMPLETIONS}.0.role"] == "assistant"
    assert _BANNED_ALIASES.isdisjoint(attrs)


def test_emit_generation_extracts_tool_calls(monkeypatch):
    captured = _capture_attrs(monkeypatch)
    span_data = SimpleNamespace(
        input=[{"role": "user", "content": "Use the tool"}],
        output=[
            {
                "type": "function_call",
                "call_id": "call_2",
                "name": "search_docs",
                "arguments": '{"query":"otel"}',
            }
        ],
        model="gpt-4o",
        usage={"prompt_tokens": 8, "completion_tokens": 2},
    )

    _otel_emitter.emit_generation(_make_span_item(), span_data)

    attrs = captured["attributes"]
    assert attrs[RESPAN_LOG_TYPE] == "chat"
    assert attrs[SpanAttributes.LLM_REQUEST_TYPE] == "chat"
    assert json.loads(attrs[f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls"]) == [
        {
            "id": "call_2",
            "type": "function",
            "function": {
                "name": "search_docs",
                "arguments": '{"query":"otel"}',
            },
        }
    ]
    assert _BANNED_ALIASES.isdisjoint(attrs)
