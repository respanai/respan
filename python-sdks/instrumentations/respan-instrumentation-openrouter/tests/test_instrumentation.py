from __future__ import annotations

import json
import logging
from types import SimpleNamespace

import pytest
from opentelemetry.semconv_ai import SpanAttributes
from respan_sdk.constants.llm_logging import LOG_TYPE_CHAT
from respan_sdk.constants.span_attributes import (
    RESPAN_LOG_TYPE,
    RESPAN_SPAN_TOOL_CALLS,
    RESPAN_SPAN_TOOLS,
)
from respan_tracing.core.tracer import RespanTracer

from respan_instrumentation_openrouter import OpenRouterInstrumentor
from respan_instrumentation_openrouter import _instrumentation
from respan_instrumentation_openrouter._constants import OPENROUTER_SYSTEM_NAME
from respan_instrumentation_openrouter._processor import OpenRouterSpanProcessor


class FakeOpenAIInstrumentor:
    created: list["FakeOpenAIInstrumentor"] = []

    def __init__(self) -> None:
        self.activated = False
        self.deactivated = False
        self._is_instrumented = True
        self.__class__.created.append(self)

    def activate(self) -> None:
        self.activated = True

    def deactivate(self) -> None:
        self.deactivated = True


class FakeTracerProvider:
    def __init__(self, processors=()) -> None:
        self._active_span_processor = SimpleNamespace(_span_processors=processors)

    def add_span_processor(self, processor) -> None:
        self._active_span_processor._span_processors = (
            *self._active_span_processor._span_processors,
            processor,
        )


@pytest.fixture(autouse=True)
def reset_fakes(monkeypatch):
    FakeOpenAIInstrumentor.created.clear()
    RespanTracer.reset_instance()
    monkeypatch.setattr(
        _instrumentation,
        "OpenAIInstrumentor",
        FakeOpenAIInstrumentor,
    )
    yield
    RespanTracer.reset_instance()


def test_name_is_openrouter() -> None:
    assert OpenRouterInstrumentor.name == "openrouter"
    assert OpenRouterInstrumentor().name == "openrouter"


def test_activate_delegates_to_openai_and_inserts_processor_before_exporter(
    monkeypatch,
) -> None:
    provider = FakeTracerProvider(processors=("exporter",))
    monkeypatch.setattr(
        _instrumentation.trace,
        "get_tracer_provider",
        lambda: provider,
    )

    instrumentor = OpenRouterInstrumentor()
    instrumentor.activate()

    assert instrumentor._is_instrumented is True
    assert len(FakeOpenAIInstrumentor.created) == 1
    assert FakeOpenAIInstrumentor.created[0].activated is True

    processors = provider._active_span_processor._span_processors
    assert isinstance(processors[0], OpenRouterSpanProcessor)
    assert processors[0] is instrumentor._processor
    assert processors[1:] == ("exporter",)

    instrumentor.deactivate()

    assert FakeOpenAIInstrumentor.created[0].deactivated is True
    assert instrumentor._is_instrumented is False
    assert provider._active_span_processor._span_processors == ("exporter",)


def test_activate_is_idempotent(monkeypatch) -> None:
    provider = FakeTracerProvider(processors=("exporter",))
    monkeypatch.setattr(
        _instrumentation.trace,
        "get_tracer_provider",
        lambda: provider,
    )

    instrumentor = OpenRouterInstrumentor()
    instrumentor.activate()
    instrumentor.activate()

    assert len(FakeOpenAIInstrumentor.created) == 1
    assert sum(
        isinstance(processor, OpenRouterSpanProcessor)
        for processor in provider._active_span_processor._span_processors
    ) == 1


def test_activate_skips_when_respan_tracing_is_disabled(monkeypatch, caplog) -> None:
    provider = FakeTracerProvider(processors=("exporter",))
    monkeypatch.setattr(
        _instrumentation.trace,
        "get_tracer_provider",
        lambda: provider,
    )
    RespanTracer(is_enabled=False)

    instrumentor = OpenRouterInstrumentor()
    with caplog.at_level(logging.INFO):
        instrumentor.activate()

    assert FakeOpenAIInstrumentor.created == []
    assert instrumentor._is_instrumented is False
    assert (
        "OpenRouter instrumentation skipped because Respan tracing is disabled"
        in caplog.text
    )


def test_activate_skips_when_openai_delegate_does_not_instrument(monkeypatch) -> None:
    class MissingOpenAIInstrumentor(FakeOpenAIInstrumentor):
        created: list["MissingOpenAIInstrumentor"] = []

        def activate(self) -> None:
            self.activated = True
            self._is_instrumented = False

    provider = FakeTracerProvider(processors=("exporter",))
    monkeypatch.setattr(
        _instrumentation,
        "OpenAIInstrumentor",
        MissingOpenAIInstrumentor,
    )
    monkeypatch.setattr(
        _instrumentation.trace,
        "get_tracer_provider",
        lambda: provider,
    )

    instrumentor = OpenRouterInstrumentor()
    instrumentor.activate()

    assert instrumentor._is_instrumented is False
    assert instrumentor._processor is None
    assert provider._active_span_processor._span_processors == ("exporter",)


def test_processor_rewrites_openai_span_to_openrouter_contract() -> None:
    span = SimpleNamespace(
        instrumentation_scope=SimpleNamespace(
            name="opentelemetry.instrumentation.openai"
        ),
        _attributes={
            SpanAttributes.LLM_SYSTEM: "openai",
            SpanAttributes.LLM_REQUEST_MODEL: "openai/gpt-4o-mini",
            f"{SpanAttributes.LLM_PROMPTS}.0.role": "user",
            f"{SpanAttributes.LLM_PROMPTS}.0.content": "Hello",
        },
    )

    OpenRouterSpanProcessor().on_end(span)

    assert span._attributes[SpanAttributes.LLM_SYSTEM] == OPENROUTER_SYSTEM_NAME
    assert span._attributes[SpanAttributes.LLM_REQUEST_TYPE] == "chat"
    assert span._attributes[RESPAN_LOG_TYPE] == LOG_TYPE_CHAT


def test_processor_can_require_openrouter_url_marker() -> None:
    span_without_marker = SimpleNamespace(
        instrumentation_scope=SimpleNamespace(
            name="opentelemetry.instrumentation.openai"
        ),
        _attributes={SpanAttributes.LLM_SYSTEM: "openai"},
    )
    processor = OpenRouterSpanProcessor(normalize_all_openai_spans=False)

    processor.on_end(span_without_marker)

    assert span_without_marker._attributes[SpanAttributes.LLM_SYSTEM] == "openai"

    span_with_marker = SimpleNamespace(
        instrumentation_scope=SimpleNamespace(
            name="opentelemetry.instrumentation.openai"
        ),
        _attributes={
            SpanAttributes.LLM_SYSTEM: "openai",
            "url.full": "https://openrouter.ai/api/v1/chat/completions",
        },
    )

    processor.on_end(span_with_marker)

    assert span_with_marker._attributes[SpanAttributes.LLM_SYSTEM] == "openrouter"


def test_processor_normalizes_tool_attrs_and_removes_off_contract_aliases() -> None:
    tool_calls = [
        {
            "id": "call_123",
            "type": "function",
            "function": {"name": "lookup", "arguments": '{"city":"Tokyo"}'},
        }
    ]
    span = SimpleNamespace(
        instrumentation_scope=SimpleNamespace(
            name="opentelemetry.instrumentation.openai"
        ),
        _attributes={
            SpanAttributes.LLM_SYSTEM: "openai",
            SpanAttributes.LLM_REQUEST_MODEL: "openai/gpt-4o-mini",
            RESPAN_SPAN_TOOLS: [{"type": "function"}],
            RESPAN_SPAN_TOOL_CALLS: tool_calls,
            "tools": [{"type": "function"}],
            "tool_calls": tool_calls,
            "model": "openai/gpt-4o-mini",
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_request_tokens": 15,
            f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls.0.id": "call_123",
            f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls.0.type": "function",
            f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls.0.function.name": (
                "lookup"
            ),
            f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls.0.function.arguments": (
                '{"city":"Tokyo"}'
            ),
        },
    )

    OpenRouterSpanProcessor().on_end(span)

    attrs = span._attributes
    assert json.loads(attrs[SpanAttributes.LLM_REQUEST_FUNCTIONS]) == [
        {"type": "function"}
    ]
    assert json.loads(attrs[f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls"]) == (
        tool_calls
    )
    assert (
        attrs[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"]
        == 'Tool call: lookup({"city":"Tokyo"})'
    )
    assert attrs[f"{SpanAttributes.LLM_COMPLETIONS}.0.role"] == "assistant"

    for alias in (
        RESPAN_SPAN_TOOLS,
        RESPAN_SPAN_TOOL_CALLS,
        "tools",
        "tool_calls",
        "model",
        "prompt_tokens",
        "completion_tokens",
        "total_request_tokens",
    ):
        assert alias not in attrs
    assert all(".tool_calls." not in key for key in attrs)


def test_processor_promotes_new_gen_ai_output_messages_to_canonical_fields() -> None:
    output_messages = [
        {
            "role": "assistant",
            "parts": [
                {
                    "type": "tool_call",
                    "name": "lookup",
                    "id": "call_123",
                    "arguments": {"city": "Tokyo"},
                }
            ],
            "finish_reason": "tool_call",
        }
    ]
    tool_definitions = [
        {
            "type": "function",
            "name": "lookup",
            "description": "Lookup a city.",
            "parameters": {"type": "object"},
        }
    ]
    span = SimpleNamespace(
        instrumentation_scope=SimpleNamespace(
            name="opentelemetry.instrumentation.openai.v1"
        ),
        _attributes={
            SpanAttributes.LLM_SYSTEM: "openai",
            SpanAttributes.LLM_REQUEST_MODEL: "openai/gpt-4o-mini",
            "gen_ai.output.messages": json.dumps(output_messages),
            "gen_ai.tool.definitions": json.dumps(tool_definitions),
        },
    )

    OpenRouterSpanProcessor().on_end(span)

    attrs = span._attributes
    assert attrs[f"{SpanAttributes.LLM_COMPLETIONS}.0.role"] == "assistant"
    assert (
        attrs[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"]
        == 'Tool call: lookup({"city": "Tokyo"})'
    )
    assert json.loads(attrs[f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls"]) == [
        {
            "type": "function",
            "function": {"name": "lookup", "arguments": '{"city": "Tokyo"}'},
            "id": "call_123",
        }
    ]
    assert json.loads(attrs[SpanAttributes.LLM_REQUEST_FUNCTIONS]) == [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "Lookup a city.",
                "parameters": {"type": "object"},
            },
        }
    ]


def test_processor_ignores_non_openai_spans() -> None:
    span = SimpleNamespace(
        instrumentation_scope=SimpleNamespace(name="other.instrumentation"),
        _attributes={SpanAttributes.LLM_SYSTEM: "anthropic", "model": "claude"},
    )

    OpenRouterSpanProcessor().on_end(span)

    assert span._attributes == {
        SpanAttributes.LLM_SYSTEM: "anthropic",
        "model": "claude",
    }
