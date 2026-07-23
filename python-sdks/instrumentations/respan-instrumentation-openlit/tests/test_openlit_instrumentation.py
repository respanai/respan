from __future__ import annotations

import json
from types import SimpleNamespace

from opentelemetry.semconv_ai import SpanAttributes

from respan_instrumentation_openlit import OpenLITInstrumentor
from respan_instrumentation_openlit._processor import translate_openlit_span
from respan_sdk.constants.span_attributes import RESPAN_LOG_METHOD, RESPAN_LOG_TYPE


class FakeSpan:
    def __init__(
        self, attributes: dict[str, object], name: str = "openai.chat"
    ) -> None:
        self._attributes = attributes
        self.name = name
        self.instrumentation_scope = SimpleNamespace(name="openlit")


def test_chat_span_is_normalized_to_canonical_contract() -> None:
    span = FakeSpan(
        {
            "gen_ai.operation.name": "chat",
            "gen_ai.provider.name": "openai",
            "gen_ai.request.model": "gpt-4.1-mini",
            "gen_ai.input.messages": json.dumps([{"role": "user", "content": "hello"}]),
            "gen_ai.output.messages": json.dumps(
                [{"role": "assistant", "content": "hi"}]
            ),
            "gen_ai.tool.definitions": json.dumps(
                [{"type": "function", "function": {"name": "weather"}}]
            ),
            "gen_ai.usage.input_tokens": 4,
            "gen_ai.usage.output_tokens": 2,
            "gen_ai.usage.total_tokens": 6,
            "model": "forbidden-alias",
            "tool_calls": "forbidden-alias",
        }
    )

    assert translate_openlit_span(span, capture_content=True)
    attrs = span._attributes
    assert attrs[RESPAN_LOG_TYPE] == "chat"
    assert attrs[RESPAN_LOG_METHOD] == "tracing_integration"
    assert attrs[SpanAttributes.LLM_SYSTEM] == "openai"
    assert attrs[f"{SpanAttributes.LLM_PROMPTS}.0.role"] == "user"
    assert attrs[f"{SpanAttributes.LLM_PROMPTS}.0.content"] == "hello"
    assert attrs[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"] == "hi"
    assert (
        json.loads(attrs[SpanAttributes.LLM_REQUEST_FUNCTIONS])[0]["function"]["name"]
        == "weather"
    )
    assert attrs[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] == 4
    assert attrs[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS] == 2
    assert attrs[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] == 6
    assert "traceloop.span.kind" not in attrs
    assert "model" not in attrs
    assert "tool_calls" not in attrs


def test_tool_span_uses_entity_input_and_output() -> None:
    span = FakeSpan(
        {
            "gen_ai.operation.name": "execute_tool",
            "gen_ai.tool.name": "weather",
            "gen_ai.tool.call.arguments": '{"city":"Paris"}',
            "gen_ai.tool.call.result": '{"temperature":18}',
        },
        name="weather.execute_tool",
    )

    translate_openlit_span(span, capture_content=True)
    attrs = span._attributes
    assert attrs[RESPAN_LOG_TYPE] == "tool"
    assert attrs[SpanAttributes.TRACELOOP_ENTITY_NAME] == "weather"
    assert json.loads(attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT]) == {"city": "Paris"}
    assert json.loads(attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]) == {
        "temperature": 18
    }
    assert not any(key.endswith("tool_calls") for key in attrs)


def test_capture_content_false_removes_sensitive_payloads_but_keeps_usage() -> None:
    span = FakeSpan(
        {
            "gen_ai.operation.name": "chat",
            "gen_ai.input.messages": '[{"role":"user","content":"secret"}]',
            "gen_ai.output.messages": '[{"role":"assistant","content":"secret"}]',
            "gen_ai.tool.definitions": '[{"name":"secret"}]',
            "gen_ai.usage.input_tokens": 8,
            "gen_ai.usage.output_tokens": 3,
        }
    )

    translate_openlit_span(span, capture_content=False)
    attrs = span._attributes
    assert "secret" not in json.dumps(attrs)
    assert attrs[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] == 11
    assert RESPAN_LOG_TYPE in attrs


def test_non_openlit_span_is_ignored() -> None:
    span = FakeSpan({"gen_ai.operation.name": "chat"})
    span.instrumentation_scope = SimpleNamespace(name="another.instrumentation")
    before = dict(span._attributes)
    assert not translate_openlit_span(span, capture_content=True)
    assert span._attributes == before


def test_lifecycle_is_idempotent_and_only_uninstruments_owned_hooks(
    monkeypatch,
) -> None:
    import respan_instrumentation_openlit._instrumentation as lifecycle

    class UpstreamInstrumentor:
        _is_instrumented_by_opentelemetry = False
        uninstrument_calls = 0

        def uninstrument(self) -> None:
            self._is_instrumented_by_opentelemetry = False
            self.uninstrument_calls += 1

    class Active:
        _span_processors: tuple[object, ...] = ()

    class Provider:
        _active_span_processor = Active()

    upstream = UpstreamInstrumentor()
    init_calls = 0

    def init(**kwargs) -> None:
        nonlocal init_calls
        assert kwargs["disable_metrics"] is True
        init_calls += 1
        upstream._is_instrumented_by_opentelemetry = True

    provider = Provider()
    monkeypatch.setattr(lifecycle, "_instrumentors", lambda: {"openai": upstream})
    original_import_module = lifecycle.importlib.import_module

    def import_module(name: str):
        if name == "openlit":
            return SimpleNamespace(init=init)
        return original_import_module(name)

    monkeypatch.setattr(lifecycle.importlib, "import_module", import_module)
    monkeypatch.setattr(lifecycle.trace, "get_tracer_provider", lambda: provider)
    monkeypatch.setattr(lifecycle, "_REFCOUNT", 0)
    monkeypatch.setattr(lifecycle, "_PROCESSOR", None)
    monkeypatch.setattr(lifecycle, "_PROVIDER", None)
    monkeypatch.setattr(lifecycle, "_OWNED_INSTRUMENTORS", [])

    instrumentor = OpenLITInstrumentor()
    instrumentor.activate()
    instrumentor.activate()
    assert init_calls == 1
    assert len(provider._active_span_processor._span_processors) == 1

    instrumentor.deactivate()
    instrumentor.deactivate()
    assert upstream.uninstrument_calls == 1
    assert provider._active_span_processor._span_processors == ()
