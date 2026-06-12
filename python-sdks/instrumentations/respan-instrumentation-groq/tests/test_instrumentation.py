from __future__ import annotations

import types
from pathlib import Path
from types import SimpleNamespace

import pytest

import respan_instrumentation_groq._instrumentation as groq_instrumentation
from respan_instrumentation_groq import GroqInstrumentor
from respan_instrumentation_groq._processor import GroqSpanProcessor


_TRANSLATOR = object()


class FakeOpenInferenceGroqInstrumentor:
    pass


class FakeDelegate:
    _translator = _TRANSLATOR
    instances: list["FakeDelegate"] = []

    def __init__(self, instrumentor_class: type, **kwargs) -> None:
        self.instrumentor_class = instrumentor_class
        self.kwargs = kwargs
        self.activated = False
        self.deactivated = False
        FakeDelegate.instances.append(self)

    def activate(self) -> None:
        self.activated = True

    def deactivate(self) -> None:
        self.deactivated = True


class FakeTracerProvider:
    def __init__(self) -> None:
        self._active_span_processor = SimpleNamespace(
            _span_processors=(_TRANSLATOR, "export-processor")
        )

    def add_span_processor(self, processor) -> None:
        self._active_span_processor._span_processors = (
            *self._active_span_processor._span_processors,
            processor,
        )


class FakeGroqOmit:
    pass


FakeGroqOmit.__module__ = "groq._types"
FakeGroqOmit.__name__ = "Omit"


@pytest.fixture(autouse=True)
def reset_fakes(monkeypatch):
    FakeDelegate.instances.clear()
    monkeypatch.setattr(
        groq_instrumentation.RespanTracer,
        "_instance",
        SimpleNamespace(is_enabled=True),
    )


def _patch_successful_activation(monkeypatch, provider: FakeTracerProvider):
    fake_module = types.SimpleNamespace(
        GroqInstrumentor=FakeOpenInferenceGroqInstrumentor
    )
    monkeypatch.setattr(
        groq_instrumentation.importlib,
        "import_module",
        lambda module_name: fake_module,
    )
    monkeypatch.setattr(
        groq_instrumentation,
        "OpenInferenceInstrumentor",
        FakeDelegate,
    )
    monkeypatch.setattr(
        groq_instrumentation.trace,
        "get_tracer_provider",
        lambda: provider,
    )


def test_name_is_groq() -> None:
    assert GroqInstrumentor.name == "groq"
    assert GroqInstrumentor().name == "groq"


def test_activate_delegates_to_openinference_groq(monkeypatch) -> None:
    provider = FakeTracerProvider()
    _patch_successful_activation(monkeypatch, provider)

    instrumentor = GroqInstrumentor(capture_content=True)
    instrumentor.activate()

    assert instrumentor._is_instrumented is True
    assert len(FakeDelegate.instances) == 1
    delegate = FakeDelegate.instances[0]
    assert delegate.instrumentor_class is FakeOpenInferenceGroqInstrumentor
    assert delegate.kwargs == {"capture_content": True}
    assert delegate.activated is True

    processors = provider._active_span_processor._span_processors
    assert processors[0] is _TRANSLATOR
    assert processors[1] is instrumentor._processor
    assert isinstance(processors[1], GroqSpanProcessor)
    assert processors[2:] == ("export-processor",)


def test_activate_is_idempotent(monkeypatch) -> None:
    provider = FakeTracerProvider()
    _patch_successful_activation(monkeypatch, provider)

    instrumentor = GroqInstrumentor()
    instrumentor.activate()
    instrumentor.activate()

    assert len(FakeDelegate.instances) == 1
    assert sum(
        isinstance(processor, GroqSpanProcessor)
        for processor in provider._active_span_processor._span_processors
    ) == 1


def test_activate_skips_when_respan_tracing_is_disabled(monkeypatch) -> None:
    monkeypatch.setattr(
        groq_instrumentation.RespanTracer,
        "_instance",
        SimpleNamespace(is_enabled=False),
    )

    instrumentor = GroqInstrumentor()
    instrumentor.activate()

    assert instrumentor._is_instrumented is False
    assert FakeDelegate.instances == []


def test_activate_skips_when_openinference_groq_is_missing(monkeypatch) -> None:
    def raise_import_error(module_name: str):
        raise ImportError(module_name)

    monkeypatch.setattr(
        groq_instrumentation.importlib,
        "import_module",
        raise_import_error,
    )
    monkeypatch.setattr(
        groq_instrumentation,
        "OpenInferenceInstrumentor",
        FakeDelegate,
    )

    instrumentor = GroqInstrumentor()
    instrumentor.activate()

    assert instrumentor._is_instrumented is False
    assert FakeDelegate.instances == []


def test_deactivate_unregisters_processor_and_delegate(monkeypatch) -> None:
    provider = FakeTracerProvider()
    _patch_successful_activation(monkeypatch, provider)

    instrumentor = GroqInstrumentor()
    instrumentor.activate()
    processor = instrumentor._processor
    instrumentor.deactivate()

    assert instrumentor._is_instrumented is False
    assert instrumentor._delegate is None
    assert instrumentor._processor is None
    assert FakeDelegate.instances[0].deactivated is True
    assert processor not in provider._active_span_processor._span_processors
    assert provider._active_span_processor._span_processors == (
        _TRANSLATOR,
        "export-processor",
    )


def test_processor_strips_groq_aliases_and_omit_values() -> None:
    span = SimpleNamespace(
        instrumentation_scope=SimpleNamespace(name="openinference.instrumentation.groq"),
        _attributes={
            "gen_ai.request.model": "gpt-4.1-nano",
            "llm.request.functions": "[]",
            "model": "gpt-4.1-nano",
            "prompt_tokens": 10,
            "completion_tokens": 3,
            "total_request_tokens": 13,
            "tools": "[]",
            "tool_calls": "[]",
            "respan.span.tools": "[]",
            "gen_ai.request.top_p": FakeGroqOmit(),
            "llm.chat.stop_sequences": "<groq.Omit object at 0xabc>",
        },
    )

    GroqSpanProcessor().on_end(span)

    assert span._attributes == {
        "gen_ai.request.model": "gpt-4.1-nano",
        "llm.request.functions": "[]",
    }


def test_processor_normalizes_tool_call_attrs_for_backend() -> None:
    completion_tool_calls = [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "get_weather", "arguments": '{"city":"Tokyo"}'},
        }
    ]
    prompt_tool_calls = [
        {
            "id": "call_history",
            "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'},
        }
    ]
    span = SimpleNamespace(
        instrumentation_scope=SimpleNamespace(name="openinference.instrumentation.groq"),
        _attributes={
            "gen_ai.completion.0.role": "assistant",
            "gen_ai.completion.0.tool_calls": completion_tool_calls,
            "gen_ai.completion.0.tool_calls.0.id": "call_1",
            "gen_ai.completion.0.tool_calls.0.type": "function",
            "gen_ai.completion.0.tool_calls.0.function.name": "get_weather",
            "gen_ai.completion.0.tool_calls.0.function.arguments": '{"city":"Tokyo"}',
            "gen_ai.prompt.1.tool_calls.0.id": "call_history",
            "gen_ai.prompt.1.tool_calls.0.function.name": "get_weather",
            "gen_ai.prompt.1.tool_calls.0.function.arguments": '{"city":"Paris"}',
        },
    )

    GroqSpanProcessor().on_end(span)

    assert span._attributes["gen_ai.completion.0.tool_calls"] == completion_tool_calls
    assert (
        span._attributes["gen_ai.completion.0.content"]
        == 'Tool call: get_weather({"city":"Tokyo"})'
    )
    assert span._attributes["gen_ai.completion.0.role"] == "assistant"
    assert span._attributes["gen_ai.prompt.1.tool_calls"] == prompt_tool_calls
    assert (
        span._attributes["gen_ai.prompt.1.content"]
        == 'Tool call: get_weather({"city":"Paris"})'
    )
    assert span._attributes["gen_ai.prompt.1.role"] == "assistant"
    assert all(".tool_calls." not in key for key in span._attributes)


def test_processor_ignores_non_groq_spans() -> None:
    span = SimpleNamespace(
        instrumentation_scope=SimpleNamespace(name="other.instrumentation"),
        _attributes={"tools": "[]"},
    )

    GroqSpanProcessor().on_end(span)

    assert span._attributes == {"tools": "[]"}


def test_processor_detects_groq_omit_without_scope() -> None:
    span = SimpleNamespace(
        _attributes={
            "tools": "[]",
            "gen_ai.request.top_p": "<groq.Omit object at 0xabc>",
            "gen_ai.request.model": "gpt-4.1-nano",
        },
    )

    GroqSpanProcessor().on_end(span)

    assert span._attributes == {"gen_ai.request.model": "gpt-4.1-nano"}


def test_wrapper_does_not_define_semconv_constants() -> None:
    source = Path(groq_instrumentation.__file__).read_text(encoding="utf-8")

    assert "from opentelemetry.semconv_ai" not in source
    assert "from openinference.semconv.trace" not in source
    assert "from respan_sdk.constants" not in source
    assert "gen_ai." not in source
    assert "llm." not in source
