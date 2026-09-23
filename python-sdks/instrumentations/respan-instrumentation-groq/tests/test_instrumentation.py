from __future__ import annotations

import asyncio
import json
import types
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest
import respan_instrumentation_groq._instrumentation as groq_instrumentation
from opentelemetry.attributes import BoundedAttributes
from respan_instrumentation_groq import GroqInstrumentor
from respan_instrumentation_groq._processor import (
    GroqInputSpanProcessor,
    GroqSpanProcessor,
)
from respan_instrumentation_groq._streaming import (
    _AsyncStreamProxy,
    _SyncStreamProxy,
)

_TRANSLATOR = object()


class FakeOpenInferenceGroqInstrumentor:
    pass


class FakeDelegate:
    _translator = _TRANSLATOR
    instances: ClassVar[list[FakeDelegate]] = []

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
    assert processors[0] is instrumentor._input_processor
    assert isinstance(processors[0], GroqInputSpanProcessor)
    assert processors[1] is _TRANSLATOR
    assert processors[2] is instrumentor._processor
    assert isinstance(processors[2], GroqSpanProcessor)
    assert processors[3:] == ("export-processor",)


def test_activate_is_idempotent(monkeypatch) -> None:
    provider = FakeTracerProvider()
    _patch_successful_activation(monkeypatch, provider)

    instrumentor = GroqInstrumentor()
    instrumentor.activate()
    instrumentor.activate()

    assert len(FakeDelegate.instances) == 1
    assert (
        sum(
            isinstance(processor, GroqSpanProcessor)
            for processor in provider._active_span_processor._span_processors
        )
        == 1
    )


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
    assert instrumentor._input_processor is None
    assert instrumentor._processor is None
    assert FakeDelegate.instances[0].deactivated is True
    assert processor not in provider._active_span_processor._span_processors
    assert provider._active_span_processor._span_processors == (
        _TRANSLATOR,
        "export-processor",
    )


def test_processor_strips_groq_aliases_and_omit_values() -> None:
    span = SimpleNamespace(
        instrumentation_scope=SimpleNamespace(
            name="openinference.instrumentation.groq"
        ),
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
        instrumentation_scope=SimpleNamespace(
            name="openinference.instrumentation.groq"
        ),
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

    assert (
        json.loads(span._attributes["gen_ai.completion.0.tool_calls"])
        == completion_tool_calls
    )
    assert (
        span._attributes["gen_ai.completion.0.content"]
        == 'Tool call: get_weather({"city":"Tokyo"})'
    )
    assert span._attributes["gen_ai.completion.0.role"] == "assistant"
    assert (
        json.loads(span._attributes["gen_ai.prompt.1.tool_calls"]) == prompt_tool_calls
    )
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


def test_input_processor_preserves_tool_result_id_and_name() -> None:
    span = SimpleNamespace(
        instrumentation_scope=SimpleNamespace(
            name="openinference.instrumentation.groq"
        ),
        _attributes={
            "llm.input_messages.2.message.role": "tool",
            "llm.input_messages.2.message.name": "get_weather",
            "llm.input_messages.2.message.tool_call_id": "call_weather_1",
        },
    )

    GroqInputSpanProcessor().on_end(span)

    assert span._attributes["gen_ai.prompt.2.name"] == "get_weather"
    assert span._attributes["gen_ai.prompt.2.tool_call_id"] == "call_weather_1"


def test_input_processor_copies_immutable_otel_attributes_before_promotion() -> None:
    span = SimpleNamespace(
        instrumentation_scope=SimpleNamespace(
            name="openinference.instrumentation.groq"
        ),
        _attributes=BoundedAttributes(
            attributes={
                "llm.input_messages.2.message.role": "tool",
                "llm.input_messages.2.message.name": "get_weather",
                "llm.input_messages.2.message.tool_call_id": "call_weather_1",
            },
            immutable=True,
        ),
    )

    GroqInputSpanProcessor().on_end(span)

    assert isinstance(span._attributes, dict)
    assert span._attributes["gen_ai.prompt.2.name"] == "get_weather"
    assert span._attributes["gen_ai.prompt.2.tool_call_id"] == "call_weather_1"


def _stream_chunks():
    usage = SimpleNamespace(prompt_tokens=8, completion_tokens=3, total_tokens=11)
    return [
        SimpleNamespace(
            model="llama-3.1-8b-instant",
            usage=None,
            x_groq=None,
            choices=[
                SimpleNamespace(
                    index=0,
                    finish_reason=None,
                    delta=SimpleNamespace(
                        role="assistant",
                        content="hello ",
                        tool_calls=None,
                    ),
                )
            ],
        ),
        SimpleNamespace(
            model="llama-3.1-8b-instant",
            usage=usage,
            x_groq=None,
            choices=[
                SimpleNamespace(
                    index=0,
                    finish_reason="stop",
                    delta=SimpleNamespace(
                        role=None,
                        content="world",
                        tool_calls=None,
                    ),
                )
            ],
        ),
    ]


def test_sync_stream_proxy_finalizes_with_assembled_content_and_usage() -> None:
    finished = []
    proxy = _SyncStreamProxy(
        iter(_stream_chunks()),
        lambda completion, error: finished.append((completion, error)),
    )

    assert len(list(proxy)) == 2
    assert len(finished) == 1
    completion, error = finished[0]
    assert error is None
    assert completion.model == "llama-3.1-8b-instant"
    assert completion.usage.total_tokens == 11
    assert completion.choices[0].message.content == "hello world"
    assert "object at 0x" not in completion.model_dump_json()


def test_async_stream_proxy_finalizes_with_assembled_content_and_usage() -> None:
    finished = []

    async def stream():
        for chunk in _stream_chunks():
            yield chunk

    async def consume() -> list[object]:
        proxy = _AsyncStreamProxy(
            stream(),
            lambda completion, error: finished.append((completion, error)),
        )
        return [chunk async for chunk in proxy]

    assert len(asyncio.run(consume())) == 2
    completion, error = finished[0]
    assert error is None
    assert completion.choices[0].message.content == "hello world"
    assert completion.usage.prompt_tokens == 8


def test_wrapper_does_not_define_semconv_constants() -> None:
    source = Path(groq_instrumentation.__file__).read_text(encoding="utf-8")

    assert "from opentelemetry.semconv_ai" not in source
    assert "from openinference.semconv.trace" not in source
    assert "from respan_sdk.constants" not in source
    assert "gen_ai." not in source
    assert "llm." not in source
