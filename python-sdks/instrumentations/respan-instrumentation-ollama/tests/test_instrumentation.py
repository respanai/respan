"""Lifecycle tests use the installed native SDK and SDK tracer provider."""

import ollama
import pytest
from opentelemetry import context
from opentelemetry.sdk.trace import TracerProvider
from respan_instrumentation_ollama import OllamaInstrumentor
from respan_instrumentation_ollama import _instrumentation as I


def test_native_identity_owned_lifecycle_shared_conflict_and_foreign():
    pr = TracerProvider()
    original = ollama.Client._request
    detach = context.detach
    a = OllamaInstrumentor(tracer_provider=pr)
    b = OllamaInstrumentor(tracer_provider=pr)
    a.activate()
    owned = ollama.Client._request
    a.activate()
    b.activate()
    assert owned is ollama.Client._request
    with pytest.raises(ValueError):
        OllamaInstrumentor(capture_content=False, tracer_provider=pr).activate()
    a.deactivate()
    assert ollama.Client._request is owned

    def foreign(*a, **kw):
        return owned(*a, **kw)

    ollama.Client._request = foreign
    b.deactivate()
    assert ollama.Client._request is foreign
    assert context.detach is detach and not I._OBSERVERS
    ollama.Client._request = original
    a.activate()
    a.deactivate()
    assert ollama.Client._request is original
    pr.shutdown()


def test_activation_partial_rollback_removes_owned_processor(monkeypatch):
    pr = TracerProvider()
    original = ollama.Client._request

    def fail(*a, **kw):
        raise ValueError("patch failure")

    monkeypatch.setattr(I, "_wrap", fail)
    a = OllamaInstrumentor(tracer_provider=pr)
    a.activate()
    assert (
        not a._is_instrumented
        and ollama.Client._request is original
        and not I._OBSERVERS
    )
    assert not pr._active_span_processor._span_processors
    pr.shutdown()


def test_absent_optional_sdk_has_no_partial_activation(monkeypatch):
    pr = TracerProvider()
    original = I.importlib.import_module

    def missing(name, *a, **kw):
        if name == "ollama":
            raise ImportError("absent")
        return original(name, *a, **kw)

    monkeypatch.setattr(I.importlib, "import_module", missing)
    a = OllamaInstrumentor(tracer_provider=pr)
    a.activate()
    assert not a._is_instrumented and not I._PATCHES
    pr.shutdown()


def test_native_processor_add_partial_failure_removes_only_owned(monkeypatch):
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    pr = TracerProvider()
    foreign = SimpleSpanProcessor(InMemorySpanExporter())
    pr.add_span_processor(foreign)
    original = pr.add_span_processor

    def add_then_fail(observer):
        original(observer)
        raise ValueError("partial native processor failure")

    monkeypatch.setattr(pr, "add_span_processor", add_then_fail)
    a = OllamaInstrumentor(tracer_provider=pr)
    a.activate()
    assert not a._is_instrumented and pr._active_span_processor._span_processors == (
        foreign,
    )
    assert not I._OBSERVERS and not I._PATCHES
    pr.shutdown()
