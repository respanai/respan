"""Lifecycle and duplicate-prevention tests for the generic wrapper."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from opentelemetry.sdk.trace import SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from respan_instrumentation_openinference._instrumentation import (
    OpenInferenceInstrumentor,
)
from respan_instrumentation_openinference._translator import OpenInferenceTranslator


class FakeTracerProvider:
    def __init__(self):
        self._active_span_processor = SimpleNamespace(
            _span_processors=("export-a", "export-b")
        )

    def add_span_processor(self, processor):
        self._active_span_processor._span_processors = (
            *self._active_span_processor._span_processors,
            processor,
        )


@pytest.fixture(autouse=True)
def reset_shared_state(monkeypatch):
    provider = FakeTracerProvider()
    monkeypatch.setattr(
        OpenInferenceInstrumentor, "_translator", OpenInferenceTranslator()
    )
    monkeypatch.setattr(OpenInferenceInstrumentor, "_translator_registered", False)
    monkeypatch.setattr(OpenInferenceInstrumentor, "_active_span_processors", [])
    monkeypatch.setattr(OpenInferenceInstrumentor, "_registrations", {})
    monkeypatch.setattr(OpenInferenceInstrumentor, "_provider", None)
    monkeypatch.setattr(
        "respan_instrumentation_openinference._instrumentation.trace.get_tracer_provider",
        lambda: provider,
    )
    return provider


def test_processor_delegate_runs_before_one_translator_and_exporters(
    reset_shared_state,
):
    class FakeOIProcessor:
        shutdown_count = 0

        def shutdown(self):
            type(self).shutdown_count += 1

    wrapper = OpenInferenceInstrumentor(FakeOIProcessor)
    wrapper.activate()

    processors = reset_shared_state._active_span_processor._span_processors
    assert processors[0] is wrapper._instrumentor
    assert processors.count(OpenInferenceInstrumentor._translator) == 1
    assert processors[2:] == ("export-a", "export-b")

    wrapper.deactivate()
    assert reset_shared_state._active_span_processor._span_processors == (
        "export-a",
        "export-b",
    )
    assert FakeOIProcessor.shutdown_count == 1


def test_standard_delegate_is_reference_counted_across_wrappers(reset_shared_state):
    class StandardDelegate:
        instrument_count = 0
        uninstrument_count = 0

        def instrument(self, **kwargs):
            assert kwargs["tracer_provider"] is reset_shared_state
            type(self).instrument_count += 1

        def uninstrument(self):
            type(self).uninstrument_count += 1

    first = OpenInferenceInstrumentor(StandardDelegate)
    second = OpenInferenceInstrumentor(StandardDelegate)
    first.activate()
    second.activate()

    assert first._instrumentor is second._instrumentor
    assert StandardDelegate.instrument_count == 1
    processors = reset_shared_state._active_span_processor._span_processors
    assert processors.count(OpenInferenceInstrumentor._translator) == 1

    first.deactivate()
    assert StandardDelegate.uninstrument_count == 0
    assert OpenInferenceInstrumentor._translator in (
        reset_shared_state._active_span_processor._span_processors
    )
    second.deactivate()
    assert StandardDelegate.uninstrument_count == 1
    assert OpenInferenceInstrumentor._translator not in (
        reset_shared_state._active_span_processor._span_processors
    )


def test_external_standard_activation_is_not_owned(reset_shared_state):
    class ExternalDelegate:
        is_instrumented_by_opentelemetry = True
        instrument_count = 0
        uninstrument_count = 0

        def instrument(self, **kwargs):
            type(self).instrument_count += 1

        def uninstrument(self):
            type(self).uninstrument_count += 1

    wrapper = OpenInferenceInstrumentor(ExternalDelegate)
    wrapper.activate()
    wrapper.deactivate()

    assert ExternalDelegate.instrument_count == 0
    assert ExternalDelegate.uninstrument_count == 0


def test_external_processor_is_adopted_without_duplicate_or_shutdown(
    reset_shared_state,
):
    class ExternalProcessor:
        shutdown_count = 0

        def shutdown(self):
            type(self).shutdown_count += 1

    external = ExternalProcessor()
    reset_shared_state.add_span_processor(external)

    wrapper = OpenInferenceInstrumentor(ExternalProcessor)
    wrapper.activate()
    processors = reset_shared_state._active_span_processor._span_processors

    assert wrapper._instrumentor is external
    assert sum(isinstance(item, ExternalProcessor) for item in processors) == 1
    assert processors.index(external) < processors.index(
        OpenInferenceInstrumentor._translator
    )

    wrapper.deactivate()
    processors = reset_shared_state._active_span_processor._span_processors
    assert sum(isinstance(item, ExternalProcessor) for item in processors) == 1
    assert ExternalProcessor.shutdown_count == 0


def test_activation_failure_rolls_back_translator_and_delegate(reset_shared_state):
    class BrokenDelegate:
        uninstrument_count = 0

        def instrument(self, **kwargs):
            self._is_instrumented_by_opentelemetry = True
            raise RuntimeError("activation failed")

        def uninstrument(self):
            type(self).uninstrument_count += 1

    wrapper = OpenInferenceInstrumentor(BrokenDelegate)
    with pytest.raises(RuntimeError, match="activation failed"):
        wrapper.activate()

    assert OpenInferenceInstrumentor._registrations == {}
    assert OpenInferenceInstrumentor._provider is None
    assert OpenInferenceInstrumentor._translator not in (
        reset_shared_state._active_span_processor._span_processors
    )
    assert BrokenDelegate.uninstrument_count == 1


def test_concurrent_same_class_activation_has_one_delegate(reset_shared_state):
    class ConcurrentDelegate:
        instrument_count = 0
        uninstrument_count = 0

        def instrument(self, **kwargs):
            type(self).instrument_count += 1

        def uninstrument(self):
            type(self).uninstrument_count += 1

    wrappers = [OpenInferenceInstrumentor(ConcurrentDelegate) for _ in range(12)]
    with ThreadPoolExecutor(max_workers=6) as executor:
        list(executor.map(lambda wrapper: wrapper.activate(), wrappers))

    assert ConcurrentDelegate.instrument_count == 1
    assert len(OpenInferenceInstrumentor._registrations) == 1
    assert (
        reset_shared_state._active_span_processor._span_processors.count(
            OpenInferenceInstrumentor._translator
        )
        == 1
    )

    with ThreadPoolExecutor(max_workers=6) as executor:
        list(executor.map(lambda wrapper: wrapper.deactivate(), wrappers))
    assert ConcurrentDelegate.uninstrument_count == 1
    assert OpenInferenceInstrumentor._registrations == {}


def test_same_wrapper_activation_and_deactivation_are_idempotent():
    class StandardDelegate:
        instrument_count = 0
        uninstrument_count = 0

        def instrument(self, **kwargs):
            type(self).instrument_count += 1

        def uninstrument(self):
            type(self).uninstrument_count += 1

    wrapper = OpenInferenceInstrumentor(StandardDelegate)
    wrapper.activate()
    wrapper.activate()
    wrapper.deactivate()
    wrapper.deactivate()

    assert StandardDelegate.instrument_count == 1
    assert StandardDelegate.uninstrument_count == 1


def test_real_provider_wrapper_orders_source_translation_and_export(
    monkeypatch,
):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    export_processor = SimpleSpanProcessor(exporter)
    provider.add_span_processor(export_processor)
    monkeypatch.setattr(
        "respan_instrumentation_openinference._instrumentation.trace.get_tracer_provider",
        lambda: provider,
    )

    class SourceProcessor(SpanProcessor):
        shutdown_count = 0

        def on_start(self, span, parent_context=None):
            del span, parent_context

        def on_end(self, span):
            span._attributes = {
                **dict(span._attributes),
                "openinference.span.kind": "LLM",
                "llm.model_name": "gpt-4.1-mini",
                "llm.invocation_parameters": '{"stream":true}',
            }

        def shutdown(self):
            type(self).shutdown_count += 1

        def force_flush(self, timeout_millis=30_000):
            del timeout_millis
            return True

    wrapper = OpenInferenceInstrumentor(SourceProcessor)
    wrapper.activate()
    try:
        processors = provider._active_span_processor._span_processors
        assert processors == (
            wrapper._instrumentor,
            OpenInferenceInstrumentor._translator,
            export_processor,
        )

        tracer = provider.get_tracer("openinference-wrapper-contract")
        with tracer.start_as_current_span("source.chat"):
            pass

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        attrs = spans[0].attributes
        assert attrs["respan.entity.log_type"] == "chat"
        assert attrs["gen_ai.request.model"] == "gpt-4.1-mini"
        assert attrs["llm.is_streaming"] is True
        assert "openinference.span.kind" not in attrs
        assert "llm.invocation_parameters" not in attrs
    finally:
        wrapper.deactivate()

    assert provider._active_span_processor._span_processors == (export_processor,)
    assert SourceProcessor.shutdown_count == 1
    provider.shutdown()
