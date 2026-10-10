from __future__ import annotations

import functools

import pytest
from _fixtures import model
from mirascope import llm
from opentelemetry import context
from respan_instrumentation_mirascope import MirascopeInstrumentor
from respan_instrumentation_mirascope import _instrumentation as implementation
from respan_instrumentation_mirascope._policy import Policy
from respan_tracing.constants.context_constants import ENABLE_CONTENT_TRACING_KEY


def test_shared_owners_and_foreign_retained_wrapper_is_inert(capture, monkeypatch):
    first, provider, exporter = capture
    second = MirascopeInstrumentor(tracer_provider=provider)
    second.activate()
    owned = llm.Model.call

    @functools.wraps(owned)
    def foreign(*args, **kwargs):
        return owned(*args, **kwargs)

    monkeypatch.setattr(llm.Model, "call", foreign)
    first.deactivate()
    native, _ = model()
    native.call("shared")
    assert len(exporter.get_finished_spans()) == 1
    second.deactivate()
    native.call("inactive")
    assert llm.Model.call is foreign
    assert len(exporter.get_finished_spans()) == 1
    first.activate()
    native.call("new generation")
    assert len(exporter.get_finished_spans()) == 2


def test_conflicting_shared_privacy_cannot_unmask(capture):
    _, provider, _ = capture
    with pytest.raises(ValueError):
        MirascopeInstrumentor(
            capture_content=False, tracer_provider=provider
        ).activate()


def test_activation_failure_after_hook_mutation_rolls_back(capture, monkeypatch):
    first, provider, _ = capture
    first.deactivate()
    original, detach = llm.Model.call, context.detach
    install = Policy.install

    def failing(policy):
        install(policy)
        raise RuntimeError("after detach mutation")

    monkeypatch.setattr(Policy, "install", failing)
    first.activate()
    assert llm.Model.call is original and context.detach is detach
    assert implementation._RUNTIME is None
    assert not any(
        isinstance(p, Policy) for p in provider._active_span_processor._span_processors
    )


def test_deactivation_cleans_pending_stream_without_native_consumption(capture):
    first, _, exporter = capture
    native, _ = model()
    response = native.stream("private pending")
    first.deactivate()
    assert "".join(response.text_stream()) == "native stream\n"
    assert len(exporter.get_finished_spans()) == 1
    assert "traceloop.entity.input" not in exporter.get_finished_spans()[0].attributes


def test_supplied_private_parent_context_bound(capture):
    first, provider, _ = capture
    private = context.set_value(ENABLE_CONTENT_TRACING_KEY, False)
    span = provider.get_tracer("application").start_span("explicit", context=private)
    assert not first.runtime.policy.ancestors(
        (span.context.trace_id, span.context.span_id)
    )
    span.end()


def test_native_start_callback_deactivation_cleans_orphan(capture):
    first, provider, exporter = capture
    from opentelemetry.sdk.trace import SpanProcessor

    class Stop(SpanProcessor):
        def on_start(self, span, parent_context=None):
            first.deactivate()

    provider.add_span_processor(Stop())
    native, source = model()
    assert native.call("private after shutdown") is source.last
    assert len(exporter.get_finished_spans()) == 1
    assert "traceloop.entity.input" not in exporter.get_finished_spans()[0].attributes
    assert not first.runtime


def test_missing_optional_openai_preserves_native_model_instrumentation(
    capture, monkeypatch
):
    import builtins

    first, _, exporter = capture
    first.deactivate()
    original = builtins.__import__

    def importing(name, *args, **kwargs):
        if name.startswith("openai"):
            raise ModuleNotFoundError("optional provider unavailable", name="openai")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", importing)
    first.activate()
    native, source = model()
    assert native.call("custom native provider") is source.last
    assert len(exporter.get_finished_spans()) == 1


def test_late_proxy_provider_cannot_capture_unobserved_private_ancestor(
    capture, monkeypatch
):
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )
    from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import (
        GEN_AI_USAGE_INPUT_TOKENS,
    )

    capture[0].deactivate()
    monkeypatch.setattr(trace, "_TRACER_PROVIDER", None)
    instrumentor = MirascopeInstrumentor()
    instrumentor.activate()
    assert instrumentor.runtime.registered is False
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(trace, "_TRACER_PROVIDER", provider)
    private = context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, False))
    try:
        with provider.get_tracer("application").start_as_current_span(
            "private-initial-parent"
        ):
            enabled = context.attach(
                context.set_value(ENABLE_CONTENT_TRACING_KEY, True)
            )
            try:
                native, source = model()
                assert native.call("PRIVATE late-provider payload") is source.last
            finally:
                context.detach(enabled)
    finally:
        context.detach(private)
        instrumentor.deactivate()
        provider.shutdown()
    child = next(s for s in exporter.get_finished_spans() if s.name == "llm")
    assert "traceloop.entity.input" not in child.attributes
    assert "traceloop.entity.output" not in child.attributes
    assert child.attributes[GEN_AI_USAGE_INPUT_TOKENS] == 9


def test_existing_private_sdk_parent_is_unknown_at_enrollment(capture):
    first, provider, exporter = capture
    first.deactivate()
    private = context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, False))
    try:
        with provider.get_tracer("application").start_as_current_span(
            "preexisting-private"
        ):
            enabled = context.attach(
                context.set_value(ENABLE_CONTENT_TRACING_KEY, True)
            )
            try:
                first.activate()
                native, source = model()
                assert native.call("PRIVATE unknown-parent payload") is source.last
            finally:
                context.detach(enabled)
    finally:
        context.detach(private)
    child = next(s for s in exporter.get_finished_spans() if s.name == "llm")
    assert "traceloop.entity.input" not in child.attributes
    assert "traceloop.entity.output" not in child.attributes
