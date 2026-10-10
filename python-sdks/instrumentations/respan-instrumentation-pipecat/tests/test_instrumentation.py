"""Lifecycle checks against released native delegate and SDK descriptors."""

import inspect

import pytest
from openinference.instrumentation.pipecat import PipecatInstrumentor as Native
from openinference.instrumentation.pipecat import _observer
from opentelemetry.sdk.trace import TracerProvider
from respan_instrumentation_pipecat import PipecatInstrumentor
from respan_instrumentation_pipecat._runtime import Runtime

try:
    from pipecat.pipeline.worker import PipelineWorker as Target
except ImportError:
    from pipecat.pipeline.task import PipelineTask as Target


@pytest.fixture
def provider():
    p = TracerProvider()
    yield p
    p.shutdown()


def test_actual_shared_owner_and_provider_rejection(provider):
    original = inspect.getattr_static(Target, "__init__")
    a = PipecatInstrumentor(tracer_provider=provider)
    b = PipecatInstrumentor(tracer_provider=provider)
    a.activate()
    a.activate()
    b.activate()
    wrapper = inspect.getattr_static(Target, "__init__")
    other = TracerProvider()
    try:
        with pytest.raises(ValueError):
            PipecatInstrumentor(tracer_provider=other).activate()
        with pytest.raises(ValueError):
            PipecatInstrumentor(
                tracer_provider=provider, capture_content=False
            ).activate()
        a.deactivate()
        assert inspect.getattr_static(Target, "__init__") is wrapper
    finally:
        b.deactivate()
        a.deactivate()
        other.shutdown()
    assert inspect.getattr_static(Target, "__init__") is original


def test_foreign_native_owner_survives(provider):
    n = Native()
    n.instrument(tracer_provider=provider)
    wrapper = inspect.getattr_static(Target, "__init__")
    a = PipecatInstrumentor(tracer_provider=provider)
    try:
        a.activate()
        a.deactivate()
        assert (
            n.is_instrumented_by_opentelemetry
            and inspect.getattr_static(Target, "__init__") is wrapper
        )
    finally:
        n.uninstrument()


def test_foreign_descriptor_context_and_observer_survive(provider):
    old = inspect.getattr_static(Target, "__init__")
    context = _observer.Context
    push = _observer.OpenInferenceObserver.on_push_frame
    a = PipecatInstrumentor(tracer_provider=provider)
    a.activate()
    native = inspect.getattr_static(Target, "__init__")

    def foreign(*args, **kwargs):
        return native(*args, **kwargs)

    async def foreign_push(*args, **kwargs):
        return await push(*args, **kwargs)

    replacement = lambda: None
    Target.__init__ = foreign
    _observer.Context = replacement
    _observer.OpenInferenceObserver.on_push_frame = foreign_push
    try:
        a.deactivate()
        assert (
            inspect.getattr_static(Target, "__init__") is foreign
            and _observer.Context is replacement
            and _observer.OpenInferenceObserver.on_push_frame is foreign_push
        )
    finally:
        Target.__init__ = old
        _observer.Context = context
        _observer.OpenInferenceObserver.on_push_frame = push


def test_partial_native_mutation_rolls_back(provider, monkeypatch):
    native = Native()
    old = inspect.getattr_static(Target, "__init__")
    fields = {k: getattr(native, k, None) for k in ["_tracer", "_config"]}
    original = native._instrument

    def failure(**kwargs):
        original(**kwargs)
        raise RuntimeError("controlled post-mutation failure")

    monkeypatch.setattr(native, "_instrument", failure)
    a = PipecatInstrumentor(tracer_provider=provider)
    a.activate()
    assert not a._is_instrumented and inspect.getattr_static(Target, "__init__") is old
    assert all(getattr(native, k, None) is v for k, v in fields.items())
    assert not native.is_instrumented_by_opentelemetry


def test_partial_runtime_hook_rolls_back(provider, monkeypatch):
    old = inspect.getattr_static(Target, "__init__")
    push = _observer.OpenInferenceObserver.on_push_frame
    context = _observer.Context
    original = Runtime.install

    def fail(self, observer):
        original(self, observer)
        raise RuntimeError("controlled after-hook failure")

    monkeypatch.setattr(Runtime, "install", fail)
    a = PipecatInstrumentor(tracer_provider=provider)
    a.activate()
    assert not a._is_instrumented and inspect.getattr_static(Target, "__init__") is old
    assert (
        _observer.OpenInferenceObserver.on_push_frame is push
        and _observer.Context is context
    )
    assert not provider._active_span_processor._span_processors


def test_foreign_provider_is_rejected(provider):
    other = TracerProvider()
    n = Native()
    n.instrument(tracer_provider=other)
    try:
        with pytest.raises(ValueError):
            PipecatInstrumentor(tracer_provider=provider).activate()
        assert n.is_instrumented_by_opentelemetry
    finally:
        n.uninstrument()
        other.shutdown()
