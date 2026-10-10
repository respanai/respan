import asyncio
import inspect

import pytest
from burr.core import ApplicationBuilder, State, action
from respan_instrumentation_burr import BurrInstrumentor
from respan_instrumentation_burr import _instrumentation as instrumentation
from wrapt import FunctionWrapper


@action(reads=[], writes=["value"])
def native(state: State):
    return state.update(value=42)


def builder():
    return ApplicationBuilder().with_actions(native).with_entrypoint("native")


def test_shared_activation_and_native_builder_list_identity(telemetry):
    first, second = BurrInstrumentor(), BurrInstrumentor()
    original = inspect.getattr_static(ApplicationBuilder, "build")
    b = builder()
    hooks = b.lifecycle_adapters
    first.activate()
    second.activate()
    assert instrumentation._ACTIVATION_COUNT == 2
    app = b.build()
    assert b.lifecycle_adapters is hooks
    assert app.run(halt_after=["native"])[2]["value"] == 42
    first.deactivate()
    assert inspect.getattr_static(ApplicationBuilder, "build") is not original
    second.deactivate()
    assert inspect.getattr_static(ApplicationBuilder, "build") is original


def test_conflicting_owner_does_not_widen_content(telemetry):
    first = BurrInstrumentor(capture_content=False)
    second = BurrInstrumentor(capture_content=True)
    first.activate()
    try:
        with pytest.raises(RuntimeError, match="different capture_content"):
            second.activate()
        assert not second._is_instrumented
        assert instrumentation._ACTIVATION_COUNT == 1
    finally:
        first.deactivate()


def test_foreign_builder_wrapper_is_retained(telemetry, monkeypatch):
    owner = BurrInstrumentor()
    owner.activate()
    installed = inspect.getattr_static(ApplicationBuilder, "build")
    foreign = FunctionWrapper(
        installed, lambda wrapped, instance, args, kwargs: wrapped(*args, **kwargs)
    )
    monkeypatch.setattr(ApplicationBuilder, "build", foreign)
    owner.deactivate()
    assert inspect.getattr_static(ApplicationBuilder, "build") is foreign
    assert builder().build().run(halt_after=["native"])[2]["value"] == 42


def test_partial_activation_rolls_back_owned_wrapper(telemetry, monkeypatch):
    original = inspect.getattr_static(ApplicationBuilder, "build")
    constructor = instrumentation.FunctionWrapper
    count = 0

    def wrapper(*args, **kwargs):
        nonlocal count
        count += 1
        if count == 2:
            raise RuntimeError("controlled-install-fault")
        return constructor(*args, **kwargs)

    monkeypatch.setattr(instrumentation, "FunctionWrapper", wrapper)
    with pytest.raises(RuntimeError, match="controlled-install-fault"):
        BurrInstrumentor().activate()
    assert inspect.getattr_static(ApplicationBuilder, "build") is original
    assert instrumentation._ADAPTER is None
    assert instrumentation._ACTIVATION_COUNT == 0


def test_native_build_error_and_list_identity_preserved(telemetry):
    owner = BurrInstrumentor()
    owner.activate()
    b = ApplicationBuilder()
    original = b.lifecycle_adapters
    try:
        with pytest.raises(ValueError):
            b.build()
        assert b.lifecycle_adapters is original
    finally:
        owner.deactivate()


def test_native_async_build_error_and_list_identity_preserved(telemetry):
    owner = BurrInstrumentor()
    owner.activate()
    b = ApplicationBuilder()
    original = b.lifecycle_adapters
    try:
        with pytest.raises(ValueError):
            asyncio.run(b.abuild())
        assert b.lifecycle_adapters is original
    finally:
        owner.deactivate()


def test_built_app_retains_disabled_hook(telemetry):
    _, exporter = telemetry
    owner = BurrInstrumentor()
    owner.activate()
    app = builder().build()
    owner.deactivate()
    assert app.run(halt_after=["native"])[2]["value"] == 42
    assert exporter.get_finished_spans() == ()


def test_policy_install_failure_restores_detach_and_native_builder(
    telemetry, monkeypatch
):
    from opentelemetry import context

    provider, _ = telemetry
    detach = context.detach
    original = inspect.getattr_static(ApplicationBuilder, "build")

    def fail(processor):
        raise RuntimeError("controlled-policy-install-fault")

    monkeypatch.setattr(provider, "add_span_processor", fail)
    with pytest.raises(RuntimeError, match="controlled-policy-install-fault"):
        BurrInstrumentor().activate()
    assert context.detach is detach
    assert inspect.getattr_static(ApplicationBuilder, "build") is original
    assert instrumentation._ACTIVATION_COUNT == 0


def test_close_removes_only_owned_policy_processors(telemetry):
    provider, _ = telemetry
    before = provider._active_span_processor._span_processors
    owner = BurrInstrumentor()
    owner.activate()
    adapter = instrumentation._ADAPTER
    assert adapter.policy in provider._active_span_processor._span_processors
    owner.deactivate()
    assert provider._active_span_processor._span_processors == before
