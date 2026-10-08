"""Released AgentOps/shared provider hooks ownership and rollback."""

import pytest
import respan_instrumentation_agentops._instrumentation as runtime
from agentops import task
from agentops.sdk.core import tracer
from opentelemetry import trace
from respan_instrumentation_agentops import AgentOpsInstrumentor


def test_reference_counts_and_restores_native_core(env):
    native = tracer.get_tracer
    prior = (tracer._initialized, tracer.provider, tracer._meter_provider)
    a, b = AgentOpsInstrumentor(), AgentOpsInstrumentor()
    a.activate()
    b.activate()
    assert a._runtime is b._runtime and tracer.initialized
    a.deactivate()

    @task
    def run():
        return "answer"

    assert run() == "answer"
    assert len(env.get_finished_spans()) == 1
    b.deactivate()
    assert (tracer._initialized, tracer.provider, tracer._meter_provider) == prior
    assert tracer.get_tracer == native
    run()
    assert len(env.get_finished_spans()) == 1


def test_incompatible_private_owner_rejected(env):
    a = AgentOpsInstrumentor()
    a.activate()
    with pytest.raises(ValueError):
        AgentOpsInstrumentor(capture_content=False).activate()
    a.deactivate()


def test_provider_conflict_rejected(env, monkeypatch):
    a = AgentOpsInstrumentor()
    a.activate()
    monkeypatch.setattr(trace, "get_tracer_provider", lambda: object())
    with pytest.raises(RuntimeError):
        AgentOpsInstrumentor().activate()
    a.deactivate()


def test_partial_install_rollback(env, monkeypatch):
    original = runtime._Runtime.patch
    native = tracer.get_tracer
    prior = (tracer._initialized, tracer.provider, tracer._meter_provider)

    def fail(self, *a):
        original(self, *a)
        if len(self.patches) == 2:
            raise RuntimeError("partial failure")

    monkeypatch.setattr(runtime._Runtime, "patch", fail)
    with pytest.raises(RuntimeError):
        AgentOpsInstrumentor().activate()
    assert (
        tracer.get_tracer == native
        and (tracer._initialized, tracer.provider, tracer._meter_provider) == prior
        and runtime._RUNTIME is None
    )
    assert not any(
        type(p).__name__ == "AgentOpsSpanProcessor"
        for p in trace.get_tracer_provider()._active_span_processor._span_processors
    )


def test_foreign_core_initialized_and_replaced_hook_survives(env):
    previous = (tracer._initialized, tracer.provider, tracer._meter_provider)
    foreign_provider = object()
    tracer._initialized = True
    tracer.provider = foreign_provider
    a = AgentOpsInstrumentor()
    a.activate()
    owned = tracer.get_tracer

    def foreign(*a, **kw):
        return owned(*a, **kw)

    tracer.get_tracer = foreign
    try:
        a.deactivate()
        assert (
            tracer.get_tracer is foreign
            and tracer.provider is foreign_provider
            and tracer.initialized
        )

        @task
        def run():
            return "answer"

        run()
        assert len(env.get_finished_spans()) == 1
        assert not any(
            k.startswith("respan.") for k in env.get_finished_spans()[0].attributes
        )
    finally:
        del tracer.get_tracer
        tracer._initialized, tracer.provider, tracer._meter_provider = previous


def test_disabled_respan_no_hooks(env, monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(
        runtime.RespanTracer, "_instance", SimpleNamespace(is_enabled=False)
    )
    a = AgentOpsInstrumentor()
    a.activate()
    assert not a._is_instrumented and runtime._RUNTIME is None


def test_foreign_meter_field_survives_owned_core_restore(env):
    prior = tracer._meter_provider
    a = AgentOpsInstrumentor()
    a.activate()
    foreign = object()
    tracer._meter_provider = foreign
    a.deactivate()
    assert tracer._meter_provider is foreign
    tracer._meter_provider = prior
