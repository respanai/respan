import logging
import threading

import pytest

from respan_instrumentation_crewai import CrewAIInstrumentor
from respan_instrumentation_crewai import _event_listener
from respan_tracing.core.tracer import RespanTracer


@pytest.fixture(autouse=True)
def reset_instrumentation():
    active_owner = CrewAIInstrumentor._active_owner
    if active_owner is not None:
        active_owner.deactivate()
    CrewAIInstrumentor._active_owner = None
    RespanTracer.reset_instance()
    yield
    active_owner = CrewAIInstrumentor._active_owner
    if active_owner is not None:
        active_owner.deactivate()
    CrewAIInstrumentor._active_owner = None
    RespanTracer.reset_instance()


def test_activate_and_deactivate_are_idempotent(monkeypatch):
    created = []

    class FakeListener:
        def __init__(self):
            self.shutdown_count = 0
            created.append(self)

        def shutdown(self):
            self.shutdown_count += 1

    monkeypatch.setattr(_event_listener, "CrewAIEventListener", FakeListener)

    instrumentor = CrewAIInstrumentor()
    instrumentor.activate()
    instrumentor.activate()

    assert len(created) == 1
    assert instrumentor._is_instrumented is True
    assert CrewAIInstrumentor._active_owner is instrumentor

    instrumentor.deactivate()
    instrumentor.deactivate()

    assert created[0].shutdown_count == 1
    assert instrumentor._is_instrumented is False
    assert CrewAIInstrumentor._active_owner is None


def test_only_one_instrumentor_instance_can_subscribe(monkeypatch):
    created = []

    class FakeListener:
        def __init__(self):
            created.append(self)

        def shutdown(self):
            return None

    monkeypatch.setattr(_event_listener, "CrewAIEventListener", FakeListener)

    first = CrewAIInstrumentor()
    second = CrewAIInstrumentor()
    first.activate()
    second.activate()

    assert len(created) == 1
    assert first._is_instrumented is True
    assert second._is_instrumented is False


def test_activate_skips_when_respan_tracing_is_disabled(monkeypatch, caplog):
    class UnexpectedListener:
        def __init__(self):
            raise AssertionError("listener must not be created")

    monkeypatch.setattr(_event_listener, "CrewAIEventListener", UnexpectedListener)
    RespanTracer(is_enabled=False)

    instrumentor = CrewAIInstrumentor()
    with caplog.at_level(logging.INFO):
        instrumentor.activate()

    assert instrumentor._is_instrumented is False
    assert (
        "CrewAI instrumentation skipped because Respan tracing is disabled"
        in caplog.text
    )


def test_activate_leaves_clean_state_when_listener_creation_fails(
    monkeypatch,
    caplog,
):
    class FailingListener:
        def __init__(self):
            raise RuntimeError("boom")

    monkeypatch.setattr(_event_listener, "CrewAIEventListener", FailingListener)

    instrumentor = CrewAIInstrumentor()
    with caplog.at_level(logging.ERROR):
        instrumentor.activate()

    assert instrumentor._listener is None
    assert instrumentor._is_instrumented is False
    assert CrewAIInstrumentor._active_owner is None
    assert "Failed to activate CrewAI instrumentation" in caplog.text


def test_deactivate_blocks_reactivation_until_shutdown_finishes(monkeypatch):
    shutdown_started = threading.Event()
    allow_shutdown = threading.Event()
    activation_finished = threading.Event()
    created = []

    class BlockingListener:
        def __init__(self):
            created.append(self)

        def shutdown(self):
            if self is created[0]:
                shutdown_started.set()
                assert allow_shutdown.wait(timeout=2)

    monkeypatch.setattr(_event_listener, "CrewAIEventListener", BlockingListener)

    first = CrewAIInstrumentor()
    first.activate()
    deactivate_thread = threading.Thread(target=first.deactivate)
    deactivate_thread.start()
    assert shutdown_started.wait(timeout=1)

    second = CrewAIInstrumentor()

    def activate_second():
        second.activate()
        activation_finished.set()

    activate_thread = threading.Thread(target=activate_second)
    activate_thread.start()
    try:
        assert not activation_finished.wait(timeout=0.1)
    finally:
        allow_shutdown.set()
        deactivate_thread.join(timeout=2)
        activate_thread.join(timeout=2)

    assert not deactivate_thread.is_alive()
    assert not activate_thread.is_alive()
    assert activation_finished.is_set()
    assert len(created) == 2
    assert second._is_instrumented is True
    second.deactivate()
