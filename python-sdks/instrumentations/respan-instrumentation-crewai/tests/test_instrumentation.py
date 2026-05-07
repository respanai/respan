import logging
import sys
from types import ModuleType, SimpleNamespace

import pytest

from respan_instrumentation_crewai import CrewAIInstrumentor
from respan_instrumentation_crewai import _instrumentation
from respan_instrumentation_crewai._instrumentation import (
    CREATE_LLM_SPANS_KWARG,
    OPENINFERENCE_CREWAI_MODULE,
    USE_EVENT_LISTENER_KWARG,
)
from respan_tracing.core.tracer import RespanTracer


def _install_fake_modules(monkeypatch):
    class FakeCrewAIInstrumentor:
        pass

    class FakeOpenInferenceInstrumentor:
        created = []

        def __init__(self, instrumentor_class, **kwargs):
            self.instrumentor_class = instrumentor_class
            self.kwargs = kwargs
            self.is_activated = False
            self.is_deactivated = False
            self.__class__.created.append(self)

        def activate(self):
            self.is_activated = True

        def deactivate(self):
            self.is_deactivated = True

    openinference_module = ModuleType("openinference")
    openinference_instrumentation_module = ModuleType("openinference.instrumentation")
    openinference_crewai_module = ModuleType(OPENINFERENCE_CREWAI_MODULE)
    openinference_crewai_module.CrewAIInstrumentor = FakeCrewAIInstrumentor
    openinference_instrumentation_module.crewai = openinference_crewai_module

    monkeypatch.setitem(sys.modules, "openinference", openinference_module)
    monkeypatch.setitem(
        sys.modules,
        "openinference.instrumentation",
        openinference_instrumentation_module,
    )
    monkeypatch.setitem(
        sys.modules,
        OPENINFERENCE_CREWAI_MODULE,
        openinference_crewai_module,
    )

    monkeypatch.setattr(
        _instrumentation,
        "OpenInferenceInstrumentor",
        FakeOpenInferenceInstrumentor,
    )

    return SimpleNamespace(
        crewai_instrumentor_class=FakeCrewAIInstrumentor,
        openinference_instrumentor_class=FakeOpenInferenceInstrumentor,
    )


@pytest.fixture(autouse=True)
def reset_tracer():
    RespanTracer.reset_instance()
    yield
    RespanTracer.reset_instance()


def test_activate_uses_openinference_crewai_defaults(monkeypatch):
    fake = _install_fake_modules(monkeypatch)

    instrumentor = CrewAIInstrumentor()
    instrumentor.activate()

    delegate = fake.openinference_instrumentor_class.created[0]
    assert delegate.instrumentor_class is fake.crewai_instrumentor_class
    assert delegate.kwargs == {
        USE_EVENT_LISTENER_KWARG: True,
        CREATE_LLM_SPANS_KWARG: True,
    }
    assert delegate.is_activated is True
    assert instrumentor._is_instrumented is True

    instrumentor.deactivate()

    assert delegate.is_deactivated is True
    assert instrumentor._is_instrumented is False


def test_activate_passes_custom_openinference_kwargs(monkeypatch):
    fake = _install_fake_modules(monkeypatch)

    instrumentor = CrewAIInstrumentor(
        use_event_listener=False,
        create_llm_spans=False,
        trace_content=False,
    )
    instrumentor.activate()

    delegate = fake.openinference_instrumentor_class.created[0]
    assert delegate.kwargs == {
        USE_EVENT_LISTENER_KWARG: False,
        CREATE_LLM_SPANS_KWARG: False,
        "trace_content": False,
    }


def test_activate_cleans_up_delegate_when_activation_fails(monkeypatch, caplog):
    fake = _install_fake_modules(monkeypatch)

    def activate_raises(self):
        self.is_activated = True
        raise RuntimeError("boom")

    monkeypatch.setattr(
        fake.openinference_instrumentor_class,
        "activate",
        activate_raises,
    )

    instrumentor = CrewAIInstrumentor()
    with caplog.at_level(logging.ERROR):
        instrumentor.activate()

    delegate = fake.openinference_instrumentor_class.created[0]
    assert delegate.is_deactivated is True
    assert instrumentor._delegate is None
    assert instrumentor._is_instrumented is False
    assert "Failed to activate CrewAI instrumentation" in caplog.text


def test_activate_skips_when_respan_tracing_is_disabled(monkeypatch, caplog):
    fake = _install_fake_modules(monkeypatch)
    RespanTracer(is_enabled=False)

    instrumentor = CrewAIInstrumentor()
    with caplog.at_level(logging.INFO):
        instrumentor.activate()

    assert fake.openinference_instrumentor_class.created == []
    assert instrumentor._is_instrumented is False
    assert (
        "CrewAI instrumentation skipped because Respan tracing is disabled"
        in caplog.text
    )


def test_activate_logs_warning_when_dependencies_are_missing(monkeypatch, caplog):
    def import_module_raises(module_name):
        if module_name == OPENINFERENCE_CREWAI_MODULE:
            raise ImportError(module_name)
        raise AssertionError(f"unexpected import: {module_name}")

    monkeypatch.setattr(
        _instrumentation.importlib,
        "import_module",
        import_module_raises,
    )
    instrumentor = CrewAIInstrumentor()

    with caplog.at_level(logging.WARNING):
        instrumentor.activate()

    assert "Failed to activate CrewAI instrumentation" in caplog.text
    assert instrumentor._is_instrumented is False
