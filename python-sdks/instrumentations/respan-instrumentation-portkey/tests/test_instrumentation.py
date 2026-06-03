import logging
import sys
from types import ModuleType, SimpleNamespace

import pytest
from respan_instrumentation_openinference._translator import OpenInferenceTranslator
from respan_sdk.constants.span_attributes import (
    RESPAN_SPAN_TOOL_CALLS,
    RESPAN_SPAN_TOOLS,
)
from respan_tracing.core.tracer import RespanTracer

from respan_instrumentation_portkey import PortkeyInstrumentor
from respan_instrumentation_portkey import _instrumentation
from respan_instrumentation_portkey._constants import OPENINFERENCE_PORTKEY_MODULE
from respan_instrumentation_portkey._processor import (
    GEN_AI_COMPLETION_TOOL_CALLS_ATTR,
    LLM_REQUEST_FUNCTIONS_ATTR,
    OTEL_SCOPE_NAME,
    PortkeySpanContractProcessor,
)


def _install_fake_modules(monkeypatch):
    class FakePortkeyInstrumentor:
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
    openinference_portkey_module = ModuleType(OPENINFERENCE_PORTKEY_MODULE)
    openinference_portkey_module.PortkeyInstrumentor = FakePortkeyInstrumentor
    openinference_instrumentation_module.portkey = openinference_portkey_module

    monkeypatch.setitem(sys.modules, "openinference", openinference_module)
    monkeypatch.setitem(
        sys.modules,
        "openinference.instrumentation",
        openinference_instrumentation_module,
    )
    monkeypatch.setitem(
        sys.modules,
        OPENINFERENCE_PORTKEY_MODULE,
        openinference_portkey_module,
    )

    monkeypatch.setattr(
        _instrumentation,
        "OpenInferenceInstrumentor",
        FakeOpenInferenceInstrumentor,
    )

    return SimpleNamespace(
        portkey_instrumentor_class=FakePortkeyInstrumentor,
        openinference_instrumentor_class=FakeOpenInferenceInstrumentor,
    )


def _make_fake_tracer_provider(processors=()):
    return SimpleNamespace(
        _active_span_processor=SimpleNamespace(_span_processors=processors),
        add_span_processor=lambda processor: None,
    )


@pytest.fixture(autouse=True)
def reset_tracer():
    RespanTracer.reset_instance()
    yield
    RespanTracer.reset_instance()


def test_activate_uses_openinference_portkey(monkeypatch):
    fake = _install_fake_modules(monkeypatch)
    translator = OpenInferenceTranslator()
    tracer_provider = _make_fake_tracer_provider(processors=(translator,))
    monkeypatch.setattr(
        _instrumentation.trace,
        "get_tracer_provider",
        lambda: tracer_provider,
    )

    instrumentor = PortkeyInstrumentor()
    instrumentor.activate()

    delegate = fake.openinference_instrumentor_class.created[0]
    assert delegate.instrumentor_class is fake.portkey_instrumentor_class
    assert delegate.kwargs == {}
    assert delegate.is_activated is True
    assert instrumentor._is_instrumented is True
    assert tracer_provider._active_span_processor._span_processors == (
        translator,
        instrumentor._contract_processor,
    )

    instrumentor.deactivate()

    assert delegate.is_deactivated is True
    assert instrumentor._is_instrumented is False
    assert tracer_provider._active_span_processor._span_processors == (translator,)


def test_activate_passes_custom_openinference_kwargs(monkeypatch):
    fake = _install_fake_modules(monkeypatch)
    tracer_provider = _make_fake_tracer_provider()
    monkeypatch.setattr(
        _instrumentation.trace,
        "get_tracer_provider",
        lambda: tracer_provider,
    )

    instrumentor = PortkeyInstrumentor(trace_content=False)
    instrumentor.activate()

    delegate = fake.openinference_instrumentor_class.created[0]
    assert delegate.kwargs == {"trace_content": False}


def test_activate_is_idempotent(monkeypatch):
    fake = _install_fake_modules(monkeypatch)
    tracer_provider = _make_fake_tracer_provider()
    monkeypatch.setattr(
        _instrumentation.trace,
        "get_tracer_provider",
        lambda: tracer_provider,
    )

    instrumentor = PortkeyInstrumentor()
    instrumentor.activate()
    instrumentor.activate()

    assert len(fake.openinference_instrumentor_class.created) == 1


def test_activate_cleans_up_delegate_when_activation_fails(monkeypatch, caplog):
    fake = _install_fake_modules(monkeypatch)
    tracer_provider = _make_fake_tracer_provider()
    monkeypatch.setattr(
        _instrumentation.trace,
        "get_tracer_provider",
        lambda: tracer_provider,
    )

    def activate_raises(self):
        self.is_activated = True
        raise RuntimeError("boom")

    monkeypatch.setattr(
        fake.openinference_instrumentor_class,
        "activate",
        activate_raises,
    )

    instrumentor = PortkeyInstrumentor()
    with caplog.at_level(logging.ERROR):
        instrumentor.activate()

    delegate = fake.openinference_instrumentor_class.created[0]
    assert delegate.is_deactivated is True
    assert instrumentor._delegate is None
    assert instrumentor._is_instrumented is False
    assert "Failed to activate Portkey instrumentation" in caplog.text


def test_activate_skips_when_respan_tracing_is_disabled(monkeypatch, caplog):
    fake = _install_fake_modules(monkeypatch)
    RespanTracer(is_enabled=False)

    instrumentor = PortkeyInstrumentor()
    with caplog.at_level(logging.INFO):
        instrumentor.activate()

    assert fake.openinference_instrumentor_class.created == []
    assert instrumentor._is_instrumented is False
    assert (
        "Portkey instrumentation skipped because Respan tracing is disabled"
        in caplog.text
    )


def test_activate_logs_warning_when_dependencies_are_missing(monkeypatch, caplog):
    def import_module_raises(module_name):
        if module_name == OPENINFERENCE_PORTKEY_MODULE:
            raise ImportError(module_name)
        raise AssertionError(f"unexpected import: {module_name}")

    monkeypatch.setattr(
        _instrumentation.importlib,
        "import_module",
        import_module_raises,
    )

    instrumentor = PortkeyInstrumentor()
    with caplog.at_level(logging.WARNING):
        instrumentor.activate()

    assert instrumentor._is_instrumented is False
    assert "Failed to activate Portkey instrumentation" in caplog.text


def test_contract_processor_removes_off_contract_aliases():
    processor = PortkeySpanContractProcessor()
    span = SimpleNamespace(
        _attributes={
            OTEL_SCOPE_NAME: OPENINFERENCE_PORTKEY_MODULE,
            "model": "gpt-4o-mini",
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_request_tokens": 15,
            "gen_ai.request.top_p": "<openai.Omit object at 0x123>",
            "tools": [{"type": "function"}],
            "tool_calls": [{"function": {"name": "lookup"}}],
            RESPAN_SPAN_TOOLS: '[{"type":"function"}]',
            RESPAN_SPAN_TOOL_CALLS: '[{"function":{"name":"lookup"}}]',
        },
        instrumentation_scope=SimpleNamespace(name=OPENINFERENCE_PORTKEY_MODULE),
    )

    processor.on_end(span)

    assert span._attributes[LLM_REQUEST_FUNCTIONS_ATTR] == '[{"type":"function"}]'
    assert (
        span._attributes[GEN_AI_COMPLETION_TOOL_CALLS_ATTR]
        == '[{"function":{"name":"lookup"}}]'
    )
    for alias in (
        "model",
        "prompt_tokens",
        "completion_tokens",
        "total_request_tokens",
        "gen_ai.request.top_p",
        "tools",
        "tool_calls",
        RESPAN_SPAN_TOOLS,
        RESPAN_SPAN_TOOL_CALLS,
    ):
        assert alias not in span._attributes


def test_contract_processor_ignores_non_portkey_spans():
    processor = PortkeySpanContractProcessor()
    span = SimpleNamespace(
        _attributes={"model": "gpt-4o-mini"},
        instrumentation_scope=SimpleNamespace(name="openinference.instrumentation.other"),
    )

    processor.on_end(span)

    assert span._attributes == {"model": "gpt-4o-mini"}
