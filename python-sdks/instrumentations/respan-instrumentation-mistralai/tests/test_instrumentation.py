import json
import logging
import sys
from types import ModuleType, SimpleNamespace

import pytest

from respan_instrumentation_mistralai import MistralAIInstrumentor
from respan_instrumentation_mistralai import _instrumentation
from respan_instrumentation_mistralai._instrumentation import (
    MISTRALAI_SDK_TRACER_NAME,
    OPENINFERENCE_MISTRALAI_MODULE,
)
from respan_sdk.constants.span_attributes import (
    RESPAN_SPAN_TOOL_CALLS,
    RESPAN_SPAN_TOOLS,
)
from respan_tracing.constants.tracing import SAMPLE_RATE_ATTR
from respan_tracing.core.tracer import RespanTracer


def _install_fake_modules(monkeypatch):
    class FakeMistralAIInstrumentor:
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
    openinference_mistralai_module = ModuleType(OPENINFERENCE_MISTRALAI_MODULE)
    openinference_mistralai_module.MistralAIInstrumentor = FakeMistralAIInstrumentor
    openinference_instrumentation_module.mistralai = openinference_mistralai_module

    monkeypatch.setitem(sys.modules, "openinference", openinference_module)
    monkeypatch.setitem(
        sys.modules,
        "openinference.instrumentation",
        openinference_instrumentation_module,
    )
    monkeypatch.setitem(
        sys.modules,
        OPENINFERENCE_MISTRALAI_MODULE,
        openinference_mistralai_module,
    )

    monkeypatch.setattr(
        _instrumentation,
        "OpenInferenceInstrumentor",
        FakeOpenInferenceInstrumentor,
    )

    return SimpleNamespace(
        mistralai_instrumentor_class=FakeMistralAIInstrumentor,
        openinference_instrumentor_class=FakeOpenInferenceInstrumentor,
    )


@pytest.fixture(autouse=True)
def reset_tracer():
    RespanTracer.reset_instance()
    yield
    RespanTracer.reset_instance()


def test_activate_uses_openinference_mistralai(monkeypatch):
    fake = _install_fake_modules(monkeypatch)

    instrumentor = MistralAIInstrumentor()
    instrumentor.activate()

    delegate = fake.openinference_instrumentor_class.created[0]
    assert delegate.instrumentor_class is fake.mistralai_instrumentor_class
    assert delegate.kwargs == {}
    assert delegate.is_activated is True
    assert instrumentor._is_instrumented is True

    instrumentor.deactivate()

    assert delegate.is_deactivated is True
    assert instrumentor._is_instrumented is False


def test_activate_passes_custom_openinference_kwargs(monkeypatch):
    fake = _install_fake_modules(monkeypatch)

    instrumentor = MistralAIInstrumentor(trace_content=False, custom_option="value")
    instrumentor.activate()

    delegate = fake.openinference_instrumentor_class.created[0]
    assert delegate.kwargs == {
        "trace_content": False,
        "custom_option": "value",
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

    instrumentor = MistralAIInstrumentor()
    with caplog.at_level(logging.ERROR):
        instrumentor.activate()

    delegate = fake.openinference_instrumentor_class.created[0]
    assert delegate.is_deactivated is True
    assert instrumentor._delegate is None
    assert instrumentor._is_instrumented is False
    assert "Failed to activate Mistral AI instrumentation" in caplog.text


def test_activate_skips_when_respan_tracing_is_disabled(monkeypatch, caplog):
    fake = _install_fake_modules(monkeypatch)
    RespanTracer(is_enabled=False)

    instrumentor = MistralAIInstrumentor()
    with caplog.at_level(logging.INFO):
        instrumentor.activate()

    assert fake.openinference_instrumentor_class.created == []
    assert instrumentor._is_instrumented is False
    assert (
        "Mistral AI instrumentation skipped because Respan tracing is disabled"
        in caplog.text
    )


def test_activate_logs_warning_when_dependencies_are_missing(monkeypatch, caplog):
    def import_module_raises(module_name):
        if module_name == OPENINFERENCE_MISTRALAI_MODULE:
            raise ImportError(module_name)
        raise AssertionError(f"unexpected import: {module_name}")

    monkeypatch.setattr(
        _instrumentation.importlib,
        "import_module",
        import_module_raises,
    )
    instrumentor = MistralAIInstrumentor()

    with caplog.at_level(logging.WARNING):
        instrumentor.activate()

    assert "Failed to activate Mistral AI instrumentation" in caplog.text
    assert instrumentor._is_instrumented is False


def test_cleanup_processor_removes_mistralai_off_contract_aliases():
    tool_calls = [
        {
            "id": "call_123",
            "type": "function",
            "function": {"name": "lookup", "arguments": "{}"},
        }
    ]
    span = SimpleNamespace(
        instrumentation_scope=SimpleNamespace(name=OPENINFERENCE_MISTRALAI_MODULE),
        _attributes={
            RESPAN_SPAN_TOOLS: "[]",
            RESPAN_SPAN_TOOL_CALLS: json.dumps(tool_calls),
            "tools": [],
            "tool_calls": tool_calls,
            "model": "mistral-large-latest",
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_request_tokens": 15,
            "llm.request.functions": "[]",
            "gen_ai.completion.0.tool_calls": tool_calls,
        },
    )

    _instrumentation._MistralAIOffContractAliasProcessor().on_end(span)

    for key in (
        RESPAN_SPAN_TOOLS,
        RESPAN_SPAN_TOOL_CALLS,
        "tools",
        "tool_calls",
        "model",
        "prompt_tokens",
        "completion_tokens",
        "total_request_tokens",
    ):
        assert key not in span._attributes
    assert span._attributes["llm.request.functions"] == "[]"
    assert json.loads(span._attributes["gen_ai.completion.0.tool_calls"]) == tool_calls


def test_cleanup_processor_ignores_non_mistralai_spans():
    span = SimpleNamespace(
        instrumentation_scope=SimpleNamespace(name="openinference.instrumentation.other"),
        _attributes={
            "tools": [],
            "gen_ai.completion.0.tool_calls": [],
        },
    )

    _instrumentation._MistralAIOffContractAliasProcessor().on_end(span)

    assert span._attributes["tools"] == []
    assert span._attributes["gen_ai.completion.0.tool_calls"] == []


def test_cleanup_processor_drops_native_mistral_sdk_spans_from_respan_export():
    span = SimpleNamespace(
        instrumentation_scope=SimpleNamespace(name=MISTRALAI_SDK_TRACER_NAME),
        _attributes={
            "gen_ai.system": "mistralai",
            "gen_ai.request.model": "mistral-large-latest",
        },
    )

    _instrumentation._MistralAIOffContractAliasProcessor().on_end(span)

    assert span._attributes[SAMPLE_RATE_ATTR] == 0
    assert span._attributes["gen_ai.system"] == "mistralai"


def test_activate_places_cleanup_after_openinference_translator(monkeypatch):
    fake = _install_fake_modules(monkeypatch)
    translator = object()
    exporter = object()

    class FakeOpenInferenceInstrumentor(fake.openinference_instrumentor_class):
        created = []

        @classmethod
        def _get_translator(cls):
            return translator

    active_span_processor = SimpleNamespace(_span_processors=(translator, exporter))
    tracer_provider = SimpleNamespace(_active_span_processor=active_span_processor)
    monkeypatch.setattr(
        _instrumentation,
        "OpenInferenceInstrumentor",
        FakeOpenInferenceInstrumentor,
    )
    monkeypatch.setattr(
        _instrumentation.trace,
        "get_tracer_provider",
        lambda: tracer_provider,
    )

    instrumentor = MistralAIInstrumentor()
    instrumentor.activate()

    processors = active_span_processor._span_processors
    assert processors[0] is translator
    assert isinstance(
        processors[1],
        _instrumentation._MistralAIOffContractAliasProcessor,
    )
    assert processors[2] is exporter

    instrumentor.deactivate()

    assert active_span_processor._span_processors == (translator, exporter)
