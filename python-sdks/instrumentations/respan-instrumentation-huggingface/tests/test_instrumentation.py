import logging
import sys
from types import ModuleType, SimpleNamespace

import pytest
from opentelemetry.semconv_ai import SpanAttributes as TLSpanAttributes

from respan_instrumentation_huggingface import HuggingFaceInstrumentor
from respan_instrumentation_huggingface import _instrumentation
from respan_instrumentation_huggingface._constants import (
    HUGGINGFACE_GEN_AI_SYSTEM,
    TRANSFORMERS_MODULE,
    TRANSFORMERS_SCOPE_NAME,
    TRANSFORMERS_TEXT_GENERATION_SPAN_NAME,
)
from respan_instrumentation_huggingface._instrumentation import (
    HuggingFaceSpanContractProcessor,
    _register_processor_first,
    _unregister_processor,
)
from respan_sdk.constants.llm_logging import LOG_TYPE_TEXT
from respan_sdk.constants.span_attributes import RESPAN_LOG_TYPE
from respan_tracing.core.tracer import RespanTracer


class FakeExportProcessor:
    pass


def _install_fake_modules(monkeypatch, tracer_provider=None):
    class FakeTransformersInstrumentor:
        created = []

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.instrument_kwargs = None
            self.is_instrumented = False
            self.is_uninstrumented = False
            self.__class__.created.append(self)

        def instrument(self, **kwargs):
            self.instrument_kwargs = kwargs
            self.is_instrumented = True

        def uninstrument(self):
            self.is_uninstrumented = True

    transformers_module = ModuleType(TRANSFORMERS_MODULE)
    transformers_module.TransformersInstrumentor = FakeTransformersInstrumentor
    monkeypatch.setitem(sys.modules, TRANSFORMERS_MODULE, transformers_module)

    if tracer_provider is None:
        tracer_provider = SimpleNamespace(
            _active_span_processor=SimpleNamespace(
                _span_processors=(FakeExportProcessor(),),
            ),
        )
    monkeypatch.setattr(
        _instrumentation.trace,
        "get_tracer_provider",
        lambda: tracer_provider,
    )

    return SimpleNamespace(
        instrumentor_class=FakeTransformersInstrumentor,
        tracer_provider=tracer_provider,
    )


@pytest.fixture(autouse=True)
def reset_tracer():
    RespanTracer.reset_instance()
    yield
    RespanTracer.reset_instance()


def test_register_processor_first_inserts_before_existing_processors():
    export_processor = FakeExportProcessor()
    contract_processor = HuggingFaceSpanContractProcessor()
    tracer_provider = SimpleNamespace(
        _active_span_processor=SimpleNamespace(
            _span_processors=(export_processor,),
        ),
    )

    _register_processor_first(tracer_provider, contract_processor)

    assert tracer_provider._active_span_processor._span_processors == (
        contract_processor,
        export_processor,
    )

    _unregister_processor(tracer_provider, contract_processor)

    assert tracer_provider._active_span_processor._span_processors == (
        export_processor,
    )


def test_activate_uses_transformers_instrumentor_and_contract_processor(monkeypatch):
    fake = _install_fake_modules(monkeypatch)

    instrumentor = HuggingFaceInstrumentor(
        use_legacy_attributes=False,
        logger_provider="fake-logger-provider",
    )
    instrumentor.activate()

    delegate = fake.instrumentor_class.created[0]
    assert delegate.kwargs == {
        "exception_logger": None,
        "use_legacy_attributes": False,
    }
    assert delegate.instrument_kwargs == {
        "tracer_provider": fake.tracer_provider,
        "logger_provider": "fake-logger-provider",
    }
    assert delegate.is_instrumented is True
    assert fake.tracer_provider._active_span_processor._span_processors[0] is (
        instrumentor._contract_processor
    )

    instrumentor.deactivate()

    assert delegate.is_uninstrumented is True
    assert instrumentor._contract_processor not in (
        fake.tracer_provider._active_span_processor._span_processors
    )


def test_activate_skips_when_respan_tracing_is_disabled(monkeypatch, caplog):
    fake = _install_fake_modules(monkeypatch)
    RespanTracer(is_enabled=False)

    instrumentor = HuggingFaceInstrumentor()
    with caplog.at_level(logging.INFO):
        instrumentor.activate()

    assert fake.instrumentor_class.created == []
    assert instrumentor._is_instrumented is False
    assert (
        "Hugging Face instrumentation skipped because Respan tracing is disabled"
        in caplog.text
    )


def test_activate_logs_warning_when_dependencies_are_missing(monkeypatch, caplog):
    def import_module_raises(module_name):
        if module_name == TRANSFORMERS_MODULE:
            raise ImportError(module_name)
        raise AssertionError(f"unexpected import: {module_name}")

    monkeypatch.setattr(
        _instrumentation.importlib,
        "import_module",
        import_module_raises,
    )
    instrumentor = HuggingFaceInstrumentor()

    with caplog.at_level(logging.WARNING):
        instrumentor.activate()

    assert "Failed to activate Hugging Face instrumentation" in caplog.text
    assert instrumentor._is_instrumented is False


def test_activate_cleans_up_when_upstream_instrument_fails(monkeypatch, caplog):
    fake = _install_fake_modules(monkeypatch)

    def instrument_raises(self, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(
        fake.instrumentor_class,
        "instrument",
        instrument_raises,
    )

    instrumentor = HuggingFaceInstrumentor()
    with caplog.at_level(logging.ERROR):
        instrumentor.activate()

    delegate = fake.instrumentor_class.created[0]
    assert delegate.is_uninstrumented is True
    assert instrumentor._contract_processor not in (
        fake.tracer_provider._active_span_processor._span_processors
    )
    assert instrumentor._is_instrumented is False
    assert "Failed to activate Hugging Face instrumentation" in caplog.text


def test_contract_processor_normalizes_transformers_text_generation_span():
    processor = HuggingFaceSpanContractProcessor()
    span = SimpleNamespace(
        name=TRANSFORMERS_TEXT_GENERATION_SPAN_NAME,
        instrumentation_scope=SimpleNamespace(name=TRANSFORMERS_SCOPE_NAME),
        _attributes={
            TLSpanAttributes.LLM_SYSTEM: "gpt2",
            TLSpanAttributes.LLM_REQUEST_TYPE: "completion",
            f"{TLSpanAttributes.LLM_PROMPTS}.0.content": "Write about tracing",
            f"{TLSpanAttributes.LLM_COMPLETIONS}.0.content": "Tracing keeps work visible.",
        },
    )

    processor.on_end(span)

    assert span._attributes[TLSpanAttributes.LLM_SYSTEM] == HUGGINGFACE_GEN_AI_SYSTEM
    assert span._attributes[TLSpanAttributes.LLM_REQUEST_TYPE] == "chat"
    assert span._attributes[RESPAN_LOG_TYPE] == LOG_TYPE_TEXT
    assert "log_type" not in span._attributes
    assert span._attributes[TLSpanAttributes.TRACELOOP_ENTITY_NAME] == (
        "huggingface.text_generation"
    )
    assert span._attributes[f"{TLSpanAttributes.LLM_PROMPTS}.0.role"] == "user"
    assert span._attributes[f"{TLSpanAttributes.LLM_COMPLETIONS}.0.role"] == "assistant"


def test_contract_processor_preserves_existing_roles():
    processor = HuggingFaceSpanContractProcessor()
    span = SimpleNamespace(
        name=TRANSFORMERS_TEXT_GENERATION_SPAN_NAME,
        instrumentation_scope=SimpleNamespace(name=TRANSFORMERS_SCOPE_NAME),
        _attributes={
            f"{TLSpanAttributes.LLM_PROMPTS}.0.content": "System prompt",
            f"{TLSpanAttributes.LLM_PROMPTS}.0.role": "system",
            f"{TLSpanAttributes.LLM_COMPLETIONS}.0.content": "Completion",
            f"{TLSpanAttributes.LLM_COMPLETIONS}.0.role": "model",
        },
    )

    processor.on_end(span)

    assert span._attributes[f"{TLSpanAttributes.LLM_PROMPTS}.0.role"] == "system"
    assert span._attributes[f"{TLSpanAttributes.LLM_COMPLETIONS}.0.role"] == "model"


def test_contract_processor_ignores_non_transformers_spans():
    processor = HuggingFaceSpanContractProcessor()
    span = SimpleNamespace(
        name="other",
        instrumentation_scope=SimpleNamespace(name="other.scope"),
        _attributes={TLSpanAttributes.LLM_SYSTEM: "openai"},
    )

    processor.on_end(span)

    assert span._attributes == {TLSpanAttributes.LLM_SYSTEM: "openai"}
