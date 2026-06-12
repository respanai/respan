import json
import logging
import sys
from types import ModuleType, SimpleNamespace

import pytest
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import LLMRequestTypeValues, SpanAttributes

from respan_instrumentation_cohere import CohereInstrumentor
from respan_instrumentation_cohere import _instrumentation
from respan_instrumentation_cohere._instrumentation import OTEL_COHERE_MODULE
from respan_instrumentation_cohere._processor import (
    COHERE_SCOPE_NAME,
    CohereSpanProcessor,
)
from respan_sdk.constants.span_attributes import RESPAN_LOG_TYPE, RESPAN_SPAN_TOOLS
from respan_tracing.core.tracer import RespanTracer


class BufferingSpanProcessor:
    pass


class FakeActiveProcessor:
    def __init__(self):
        self.export_processor = BufferingSpanProcessor()
        self._span_processors = (self.export_processor,)


class FakeTracerProvider:
    def __init__(self):
        self._active_span_processor = FakeActiveProcessor()
        self.added_processors = []

    def add_span_processor(self, processor):
        self.added_processors.append(processor)


class FakeSpan:
    def __init__(self, attrs, name="cohere.chat"):
        self.name = name
        self._attributes = dict(attrs)
        self.attributes = self._attributes
        self.instrumentation_scope = SimpleNamespace(name=COHERE_SCOPE_NAME)


def _install_fake_modules(monkeypatch):
    class FakeOTELCohereInstrumentor:
        created = []

        def __init__(self, **kwargs):
            self.constructor_kwargs = kwargs
            self.instrument_kwargs = None
            self.is_instrumented_by_opentelemetry = False
            self.is_uninstrumented = False
            self.__class__.created.append(self)

        def instrument(self, **kwargs):
            self.instrument_kwargs = kwargs
            self.is_instrumented_by_opentelemetry = True

        def uninstrument(self):
            self.is_uninstrumented = True
            self.is_instrumented_by_opentelemetry = False

    opentelemetry_module = ModuleType("opentelemetry")
    otel_instrumentation_module = ModuleType("opentelemetry.instrumentation")
    otel_cohere_module = ModuleType(OTEL_COHERE_MODULE)
    otel_cohere_module.CohereInstrumentor = FakeOTELCohereInstrumentor

    monkeypatch.setitem(sys.modules, "opentelemetry", opentelemetry_module)
    monkeypatch.setitem(
        sys.modules,
        "opentelemetry.instrumentation",
        otel_instrumentation_module,
    )
    monkeypatch.setitem(sys.modules, OTEL_COHERE_MODULE, otel_cohere_module)

    return SimpleNamespace(cohere_instrumentor_class=FakeOTELCohereInstrumentor)


@pytest.fixture(autouse=True)
def reset_tracer():
    RespanTracer.reset_instance()
    yield
    RespanTracer.reset_instance()


@pytest.fixture
def fake_tracer_provider(monkeypatch):
    provider = FakeTracerProvider()
    monkeypatch.setattr(
        _instrumentation.trace,
        "get_tracer_provider",
        lambda: provider,
    )
    return provider


def test_activate_uses_otel_cohere_and_registers_processor(
    monkeypatch,
    fake_tracer_provider,
):
    fake = _install_fake_modules(monkeypatch)

    instrumentor = CohereInstrumentor()
    instrumentor.activate()

    upstream = fake.cohere_instrumentor_class.created[0]
    assert upstream.constructor_kwargs == {
        "exception_logger": None,
        "use_legacy_attributes": True,
    }
    assert upstream.instrument_kwargs == {"tracer_provider": fake_tracer_provider}
    assert instrumentor._is_instrumented is True

    processors = fake_tracer_provider._active_span_processor._span_processors
    assert isinstance(processors[0], CohereSpanProcessor)
    assert processors[1] is fake_tracer_provider._active_span_processor.export_processor

    instrumentor.deactivate()

    assert upstream.is_uninstrumented is True
    assert instrumentor._is_instrumented is False
    assert fake_tracer_provider._active_span_processor._span_processors == (
        fake_tracer_provider._active_span_processor.export_processor,
    )


def test_activate_passes_constructor_and_instrument_kwargs(
    monkeypatch,
    fake_tracer_provider,
):
    fake = _install_fake_modules(monkeypatch)

    instrumentor = CohereInstrumentor(
        exception_logger=logging.getLogger("cohere-test"),
        use_legacy_attributes=False,
        logger_provider="logger-provider",
    )
    instrumentor.activate()

    upstream = fake.cohere_instrumentor_class.created[0]
    assert upstream.constructor_kwargs == {
        "exception_logger": logging.getLogger("cohere-test"),
        "use_legacy_attributes": False,
    }
    assert upstream.instrument_kwargs == {
        "tracer_provider": fake_tracer_provider,
        "logger_provider": "logger-provider",
    }


def test_activate_is_idempotent(monkeypatch, fake_tracer_provider):
    fake = _install_fake_modules(monkeypatch)

    instrumentor = CohereInstrumentor()
    instrumentor.activate()
    instrumentor.activate()

    assert len(fake.cohere_instrumentor_class.created) == 1
    processors = fake_tracer_provider._active_span_processor._span_processors
    assert sum(isinstance(item, CohereSpanProcessor) for item in processors) == 1


def test_activate_cleans_up_processor_when_activation_fails(
    monkeypatch,
    caplog,
    fake_tracer_provider,
):
    fake = _install_fake_modules(monkeypatch)

    def instrument_raises(self, **kwargs):
        self.instrument_kwargs = kwargs
        raise RuntimeError("boom")

    monkeypatch.setattr(
        fake.cohere_instrumentor_class,
        "instrument",
        instrument_raises,
    )

    instrumentor = CohereInstrumentor()
    with caplog.at_level(logging.ERROR):
        instrumentor.activate()

    upstream = fake.cohere_instrumentor_class.created[0]
    assert upstream.is_uninstrumented is False
    assert instrumentor._instrumentor is None
    assert instrumentor._processor is None
    assert instrumentor._is_instrumented is False
    assert fake_tracer_provider._active_span_processor._span_processors == (
        fake_tracer_provider._active_span_processor.export_processor,
    )
    assert "Failed to activate Cohere instrumentation" in caplog.text


def test_activate_cleans_up_when_upstream_declines_to_instrument(
    monkeypatch,
    caplog,
    fake_tracer_provider,
):
    fake = _install_fake_modules(monkeypatch)

    def instrument_does_not_activate(self, **kwargs):
        self.instrument_kwargs = kwargs

    monkeypatch.setattr(
        fake.cohere_instrumentor_class,
        "instrument",
        instrument_does_not_activate,
    )

    instrumentor = CohereInstrumentor()
    with caplog.at_level(logging.WARNING):
        instrumentor.activate()

    upstream = fake.cohere_instrumentor_class.created[0]
    assert upstream.is_uninstrumented is False
    assert instrumentor._instrumentor is None
    assert instrumentor._processor is None
    assert instrumentor._is_instrumented is False
    assert fake_tracer_provider._active_span_processor._span_processors == (
        fake_tracer_provider._active_span_processor.export_processor,
    )
    assert "upstream instrumentor did not activate" in caplog.text


def test_activate_skips_when_respan_tracing_is_disabled(
    monkeypatch,
    caplog,
    fake_tracer_provider,
):
    fake = _install_fake_modules(monkeypatch)
    RespanTracer(is_enabled=False)

    instrumentor = CohereInstrumentor()
    with caplog.at_level(logging.INFO):
        instrumentor.activate()

    assert fake.cohere_instrumentor_class.created == []
    assert instrumentor._is_instrumented is False
    assert fake_tracer_provider._active_span_processor._span_processors == (
        fake_tracer_provider._active_span_processor.export_processor,
    )
    assert (
        "Cohere instrumentation skipped because Respan tracing is disabled"
        in caplog.text
    )


def test_activate_logs_warning_when_dependencies_are_missing(
    monkeypatch,
    caplog,
    fake_tracer_provider,
):
    def import_module_raises(module_name):
        if module_name == OTEL_COHERE_MODULE:
            raise ImportError(module_name)
        raise AssertionError(f"unexpected import: {module_name}")

    monkeypatch.setattr(
        _instrumentation.importlib,
        "import_module",
        import_module_raises,
    )
    instrumentor = CohereInstrumentor()

    with caplog.at_level(logging.WARNING):
        instrumentor.activate()

    assert "Failed to activate Cohere instrumentation" in caplog.text
    assert instrumentor._is_instrumented is False
    assert fake_tracer_provider._active_span_processor._span_processors == (
        fake_tracer_provider._active_span_processor.export_processor,
    )


def test_processor_normalizes_chat_contract_and_strips_aliases():
    span = FakeSpan(
        {
            SpanAttributes.LLM_SYSTEM: "Cohere",
            SpanAttributes.LLM_REQUEST_TYPE: LLMRequestTypeValues.CHAT.value,
            SpanAttributes.LLM_USAGE_PROMPT_TOKENS: 12,
            SpanAttributes.LLM_USAGE_COMPLETION_TOKENS: 5,
            f"{SpanAttributes.LLM_REQUEST_FUNCTIONS}.0.name": "lookup_order",
            f"{SpanAttributes.LLM_REQUEST_FUNCTIONS}.0.description": "Lookup an order.",
            f"{SpanAttributes.LLM_REQUEST_FUNCTIONS}.0.parameters": json.dumps(
                {
                    "type": "object",
                    "properties": {"order_id": {"type": "string"}},
                }
            ),
            f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls.0.id": "call_1",
            f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls.0.name": "lookup_order",
            f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls.0.arguments": '{"order_id":"A1"}',
            "model": "command-a-03-2025",
            "tools": "[]",
            "tool_calls": "[]",
            RESPAN_SPAN_TOOLS: "[]",
        }
    )

    CohereSpanProcessor().on_end(span)

    assert span._attributes[RESPAN_LOG_TYPE] == "chat"
    assert span._attributes[SpanAttributes.LLM_SYSTEM] == "cohere"
    assert span._attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 12
    assert span._attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] == 5
    assert span._attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] == 17
    assert json.loads(span._attributes[SpanAttributes.LLM_REQUEST_FUNCTIONS]) == [
        {
            "name": "lookup_order",
            "description": "Lookup an order.",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
            },
        }
    ]
    assert json.loads(
        span._attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls"]
    ) == [
        {
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "lookup_order",
                "arguments": '{"order_id":"A1"}',
            },
        }
    ]
    assert (
        f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls.0.name" not in span._attributes
    )
    assert f"{SpanAttributes.LLM_REQUEST_FUNCTIONS}.0.name" not in span._attributes
    assert "model" not in span._attributes
    assert "tools" not in span._attributes
    assert "tool_calls" not in span._attributes
    assert RESPAN_SPAN_TOOLS not in span._attributes


def test_processor_maps_embedding_and_rerank_log_types():
    embedding_span = FakeSpan(
        {
            SpanAttributes.LLM_SYSTEM: "Cohere",
            SpanAttributes.LLM_REQUEST_TYPE: LLMRequestTypeValues.EMBEDDING.value,
        },
        name="cohere.embed",
    )
    rerank_span = FakeSpan(
        {
            SpanAttributes.LLM_SYSTEM: "Cohere",
            SpanAttributes.LLM_REQUEST_TYPE: LLMRequestTypeValues.RERANK.value,
        },
        name="cohere.rerank",
    )

    processor = CohereSpanProcessor()
    processor.on_end(embedding_span)
    processor.on_end(rerank_span)

    assert embedding_span._attributes[RESPAN_LOG_TYPE] == "embedding"
    assert rerank_span._attributes[RESPAN_LOG_TYPE] == "task"


def test_processor_ignores_non_cohere_spans():
    span = FakeSpan({SpanAttributes.LLM_SYSTEM: "openai"}, name="openai.chat")
    span.instrumentation_scope = SimpleNamespace(
        name="opentelemetry.instrumentation.openai"
    )

    CohereSpanProcessor().on_end(span)

    assert RESPAN_LOG_TYPE not in span._attributes
    assert span._attributes[SpanAttributes.LLM_SYSTEM] == "openai"
