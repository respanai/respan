import json
import logging
import sys
from types import ModuleType, SimpleNamespace

import pytest
from opentelemetry.semconv_ai import SpanAttributes as TLSpanAttributes
from respan_sdk.constants.span_attributes import (
    RESPAN_SPAN_TOOL_CALLS,
    RESPAN_SPAN_TOOLS,
)
from respan_instrumentation_smolagents._constants import (
    SPAN_ALIAS_COMPLETION_TOKENS,
    SPAN_ALIAS_MODEL,
    SPAN_ALIAS_PROMPT_TOKENS,
    SPAN_ALIAS_TOOL_CALLS,
    SPAN_ALIAS_TOOLS,
    SPAN_ALIAS_TOTAL_REQUEST_TOKENS,
)

from respan_instrumentation_smolagents import SmolagentsInstrumentor
from respan_instrumentation_smolagents import _instrumentation
from respan_instrumentation_smolagents._constants import (
    GEN_AI_COMPLETION_CONTENT_ATTR,
    GEN_AI_COMPLETION_ROLE_ATTR,
    GEN_AI_COMPLETION_TOOL_CALLS_ATTR,
    LLM_REQUEST_FUNCTIONS_ATTR,
    OPENINFERENCE_INPUT_MESSAGES_ATTR,
    OPENINFERENCE_INSTRUMENTATION_MODULE,
    OPENINFERENCE_MESSAGE_CONTENT_ATTR,
    OPENINFERENCE_MESSAGE_CONTENT_TEXT_ATTR,
    OPENINFERENCE_MESSAGE_CONTENT_TYPE_ATTR,
    OPENINFERENCE_MESSAGE_CONTENTS_ATTR,
    OPENINFERENCE_MESSAGE_ROLE_ATTR,
    OPENINFERENCE_SMOLAGENTS_MODULE,
    OPENINFERENCE_OUTPUT_MESSAGES_ATTR,
    OTEL_SCOPE_NAME,
    SMOLAGENTS_FINAL_ANSWER_ARGUMENT,
    SMOLAGENTS_FINAL_ANSWER_TOOL_NAME,
    TOOL_CALL_FUNCTION_ARGUMENTS_FIELD,
    TOOL_CALL_FUNCTION_FIELD,
    TOOL_CALL_FUNCTION_NAME_FIELD,
)
from respan_instrumentation_smolagents._processor import (
    SmolagentsSpanContentProcessor,
    SmolagentsSpanContractProcessor,
)
from respan_tracing.core.tracer import RespanTracer

NON_SMOLAGENTS_SCOPE_NAME = f"{OPENINFERENCE_INSTRUMENTATION_MODULE}.crewai"


def _install_fake_modules(monkeypatch):
    class FakeSmolagentsInstrumentor:
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
    openinference_instrumentation_module = ModuleType(
        OPENINFERENCE_INSTRUMENTATION_MODULE
    )
    openinference_smolagents_module = ModuleType(OPENINFERENCE_SMOLAGENTS_MODULE)
    openinference_smolagents_module.SmolagentsInstrumentor = FakeSmolagentsInstrumentor
    openinference_instrumentation_module.smolagents = openinference_smolagents_module

    monkeypatch.setitem(sys.modules, "openinference", openinference_module)
    monkeypatch.setitem(
        sys.modules,
        OPENINFERENCE_INSTRUMENTATION_MODULE,
        openinference_instrumentation_module,
    )
    monkeypatch.setitem(
        sys.modules,
        OPENINFERENCE_SMOLAGENTS_MODULE,
        openinference_smolagents_module,
    )

    monkeypatch.setattr(
        _instrumentation,
        "OpenInferenceInstrumentor",
        FakeOpenInferenceInstrumentor,
    )

    return SimpleNamespace(
        smolagents_instrumentor_class=FakeSmolagentsInstrumentor,
        openinference_instrumentor_class=FakeOpenInferenceInstrumentor,
    )


def _make_fake_tracer_provider(processors=()):
    return SimpleNamespace(
        _active_span_processor=SimpleNamespace(_span_processors=processors),
        add_span_processor=lambda processor: None,
    )


def _oi_message_attr(prefix: str, index: int, attr: str) -> str:
    return f"{prefix}.{index}.{attr}"


def _oi_message_content_attr(
    prefix: str,
    message_index: int,
    content_index: int,
    attr: str,
) -> str:
    return (
        f"{prefix}.{message_index}.{OPENINFERENCE_MESSAGE_CONTENTS_ATTR}."
        f"{content_index}.{attr}"
    )


@pytest.fixture(autouse=True)
def reset_tracer():
    RespanTracer.reset_instance()
    yield
    RespanTracer.reset_instance()


def test_activate_uses_openinference_smolagents(monkeypatch):
    fake = _install_fake_modules(monkeypatch)
    tracer_provider = _make_fake_tracer_provider()
    monkeypatch.setattr(
        _instrumentation.trace,
        "get_tracer_provider",
        lambda: tracer_provider,
    )

    instrumentor = SmolagentsInstrumentor()
    instrumentor.activate()

    delegate = fake.openinference_instrumentor_class.created[0]
    assert delegate.instrumentor_class is fake.smolagents_instrumentor_class
    assert delegate.kwargs == {}
    assert delegate.is_activated is True
    assert instrumentor._is_instrumented is True

    instrumentor.deactivate()

    assert delegate.is_deactivated is True
    assert instrumentor._is_instrumented is False


def test_activate_passes_custom_openinference_kwargs(monkeypatch):
    fake = _install_fake_modules(monkeypatch)
    tracer_provider = _make_fake_tracer_provider()
    monkeypatch.setattr(
        _instrumentation.trace,
        "get_tracer_provider",
        lambda: tracer_provider,
    )

    instrumentor = SmolagentsInstrumentor(trace_content=False)
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

    instrumentor = SmolagentsInstrumentor()
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

    instrumentor = SmolagentsInstrumentor()
    with caplog.at_level(logging.ERROR):
        instrumentor.activate()

    delegate = fake.openinference_instrumentor_class.created[0]
    assert delegate.is_deactivated is True
    assert instrumentor._delegate is None
    assert instrumentor._is_instrumented is False
    assert "Failed to activate smolagents instrumentation" in caplog.text


def test_activate_skips_when_respan_tracing_is_disabled(monkeypatch, caplog):
    fake = _install_fake_modules(monkeypatch)
    RespanTracer(is_enabled=False)

    instrumentor = SmolagentsInstrumentor()
    with caplog.at_level(logging.INFO):
        instrumentor.activate()

    assert fake.openinference_instrumentor_class.created == []
    assert instrumentor._is_instrumented is False
    assert (
        "smolagents instrumentation skipped because Respan tracing is disabled"
        in caplog.text
    )


def test_activate_logs_warning_when_dependencies_are_missing(monkeypatch, caplog):
    def import_module_raises(module_name):
        if module_name == OPENINFERENCE_SMOLAGENTS_MODULE:
            raise ImportError(module_name)
        raise AssertionError(f"unexpected import: {module_name}")

    monkeypatch.setattr(
        _instrumentation.importlib,
        "import_module",
        import_module_raises,
    )
    instrumentor = SmolagentsInstrumentor()

    with caplog.at_level(logging.WARNING):
        instrumentor.activate()

    assert "Failed to activate smolagents instrumentation" in caplog.text
    assert instrumentor._is_instrumented is False


def test_activate_registers_content_and_contract_processors_around_translator(
    monkeypatch,
):
    fake = _install_fake_modules(monkeypatch)

    class FakeOpenInferenceTranslator:
        pass

    translator = FakeOpenInferenceTranslator()
    tracer_provider = _make_fake_tracer_provider(
        processors=(translator, "exporter"),
    )
    monkeypatch.setattr(
        _instrumentation,
        "OpenInferenceTranslator",
        FakeOpenInferenceTranslator,
    )
    monkeypatch.setattr(
        _instrumentation.trace,
        "get_tracer_provider",
        lambda: tracer_provider,
    )

    instrumentor = SmolagentsInstrumentor()
    instrumentor.activate()

    processors = tracer_provider._active_span_processor._span_processors
    assert isinstance(processors[0], SmolagentsSpanContentProcessor)
    assert processors[1] is translator
    assert isinstance(processors[2], SmolagentsSpanContractProcessor)
    assert processors[3] == "exporter"
    assert fake.openinference_instrumentor_class.created[0].is_activated is True

    instrumentor.deactivate()

    processors = tracer_provider._active_span_processor._span_processors
    assert processors == (translator, "exporter")


def test_content_processor_flattens_openinference_message_content():
    processor = SmolagentsSpanContentProcessor()
    span = SimpleNamespace(
        _attributes={
            OTEL_SCOPE_NAME: OPENINFERENCE_SMOLAGENTS_MODULE,
            _oi_message_attr(
                OPENINFERENCE_INPUT_MESSAGES_ATTR,
                0,
                OPENINFERENCE_MESSAGE_ROLE_ATTR,
            ): "user",
            _oi_message_content_attr(
                OPENINFERENCE_INPUT_MESSAGES_ATTR,
                0,
                0,
                OPENINFERENCE_MESSAGE_CONTENT_TYPE_ATTR,
            ): "text",
            _oi_message_content_attr(
                OPENINFERENCE_INPUT_MESSAGES_ATTR,
                0,
                0,
                OPENINFERENCE_MESSAGE_CONTENT_TEXT_ATTR,
            ): "hello",
            _oi_message_content_attr(
                OPENINFERENCE_INPUT_MESSAGES_ATTR,
                0,
                1,
                OPENINFERENCE_MESSAGE_CONTENT_TYPE_ATTR,
            ): "text",
            _oi_message_content_attr(
                OPENINFERENCE_INPUT_MESSAGES_ATTR,
                0,
                1,
                OPENINFERENCE_MESSAGE_CONTENT_TEXT_ATTR,
            ): "world",
            _oi_message_attr(
                OPENINFERENCE_OUTPUT_MESSAGES_ATTR,
                0,
                OPENINFERENCE_MESSAGE_ROLE_ATTR,
            ): "assistant",
            _oi_message_content_attr(
                OPENINFERENCE_OUTPUT_MESSAGES_ATTR,
                0,
                0,
                OPENINFERENCE_MESSAGE_CONTENT_TYPE_ATTR,
            ): "text",
            _oi_message_content_attr(
                OPENINFERENCE_OUTPUT_MESSAGES_ATTR,
                0,
                0,
                OPENINFERENCE_MESSAGE_CONTENT_TEXT_ATTR,
            ): "done",
        }
    )

    processor.on_end(span)

    assert (
        span._attributes[
            _oi_message_attr(
                OPENINFERENCE_INPUT_MESSAGES_ATTR,
                0,
                OPENINFERENCE_MESSAGE_CONTENT_ATTR,
            )
        ]
        == "hello\nworld"
    )
    assert (
        span._attributes[
            _oi_message_attr(
                OPENINFERENCE_OUTPUT_MESSAGES_ATTR,
                0,
                OPENINFERENCE_MESSAGE_CONTENT_ATTR,
            )
        ]
        == "done"
    )


def test_contract_processor_normalizes_and_removes_aliases_from_smolagents_spans():
    processor = SmolagentsSpanContractProcessor()
    nested_tool_call_attr = f"{GEN_AI_COMPLETION_TOOL_CALLS_ATTR}.0.id"
    span = SimpleNamespace(
        _attributes={
            OTEL_SCOPE_NAME: OPENINFERENCE_SMOLAGENTS_MODULE,
            SPAN_ALIAS_MODEL: "openai/gpt-4o-mini",
            SPAN_ALIAS_PROMPT_TOKENS: 10,
            SPAN_ALIAS_COMPLETION_TOKENS: 5,
            SPAN_ALIAS_TOTAL_REQUEST_TOKENS: 15,
            SPAN_ALIAS_TOOLS: [{"type": "function"}],
            SPAN_ALIAS_TOOL_CALLS: [{"id": "call_1"}],
            RESPAN_SPAN_TOOLS: '[{"type":"function"}]',
            RESPAN_SPAN_TOOL_CALLS: '[{"id":"call_1"}]',
            TLSpanAttributes.LLM_REQUEST_MODEL: "openai/gpt-4o-mini",
            LLM_REQUEST_FUNCTIONS_ATTR: [{"type": "function"}],
            GEN_AI_COMPLETION_TOOL_CALLS_ATTR: [{"id": "call_1"}],
            nested_tool_call_attr: "call_1",
        }
    )

    processor.on_end(span)

    assert SPAN_ALIAS_MODEL not in span._attributes
    assert SPAN_ALIAS_PROMPT_TOKENS not in span._attributes
    assert SPAN_ALIAS_COMPLETION_TOKENS not in span._attributes
    assert SPAN_ALIAS_TOTAL_REQUEST_TOKENS not in span._attributes
    assert SPAN_ALIAS_TOOLS not in span._attributes
    assert SPAN_ALIAS_TOOL_CALLS not in span._attributes
    assert RESPAN_SPAN_TOOLS not in span._attributes
    assert RESPAN_SPAN_TOOL_CALLS not in span._attributes
    assert span._attributes[TLSpanAttributes.LLM_REQUEST_MODEL] == "openai/gpt-4o-mini"
    assert json.loads(span._attributes[LLM_REQUEST_FUNCTIONS_ATTR]) == [
        {"type": "function"}
    ]
    assert span._attributes[GEN_AI_COMPLETION_TOOL_CALLS_ATTR] == [{"id": "call_1"}]
    assert nested_tool_call_attr not in span._attributes
    assert span._attributes[GEN_AI_COMPLETION_ROLE_ATTR] == "assistant"
    assert span._attributes[GEN_AI_COMPLETION_CONTENT_ATTR] == ""


def test_contract_processor_backfills_canonical_tool_fields_from_helpers():
    processor = SmolagentsSpanContractProcessor()
    span = SimpleNamespace(
        _attributes={
            OTEL_SCOPE_NAME: OPENINFERENCE_SMOLAGENTS_MODULE,
            RESPAN_SPAN_TOOLS: '[{"type":"function"}]',
            RESPAN_SPAN_TOOL_CALLS: '[{"id":"call_1"}]',
        }
    )

    processor.on_end(span)

    assert json.loads(span._attributes[LLM_REQUEST_FUNCTIONS_ATTR]) == [
        {"type": "function"}
    ]
    assert span._attributes[GEN_AI_COMPLETION_TOOL_CALLS_ATTR] == [{"id": "call_1"}]
    assert RESPAN_SPAN_TOOLS not in span._attributes
    assert RESPAN_SPAN_TOOL_CALLS not in span._attributes


def test_contract_processor_promotes_final_answer_tool_call_to_content():
    processor = SmolagentsSpanContractProcessor()
    span = SimpleNamespace(
        _attributes={
            OTEL_SCOPE_NAME: OPENINFERENCE_SMOLAGENTS_MODULE,
            RESPAN_SPAN_TOOL_CALLS: json.dumps(
                [
                    {
                        "id": "call_final",
                        "type": "function",
                        TOOL_CALL_FUNCTION_FIELD: {
                            TOOL_CALL_FUNCTION_NAME_FIELD: (
                                SMOLAGENTS_FINAL_ANSWER_TOOL_NAME
                            ),
                            TOOL_CALL_FUNCTION_ARGUMENTS_FIELD: json.dumps(
                                {
                                    SMOLAGENTS_FINAL_ANSWER_ARGUMENT: (
                                        "The final total is $63."
                                    )
                                }
                            ),
                        },
                    }
                ]
            ),
        }
    )

    processor.on_end(span)

    assert span._attributes[GEN_AI_COMPLETION_CONTENT_ATTR] == (
        "The final total is $63."
    )
    assert span._attributes[GEN_AI_COMPLETION_ROLE_ATTR] == "assistant"
    assert GEN_AI_COMPLETION_TOOL_CALLS_ATTR not in span._attributes
    assert RESPAN_SPAN_TOOL_CALLS not in span._attributes


def test_contract_processor_ignores_non_smolagents_spans():
    processor = SmolagentsSpanContractProcessor()
    span = SimpleNamespace(
        _attributes={
            OTEL_SCOPE_NAME: NON_SMOLAGENTS_SCOPE_NAME,
            SPAN_ALIAS_MODEL: "openai/gpt-4o-mini",
            SPAN_ALIAS_TOOLS: [{"type": "function"}],
        }
    )

    processor.on_end(span)

    assert span._attributes[SPAN_ALIAS_MODEL] == "openai/gpt-4o-mini"
    assert span._attributes[SPAN_ALIAS_TOOLS] == [{"type": "function"}]
