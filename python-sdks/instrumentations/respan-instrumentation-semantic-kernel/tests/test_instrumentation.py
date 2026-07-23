import json
import logging
import os
import sys
from types import ModuleType, SimpleNamespace

import pytest
from opentelemetry.semconv._incubating.attributes import gen_ai_attributes as GenAIAttributes
from opentelemetry.semconv_ai import SpanAttributes

from respan_instrumentation_semantic_kernel import SemanticKernelInstrumentor
from respan_instrumentation_semantic_kernel import _instrumentation
from respan_instrumentation_semantic_kernel._constants import (
    SEMANTIC_KERNEL_ENABLE_OTEL_DIAGNOSTICS_ENV,
    SEMANTIC_KERNEL_ENABLE_OTEL_DIAGNOSTICS_SENSITIVE_ENV,
    SEMANTIC_KERNEL_MODEL_DIAGNOSTICS_LOGGER,
    SEMANTIC_KERNEL_ROOT_MODULE,
)
from respan_instrumentation_semantic_kernel._processor import (
    SemanticKernelLogRecordHandler,
    SemanticKernelSpanProcessor,
    enrich_semantic_kernel_span,
)
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
    def __init__(self, attrs, name="chat.completions gpt-4o-mini", scope=None):
        self.name = name
        self._attributes = dict(attrs)
        self.attributes = self._attributes
        self.events = ()
        self.instrumentation_scope = SimpleNamespace(
            name=scope or "semantic_kernel.utils.telemetry.model_diagnostics.decorators"
        )


class FakeRecordingSpan:
    def __init__(self):
        self.attributes = {}

    def set_attribute(self, key, value):
        self.attributes[key] = value

    def is_recording(self):
        return True


def _install_fake_semantic_kernel(monkeypatch):
    semantic_kernel_module = ModuleType(SEMANTIC_KERNEL_ROOT_MODULE)

    decorators_module = ModuleType(
        "semantic_kernel.utils.telemetry.model_diagnostics.decorators"
    )
    decorators_module.MODEL_DIAGNOSTICS_SETTINGS = SimpleNamespace(
        enable_otel_diagnostics=False,
        enable_otel_diagnostics_sensitive=False,
    )

    function_tracer_module = ModuleType(
        "semantic_kernel.utils.telemetry.model_diagnostics.function_tracer"
    )
    function_tracer_module.MODEL_DIAGNOSTICS_SETTINGS = SimpleNamespace(
        enable_otel_diagnostics=False,
        enable_otel_diagnostics_sensitive=False,
    )

    monkeypatch.setitem(sys.modules, SEMANTIC_KERNEL_ROOT_MODULE, semantic_kernel_module)
    monkeypatch.setitem(
        sys.modules,
        "semantic_kernel.utils.telemetry.model_diagnostics.decorators",
        decorators_module,
    )
    monkeypatch.setitem(
        sys.modules,
        "semantic_kernel.utils.telemetry.model_diagnostics.function_tracer",
        function_tracer_module,
    )
    return SimpleNamespace(
        decorators_module=decorators_module,
        function_tracer_module=function_tracer_module,
    )


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


def test_activate_registers_processor_handler_and_restores_state(
    monkeypatch,
    fake_tracer_provider,
):
    fake = _install_fake_semantic_kernel(monkeypatch)
    monkeypatch.delenv(SEMANTIC_KERNEL_ENABLE_OTEL_DIAGNOSTICS_ENV, raising=False)
    monkeypatch.setenv(SEMANTIC_KERNEL_ENABLE_OTEL_DIAGNOSTICS_SENSITIVE_ENV, "old")
    diagnostics_logger = logging.getLogger(SEMANTIC_KERNEL_MODEL_DIAGNOSTICS_LOGGER)
    previous_level = diagnostics_logger.level
    previous_handlers = list(diagnostics_logger.handlers)

    instrumentor = SemanticKernelInstrumentor()
    instrumentor.activate()

    processors = fake_tracer_provider._active_span_processor._span_processors
    assert isinstance(processors[0], SemanticKernelSpanProcessor)
    assert processors[1] is fake_tracer_provider._active_span_processor.export_processor
    assert any(isinstance(handler, SemanticKernelLogRecordHandler) for handler in diagnostics_logger.handlers)
    assert os.environ[SEMANTIC_KERNEL_ENABLE_OTEL_DIAGNOSTICS_ENV] == "true"
    assert os.environ[SEMANTIC_KERNEL_ENABLE_OTEL_DIAGNOSTICS_SENSITIVE_ENV] == "true"
    assert fake.decorators_module.MODEL_DIAGNOSTICS_SETTINGS.enable_otel_diagnostics is True
    assert (
        fake.decorators_module.MODEL_DIAGNOSTICS_SETTINGS.enable_otel_diagnostics_sensitive
        is True
    )

    instrumentor.deactivate()

    assert fake_tracer_provider._active_span_processor._span_processors == (
        fake_tracer_provider._active_span_processor.export_processor,
    )
    assert SEMANTIC_KERNEL_ENABLE_OTEL_DIAGNOSTICS_ENV not in os.environ
    assert os.environ[SEMANTIC_KERNEL_ENABLE_OTEL_DIAGNOSTICS_SENSITIVE_ENV] == "old"
    assert fake.decorators_module.MODEL_DIAGNOSTICS_SETTINGS.enable_otel_diagnostics is False
    assert (
        fake.decorators_module.MODEL_DIAGNOSTICS_SETTINGS.enable_otel_diagnostics_sensitive
        is False
    )
    assert diagnostics_logger.level == previous_level
    assert diagnostics_logger.handlers == previous_handlers


def test_activate_is_idempotent(monkeypatch, fake_tracer_provider):
    _install_fake_semantic_kernel(monkeypatch)

    instrumentor = SemanticKernelInstrumentor()
    instrumentor.activate()
    instrumentor.activate()

    processors = fake_tracer_provider._active_span_processor._span_processors
    assert sum(isinstance(item, SemanticKernelSpanProcessor) for item in processors) == 1


def test_activate_skips_when_respan_tracing_is_disabled(
    monkeypatch,
    caplog,
    fake_tracer_provider,
):
    _install_fake_semantic_kernel(monkeypatch)
    RespanTracer(is_enabled=False)

    instrumentor = SemanticKernelInstrumentor()
    with caplog.at_level(logging.INFO):
        instrumentor.activate()

    assert instrumentor._is_instrumented is False
    assert fake_tracer_provider._active_span_processor._span_processors == (
        fake_tracer_provider._active_span_processor.export_processor,
    )
    assert (
        "Semantic Kernel instrumentation skipped because Respan tracing is disabled"
        in caplog.text
    )


def test_activate_logs_warning_when_dependency_missing(
    monkeypatch,
    caplog,
    fake_tracer_provider,
):
    def import_module_raises(module_name):
        if module_name == SEMANTIC_KERNEL_ROOT_MODULE:
            raise ImportError(module_name)
        raise AssertionError(f"unexpected import: {module_name}")

    monkeypatch.setattr(
        _instrumentation.importlib,
        "import_module",
        import_module_raises,
    )

    instrumentor = SemanticKernelInstrumentor()
    with caplog.at_level(logging.WARNING):
        instrumentor.activate()

    assert "Failed to activate Semantic Kernel instrumentation" in caplog.text
    assert instrumentor._is_instrumented is False
    assert fake_tracer_provider._active_span_processor._span_processors == (
        fake_tracer_provider._active_span_processor.export_processor,
    )


def test_log_handler_promotes_prompt_and_completion_to_current_span(monkeypatch):
    current_span = FakeRecordingSpan()
    monkeypatch.setattr(
        "respan_instrumentation_semantic_kernel._processor.trace.get_current_span",
        lambda: current_span,
    )
    handler = SemanticKernelLogRecordHandler()

    prompt_record = logging.LogRecord(
        name="semantic_kernel.utils.telemetry.model_diagnostics.decorators",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg=json.dumps({"role": "user", "content": "Weather in Tokyo?"}),
        args=(),
        exc_info=None,
    )
    prompt_record.__dict__["event.name"] = "gen_ai.user.message"
    prompt_record.CHAT_MESSAGE_INDEX = 0
    handler.emit(prompt_record)

    completion_record = logging.LogRecord(
        name="semantic_kernel.utils.telemetry.model_diagnostics.decorators",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg=json.dumps({
            "message": {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city":"Tokyo"}',
                        },
                    }
                ],
            }
        }),
        args=(),
        exc_info=None,
    )
    completion_record.__dict__["event.name"] = "gen_ai.choice"
    handler.emit(completion_record)

    assert current_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.role"] == "user"
    assert (
        current_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"]
        == "Weather in Tokyo?"
    )
    assert (
        current_span.attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.role"]
        == "assistant"
    )
    assert json.loads(
        current_span.attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls"]
    )[0]["function"]["name"] == "get_weather"


def test_enrich_chat_span_maps_contract_fields_and_strips_aliases():
    span = FakeSpan({
        GenAIAttributes.GEN_AI_OPERATION_NAME: "chat.completions",
        SpanAttributes.LLM_SYSTEM: "openai",
        SpanAttributes.LLM_REQUEST_MODEL: "gpt-4o-mini",
        GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS: 12,
        GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS: 7,
        f"{SpanAttributes.LLM_PROMPTS}.0.role": "user",
        f"{SpanAttributes.LLM_PROMPTS}.0.content": "Hello",
        f"{SpanAttributes.LLM_COMPLETIONS}.0.role": "assistant",
        f"{SpanAttributes.LLM_COMPLETIONS}.0.content": "Hi",
        "model": "bad",
        "prompt_tokens": 12,
        "tools": [],
        "tool_calls": [],
        "respan.span.tools": "[]",
        "respan.span.tool_calls": "[]",
    })

    assert enrich_semantic_kernel_span(span) is True

    attrs = span._attributes
    assert attrs["respan.entity.log_type"] == "chat"
    assert attrs["respan.entity.log_method"] == "tracing_integration"
    assert attrs[SpanAttributes.LLM_REQUEST_TYPE] == "chat"
    assert attrs[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] == 12
    assert attrs[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS] == 7
    assert attrs[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] == 19
    assert json.loads(attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT]) == [
        {"role": "user", "content": "Hello"}
    ]
    assert json.loads(attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]) == [
        {"role": "assistant", "content": "Hi"}
    ]
    for key in (
        "model",
        "prompt_tokens",
        "tools",
        "tool_calls",
        "respan.span.tools",
        "respan.span.tool_calls",
        SpanAttributes.TRACELOOP_SPAN_KIND,
    ):
        assert key not in attrs


def test_enrich_legacy_event_payloads():
    span = FakeSpan(
        {
            GenAIAttributes.GEN_AI_OPERATION_NAME: "chat.completions",
            SpanAttributes.LLM_SYSTEM: "openai",
            SpanAttributes.LLM_REQUEST_MODEL: "gpt-4o-mini",
            "gen_ai.response.prompt_tokens": 3,
            "gen_ai.response.completion_tokens": 4,
        }
    )
    span.events = (
        SimpleNamespace(
            name="gen_ai.content.prompt",
            attributes={"gen_ai.prompt": '[{"role":"user","content":"Hi"}]'},
        ),
        SimpleNamespace(
            name="gen_ai.content.completion",
            attributes={"gen_ai.completion": '[{"role":"assistant","content":"Hello"}]'},
        ),
    )

    enrich_semantic_kernel_span(span)

    attrs = span._attributes
    assert attrs[f"{SpanAttributes.LLM_PROMPTS}.0.content"] == "Hi"
    assert attrs[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"] == "Hello"
    assert attrs[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 3
    assert attrs[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] == 4
    assert "gen_ai.response.prompt_tokens" not in attrs
    assert "gen_ai.response.completion_tokens" not in attrs


def test_enrich_tool_span_maps_contract_fields_and_strips_raw_tool_attrs():
    span = FakeSpan(
        {
            GenAIAttributes.GEN_AI_OPERATION_NAME: "execute_tool",
            GenAIAttributes.GEN_AI_TOOL_NAME: "Weather-get_weather",
            GenAIAttributes.GEN_AI_TOOL_CALL_ARGUMENTS: '{"city":"Tokyo"}',
            GenAIAttributes.GEN_AI_TOOL_CALL_RESULT: "sunny",
            GenAIAttributes.GEN_AI_TOOL_DESCRIPTION: "Get weather",
            "tool_calls": [{"bad": True}],
        },
        name="execute_tool Weather-get_weather",
        scope="semantic_kernel.utils.telemetry.model_diagnostics.function_tracer",
    )

    enrich_semantic_kernel_span(span)

    attrs = span._attributes
    assert attrs["respan.entity.log_type"] == "tool"
    assert attrs[SpanAttributes.TRACELOOP_ENTITY_NAME] == "Weather-get_weather"
    assert attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] == '{"city":"Tokyo"}'
    assert attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] == "sunny"
    for key in (
        GenAIAttributes.GEN_AI_TOOL_NAME,
        GenAIAttributes.GEN_AI_TOOL_CALL_ARGUMENTS,
        GenAIAttributes.GEN_AI_TOOL_CALL_RESULT,
        GenAIAttributes.GEN_AI_TOOL_DESCRIPTION,
        "tool_calls",
        SpanAttributes.TRACELOOP_SPAN_KIND,
    ):
        assert key not in attrs


def test_processor_ignores_non_semantic_kernel_span():
    span = FakeSpan(
        {
            GenAIAttributes.GEN_AI_OPERATION_NAME: "chat.completions",
            "tool_calls": [],
        },
        scope="other.instrumentation",
    )

    SemanticKernelSpanProcessor().on_end(span)

    assert span._attributes == span.attributes
    assert "tool_calls" in span._attributes
