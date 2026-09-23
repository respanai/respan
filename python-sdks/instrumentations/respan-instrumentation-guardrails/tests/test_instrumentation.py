import logging
import sys
from types import ModuleType, SimpleNamespace

import pytest
from opentelemetry.semconv_ai import LLMRequestTypeValues, SpanAttributes
from respan_instrumentation_guardrails import GuardrailsInstrumentor, _instrumentation
from respan_instrumentation_guardrails._instrumentation import (
    GUARDRAILS_RUNTIME_MODULE,
    GuardrailsSpanProcessor,
)
from respan_sdk.constants.llm_logging import LOG_TYPE_CHAT, LOG_TYPE_GUARDRAIL
from respan_sdk.constants.span_attributes import (
    LLM_REQUEST_MODEL,
    LLM_REQUEST_TYPE,
    LLM_USAGE_COMPLETION_TOKENS,
    LLM_USAGE_PROMPT_TOKENS,
    RESPAN_LOG_TYPE,
)
from respan_tracing.core.tracer import RespanTracer


class FakeSpan:
    def __init__(self, name, *, trace_id=None):
        self.name = name
        self.attributes = {}
        self.exceptions = []
        self.status = None
        self._trace_id = trace_id

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def set_attribute(self, key, value):
        self.attributes[key] = value

    def record_exception(self, exception):
        self.exceptions.append(exception)

    def set_status(self, status):
        self.status = status

    def get_span_context(self):
        return SimpleNamespace(trace_id=self._trace_id)


class FakeTracer:
    def __init__(self):
        self.spans = []

    def start_as_current_span(self, name):
        span = FakeSpan(name=name)
        self.spans.append(span)
        return span


class FakeReadableSpan:
    def __init__(self, name, attributes, *, trace_id=None):
        self.name = name
        self._attributes = dict(attributes)
        self._trace_id = trace_id

    def get_span_context(self):
        return SimpleNamespace(trace_id=self._trace_id)


def _install_fake_guardrails(monkeypatch):
    class FakeGuard:
        def __call__(self, *args, **kwargs):
            return SimpleNamespace(
                validation_passed=True,
                validated_output={"mode": "call"},
                raw_llm_output='{"mode": "call"}',
            )

        def parse(self, *args, **kwargs):
            return SimpleNamespace(
                validation_passed=True,
                validated_output={"mode": "parse"},
                raw_llm_output=kwargs.get("llm_output", ""),
            )

        def validate(self, llm_output, *args, **kwargs):
            return SimpleNamespace(
                validation_passed=True,
                validated_output=llm_output,
                raw_llm_output=llm_output,
            )

    guardrails_module = ModuleType(GUARDRAILS_RUNTIME_MODULE)
    guardrails_module.Guard = FakeGuard
    monkeypatch.setitem(sys.modules, GUARDRAILS_RUNTIME_MODULE, guardrails_module)
    return FakeGuard


@pytest.fixture(autouse=True)
def reset_tracer():
    RespanTracer.reset_instance()
    yield
    RespanTracer.reset_instance()


def test_activate_uses_native_guardrails_spans_without_wrapping(monkeypatch):
    fake_guard_class = _install_fake_guardrails(monkeypatch)
    original_parse = fake_guard_class.parse

    instrumentor = GuardrailsInstrumentor()
    instrumentor.activate()

    result = fake_guard_class().parse(
        llm_output='{"issue": "late shipment"}',
        num_reasks=0,
    )

    assert result.validation_passed is True
    assert fake_guard_class.parse is original_parse
    assert instrumentor.is_instrumented is True


def test_call_and_validate_are_not_wrapped(monkeypatch):
    fake_guard_class = _install_fake_guardrails(monkeypatch)
    original_call = fake_guard_class.__call__
    original_validate = fake_guard_class.validate

    instrumentor = GuardrailsInstrumentor()
    instrumentor.activate()

    assert fake_guard_class.__call__ is original_call
    assert fake_guard_class.validate is original_validate


def test_span_processor_bridges_propagated_context_on_start(monkeypatch):
    monkeypatch.setattr(
        _instrumentation,
        "read_propagated_attributes",
        lambda: {
            "respan.customer_params.customer_identifier": "customer-1",
            "respan.threads.thread_identifier": "thread-1",
            "respan.metadata.run_id": "marker-1",
        },
    )
    span = FakeSpan("step")

    GuardrailsSpanProcessor().on_start(span)

    assert span.attributes["respan.customer_params.customer_identifier"] == "customer-1"
    assert span.attributes["respan.threads.thread_identifier"] == "thread-1"
    assert span.attributes["respan.metadata.run_id"] == "marker-1"


def test_span_processor_reuses_trace_context_when_nested_runtime_loses_it(
    monkeypatch,
):
    propagated = {
        "respan.customer_params.customer_identifier": "customer-1",
        "respan.threads.thread_identifier": "thread-1",
        "respan.metadata.run_id": "marker-1",
    }
    current = {"value": propagated}
    monkeypatch.setattr(
        _instrumentation,
        "read_propagated_attributes",
        lambda: current["value"],
    )
    processor = GuardrailsSpanProcessor()
    root = FakeSpan("workflow", trace_id=123)
    child = FakeSpan("step", trace_id=123)

    processor.on_start(root)
    current["value"] = {}
    processor.on_start(child)

    assert child.attributes == propagated

    processor.on_end(
        FakeReadableSpan("step", {"type": "guardrails/guard/step"}, trace_id=123)
    )
    processor.on_end(FakeReadableSpan("workflow", {}, trace_id=123))
    assert processor._propagated_by_trace == {}
    assert processor._active_spans_by_trace == {}


def test_deactivate_leaves_native_methods_unchanged(monkeypatch):
    fake_guard_class = _install_fake_guardrails(monkeypatch)
    original_parse = fake_guard_class.parse

    instrumentor = GuardrailsInstrumentor()
    instrumentor.activate()
    assert fake_guard_class.parse is original_parse

    instrumentor.deactivate()
    assert fake_guard_class.parse is original_parse
    assert instrumentor._is_instrumented is False


def test_activate_skips_when_respan_tracing_is_disabled(monkeypatch, caplog):
    fake_guard_class = _install_fake_guardrails(monkeypatch)
    original_parse = fake_guard_class.parse
    RespanTracer(is_enabled=False)

    instrumentor = GuardrailsInstrumentor()
    with caplog.at_level(logging.INFO):
        instrumentor.activate()

    assert fake_guard_class.parse is original_parse
    assert instrumentor._is_instrumented is False
    assert (
        "Guardrails instrumentation skipped because Respan tracing is disabled"
        in caplog.text
    )


def test_activate_logs_warning_when_guardrails_runtime_is_missing(monkeypatch, caplog):
    def import_module_raises(module_name):
        if module_name == GUARDRAILS_RUNTIME_MODULE:
            raise ImportError(module_name)
        raise AssertionError(f"unexpected import: {module_name}")

    monkeypatch.setattr(
        _instrumentation.importlib,
        "import_module",
        import_module_raises,
    )
    instrumentor = GuardrailsInstrumentor()

    with caplog.at_level(logging.WARNING):
        instrumentor.activate()

    assert "Failed to activate Guardrails instrumentation" in caplog.text
    assert instrumentor._is_instrumented is False


def test_span_processor_translates_guardrails_llm_call_span():
    span = FakeReadableSpan(
        name="call",
        attributes={
            "type": "guardrails/guard/step/call",
            "llm.input_messages.0.message.role": "user",
            "llm.input_messages.0.message.content": "Return JSON",
            "llm.output_messages.0.message.role": "assistant",
            "llm.output_messages.0.message.content": '{"ok": true}',
            "llm.invocation_parameters": "{'model': 'gpt-4o', 'temperature': 0}",
            "llm.token_count.prompt": "27",
            "llm.token_count.completion": "105",
            "llm.token_count.total": "132",
            "input.value": '{"messages": []}',
            "output.value": '{"output": "{\\"ok\\": true}"}',
        },
    )

    GuardrailsSpanProcessor().on_end(span)

    assert span._attributes[RESPAN_LOG_TYPE] == LOG_TYPE_CHAT
    assert SpanAttributes.TRACELOOP_SPAN_KIND not in span._attributes
    assert span._attributes[LLM_REQUEST_TYPE] == LLMRequestTypeValues.CHAT.value
    assert span._attributes[LLM_REQUEST_MODEL] == "gpt-4o"
    assert span._attributes[SpanAttributes.LLM_REQUEST_TEMPERATURE] == 0
    assert span._attributes[LLM_USAGE_PROMPT_TOKENS] == 27
    assert span._attributes[LLM_USAGE_COMPLETION_TOKENS] == 105
    assert span._attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] == 132
    assert span._attributes["gen_ai.prompt.0.role"] == "user"
    assert span._attributes["gen_ai.prompt.0.content"] == "Return JSON"
    assert span._attributes["gen_ai.completion.0.role"] == "assistant"
    assert span._attributes["gen_ai.completion.0.content"] == '{"ok": true}'
    assert span._attributes["traceloop.entity.name"] == "guardrails.call"
    assert span._attributes["traceloop.entity.input"] == '{"messages": []}'
    assert (
        span._attributes["traceloop.entity.output"] == '{"output": "{\\"ok\\": true}"}'
    )


def test_span_processor_marks_guardrails_non_llm_span_as_guardrail():
    span = FakeReadableSpan(
        name="guard",
        attributes={
            "type": "guardrails/guard",
            "input.value": "input",
            "output.value": "output",
        },
    )

    GuardrailsSpanProcessor().on_end(span)

    assert span._attributes[RESPAN_LOG_TYPE] == LOG_TYPE_GUARDRAIL
    assert SpanAttributes.TRACELOOP_SPAN_KIND not in span._attributes
    assert span._attributes["traceloop.entity.name"] == "guardrails.guard"
    assert span._attributes["traceloop.entity.input"] == "input"
    assert span._attributes["traceloop.entity.output"] == "output"


def test_span_processor_keeps_local_validation_step_call_as_guardrail():
    span = FakeReadableSpan(
        name="call",
        attributes={
            "type": "guardrails/guard/step/call",
            "input.value": '{"args": []}',
            "output.value": '{"output": "validated"}',
        },
    )

    GuardrailsSpanProcessor().on_end(span)

    assert span._attributes[RESPAN_LOG_TYPE] == LOG_TYPE_GUARDRAIL
    assert SpanAttributes.TRACELOOP_SPAN_KIND not in span._attributes
    assert LLM_REQUEST_TYPE not in span._attributes
    assert LLM_USAGE_PROMPT_TOKENS not in span._attributes
    assert span._attributes["traceloop.entity.name"] == "guardrails.call"


def test_span_processor_corrects_local_guard_llm_call_count() -> None:
    span = FakeReadableSpan(
        name="guard",
        attributes={
            "type": "guardrails/guard",
            "token_consumption": 0,
            "number_of_llm_calls": 1,
        },
    )

    GuardrailsSpanProcessor().on_end(span)

    assert span._attributes["number_of_llm_calls"] == 0


def test_span_processor_ignores_unrelated_span():
    span = FakeReadableSpan(name="http", attributes={"http.method": "POST"})

    GuardrailsSpanProcessor().on_end(span)

    assert span._attributes == {"http.method": "POST"}
