"""Guardrails AI instrumentation plugin for Respan."""

import ast
import importlib
import json
import logging
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor
from opentelemetry.semconv_ai import LLMRequestTypeValues, SpanAttributes
from respan_sdk.constants.llm_logging import LOG_TYPE_CHAT, LOG_TYPE_GUARDRAIL
from respan_sdk.constants.span_attributes import (
    LLM_REQUEST_MODEL,
    LLM_REQUEST_TYPE,
    LLM_USAGE_COMPLETION_TOKENS,
    LLM_USAGE_PROMPT_TOKENS,
    RESPAN_LOG_TYPE,
)
from respan_tracing.core.tracer import RespanTracer
from respan_tracing.utils.span_factory import read_propagated_attributes

logger = logging.getLogger(__name__)

GUARDRAILS_INSTRUMENTATION_NAME = "guardrails"
GUARDRAILS_RUNTIME_MODULE = "guardrails"
GUARDRAILS_GUARD_CLASS = "Guard"

_GUARDRAILS_SPAN_TYPE = "type"
_GUARDRAILS_SPAN_TYPE_PREFIX = "guardrails/"
_GUARDRAILS_GUARD_TYPE = "guardrails/guard"
_GUARDRAILS_STEP_TYPE = "guardrails/guard/step"
_GUARDRAILS_LLM_CALL_TYPE = "guardrails/guard/step/call"
_GUARDRAILS_INPUT_VALUE = "input.value"
_GUARDRAILS_OUTPUT_VALUE = "output.value"
_GUARDRAILS_LLM_INVOCATION_PARAMETERS = "llm.invocation_parameters"
_GUARDRAILS_LLM_INPUT_MESSAGES_PREFIX = "llm.input_messages."
_GUARDRAILS_LLM_OUTPUT_MESSAGES_PREFIX = "llm.output_messages."
_GUARDRAILS_LLM_TOKEN_COUNT_PROMPT = "llm.token_count.prompt"
_GUARDRAILS_LLM_TOKEN_COUNT_COMPLETION = "llm.token_count.completion"
_GUARDRAILS_LLM_TOKEN_COUNT_TOTAL = "llm.token_count.total"
_GEN_AI_PROMPT_PREFIX = f"{SpanAttributes.LLM_PROMPTS}."
_GEN_AI_COMPLETION_PREFIX = f"{SpanAttributes.LLM_COMPLETIONS}."
_LLM_USAGE_TOTAL_TOKENS = SpanAttributes.LLM_USAGE_TOTAL_TOKENS


def _load_guardrails_guard_class() -> type:
    guardrails_module = importlib.import_module(GUARDRAILS_RUNTIME_MODULE)
    return getattr(guardrails_module, GUARDRAILS_GUARD_CLASS)


def _parse_invocation_parameters(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or value == "":
        return {}

    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return {}

    return parsed if isinstance(parsed, dict) else {}


def _int_value(value: Any) -> int | None:
    if value in {None, ""}:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _translate_guardrails_message_attrs(
    attrs: dict[str, Any],
    source_prefix: str,
    target_prefix: str,
) -> None:
    for key, value in list(attrs.items()):
        if not key.startswith(source_prefix):
            continue

        suffix = key[len(source_prefix) :]
        parts = suffix.split(".", 2)
        if len(parts) != 3 or not parts[0].isdigit() or parts[1] != "message":
            continue

        field_name = parts[2]
        if field_name not in {"role", "content"}:
            continue

        attrs.setdefault(f"{target_prefix}{parts[0]}.{field_name}", value)


def _has_guardrails_llm_attrs(attrs: dict[str, Any]) -> bool:
    if attrs.get(_GUARDRAILS_LLM_INVOCATION_PARAMETERS):
        return True
    if attrs.get(_GUARDRAILS_LLM_TOKEN_COUNT_PROMPT) is not None:
        return True
    if attrs.get(_GUARDRAILS_LLM_TOKEN_COUNT_COMPLETION) is not None:
        return True
    if attrs.get(_GUARDRAILS_LLM_TOKEN_COUNT_TOTAL) is not None:
        return True
    return any(
        key.startswith(
            (
                _GUARDRAILS_LLM_INPUT_MESSAGES_PREFIX,
                _GUARDRAILS_LLM_OUTPUT_MESSAGES_PREFIX,
            )
        )
        for key in attrs
    )


def _guardrails_operation_name(guardrails_type: str, span_name: str) -> str:
    if guardrails_type == _GUARDRAILS_GUARD_TYPE:
        return "guardrails.guard"
    if guardrails_type == _GUARDRAILS_STEP_TYPE:
        return "guardrails.step"
    if guardrails_type == _GUARDRAILS_LLM_CALL_TYPE:
        return "guardrails.call"
    if guardrails_type.endswith("/validator"):
        return "guardrails.validator"
    return f"guardrails.{span_name}"


class GuardrailsSpanProcessor(SpanProcessor):
    """Normalize Guardrails internal OTEL spans for the Respan backend."""

    def __init__(self) -> None:
        self._propagated_by_trace: dict[int, dict[str, Any]] = {}
        self._active_spans_by_trace: dict[int, int] = {}

    @staticmethod
    def _trace_id(span: Any) -> int | None:
        get_span_context = getattr(span, "get_span_context", None)
        if not callable(get_span_context):
            return None
        return getattr(get_span_context(), "trace_id", None)

    def on_start(self, span: Any, parent_context: Any = None) -> None:
        del parent_context
        trace_id = self._trace_id(span)
        if trace_id is not None:
            self._active_spans_by_trace[trace_id] = (
                self._active_spans_by_trace.get(trace_id, 0) + 1
            )

        propagated = read_propagated_attributes()
        if propagated and trace_id is not None:
            cached = self._propagated_by_trace.setdefault(trace_id, {})
            cached.update(propagated)
        elif trace_id is not None:
            propagated = self._propagated_by_trace.get(trace_id, {})

        for key, value in propagated.items():
            attributes = getattr(span, "attributes", None) or {}
            if attributes.get(key) is None:
                span.set_attribute(key, value)

    def on_end(self, span: ReadableSpan) -> None:
        trace_id = self._trace_id(span)
        try:
            self._normalize_span(span)
        finally:
            if trace_id is not None:
                remaining = self._active_spans_by_trace.get(trace_id, 1) - 1
                if remaining <= 0:
                    self._active_spans_by_trace.pop(trace_id, None)
                    self._propagated_by_trace.pop(trace_id, None)
                else:
                    self._active_spans_by_trace[trace_id] = remaining

    def _normalize_span(self, span: ReadableSpan) -> None:
        original_attrs = getattr(span, "_attributes", None)
        if original_attrs is None:
            return

        attrs = dict(original_attrs)
        guardrails_type = attrs.get(_GUARDRAILS_SPAN_TYPE)
        if not (
            isinstance(guardrails_type, str)
            and guardrails_type.startswith(_GUARDRAILS_SPAN_TYPE_PREFIX)
        ):
            return

        attrs.setdefault(
            SpanAttributes.TRACELOOP_ENTITY_NAME,
            _guardrails_operation_name(guardrails_type, span.name),
        )
        attrs.setdefault(SpanAttributes.TRACELOOP_ENTITY_PATH, "")

        input_value = attrs.get(_GUARDRAILS_INPUT_VALUE)
        if input_value is not None:
            attrs.setdefault(SpanAttributes.TRACELOOP_ENTITY_INPUT, str(input_value))

        output_value = attrs.get(_GUARDRAILS_OUTPUT_VALUE)
        if output_value is not None:
            attrs.setdefault(SpanAttributes.TRACELOOP_ENTITY_OUTPUT, str(output_value))

        if (
            guardrails_type != _GUARDRAILS_LLM_CALL_TYPE
            or not _has_guardrails_llm_attrs(attrs)
        ):
            attrs.setdefault(RESPAN_LOG_TYPE, LOG_TYPE_GUARDRAIL)
            attrs.pop(SpanAttributes.TRACELOOP_SPAN_KIND, None)
            if guardrails_type == _GUARDRAILS_GUARD_TYPE:
                token_consumption = _int_value(attrs.get("token_consumption"))
                if (
                    not token_consumption
                    and attrs.get("number_of_llm_calls") is not None
                ):
                    attrs["number_of_llm_calls"] = 0
            span._attributes = attrs
            return

        attrs[RESPAN_LOG_TYPE] = LOG_TYPE_CHAT
        attrs.pop(SpanAttributes.TRACELOOP_SPAN_KIND, None)
        attrs.setdefault(LLM_REQUEST_TYPE, LLMRequestTypeValues.CHAT.value)

        invocation_parameters = _parse_invocation_parameters(
            attrs.get(_GUARDRAILS_LLM_INVOCATION_PARAMETERS)
        )
        model = invocation_parameters.get("model")
        if model:
            attrs.setdefault(LLM_REQUEST_MODEL, model)

        temperature = invocation_parameters.get("temperature")
        if temperature is not None:
            attrs.setdefault(SpanAttributes.LLM_REQUEST_TEMPERATURE, temperature)

        prompt_tokens = _int_value(attrs.get(_GUARDRAILS_LLM_TOKEN_COUNT_PROMPT))
        if prompt_tokens is not None:
            attrs.setdefault(LLM_USAGE_PROMPT_TOKENS, prompt_tokens)

        completion_tokens = _int_value(
            attrs.get(_GUARDRAILS_LLM_TOKEN_COUNT_COMPLETION)
        )
        if completion_tokens is not None:
            attrs.setdefault(LLM_USAGE_COMPLETION_TOKENS, completion_tokens)

        total_tokens = _int_value(attrs.get(_GUARDRAILS_LLM_TOKEN_COUNT_TOTAL))
        if total_tokens is None and (
            prompt_tokens is not None or completion_tokens is not None
        ):
            total_tokens = (prompt_tokens or 0) + (completion_tokens or 0)
        if total_tokens is not None:
            attrs.setdefault(_LLM_USAGE_TOTAL_TOKENS, total_tokens)

        _translate_guardrails_message_attrs(
            attrs=attrs,
            source_prefix=_GUARDRAILS_LLM_INPUT_MESSAGES_PREFIX,
            target_prefix=_GEN_AI_PROMPT_PREFIX,
        )
        _translate_guardrails_message_attrs(
            attrs=attrs,
            source_prefix=_GUARDRAILS_LLM_OUTPUT_MESSAGES_PREFIX,
            target_prefix=_GEN_AI_COMPLETION_PREFIX,
        )

        span._attributes = attrs

    def shutdown(self) -> None:
        self._propagated_by_trace.clear()
        self._active_spans_by_trace.clear()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


class GuardrailsInstrumentor:
    """Respan instrumentor for Guardrails AI.

    Normalizes Guardrails' native OTEL spans into the Respan contract without
    adding a duplicate public-method wrapper span.
    """

    name = GUARDRAILS_INSTRUMENTATION_NAME
    _span_processor = GuardrailsSpanProcessor()
    _span_processor_registered = False

    def __init__(self) -> None:
        self._guard_class: type | None = None
        self._is_instrumented = False

    @property
    def is_instrumented(self) -> bool:
        return self._is_instrumented

    @staticmethod
    def _is_respan_tracing_enabled() -> bool:
        tracer = getattr(RespanTracer, "_instance", None)
        if tracer is None:
            return True
        return bool(getattr(tracer, "is_enabled", True))

    @classmethod
    def _register_span_processor(cls) -> None:
        tracer_provider = trace.get_tracer_provider()
        active_span_processor = getattr(
            tracer_provider,
            "_active_span_processor",
            None,
        )
        processors = (
            getattr(active_span_processor, "_span_processors", None)
            if active_span_processor is not None
            else None
        )

        if processors is not None:
            active_span_processor._span_processors = (
                cls._span_processor,
                *(
                    processor
                    for processor in processors
                    if processor is not cls._span_processor
                ),
            )
            cls._span_processor_registered = True
            return

        if hasattr(tracer_provider, "add_span_processor"):
            tracer_provider.add_span_processor(cls._span_processor)
            cls._span_processor_registered = True

    @classmethod
    def _remove_span_processor(cls) -> None:
        tracer_provider = trace.get_tracer_provider()
        active_span_processor = getattr(
            tracer_provider,
            "_active_span_processor",
            None,
        )
        processors = (
            getattr(active_span_processor, "_span_processors", None)
            if active_span_processor is not None
            else None
        )
        if processors is not None:
            active_span_processor._span_processors = tuple(
                processor
                for processor in processors
                if processor is not cls._span_processor
            )
        cls._span_processor_registered = False

    def activate(self) -> None:
        """Instrument Guardrails public Guard methods."""
        if self._is_instrumented:
            return

        if not self._is_respan_tracing_enabled():
            logger.info(
                "Guardrails instrumentation skipped because Respan tracing is disabled"
            )
            return

        try:
            guard_class = _load_guardrails_guard_class()
        except (AttributeError, ImportError) as exc:
            logger.warning(
                "Failed to activate Guardrails instrumentation — missing runtime dependency: %s",
                exc,
            )
            return

        self._guard_class = guard_class
        self._register_span_processor()
        self._is_instrumented = True
        logger.info("Guardrails instrumentation activated")

    def deactivate(self) -> None:
        """Restore original Guardrails methods."""
        self._guard_class = None
        self._remove_span_processor()
        self._is_instrumented = False
        logger.info("Guardrails instrumentation deactivated")
