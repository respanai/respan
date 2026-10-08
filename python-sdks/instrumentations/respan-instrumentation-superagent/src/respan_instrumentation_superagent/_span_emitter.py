"""Live Superagent spans in the active OpenTelemetry provider."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from opentelemetry import context, trace
from opentelemetry.semconv._incubating.attributes.error_attributes import ERROR_MESSAGE
from opentelemetry.semconv.attributes.error_attributes import ERROR_TYPE
from opentelemetry.semconv.attributes.http_attributes import HTTP_RESPONSE_STATUS_CODE
from opentelemetry.semconv_ai import SpanAttributes
from respan_sdk.constants.llm_logging import LOG_TYPE_GUARDRAIL, LOG_TYPE_TOOL
from respan_sdk.constants.span_attributes import (
    RESPAN_LOG_TYPE,
    RESPAN_METADATA_GUARDRAIL_NAME,
    RESPAN_METADATA_TRIGGERED,
)
from respan_tracing.constants.context_constants import ENABLE_CONTENT_TRACING_KEY
from respan_tracing.decorators.base import _should_send_prompts
from respan_tracing.utils.span_factory import read_propagated_attributes

from ._constants import (
    GUARD_METHOD,
    SUPERAGENT_INSTRUMENTATION_NAME,
    SUPERAGENT_METADATA_CLASSIFICATION,
    SUPERAGENT_METADATA_INTEGRATION,
    SUPERAGENT_METADATA_METHOD,
    SUPERAGENT_METADATA_MODEL,
    SUPERAGENT_METADATA_REDACT_FINDINGS,
    SUPERAGENT_METADATA_USAGE,
)
from ._serialization import (
    extract_model,
    normalize_call_input,
    safe_error_message,
    safe_json_dumps,
)

logger = logging.getLogger(__name__)


def _get_attr(value: Any, name: str) -> Any:
    if isinstance(value, dict):
        return value.get(name)
    try:
        return getattr(value, name, None)
    except BaseException:  # noqa: BLE001 - telemetry cannot change the SDK result
        return None


def _error_status_code(error: BaseException) -> int | None:
    status = _get_attr(error, "status_code")
    if not isinstance(status, int) or isinstance(status, bool):
        status = _get_attr(_get_attr(error, "response"), "status_code")
    return (
        status
        if isinstance(status, int)
        and not isinstance(status, bool)
        and 400 <= status <= 599
        else None
    )


def _add_result_metadata(
    attrs: dict[str, Any], method_name: str, result: Any, *, capture: bool
) -> None:
    usage = _get_attr(result, "usage")
    if usage is not None:
        counts = {
            key: _get_attr(usage, key)
            for key in [
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "input_tokens",
                "output_tokens",
                "reasoning_tokens",
                "cost",
            ]
        }
        counts = {
            key: value
            for key, value in counts.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        if counts:
            attrs[SUPERAGENT_METADATA_USAGE] = safe_json_dumps(counts)
    if method_name == GUARD_METHOD:
        classification = _get_attr(result, "classification")
        if classification in {"pass", "block"}:
            attrs[SUPERAGENT_METADATA_CLASSIFICATION] = classification
            attrs[RESPAN_METADATA_TRIGGERED] = classification == "block"
        attrs[RESPAN_METADATA_GUARDRAIL_NAME] = "superagent.guard"
    if method_name == "redact" and capture:
        findings = _get_attr(result, "findings")
        if findings is not None:
            attrs[SUPERAGENT_METADATA_REDACT_FINDINGS] = safe_json_dumps(findings)


def build_superagent_span_attributes(
    *,
    method_name: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    result: Any,
    capture: bool = True,
) -> dict[str, Any]:
    operation = method_name
    attrs = {
        RESPAN_LOG_TYPE: LOG_TYPE_GUARDRAIL
        if method_name == GUARD_METHOD
        else LOG_TYPE_TOOL,
        SpanAttributes.TRACELOOP_ENTITY_NAME: operation,
        SpanAttributes.TRACELOOP_ENTITY_PATH: operation,
        SUPERAGENT_METADATA_INTEGRATION: SUPERAGENT_INSTRUMENTATION_NAME,
        SUPERAGENT_METADATA_METHOD: method_name,
    }
    model = extract_model(args=args, kwargs=kwargs)
    if model:
        attrs[SUPERAGENT_METADATA_MODEL] = model
    if capture:
        attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = safe_json_dumps(
            normalize_call_input(method_name=method_name, args=args, kwargs=kwargs)
        )
        attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = safe_json_dumps(result)
    _add_result_metadata(attrs, method_name, result, capture=capture)
    return attrs


@dataclass
class CallSpan:
    span: Any
    method_name: str
    capture: bool


@contextmanager
def call_scope(call):
    if call is None:
        yield
        return
    ctx = trace.set_span_in_context(call.span)
    if not call.capture:
        ctx = context.set_value(ENABLE_CONTENT_TRACING_KEY, False, ctx)
    token = context.attach(ctx)
    try:
        yield
    finally:
        context.detach(token)


def start_superagent_span(
    *,
    method_name: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    provider=None,
    start_time_ns: int | None = None,
) -> CallSpan | None:
    if context.get_value(context._SUPPRESS_INSTRUMENTATION_KEY):
        return None
    try:
        capture = _should_send_prompts()
        attrs = build_superagent_span_attributes(
            method_name=method_name,
            args=args,
            kwargs=kwargs,
            result=None,
            capture=capture,
        )
        attrs.pop(SpanAttributes.TRACELOOP_ENTITY_OUTPUT, None)
        attrs.update(read_propagated_attributes())
        span = (
            (provider or trace.get_tracer_provider())
            .get_tracer(__name__)
            .start_span(
                f"superagent.{method_name}", attributes=attrs, start_time=start_time_ns
            )
        )
        return CallSpan(span, method_name, capture)
    except Exception:
        logger.debug("Could not start Superagent span", exc_info=True)
        return None


def finish_superagent_span(
    call: CallSpan | None,
    *,
    result: Any = None,
    error: BaseException | None = None,
    end_time_ns: int | None = None,
) -> None:
    if call is None:
        return
    span = call.span
    try:
        capture = call.capture and _should_send_prompts()
        if not capture:
            # A later content veto removes the already captured request too.
            attributes = getattr(span, "_attributes", None)
            if attributes is not None:
                attributes.pop(SpanAttributes.TRACELOOP_ENTITY_INPUT, None)
        attrs = {}
        if error is None:
            if capture:
                attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = safe_json_dumps(result)
            _add_result_metadata(attrs, call.method_name, result, capture=capture)
        else:
            message = safe_error_message(error) if capture else type(error).__name__
            attrs[ERROR_TYPE] = type(error).__name__
            attrs[ERROR_MESSAGE] = message
            status = _error_status_code(error)
            if status is not None:
                attrs[HTTP_RESPONSE_STATUS_CODE] = status
            span.set_status(trace.Status(trace.StatusCode.ERROR, message))
        span.set_attributes(attrs)
    except Exception:
        logger.debug("Could not finish Superagent attributes", exc_info=True)
    finally:
        try:
            span.end(end_time=end_time_ns)
        except Exception:
            logger.debug("Could not end Superagent span", exc_info=True)


def emit_superagent_span(
    *,
    method_name: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    result: Any,
    start_time_ns: int,
    end_time_ns: int,
    error: BaseException | None = None,
) -> bool:
    """Compatibility helper; normal instrumentation starts its span at call time."""
    call = start_superagent_span(
        method_name=method_name, args=args, kwargs=kwargs, start_time_ns=start_time_ns
    )
    finish_superagent_span(call, result=result, error=error, end_time_ns=end_time_ns)
    return call is not None
