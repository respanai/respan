"""Capture policy at native OpenLIT execution boundaries."""

import os

from opentelemetry import context
from opentelemetry.semconv_ai import SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY
from respan_tracing.constants.context_constants import ENABLE_CONTENT_TRACING_KEY


def suppressed(parent_context=None):
    return bool(
        context.get_value(context._SUPPRESS_INSTRUMENTATION_KEY)
        or context.get_value(SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY)
        or context.get_value(context._SUPPRESS_INSTRUMENTATION_KEY, parent_context)
        or context.get_value(
            SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY, parent_context
        )
    )


def content_allowed(configured=True, parent_context=None):
    return (
        configured
        and os.getenv("TRACELOOP_TRACE_CONTENT", "true").strip().lower()
        not in {"false", "0", "off", "no"}
        and os.getenv("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "true")
        .strip()
        .lower()
        not in {"false", "0", "off", "no"}
        and context.get_value(ENABLE_CONTENT_TRACING_KEY) is not False
        and context.get_value(ENABLE_CONTENT_TRACING_KEY, parent_context) is not False
        and not suppressed(parent_context)
    )


def allowed(span, configured=True):
    if span is None:
        return False
    recording = getattr(span, "is_recording", None)
    if callable(recording) and not recording():
        return False
    from ._instrumentation import _PROCESSOR

    processor = getattr(span, "_respan_openlit_processor", _PROCESSOR)
    permitted = processor.allowed(span) if processor is not None else True
    return content_allowed(configured) and permitted
