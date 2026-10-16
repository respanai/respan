"""Capture policy at native OpenLIT execution boundaries."""

import os

from opentelemetry import context
from opentelemetry.semconv_ai import SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY
from respan_tracing.constants.generic_constants import TRACE_CONTENT_OFF_VALUES
from respan_tracing.utils.context import is_content_tracing_enabled


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
        and os.getenv("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "true")
        .strip()
        .lower()
        not in TRACE_CONTENT_OFF_VALUES
        and is_content_tracing_enabled()
        and is_content_tracing_enabled(parent_context)
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
