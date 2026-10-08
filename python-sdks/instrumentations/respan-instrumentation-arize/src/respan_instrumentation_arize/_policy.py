"""Private content policy and OTel suppression."""

import os

from opentelemetry import context
from opentelemetry.semconv_ai import (
    SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY,
    SpanAttributes,
)
from respan_tracing.constants.context_constants import ENABLE_CONTENT_TRACING_KEY


def content_allowed(setting=True):
    return (
        setting
        and context.get_value(ENABLE_CONTENT_TRACING_KEY) is not False
        and os.getenv("TRACELOOP_TRACE_CONTENT", "true").strip().lower()
        not in {"false", "0", "off", "no"}
    )


def suppressed():
    return bool(
        context.get_value(context._SUPPRESS_INSTRUMENTATION_KEY)
        or context.get_value(SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY)
    )


def clear_content(span):
    attributes = getattr(span, "_attributes", None)
    if attributes is not None:
        for key in list(attributes):
            if key in (
                SpanAttributes.TRACELOOP_ENTITY_INPUT,
                SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
            ):
                attributes.pop(key, None)
