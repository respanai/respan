"""Private start snapshots with final payload veto and OTel suppression."""

from opentelemetry import context
from opentelemetry.semconv_ai import (
    SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY,
    SpanAttributes,
)
from respan_tracing.utils.context import is_content_tracing_enabled


def content_allowed(setting: bool = True) -> bool:
    return setting and is_content_tracing_enabled()


def suppressed() -> bool:
    return bool(
        context.get_value(context._SUPPRESS_INSTRUMENTATION_KEY)
        or context.get_value(SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY)
    )


def clear_content(span) -> None:
    attributes = getattr(span, "_attributes", None)
    if attributes is not None:
        for key in list(attributes):
            if key in (
                SpanAttributes.TRACELOOP_ENTITY_INPUT,
                SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
                SpanAttributes.LLM_REQUEST_FUNCTIONS,
            ) or key.startswith(
                (f"{SpanAttributes.LLM_PROMPTS}.", f"{SpanAttributes.LLM_COMPLETIONS}.")
            ):
                attributes.pop(key, None)
