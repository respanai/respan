"""Private start snapshots and current opt-outs for native AgentOps spans."""

import os

from opentelemetry import context
from opentelemetry.semconv_ai import SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY
from respan_tracing.constants.context_constants import ENABLE_CONTENT_TRACING_KEY


def content_allowed(setting=True):
    return (
        setting
        and context.get_value(ENABLE_CONTENT_TRACING_KEY) is not False
        and os.getenv("TRACELOOP_TRACE_CONTENT", "true").strip().lower()
        not in {"false", "0", "no", "off"}
    )


def suppressed():
    return bool(
        context.get_value(context._SUPPRESS_INSTRUMENTATION_KEY)
        or context.get_value(SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY)
    )
