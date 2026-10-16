"""Private start snapshots and current opt-outs for native AgentOps spans."""

from opentelemetry import context
from opentelemetry.semconv_ai import SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY
from respan_tracing import is_content_tracing_enabled


def content_allowed(setting=True):
    return setting and is_content_tracing_enabled()


def suppressed():
    return bool(
        context.get_value(context._SUPPRESS_INSTRUMENTATION_KEY)
        or context.get_value(SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY)
    )
