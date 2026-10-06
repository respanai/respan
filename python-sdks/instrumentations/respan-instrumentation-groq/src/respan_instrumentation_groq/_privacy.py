"""Snapshot Respan's content policy before lazy SDK calls leave their context."""

import os

from opentelemetry import context
from respan_tracing.constants.context_constants import ENABLE_CONTENT_TRACING_KEY


def content_enabled(parent_context=None) -> bool:
    if (
        context.get_value(ENABLE_CONTENT_TRACING_KEY) is False
        or context.get_value(ENABLE_CONTENT_TRACING_KEY, parent_context) is False
    ):
        return False
    return os.getenv("TRACELOOP_TRACE_CONTENT", "true").strip().lower() not in {
        "false",
        "0",
        "off",
    } or bool(context.get_value("override_enable_content_tracing", parent_context))
