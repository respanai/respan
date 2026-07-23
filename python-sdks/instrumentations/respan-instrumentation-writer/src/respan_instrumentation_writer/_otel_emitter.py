"""Emit Writer SDK calls as OTEL ReadableSpan objects."""

from __future__ import annotations

import logging
import time
from typing import Any

from opentelemetry import trace

from respan_sdk.utils.data_processing.id_processing import (
    format_span_id,
    format_trace_id,
)
from respan_tracing.utils.span_factory import build_readable_span, inject_span

logger = logging.getLogger(__name__)


def _current_trace_parent_ids() -> tuple[str | None, str | None]:
    try:
        current_span = trace.get_current_span()
        span_context = current_span.get_span_context()
    except Exception:
        return None, None

    trace_id = getattr(span_context, "trace_id", 0) or 0
    span_id = getattr(span_context, "span_id", 0) or 0
    if trace_id == 0 or span_id == 0:
        return None, None
    return format_trace_id(trace_id=trace_id), format_span_id(span_id=span_id)


def emit_writer_span(
    *,
    name: str,
    attrs: dict[str, Any],
    start_ns: int,
    end_ns: int | None = None,
    error_message: str | None = None,
    status_code: int = 200,
) -> None:
    """Build a ReadableSpan for a Writer SDK call and inject it."""
    try:
        span_attrs = dict(attrs)
        if error_message:
            span_attrs["error.message"] = error_message
            status_code = status_code if status_code >= 400 else 500

        trace_id, parent_id = _current_trace_parent_ids()
        span = build_readable_span(
            name=name,
            trace_id=trace_id,
            parent_id=parent_id,
            start_time_ns=start_ns,
            end_time_ns=end_ns if end_ns is not None else time.time_ns(),
            attributes=span_attrs,
            error_message=error_message,
            status_code=status_code,
        )
        inject_span(span=span)
    except Exception:
        logger.debug("Failed to emit Writer span", exc_info=True)
