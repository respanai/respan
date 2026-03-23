"""Shared utilities for constructing and injecting ReadableSpan objects.

Used by plugin instrumentations (e.g. openai-agents) and the ``Respan``
unified entry point (e.g. ``log_batch_results``) to emit spans into the
single OTEL pipeline without going through the V2 dict exporter.

Key functions:

- ``build_readable_span()`` — construct a ``ReadableSpan`` with explicit IDs,
  timestamps, and attributes that passes ``is_processable_span()``.
- ``inject_span()`` — push a ``ReadableSpan`` through the active
  ``TracerProvider``'s processor chain.
- ``read_propagated_attributes()`` — read the ``_PROPAGATED_ATTRIBUTES``
  ContextVar and map values to OTEL span attribute keys.
"""

import contextvars
import json
import logging
import time
from contextlib import contextmanager
from typing import Any, Dict, Optional, Sequence

from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.trace import SpanContext, SpanKind, TraceFlags
from opentelemetry.trace.status import Status, StatusCode

from respan_sdk.constants.span_attributes import (
    RESPAN_PROMPT,
    RESPAN_METADATA,
    RESPAN_SPAN_ATTRIBUTES_MAP,
)
from respan_sdk.utils.data_processing.id_processing import (
    ensure_trace_id,
    ensure_span_id,
)
from respan_sdk.utils.time import iso_to_ns

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Context-propagated attributes
# ---------------------------------------------------------------------------

_PROPAGATED_ATTRIBUTES: contextvars.ContextVar[Dict[str, Any]] = contextvars.ContextVar(
    "respan_propagated_attributes", default={}
)

# Keys accepted by propagate_attributes() — derived from the canonical
# RESPAN_SPAN_ATTRIBUTES_MAP so there is a single source of truth.
_SUPPORTED_ATTRIBUTE_KEYS = frozenset(RESPAN_SPAN_ATTRIBUTES_MAP.keys())


@contextmanager
def propagate_attributes(**kwargs):
    """Context manager that attaches attributes to all spans exported within its scope.

    Attributes are stored in a ``ContextVar`` and merged into every span
    at export time.  This works correctly with ``asyncio`` — each task gets its
    own copy of the context.

    Supported attributes:
        customer_identifier, customer_email, customer_name,
        thread_identifier, custom_identifier, group_identifier,
        environment, metadata (dict — merged, not replaced),
        prompt (dict with prompt_id + variables — triggers server-side
        template resolution).

    Example::

        with propagate_attributes(customer_identifier="user_123"):
            result = await Runner.run(agent, "Hello")

        with propagate_attributes(prompt={"prompt_id": "abc", "variables": {"x": "y"}}):
            result = await Runner.run(agent, "Hello")
    """
    # Merge with any already-active attributes (supports nesting)
    parent = _PROPAGATED_ATTRIBUTES.get()
    merged = {**parent}
    for key, value in kwargs.items():
        if key not in _SUPPORTED_ATTRIBUTE_KEYS:
            logger.warning("Ignoring unsupported attribute: %s", key)
            continue
        if RESPAN_SPAN_ATTRIBUTES_MAP.get(key) == RESPAN_METADATA and isinstance(value, dict):
            # Merge metadata dicts instead of replacing
            merged[key] = {**merged.get(key, {}), **value}
        else:
            merged[key] = value

    token = _PROPAGATED_ATTRIBUTES.set(merged)
    try:
        yield
    finally:
        _PROPAGATED_ATTRIBUTES.reset(token)


# ---------------------------------------------------------------------------
# Propagated attributes bridge
# ---------------------------------------------------------------------------


def read_propagated_attributes() -> Dict[str, Any]:
    """Read ``_PROPAGATED_ATTRIBUTES`` ContextVar and map to OTEL span attribute keys.

    Returns a dict suitable for merging into a span's attributes, e.g.::

        {"respan.customer_params.customer_identifier": "user_123",
         "respan.threads.thread_identifier": "conv_abc",
         "respan.metadata.plan": "pro"}
    """
    ctx_attrs = _PROPAGATED_ATTRIBUTES.get()
    if not ctx_attrs:
        return {}

    result: Dict[str, Any] = {}
    for key, value in ctx_attrs.items():
        if key not in RESPAN_SPAN_ATTRIBUTES_MAP:
            continue
        attr_key = RESPAN_SPAN_ATTRIBUTES_MAP[key]
        if attr_key == RESPAN_METADATA and isinstance(value, dict):
            # Metadata is stored as individual respan.metadata.<key> attributes
            for mk, mv in value.items():
                result[f"{RESPAN_METADATA}.{mk}"] = str(mv) if not isinstance(mv, str) else mv
        elif attr_key == RESPAN_PROMPT and isinstance(value, dict):
            # Prompt config: store as JSON string for the exporter to pick up
            result[RESPAN_PROMPT] = json.dumps(value, default=str)
        else:
            result[attr_key] = value
    return result


# ---------------------------------------------------------------------------
# Span construction
# ---------------------------------------------------------------------------


def build_readable_span(
    name: str,
    *,
    trace_id: Optional[str] = None,
    span_id: Optional[str] = None,
    parent_id: Optional[str] = None,
    start_time_iso: Optional[str] = None,
    end_time_iso: Optional[str] = None,
    start_time_ns: Optional[int] = None,
    end_time_ns: Optional[int] = None,
    attributes: Optional[Dict[str, Any]] = None,
    status_code: int = 200,
    error_message: Optional[str] = None,
    kind: SpanKind = SpanKind.INTERNAL,
    merge_propagated: bool = True,
) -> ReadableSpan:
    """Construct a ``ReadableSpan`` with explicit IDs and attributes.

    The span is configured to pass ``is_processable_span()`` — it will have
    at least a ``traceloop.entity.path`` or ``traceloop.span.kind`` attribute.

    Args:
        name: Span name (e.g. ``"response"``, ``"batch:custom_id"``).
        trace_id: 32-char hex trace ID.  Auto-generated if ``None``.
        span_id: 16-char hex span ID.  Auto-generated if ``None``.
        parent_id: 16-char hex parent span ID.  ``None`` → root span.
        start_time_iso: ISO-8601 start time.
        end_time_iso: ISO-8601 end time.
        start_time_ns: Start time in nanoseconds (takes precedence over ISO).
        end_time_ns: End time in nanoseconds (takes precedence over ISO).
        attributes: Span attributes dict (traceloop.*, gen_ai.*, etc.).
        status_code: HTTP-style status code (< 400 → OK, >= 400 → ERROR).
        error_message: Error description (sets span status to ERROR).
        kind: OTEL SpanKind (default INTERNAL).
        merge_propagated: If True, merge propagated attributes from ContextVar.

    Returns:
        A fully-formed ``ReadableSpan`` ready to be injected via ``inject_span()``.
    """
    # Resolve IDs
    tid = ensure_trace_id(trace_id)
    sid = ensure_span_id(span_id)

    # Build parent SpanContext (or None for root)
    parent = None
    if parent_id:
        pid = ensure_span_id(parent_id)
        parent = SpanContext(
            trace_id=tid,
            span_id=pid,
            is_remote=False,
            trace_flags=TraceFlags(TraceFlags.SAMPLED),
        )

    # Build this span's context
    ctx = SpanContext(
        trace_id=tid,
        span_id=sid,
        is_remote=False,
        trace_flags=TraceFlags(TraceFlags.SAMPLED),
    )

    # Resolve timestamps
    if start_time_ns is None:
        start_time_ns = iso_to_ns(start_time_iso) or time.time_ns()
    if end_time_ns is None:
        end_time_ns = iso_to_ns(end_time_iso) or time.time_ns()

    # Build attributes
    attrs: Dict[str, Any] = dict(attributes or {})

    # Merge propagated attributes (customer_identifier, thread_id, etc.)
    if merge_propagated:
        propagated = read_propagated_attributes()
        for k, v in propagated.items():
            attrs.setdefault(k, v)

    # Determine status
    if error_message:
        status = Status(StatusCode.ERROR, error_message)
    elif status_code >= 400:
        status = Status(StatusCode.ERROR, f"HTTP {status_code}")
    else:
        status = Status(StatusCode.OK)

    # Get the global TracerProvider's resource for consistency
    tp = trace.get_tracer_provider()
    resource = getattr(tp, "resource", None)

    span = ReadableSpan(
        name=name,
        context=ctx,
        parent=parent,
        resource=resource,
        attributes=attrs,
        kind=kind,
        start_time=start_time_ns,
        end_time=end_time_ns,
        status=status,
        events=(),
        links=(),
    )
    return span


# ---------------------------------------------------------------------------
# Span injection
# ---------------------------------------------------------------------------


def inject_span(span: ReadableSpan) -> bool:
    """Push *span* through the active ``TracerProvider``'s processor chain.

    This is how plugin-constructed spans enter the OTEL pipeline without
    needing a live tracer context.  The span passes through
    ``RespanSpanProcessor.on_end()`` → ``is_processable_span()`` filter →
    ``RespanSpanExporter`` → ``/v2/traces``.

    Returns ``True`` on success, ``False`` if no processor is available.
    """
    tp = trace.get_tracer_provider()
    processor = getattr(tp, "_active_span_processor", None)
    if processor is None:
        logger.warning("No active span processor — span %r not exported", span.name)
        return False
    try:
        processor.on_end(span)
        return True
    except Exception:
        logger.exception("Failed to inject span %r", span.name)
        return False
