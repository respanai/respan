"""Respan instrumentation for Google ADK traces.

This instrumentor wraps OpenTelemetry span processors to intercept ADK spans
and export them via a provided exporter (e.g. RespanSpanExporterV2).
"""
from __future__ import annotations

import logging
from collections import OrderedDict
from threading import Lock
from typing import Any, Dict, Iterable, List, Optional

import wrapt
from opentelemetry.sdk.trace.export import SpanExportResult
from respan_tracing.exporters.span_exporter_v2 import _PROPAGATED_ATTRIBUTES

from respan_instrumentation_google_adk.converter import AdkSpanConverter
from respan_instrumentation_google_adk.utils import group_spans_by_trace, is_adk_span, otel_span_to_dict

logger = logging.getLogger(__name__)

_PATCHED = False

_DEFAULT_TRACE_BUFFER_MAX_SPANS = 8192


class _SpanDedupeCache:
    """Cache for deduplicating spans by trace_id:span_id."""

    def __init__(self, max_size: int = 10000) -> None:
        self._max_size = max_size
        self._data: "OrderedDict[str, None]" = OrderedDict()
        self._lock = Lock()

    def add(self, trace_id: Optional[str], span_id: Optional[str]) -> bool:
        """Add span to cache. Returns True if span was not already in cache."""
        if not trace_id or not span_id:
            return True
        key = f"{trace_id}:{span_id}"
        with self._lock:
            if key in self._data:
                return False
            self._data[key] = None
            if len(self._data) > self._max_size:
                self._data.popitem(last=False)
        return True


_ACTIVE_EXPORTER: Optional[Any] = None
_ACTIVE_CONVERTER: Optional[AdkSpanConverter] = None
_ACTIVE_DEDUPE: Optional[_SpanDedupeCache] = _SpanDedupeCache()
_ACTIVE_PASSTHROUGH = False
# None = no limit (not recommended for production).
_ACTIVE_TRACE_BUFFER_MAX_SPANS: Optional[int] = _DEFAULT_TRACE_BUFFER_MAX_SPANS

# Buffer child spans per trace so the converter sees the full trace at once.
# The ADK root span ("invocation") typically ends last; when it arrives, the
# entire buffered trace is flushed through the converter together.
_TRACE_BUFFER: Dict[str, List[object]] = {}
_TRACE_BUFFER_LOCK = Lock()

# Snapshot of propagated context attributes captured on the user thread.
# Keyed by trace_id.  Without this, propagate_attributes() context would be
# lost because BatchSpanProcessor._export runs on a background thread where
# the ContextVar is empty.  Mirrors the pattern from the OpenAI Agents SDK
# instrumentation (_RespanTracingProcessor._captured_ctx).
_TRACE_CTX_CACHE: Dict[str, Dict[str, Any]] = {}

_ADK_ROOT_SPAN_NAMES = {"invocation"}


def _export_adk_spans(
    spans: Iterable[object],
    captured_ctx: Optional[Dict[str, Any]] = None,
) -> SpanExportResult:
    """Export ADK spans to Respan via the active exporter."""
    exporter = _ACTIVE_EXPORTER
    converter = _ACTIVE_CONVERTER
    dedupe = _ACTIVE_DEDUPE
    if exporter is None or converter is None:
        return SpanExportResult.SUCCESS

    adk_span_dicts: List[Dict[str, object]] = []
    for span in spans:
        if not is_adk_span(span=span):
            continue
        span_dict = otel_span_to_dict(span=span)
        if dedupe and not dedupe.add(
            trace_id=span_dict.get("trace_id"),
            span_id=span_dict.get("span_id"),
        ):
            continue
        adk_span_dicts.append(span_dict)

    if not adk_span_dicts:
        return SpanExportResult.SUCCESS

    payloads: List[Dict[str, object]] = []
    grouped = group_spans_by_trace(spans=adk_span_dicts)
    for trace_spans in grouped.values():
        payloads.extend(converter.convert(trace_or_spans=trace_spans))

    if not payloads:
        return SpanExportResult.SUCCESS

    # Restore captured propagate_attributes context if the current thread
    # doesn't have it (e.g. the context manager already exited, or we're on
    # the BatchSpanProcessor background thread).
    current_ctx = _PROPAGATED_ATTRIBUTES.get()
    if captured_ctx and not current_ctx:
        token = _PROPAGATED_ATTRIBUTES.set(captured_ctx)
        try:
            exporter.export(payloads)
        finally:
            _PROPAGATED_ATTRIBUTES.reset(token)
    else:
        exporter.export(payloads)

    return SpanExportResult.SUCCESS


def _flush_pending_trace_buffer() -> None:
    """Export all pending per-trace buffers (e.g. on deactivate). Best-effort."""
    with _TRACE_BUFFER_LOCK:
        pending = list(_TRACE_BUFFER.items())
        _TRACE_BUFFER.clear()
        pending_ctx = dict(_TRACE_CTX_CACHE)
        _TRACE_CTX_CACHE.clear()
    for trace_id, spans in pending:
        if not spans:
            continue
        try:
            _export_adk_spans(
                spans=spans,
                captured_ctx=pending_ctx.get(trace_id),
            )
        except Exception as exc:
            logger.warning(
                "Failed to export buffered ADK spans on shutdown: %s",
                exc,
                exc_info=True,
            )


def _batch_export_wrapper(wrapped, instance, args, kwargs):
    """Wrapper for BatchSpanProcessor._export method.

    ADK spans are normally intercepted earlier in on_end (on the user thread)
    and never reach the batch queue.  This wrapper is a safety net that filters
    out any ADK spans that slip through (e.g. race conditions) to prevent
    double-export.
    """
    if _ACTIVE_EXPORTER is None:
        return wrapped(*args, **kwargs)
    spans = list(args[0]) if args else list(kwargs.get("spans", []))
    if not spans:
        return wrapped(*args, **kwargs)

    other_spans = [s for s in spans if not is_adk_span(span=s)]

    if _ACTIVE_PASSTHROUGH:
        return wrapped(*args, **kwargs)

    if other_spans:
        kwargs.pop("spans", None)
        return wrapped(other_spans, **kwargs)

    return SpanExportResult.SUCCESS


def _on_end_wrapper(wrapped, instance, args, kwargs):
    """Wrapper for span processor on_end.

    Intercepts ADK spans on the **user thread** (where ContextVars from
    ``propagate_attributes`` are available), buffers them per trace, and
    flushes the entire trace through the converter when the root
    "invocation" span arrives.
    """
    if _ACTIVE_EXPORTER is None:
        return wrapped(*args, **kwargs)
    span = args[0] if args else kwargs.get("span")
    if span is None or not is_adk_span(span=span):
        return wrapped(*args, **kwargs)

    # Determine trace_id and whether this is the root span
    span_name = getattr(span, "name", None) or ""
    ctx = getattr(span, "context", None)
    trace_id = format(ctx.trace_id, "032x") if ctx else None
    is_root = span_name in _ADK_ROOT_SPAN_NAMES

    # Capture propagated attributes on the user thread.  First write wins
    # (the outermost propagate_attributes scope is the most complete).
    prop_ctx = _PROPAGATED_ATTRIBUTES.get()
    if prop_ctx and trace_id:
        with _TRACE_BUFFER_LOCK:
            if trace_id not in _TRACE_CTX_CACHE:
                _TRACE_CTX_CACHE[trace_id] = prop_ctx

    if trace_id:
        all_spans: Optional[List[object]] = None
        captured_ctx: Optional[Dict[str, Any]] = None
        overflow_flush = False
        with _TRACE_BUFFER_LOCK:
            buf = _TRACE_BUFFER.setdefault(trace_id, [])
            buf.append(span)
            max_spans = _ACTIVE_TRACE_BUFFER_MAX_SPANS
            overflow_flush = max_spans is not None and len(buf) > max_spans
            if overflow_flush:
                all_spans = _TRACE_BUFFER.pop(trace_id)
                captured_ctx = _TRACE_CTX_CACHE.pop(trace_id, None)
            elif not is_root:
                if _ACTIVE_PASSTHROUGH:
                    return wrapped(*args, **kwargs)
                return None
            else:
                all_spans = _TRACE_BUFFER.pop(trace_id)
                captured_ctx = _TRACE_CTX_CACHE.pop(trace_id, None)

        if overflow_flush:
            logger.warning(
                "ADK trace buffer exceeded trace_buffer_max_spans (%s); "
                "flushing partial trace %s…",
                max_spans,
                trace_id[:16],
            )
        try:
            _export_adk_spans(
                spans=all_spans or [],
                captured_ctx=captured_ctx,
            )
        except Exception as exc:
            logger.warning("Failed to export ADK spans: %s", exc, exc_info=True)
    else:
        # No trace_id — export individually (fallback)
        try:
            _export_adk_spans(spans=[span], captured_ctx=prop_ctx or None)
        except Exception as exc:
            logger.warning("Failed to export ADK span: %s", exc, exc_info=True)

    if _ACTIVE_PASSTHROUGH:
        return wrapped(*args, **kwargs)
    return None


class GoogleAdkInstrumentor:
    """Instrument OpenTelemetry exporters to send Google ADK traces to Respan.

    Usage::

        from respan import Respan
        from respan_instrumentation_google_adk import GoogleAdkInstrumentor

        respan = Respan(instrumentations=[GoogleAdkInstrumentor(environment="development")])
    """

    name = "google-adk"

    def __init__(
        self,
        environment: Optional[str] = None,
        customer_identifier: Optional[str] = None,
        passthrough: bool = False,
        dedupe_max_size: int = 10000,
        trace_buffer_max_spans: Optional[int] = _DEFAULT_TRACE_BUFFER_MAX_SPANS,
    ) -> None:
        self._environment = environment
        self._customer_identifier = customer_identifier
        self._passthrough = passthrough
        self._dedupe_max_size = dedupe_max_size
        self._trace_buffer_max_spans = trace_buffer_max_spans
        self._converter: Optional[AdkSpanConverter] = None

    def activate(self, exporter: Any) -> None:
        """Activate ADK instrumentation with the provided exporter.

        Args:
            exporter: A ``RespanSpanExporterV2`` instance (or compatible object
                      with an ``export(payloads)`` method).
        """
        self._converter = AdkSpanConverter(
            environment=self._environment,
            customer_identifier=self._customer_identifier,
        )

        global _ACTIVE_EXPORTER, _ACTIVE_CONVERTER, _ACTIVE_DEDUPE, _ACTIVE_PASSTHROUGH
        global _ACTIVE_TRACE_BUFFER_MAX_SPANS
        _ACTIVE_EXPORTER = exporter
        _ACTIVE_CONVERTER = self._converter
        _ACTIVE_DEDUPE = _SpanDedupeCache(max_size=self._dedupe_max_size)
        _ACTIVE_PASSTHROUGH = self._passthrough
        _ACTIVE_TRACE_BUFFER_MAX_SPANS = self._trace_buffer_max_spans

        self._patch_span_processors()
        logger.info("Respan Google ADK instrumentation activated")

    def deactivate(self) -> None:
        """Deactivate ADK instrumentation."""
        global _ACTIVE_EXPORTER, _ACTIVE_CONVERTER, _ACTIVE_DEDUPE, _ACTIVE_PASSTHROUGH
        global _ACTIVE_TRACE_BUFFER_MAX_SPANS
        _flush_pending_trace_buffer()
        _ACTIVE_EXPORTER = None
        _ACTIVE_CONVERTER = None
        _ACTIVE_DEDUPE = None
        _ACTIVE_PASSTHROUGH = False
        _ACTIVE_TRACE_BUFFER_MAX_SPANS = _DEFAULT_TRACE_BUFFER_MAX_SPANS
        self._converter = None
        logger.info("Respan Google ADK instrumentation deactivated")

    def _patch_span_processors(self) -> None:
        global _PATCHED
        if _PATCHED:
            return

        try:
            from opentelemetry.sdk.trace import export as trace_export

            # Always patch on_end to intercept ADK spans on the user thread.
            # This is where we capture propagate_attributes context and buffer
            # spans per-trace.
            wrapt.wrap_function_wrapper(
                module="opentelemetry.sdk.trace.export",
                name="BatchSpanProcessor.on_end",
                wrapper=_on_end_wrapper,
            )

            # Also patch _export as a safety net: any ADK spans that somehow
            # reach the batch queue are filtered out to prevent double-export.
            if hasattr(trace_export.BatchSpanProcessor, "_export"):
                wrapt.wrap_function_wrapper(
                    module="opentelemetry.sdk.trace.export",
                    name="BatchSpanProcessor._export",
                    wrapper=_batch_export_wrapper,
                )
        except Exception as exc:
            logger.debug("Failed to patch BatchSpanProcessor: %s", exc)

        wrapt.wrap_function_wrapper(
            module="opentelemetry.sdk.trace.export",
            name="SimpleSpanProcessor.on_end",
            wrapper=_on_end_wrapper,
        )

        _PATCHED = True
        logger.debug("Patched OpenTelemetry span processors for ADK export")
