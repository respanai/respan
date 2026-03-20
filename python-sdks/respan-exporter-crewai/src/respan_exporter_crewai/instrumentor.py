"""Respan OpenTelemetry redirect for CrewAI traces.

This instrumentor wraps OpenTelemetry span processors to intercept CrewAI spans
and export them to the Respan tracing ingest endpoint.
"""
from __future__ import annotations

import logging
from collections import OrderedDict
from threading import Lock
from typing import Collection, Dict, Iterable, List, Optional, Sequence

import wrapt
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.sdk.trace.export import SpanExportResult

from respan_exporter_crewai.exporter import RespanCrewAIExporter
from respan_exporter_crewai.utils import group_spans_by_trace, is_crewai_span, otel_span_to_dict

logger = logging.getLogger(__name__)

_INSTRUMENTS = ("crewai >= 0.80.0", "openinference-instrumentation-crewai >= 0.1.0")

_PATCHED = False


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


_ACTIVE_EXPORTER: Optional[RespanCrewAIExporter] = None
_ACTIVE_DEDUPE = _SpanDedupeCache()
_ACTIVE_PASSTHROUGH = False


def _export_crewai_spans(spans: Iterable[object]) -> SpanExportResult:
    """Export CrewAI spans to Respan."""
    exporter = _ACTIVE_EXPORTER
    dedupe = _ACTIVE_DEDUPE
    if exporter is None:
        return SpanExportResult.SUCCESS

    crewai_span_dicts: List[Dict[str, object]] = []
    for span in spans:
        if not is_crewai_span(span=span):
            continue
        span_dict = otel_span_to_dict(span=span)
        if dedupe and not dedupe.add(
            trace_id=span_dict.get("trace_id"),
            span_id=span_dict.get("span_id"),
        ):
            continue
        crewai_span_dicts.append(span_dict)

    if not crewai_span_dicts:
        return SpanExportResult.SUCCESS

    payloads: List[Dict[str, object]] = []
    grouped = group_spans_by_trace(spans=crewai_span_dicts)
    for trace_spans in grouped.values():
        payloads.extend(exporter.build_payload(trace_or_spans=trace_spans))

    if not payloads:
        return SpanExportResult.SUCCESS

    if not exporter.api_key:
        logger.warning("Respan API key is not set; skipping CrewAI export")
        return SpanExportResult.SUCCESS

    exporter._send(payloads=payloads)
    return SpanExportResult.SUCCESS


def _batch_export_wrapper(wrapped, instance, args, kwargs):
    """Wrapper for BatchSpanProcessor._export method."""
    spans = list(args[0]) if args else list(kwargs.get("spans", []))
    if not spans:
        return wrapped(*args, **kwargs)

    crewai_spans = []
    other_spans = []
    for span in spans:
        if is_crewai_span(span=span):
            crewai_spans.append(span)
        else:
            other_spans.append(span)

    if not crewai_spans:
        return wrapped(*args, **kwargs)

    try:
        export_result = _export_crewai_spans(spans=crewai_spans)
    except Exception as exc:
        logger.warning("Failed to export CrewAI spans: %s", exc, exc_info=True)
        export_result = SpanExportResult.FAILURE

    if _ACTIVE_PASSTHROUGH:
        return wrapped(*args, **kwargs)

    if other_spans:
        return wrapped(other_spans, **kwargs)

    return export_result


def _on_end_wrapper(wrapped, instance, args, kwargs):
    """Wrapper for SimpleSpanProcessor.on_end method."""
    span = args[0] if args else kwargs.get("span")
    if span is None or not is_crewai_span(span=span):
        return wrapped(*args, **kwargs)

    try:
        _export_crewai_spans(spans=[span])
    except Exception as exc:
        logger.warning("Failed to export CrewAI span: %s", exc, exc_info=True)

    if _ACTIVE_PASSTHROUGH:
        return wrapped(*args, **kwargs)
    return None


class RespanCrewAIInstrumentor(BaseInstrumentor):
    """Instrument OpenTelemetry exporters to send CrewAI traces to Respan."""

    def __init__(self) -> None:
        super().__init__()
        self._exporter: Optional[RespanCrewAIExporter] = None
        self._passthrough = False
        self._dedupe = _SpanDedupeCache()

    def instrumentation_dependencies(self) -> Collection[str]:
        return _INSTRUMENTS

    def _instrument(self, **kwargs) -> None:
        self._exporter = RespanCrewAIExporter(
            api_key=kwargs.get("api_key"),
            endpoint=kwargs.get("endpoint"),
            base_url=kwargs.get("base_url"),
            environment=kwargs.get("environment"),
            customer_identifier=kwargs.get("customer_identifier"),
            timeout=kwargs.get("timeout", 10),
        )
        self._passthrough = bool(kwargs.get("passthrough", False))
        self._dedupe = _SpanDedupeCache(max_size=kwargs.get("dedupe_max_size", 10000))

        global _ACTIVE_EXPORTER, _ACTIVE_DEDUPE, _ACTIVE_PASSTHROUGH
        _ACTIVE_EXPORTER = self._exporter
        _ACTIVE_DEDUPE = self._dedupe
        _ACTIVE_PASSTHROUGH = self._passthrough

        self._patch_span_processors()
        logger.info("Respan CrewAI instrumentation enabled")

    def _uninstrument(self, **kwargs) -> None:
        global _ACTIVE_EXPORTER, _ACTIVE_DEDUPE, _ACTIVE_PASSTHROUGH
        _ACTIVE_EXPORTER = None
        _ACTIVE_DEDUPE = None
        _ACTIVE_PASSTHROUGH = True
        logger.info("Respan CrewAI instrumentation disabled")

    def _patch_span_processors(self) -> None:
        global _PATCHED
        if _PATCHED:
            return

        try:
            from opentelemetry.sdk.trace import export as trace_export

            if hasattr(trace_export.BatchSpanProcessor, "_export"):
                wrapt.wrap_function_wrapper(
                    module="opentelemetry.sdk.trace.export",
                    name="BatchSpanProcessor._export",
                    wrapper=_batch_export_wrapper,
                )
            else:
                wrapt.wrap_function_wrapper(
                    module="opentelemetry.sdk.trace.export",
                    name="BatchSpanProcessor.on_end",
                    wrapper=_on_end_wrapper,
                )
        except Exception as exc:
            logger.debug("Failed to patch BatchSpanProcessor: %s", exc)

        wrapt.wrap_function_wrapper(
            module="opentelemetry.sdk.trace.export",
            name="SimpleSpanProcessor.on_end",
            wrapper=_on_end_wrapper,
        )

        _PATCHED = True
        logger.debug("Patched OpenTelemetry span processors for CrewAI export")
