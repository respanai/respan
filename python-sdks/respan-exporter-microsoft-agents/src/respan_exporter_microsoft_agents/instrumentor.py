"""Respan OpenTelemetry redirect for microsoft_agents traces."""
from __future__ import annotations

import logging
from collections import OrderedDict
from threading import Lock
from typing import Collection, Dict, Iterable, List, Optional

import wrapt
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.sdk.trace.export import SpanExportResult

from respan_exporter_microsoft_agents.exporter import RespanMicrosoftAgentsExporter
from respan_exporter_microsoft_agents.utils import group_spans_by_trace, is_microsoft_agents_span, otel_span_to_dict

logger = logging.getLogger(__name__)

_INSTRUMENTS = ("opentelemetry-sdk >= 1.20.0",)
_PATCHED = False


class _SpanDedupeCache:
    """Cache for deduplicating spans."""

    def __init__(self, max_size: int = 10000) -> None:
        self._max_size = max_size
        self._data: "OrderedDict[str, None]" = OrderedDict()
        self._lock = Lock()

    def add(self, trace_id: Optional[str], span_id: Optional[str]) -> bool:
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


_ACTIVE_EXPORTER: Optional[RespanMicrosoftAgentsExporter] = None
_ACTIVE_DEDUPE = _SpanDedupeCache()
_ACTIVE_PASSTHROUGH = False


def _export_microsoft_agents_spans(spans: Iterable[object]) -> SpanExportResult:
    """Export microsoft_agents spans to Respan."""
    exporter = _ACTIVE_EXPORTER
    dedupe = _ACTIVE_DEDUPE
    if exporter is None:
        return SpanExportResult.SUCCESS

    span_dicts: List[Dict[str, object]] = []
    for span in spans:
        if not is_microsoft_agents_span(span=span):
            continue
        span_dict = otel_span_to_dict(span=span)
        if dedupe and not dedupe.add(trace_id=span_dict.get("trace_id"), span_id=span_dict.get("span_id")):
            continue
        span_dicts.append(span_dict)

    if not span_dicts:
        return SpanExportResult.SUCCESS

    payloads: List[Dict[str, object]] = []
    grouped = group_spans_by_trace(spans=span_dicts)
    for trace_spans in grouped.values():
        payloads.extend(exporter.build_payload(trace_or_spans=trace_spans))

    if not payloads:
        return SpanExportResult.SUCCESS

    if not exporter.api_key:
        logger.warning("Respan API key is not set; skipping microsoft_agents export")
        return SpanExportResult.SUCCESS

    exporter._send(payloads=payloads)
    return SpanExportResult.SUCCESS


def _batch_export_wrapper(wrapped, instance, args, kwargs):
    spans = list(args[0]) if args else list(kwargs.get("spans", []))
    if not spans:
        return wrapped(*args, **kwargs)

    microsoft_agents_spans = []
    other_spans = []
    for s in spans:
        if is_microsoft_agents_span(span=s):
            microsoft_agents_spans.append(s)
        else:
            other_spans.append(s)

    if not microsoft_agents_spans:
        return wrapped(*args, **kwargs)

    try:
        _export_microsoft_agents_spans(spans=microsoft_agents_spans)
    except Exception as exc:
        logger.warning("Failed to export microsoft_agents spans: %s", exc, exc_info=True)

    if _ACTIVE_PASSTHROUGH:
        return wrapped(*args, **kwargs)
    if other_spans:
        return wrapped(other_spans, **kwargs)
    return SpanExportResult.SUCCESS


def _on_end_wrapper(wrapped, instance, args, kwargs):
    span = args[0] if args else kwargs.get("span")
    if span is None or not is_microsoft_agents_span(span=span):
        return wrapped(*args, **kwargs)

    try:
        _export_microsoft_agents_spans(spans=[span])
    except Exception as exc:
        logger.warning("Failed to export microsoft_agents span: %s", exc, exc_info=True)

    if _ACTIVE_PASSTHROUGH:
        return wrapped(*args, **kwargs)
    return None


class RespanMicrosoftAgentsInstrumentor(BaseInstrumentor):
    """Instrument OTel exporters to send microsoft_agents traces to Respan."""

    def __init__(self) -> None:
        super().__init__()
        self._exporter: Optional[RespanMicrosoftAgentsExporter] = None
        self._passthrough = False
        self._dedupe = _SpanDedupeCache()

    def instrumentation_dependencies(self) -> Collection[str]:
        return _INSTRUMENTS

    def _instrument(self, **kwargs) -> None:
        self._exporter = RespanMicrosoftAgentsExporter(
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
        logger.info("Respan microsoft_agents instrumentation enabled")

    def _uninstrument(self, **kwargs) -> None:
        global _ACTIVE_EXPORTER, _ACTIVE_DEDUPE, _ACTIVE_PASSTHROUGH
        _ACTIVE_EXPORTER = None
        _ACTIVE_DEDUPE = _SpanDedupeCache()
        _ACTIVE_PASSTHROUGH = True
        logger.info("Respan microsoft_agents instrumentation disabled")

    def _patch_span_processors(self) -> None:
        global _PATCHED
        if _PATCHED:
            return

        try:
            from opentelemetry.sdk.trace import export as trace_export
            if hasattr(trace_export.BatchSpanProcessor, "_export"):
                wrapt.wrap_function_wrapper("opentelemetry.sdk.trace.export", "BatchSpanProcessor._export", _batch_export_wrapper)
            else:
                wrapt.wrap_function_wrapper("opentelemetry.sdk.trace.export", "BatchSpanProcessor.on_end", _on_end_wrapper)
        except Exception as exc:
            logger.debug("Failed to patch BatchSpanProcessor: %s", exc)

        wrapt.wrap_function_wrapper("opentelemetry.sdk.trace.export", "SimpleSpanProcessor.on_end", _on_end_wrapper)

        _PATCHED = True
        logger.debug("Patched OTel span processors for microsoft_agents export")
