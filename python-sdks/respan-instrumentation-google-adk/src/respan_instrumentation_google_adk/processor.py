"""Wrapt-based processor patching for ADK span enrichment.

Patches RespanSpanProcessor.on_end(), BatchSpanProcessor._export(),
and SimpleSpanProcessor.on_end() to intercept ADK spans, enrich them
with traceloop attributes, and pass enriched versions through the
normal OTEL pipeline.

Follows the same wrapt pattern as respan-exporter-agno.
"""

import logging

import wrapt

from .adk_detection import is_adk_span
from .enrichment import enrich_adk_batch

logger = logging.getLogger(__name__)

_PATCHED = False


def _respan_processor_on_end_wrapper(wrapped, instance, args, kwargs):
    """Bypass is_processable_span() for ADK spans.

    RespanSpanProcessor.on_end() filters spans via is_processable_span()
    before passing them to the inner BatchSpanProcessor. Raw ADK spans
    (e.g. "invocation", "agent_run") lack traceloop attributes and would
    be dropped. This wrapper lets them through to the inner processor
    where they'll be enriched in the _export() batch.
    """
    span = args[0] if args else kwargs.get("span")
    if span is not None and is_adk_span(span):
        # Pass directly to inner processor, bypassing is_processable_span()
        inner = getattr(instance, "processor", None)
        if inner is not None:
            inner.on_end(span)
            return
    return wrapped(*args, **kwargs)


def _batch_export_wrapper(wrapped, instance, args, kwargs):
    """Enrich ADK spans in a batch before export.

    Intercepts BatchSpanProcessor._export() to enrich ADK spans with
    traceloop attributes (cross-span enrichment: propagate agent name,
    I/O from child spans to root). Non-ADK spans pass through unchanged.
    """
    spans = args[0] if args else kwargs.get("spans", [])
    if not spans:
        return wrapped(*args, **kwargs)

    enriched = enrich_adk_batch(list(spans))

    # Replace the spans argument and call original
    if args:
        return wrapped(enriched, *args[1:], **kwargs)
    return wrapped(enriched, **{k: v for k, v in kwargs.items() if k != "spans"})


def _on_end_wrapper(wrapped, instance, args, kwargs):
    """Per-span enrichment for SimpleSpanProcessor.

    Limited: no cross-span propagation (single-span context only).
    """
    span = args[0] if args else kwargs.get("span")
    if span is None or not is_adk_span(span):
        return wrapped(*args, **kwargs)

    enriched = enrich_adk_batch([span])
    if enriched:
        if args:
            return wrapped(enriched[0], *args[1:], **kwargs)
        return wrapped(enriched[0], **{k: v for k, v in kwargs.items() if k != "span"})
    return wrapped(*args, **kwargs)


def patch_span_processors() -> None:
    """Patch OTel span processors with wrapt for ADK enrichment.

    Patches 3 points:
    1. RespanSpanProcessor.on_end — let ADK spans bypass is_processable_span()
    2. BatchSpanProcessor._export — batch-level enrichment
    3. SimpleSpanProcessor.on_end — per-span fallback

    Idempotent: calling multiple times is safe.
    """
    global _PATCHED
    if _PATCHED:
        return

    # 1. Let ADK spans bypass RespanSpanProcessor filtering
    try:
        wrapt.wrap_function_wrapper(
            module="respan_tracing.processors.base",
            name="RespanSpanProcessor.on_end",
            wrapper=_respan_processor_on_end_wrapper,
        )
    except Exception as exc:
        logger.debug("Failed to patch RespanSpanProcessor: %s", exc)

    # 2. Enrich ADK spans in batch before export
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

    # 3. SimpleSpanProcessor fallback
    try:
        wrapt.wrap_function_wrapper(
            module="opentelemetry.sdk.trace.export",
            name="SimpleSpanProcessor.on_end",
            wrapper=_on_end_wrapper,
        )
    except Exception as exc:
        logger.debug("Failed to patch SimpleSpanProcessor: %s", exc)

    _PATCHED = True
    logger.debug("Patched OTel span processors for ADK enrichment")
