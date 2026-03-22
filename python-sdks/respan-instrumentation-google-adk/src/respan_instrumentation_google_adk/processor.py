"""Wrapt-based processor patching for ADK span enrichment.

Patches RespanSpanProcessor.on_end() to bypass span filtering for ADK spans,
and RespanSpanExporter.export() to enrich ADK spans in batch before export.

The exporter-level patch ensures cross-span enrichment (agent name, I/O
propagation) works regardless of which OTel processor is used (Batch or
Simple), and regardless of internal OTel SDK implementation details.

Follows the same wrapt pattern as respan-exporter-agno.
"""

import wrapt

from respan_tracing.utils.logging import get_respan_logger

from .adk_detection import is_adk_span
from .enrichment import enrich_adk_batch

logger = get_respan_logger("instrumentation.google_adk")

_PATCHED = False


def _respan_processor_on_end_wrapper(wrapped, instance, args, kwargs):
    """Bypass is_processable_span() for ADK spans.

    RespanSpanProcessor.on_end() filters spans via is_processable_span()
    before passing them to the inner BatchSpanProcessor. Raw ADK spans
    (e.g. "invocation", "agent_run") lack traceloop attributes and would
    be dropped. This wrapper lets them through to the inner processor
    where they'll be enriched at export time.
    """
    span = args[0] if args else kwargs.get("span")
    if span is not None and is_adk_span(span):
        # Pass directly to inner processor, bypassing is_processable_span()
        inner = getattr(instance, "processor", None)
        if inner is not None:
            inner.on_end(span)
            return
    return wrapped(*args, **kwargs)


def _exporter_export_wrapper(wrapped, instance, args, kwargs):
    """Enrich ADK spans in batch before the exporter serializes them.

    Intercepts RespanSpanExporter.export() to enrich ADK spans with
    traceloop attributes (cross-span enrichment: propagate agent name,
    I/O from child spans to root). Non-ADK spans pass through unchanged.

    This runs BEFORE the exporter's own enrichment (root promotion,
    gen_ai.system → llm.request.type), so enriched spans may be
    double-wrapped in EnrichedSpan — the proxy pattern handles this
    correctly via property chaining.
    """
    spans = args[0] if args else kwargs.get("spans", [])
    if not spans:
        return wrapped(*args, **kwargs)

    enriched = enrich_adk_batch(list(spans))

    # Replace the spans argument and call original
    if args:
        return wrapped(enriched, *args[1:], **kwargs)
    return wrapped(enriched, **{k: v for k, v in kwargs.items() if k != "spans"})


def patch_span_processors() -> None:
    """Patch OTel span processors with wrapt for ADK enrichment.

    Patches 2 points:
    1. RespanSpanProcessor.on_end — let ADK spans bypass is_processable_span()
    2. RespanSpanExporter.export — batch-level enrichment before serialization

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
        wrapt.wrap_function_wrapper(
            module="respan_tracing.exporters.respan",
            name="RespanSpanExporter.export",
            wrapper=_exporter_export_wrapper,
        )
    except Exception as exc:
        logger.debug("Failed to patch RespanSpanExporter: %s", exc)

    _PATCHED = True
    logger.debug("Patched OTel span processors for ADK enrichment")
