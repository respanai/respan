from contextlib import contextmanager
from datetime import datetime, timezone
import logging
from typing import Any, Dict, List, Optional, Union
from opentelemetry import trace, context as context_api
from opentelemetry.trace.span import Span
from pydantic import ValidationError
from respan_sdk.constants.span_attributes import (
    RESPAN_LINK_TIMESTAMP,
    RESPAN_METADATA,
    RESPAN_SPAN_ATTRIBUTES_MAP,
)
from respan_sdk.respan_types.span_types import SpanLink
from respan_sdk.respan_types.param_types import RespanParams
from respan_sdk.utils.data_processing.id_processing import (
    SPAN_ID_HEX_LENGTH,
    TRACE_ID_HEX_LENGTH,
    format_span_id,
    format_trace_id,
    normalize_hex_id,
)
from respan_tracing.utils.logging import get_respan_logger


from respan_tracing.constants.generic_constants import LOGGER_NAME_SPAN
from respan_tracing.constants.context_constants import PENDING_SPAN_LINKS_KEY

__all__ = ["SpanLink", "span_link_to_otel", "span_to_link", "respan_span_attributes", "attach_span_links"]

logger = get_respan_logger(LOGGER_NAME_SPAN)


def span_link_to_otel(link: SpanLink) -> trace.Link:
    """Convert a SpanLink data model into an OpenTelemetry Link.

    Validates hex identifiers and builds the OTel SpanContext + Link.
    """
    normalized_trace_id = normalize_hex_id(link.trace_id, TRACE_ID_HEX_LENGTH, "trace_id")
    normalized_span_id = normalize_hex_id(link.span_id, SPAN_ID_HEX_LENGTH, "span_id")
    trace_flags = trace.TraceFlags(trace.TraceFlags.SAMPLED if link.is_sampled else 0)
    span_context = trace.SpanContext(
        trace_id=int(normalized_trace_id, 16),
        span_id=int(normalized_span_id, 16),
        is_remote=link.is_remote,
        trace_flags=trace_flags,
        trace_state=trace.TraceState(),
    )
    # Merge timestamp into attributes if provided (enables efficient CH lookups)
    attrs = dict(link.attributes)
    if link.timestamp:
        attrs[RESPAN_LINK_TIMESTAMP] = link.timestamp
    return trace.Link(context=span_context, attributes=attrs)


def span_to_link(
    span: Span,
    attributes: Optional[Dict[str, Any]] = None,
) -> SpanLink:
    """Create a SpanLink from a live OTel span, auto-capturing its timestamp.

    Extracts trace_id, span_id, and start_time from the span. The start_time
    is converted to ISO 8601 and stored as ``timestamp`` so downstream consumers
    (e.g., CH point-lookups) can use it without a full scan.

    Args:
        span: A live OpenTelemetry span (must have a valid SpanContext).
        attributes: Optional extra attributes to include in the link.

    Returns:
        A SpanLink with auto-captured identifiers and timestamp.

    Raises:
        ValueError: If the span has an invalid (zero) SpanContext.
    """
    ctx = span.get_span_context()
    if not ctx or not ctx.is_valid:
        raise ValueError("Cannot create link from span with invalid SpanContext")

    trace_id = format_trace_id(ctx.trace_id)
    span_id = format_span_id(ctx.span_id)

    # Auto-capture timestamp from span start_time (SDK spans expose this)
    timestamp = None
    start_time_ns = getattr(span, "start_time", None)
    if start_time_ns:
        dt = datetime.fromtimestamp(
            start_time_ns // 10**9, tz=timezone.utc
        ).replace(microsecond=(start_time_ns % 10**9) // 1000)
        timestamp = dt.isoformat()

    return SpanLink(
        trace_id=trace_id,
        span_id=span_id,
        attributes=attributes or {},
        timestamp=timestamp,
        is_remote=ctx.is_remote,
        is_sampled=bool(ctx.trace_flags & trace.TraceFlags.SAMPLED),
    )


def attach_span_links(links: List[SpanLink]) -> None:
    """Attach span links to the current OTel context for the next decorated span.

    Links stored via this function are consumed (cleared) when the next
    decorator-created span starts. This decouples link producers from span
    consumers — any caller can attach links without the decorated function
    needing to know.

    Args:
        links: List of SpanLink objects to attach to the next span.
    """
    if not links:
        return
    # Merge with any previously attached links in this context
    existing = context_api.get_value(PENDING_SPAN_LINKS_KEY) or []
    merged = list(existing) + list(links)
    context_api.attach(context_api.set_value(PENDING_SPAN_LINKS_KEY, merged))


def consume_span_links() -> List[trace.Link]:
    """Read and clear pending span links from the OTel context.

    Returns converted OTel Link objects. Clears the context key so links
    are only consumed once.
    """
    pending: List[SpanLink] = context_api.get_value(PENDING_SPAN_LINKS_KEY) or []
    if not pending:
        return []
    # Clear pending links from context
    context_api.attach(context_api.set_value(PENDING_SPAN_LINKS_KEY, None))
    return [span_link_to_otel(link) for link in pending]


@contextmanager
def respan_span_attributes(respan_params: Union[Dict[str, Any], RespanParams]):
    """Adds Respan-specific attributes to the current active span.
    
    Args:
        respan_params: Dictionary of parameters to set as span attributes.
                          Must conform to RespanParams model structure.
    
    Notes:
        - If no active span is found, a warning will be logged and the context will continue
        - If params validation fails, a warning will be logged and the context will continue
        - If an attribute cannot be set, a warning will be logged and the context will continue
    """
    from respan_tracing.core.tracer import RespanTracer

    if not RespanTracer.is_initialized():
        logger.warning("Respan Telemetry not initialized. Attributes will not be set.")
        yield
        return
        

    current_span = trace.get_current_span()
    
    if not isinstance(current_span, Span):
        logger.warning("No active span found. Attributes will not be set.")
        yield
        return

    try:
        # Keep your original validation
        validated_params = (
            respan_params 
            if isinstance(respan_params, RespanParams) 
            else RespanParams.model_validate(respan_params)
        )
        
        for key, value in validated_params.model_dump(mode="json").items():
            attr_key = RESPAN_SPAN_ATTRIBUTES_MAP.get(key)
            if attr_key and attr_key != RESPAN_METADATA:
                try:
                    current_span.set_attribute(attr_key, value)
                except (ValueError, TypeError) as e:
                    logger.warning(
                        f"Failed to set span attribute {attr_key}={value}: {str(e)}"
                    )
            # Treat metadata as a special case — expand to per-key attributes
            if attr_key == RESPAN_METADATA:
                for metadata_key, metadata_value in value.items():
                    current_span.set_attribute(f"{RESPAN_METADATA}.{metadata_key}", metadata_value)
        yield
    except ValidationError as e:
        logger.warning(f"Failed to validate params: {str(e.errors(include_url=False))}")
        yield
    except Exception as e:
        logger.exception(f"Unexpected error in span attribute context: {str(e)}")
        raise
