from contextlib import contextmanager
import logging
from typing import Any, Dict, List, Union
from opentelemetry import trace, context as context_api
from opentelemetry.trace.span import Span
from pydantic import ValidationError
from respan_sdk.respan_types.span_types import (
    RESPAN_SPAN_ATTRIBUTES_MAP,
    RespanSpanAttributes,
    SpanLink,
)
LINK_TIMESTAMP_ATTR = RespanSpanAttributes.LINK_TIMESTAMP.value
from respan_sdk.respan_types.param_types import RespanParams
from respan_sdk.utils.data_processing.id_processing import (
    SPAN_ID_HEX_LENGTH,
    TRACE_ID_HEX_LENGTH,
    normalize_hex_id,
)
from respan_tracing.utils.logging import get_respan_logger


from ..constants.generic_constants import LOGGER_NAME_SPAN
from ..constants.context_constants import PENDING_SPAN_LINKS_KEY

__all__ = ["SpanLink", "span_link_to_otel", "respan_span_attributes", "attach_span_links"]

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
        attrs[LINK_TIMESTAMP_ATTR] = link.timestamp
    return trace.Link(context=span_context, attributes=attrs)


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
            if key in RESPAN_SPAN_ATTRIBUTES_MAP and key != "metadata":
                try:
                    current_span.set_attribute(RESPAN_SPAN_ATTRIBUTES_MAP[key], value)
                except (ValueError, TypeError) as e:
                    logger.warning(
                        f"Failed to set span attribute {RESPAN_SPAN_ATTRIBUTES_MAP[key]}={value}: {str(e)}"
                    )
            # Treat metadata as a special case
            if key == "metadata":
                for metadata_key, metadata_value in value.items():
                    current_span.set_attribute(f"{RespanSpanAttributes.RESPAN_METADATA.value}.{metadata_key}", metadata_value)
        yield
    except ValidationError as e:
        logger.warning(f"Failed to validate params: {str(e.errors(include_url=False))}")
        yield
    except Exception as e:
        logger.exception(f"Unexpected error in span attribute context: {str(e)}")
        raise
