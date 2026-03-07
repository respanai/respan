from contextlib import contextmanager
from dataclasses import dataclass, field
import logging
from typing import Any, Dict, Union
from opentelemetry import trace
from opentelemetry.trace.span import Span
from respan_sdk.respan_types.span_types import RESPAN_SPAN_ATTRIBUTES_MAP, RespanSpanAttributes
from respan_sdk.respan_types.param_types import RespanParams
from pydantic import ValidationError
from respan_tracing.utils.logging import get_respan_logger


from ..constants.generic_constants import LOGGER_NAME_SPAN

__all__ = ["SpanLink", "respan_span_attributes"]

logger = get_respan_logger(LOGGER_NAME_SPAN)


def _normalize_hex_identifier(identifier: str, expected_length: int, field_name: str) -> str:
    """Normalize a hex trace/span identifier and validate its shape."""
    if not isinstance(identifier, str):
        raise TypeError(f"{field_name} must be a string")

    normalized = identifier.lower().removeprefix("0x")
    if len(normalized) != expected_length:
        raise ValueError(
            f"{field_name} must be {expected_length} hex characters, got {len(normalized)}"
        )

    try:
        int(normalized, 16)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be a hexadecimal string") from exc

    return normalized


@dataclass(frozen=True)
class SpanLink:
    """Serializable link definition for attaching causal links to new spans."""

    trace_id: str
    span_id: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    is_remote: bool = True
    is_sampled: bool = True

    def to_otel_link(self) -> trace.Link:
        """Convert the SDK link shape into an OpenTelemetry Link."""
        normalized_trace_id = _normalize_hex_identifier(self.trace_id, 32, "trace_id")
        normalized_span_id = _normalize_hex_identifier(self.span_id, 16, "span_id")
        trace_flags = trace.TraceFlags(trace.TraceFlags.SAMPLED if self.is_sampled else 0)
        span_context = trace.SpanContext(
            trace_id=int(normalized_trace_id, 16),
            span_id=int(normalized_span_id, 16),
            is_remote=self.is_remote,
            trace_flags=trace_flags,
            trace_state=trace.TraceState(),
        )
        return trace.Link(context=span_context, attributes=self.attributes)


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
