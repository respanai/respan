from typing import Any, Dict, Optional

from pydantic import ConfigDict, Field

from respan_sdk.respan_types.base_types import RespanBaseModel


class SpanLink(RespanBaseModel):
    """Serializable link definition for attaching causal links to new spans.

    A lightweight data holder with no OTel dependency.  The conversion to an
    OpenTelemetry ``trace.Link`` is performed by ``respan_tracing`` at runtime.

    Args:
        trace_id: Hex trace ID of the linked span.
        span_id: Hex span ID of the linked span.
        attributes: Extra key-value pairs to attach to the OTel link.
        timestamp: Optional ISO 8601 timestamp of the linked trace. When set,
            automatically merged into link attributes as
            ``respan.link.timestamp``. This enables efficient ClickHouse
            point-lookups when navigating to the linked trace (the primary
            key includes timestamp).
        is_remote: Whether the linked span is remote (default True).
        is_sampled: Whether the linked span was sampled (default True).
    """

    model_config = ConfigDict(frozen=True)

    trace_id: str
    span_id: str
    attributes: Dict[str, Any] = Field(default_factory=dict)
    timestamp: Optional[str] = None
    is_remote: bool = True
    is_sampled: bool = True
