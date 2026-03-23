from enum import Enum
from typing import Any, Dict, Optional

from pydantic import ConfigDict, Field

from respan_sdk.constants.span_attributes import (
    RESPAN_SPAN_CUSTOM_ID,
    RESPAN_CUSTOMER_PARAMS_ID,
    RESPAN_CUSTOMER_PARAMS_EMAIL,
    RESPAN_CUSTOMER_PARAMS_NAME,
    RESPAN_EVALUATION_PARAMS_ID,
    RESPAN_THREADS_ID,
    RESPAN_TRACE_GROUP_ID,
    RESPAN_METADATA,
    RESPAN_LOG_METHOD,
    RESPAN_LOG_TYPE,
    RESPAN_LOG_ID,
    RESPAN_LOG_PARENT_ID,
    RESPAN_LOG_ROOT_ID,
    RESPAN_LOG_SOURCE,
)
from respan_sdk.respan_types.base_types import RespanBaseModel


class RespanSpanAttributes(str, Enum):
    """Backward-compatible enum re-exporting constants from ``span_attributes.py``.

    The canonical source of truth is ``respan_sdk.constants.span_attributes``.
    Prefer importing flat constants directly for new code.  This enum exists
    only so existing callers (e.g. respan-backend) that use
    ``RespanSpanAttributes.RESPAN_METADATA.value`` continue to work.
    """

    RESPAN_SPAN_CUSTOM_ID = RESPAN_SPAN_CUSTOM_ID
    RESPAN_CUSTOMER_PARAMS_ID = RESPAN_CUSTOMER_PARAMS_ID
    RESPAN_CUSTOMER_PARAMS_EMAIL = RESPAN_CUSTOMER_PARAMS_EMAIL
    RESPAN_CUSTOMER_PARAMS_NAME = RESPAN_CUSTOMER_PARAMS_NAME
    RESPAN_EVALUATION_PARAMS_ID = RESPAN_EVALUATION_PARAMS_ID
    RESPAN_THREADS_ID = RESPAN_THREADS_ID
    RESPAN_TRACE_GROUP_ID = RESPAN_TRACE_GROUP_ID
    RESPAN_METADATA = RESPAN_METADATA
    LOG_METHOD = RESPAN_LOG_METHOD
    LOG_TYPE = RESPAN_LOG_TYPE
    LOG_ID = RESPAN_LOG_ID
    LOG_PARENT_ID = RESPAN_LOG_PARENT_ID
    LOG_ROOT_ID = RESPAN_LOG_ROOT_ID
    LOG_SOURCE = RESPAN_LOG_SOURCE


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
