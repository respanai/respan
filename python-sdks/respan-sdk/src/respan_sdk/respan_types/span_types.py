from enum import Enum
from typing import Any, Dict, Optional

from pydantic import ConfigDict, Field

from respan_sdk.respan_types.base_types import RespanBaseModel


class RespanSpanAttributes(str, Enum):
    """Respan span attribute key constants.

    Single source of truth for all ``respan.*`` attribute keys used across
    the pipeline.  Both the backend (``RespanSpanAttributes.X.value``) and
    SDK flat-constant importers (``span_attributes.py``) reference this enum.
    """

    # Span params
    RESPAN_SPAN_CUSTOM_ID = "respan.span_params.custom_identifier"

    # Customer params
    RESPAN_CUSTOMER_PARAMS_ID = "respan.customer_params.customer_identifier"
    RESPAN_CUSTOMER_PARAMS_EMAIL = "respan.customer_params.email"
    RESPAN_CUSTOMER_PARAMS_NAME = "respan.customer_params.name"

    # Evaluation params
    RESPAN_EVALUATION_PARAMS_ID = "respan.evaluation_params.evaluation_identifier"

    # Threads
    RESPAN_THREADS_ID = "respan.threads.thread_identifier"

    # Trace
    RESPAN_TRACE_GROUP_ID = "respan.trace.trace_group_identifier"

    # Metadata
    RESPAN_METADATA = "respan.metadata"

    # Prompt & environment
    RESPAN_PROMPT = "respan.prompt"
    RESPAN_ENVIRONMENT = "respan.environment"

    # Span links
    RESPAN_LINK_TIMESTAMP = "respan.link.timestamp"

    # Logging
    LOG_METHOD = "respan.entity.log_method"
    LOG_TYPE = "respan.entity.log_type"
    LOG_ID = "respan.entity.log_id"
    LOG_PARENT_ID = "respan.entity.log_parent_id"
    LOG_ROOT_ID = "respan.entity.log_root_id"
    LOG_SOURCE = "respan.entity.log_source"


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
