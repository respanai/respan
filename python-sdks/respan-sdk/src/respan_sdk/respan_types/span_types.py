from enum import Enum
from typing import Any, Dict

from pydantic import ConfigDict, Field

from respan_sdk.respan_types.base_types import RespanBaseModel


class SpanLink(RespanBaseModel):
    """Serializable link definition for attaching causal links to new spans.

    A lightweight data holder with no OTel dependency.  The conversion to an
    OpenTelemetry ``trace.Link`` is performed by ``respan_tracing`` at runtime.
    """

    model_config = ConfigDict(frozen=True)

    trace_id: str
    span_id: str
    attributes: Dict[str, Any] = Field(default_factory=dict)
    is_remote: bool = True
    is_sampled: bool = True


class RespanSpanAttributes(Enum):
    # Span attributes
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
    RESPAN_METADATA = "respan.metadata" # This is a pattern, it can be  any "respan.metadata.key" where key is customizable

    # Logging
    LOG_METHOD = "respan.entity.log_method"
    LOG_TYPE = "respan.entity.log_type"
    LOG_ID = "respan.entity.log_id"
    LOG_PARENT_ID = "respan.entity.log_parent_id"
    LOG_ROOT_ID = "respan.entity.log_root_id"
    LOG_SOURCE = "respan.entity.log_source"

RESPAN_SPAN_ATTRIBUTES_MAP = {
    "customer_identifier": RespanSpanAttributes.RESPAN_CUSTOMER_PARAMS_ID.value,
    "customer_email": RespanSpanAttributes.RESPAN_CUSTOMER_PARAMS_EMAIL.value,
    "customer_name": RespanSpanAttributes.RESPAN_CUSTOMER_PARAMS_NAME.value,
    "evaluation_identifier": RespanSpanAttributes.RESPAN_EVALUATION_PARAMS_ID.value,
    "thread_identifier": RespanSpanAttributes.RESPAN_THREADS_ID.value,
    "custom_identifier": RespanSpanAttributes.RESPAN_SPAN_CUSTOM_ID.value,
    "trace_group_identifier": RespanSpanAttributes.RESPAN_TRACE_GROUP_ID.value,
    "metadata": RespanSpanAttributes.RESPAN_METADATA.value,
}