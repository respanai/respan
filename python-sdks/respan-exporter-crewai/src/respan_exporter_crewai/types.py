"""Type definitions for Respan CrewAI exporter."""
from datetime import datetime
from typing import Any, Dict, Optional, Union

from respan_sdk.respan_types.base_types import RespanBaseModel


class TraceContext(RespanBaseModel):
    """Context information for a trace, extracted from trace object and root span."""

    trace_id: str
    trace_name: Optional[str]
    workflow_name: Optional[str]
    metadata: Dict[str, Any]
    session_identifier: Optional[Union[str, int]] = None
    trace_group_identifier: Optional[Union[str, int]] = None
    start_time: Optional[datetime] = None
    customer_identifier: Optional[Union[str, int]] = None
