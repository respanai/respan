"""Type definitions for Respan LangChain exporter."""
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from pydantic import Field
from respan_sdk.respan_types.base_types import RespanBaseModel


class TraceContext(RespanBaseModel):
    """Context information for a trace, extracted from root chain span."""

    trace_id: str
    trace_name: Optional[str]
    workflow_name: Optional[str]
    metadata: Dict[str, Any]
    session_identifier: Optional[Union[str, int]] = None
    trace_group_identifier: Optional[Union[str, int]] = None
    start_time: Optional[datetime] = None
    customer_identifier: Optional[Union[str, int]] = None


class SpanRecord(RespanBaseModel):
    """Internal record for a LangChain callback span."""

    run_id: str
    parent_run_id: Optional[str] = None
    span_type: str  # workflow, agent, task, tool, generation
    name: Optional[str] = None
    input: Any = None
    output: Any = None
    error: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    model: Optional[str] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    prompt_messages: Any = None
    completion_message: Any = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)
