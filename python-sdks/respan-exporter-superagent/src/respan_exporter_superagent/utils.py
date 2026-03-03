import logging
from datetime import datetime
from typing import Any, Dict, Optional

from respan_sdk.constants.llm_logging import LOG_TYPE_TOOL
from respan_sdk.respan_types import RespanParams
from respan_sdk.utils.export import send_payloads, validate_payload
from respan_sdk.utils.serialization import safe_json_dumps
from respan_sdk.utils.time import now_utc


logger = logging.getLogger(__name__)


def build_payload(
    *,
    method_name: str,
    start_time: datetime,
    end_time: datetime,
    status: str,
    input_value: Any,
    output_value: Any,
    error_message: Optional[str],
    export_params: Optional[RespanParams],
) -> Dict[str, Any]:
    params = export_params or RespanParams()

    payload: Dict[str, Any] = {
        "span_workflow_name": params.span_workflow_name or "superagent",
        "span_name": params.span_name or f"superagent.{method_name}",
        "log_type": params.log_type or LOG_TYPE_TOOL,
        "start_time": start_time.isoformat(),
        "timestamp": end_time.isoformat(),
        "latency": (end_time - start_time).total_seconds(),
        "status": status,
    }

    if input_value is not None:
        payload["input"] = safe_json_dumps(input_value) if not isinstance(input_value, str) else input_value
    if output_value is not None:
        payload["output"] = safe_json_dumps(output_value) if not isinstance(output_value, str) else output_value
    if error_message:
        payload["error_message"] = error_message

    if params.trace_unique_id:
        payload["trace_unique_id"] = params.trace_unique_id
        payload["trace_name"] = params.trace_name or payload["span_workflow_name"]

    if params.span_unique_id:
        payload["span_unique_id"] = params.span_unique_id
    if params.span_parent_id:
        payload["span_parent_id"] = params.span_parent_id

    if params.session_identifier:
        payload["session_identifier"] = params.session_identifier

    if params.customer_identifier:
        payload["customer_identifier"] = params.customer_identifier

    metadata: Dict[str, Any] = {}
    if params.metadata:
        metadata.update(params.metadata)
    metadata["integration"] = "superagent"
    metadata["method"] = method_name

    if metadata:
        payload["metadata"] = metadata

    return payload

