"""Tracing payload transformation helper utilities."""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from respan_sdk.constants.llm_logging import LOG_TYPE_CHAT, LOG_TYPE_TASK, LOG_TYPE_WORKFLOW
from respan_sdk.respan_types.log_types import RespanFullLogParams

from respan_exporter_haystack.utils.serialization_utils import serialize_data

logger = logging.getLogger(__name__)


def format_span_for_api(
    span_data: Dict[str, Any],
    workflow_name: str,
    workflow_metadata: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Format traced span data for Respan trace ingest payloads."""
    operation_name = span_data["operation_name"]
    tags = span_data.get("tags", {})
    data = span_data.get("data", {})

    component_name = tags.get("haystack.component.name", "") or ""
    component_type = tags.get("haystack.component.type", "") or ""
    component_type_lower = component_type.lower()
    component_name_lower = component_name.lower()

    # Skip internal tracer component from user-visible traces.
    if "respanconnector" in component_type_lower:
        return None

    start_time = datetime.fromtimestamp(
        timestamp=span_data["start_time"], tz=timezone.utc
    ).isoformat()
    end_time = datetime.fromtimestamp(timestamp=span_data["end_time"], tz=timezone.utc).isoformat()

    if component_name:
        span_name = component_name
    elif operation_name == "haystack.pipeline.run":
        span_name = workflow_name
    else:
        span_name = operation_name

    if "generator" in component_type_lower or "llm" in component_name_lower:
        log_type = LOG_TYPE_CHAT
    elif "builder" in component_type_lower or "prompt" in component_name_lower:
        log_type = LOG_TYPE_TASK
    elif operation_name == "haystack.pipeline.run":
        log_type = LOG_TYPE_WORKFLOW
    else:
        log_type = LOG_TYPE_TASK

    metadata = {**workflow_metadata}
    if component_name:
        metadata["component_name"] = component_name
    if component_type:
        metadata["component_type"] = component_type

    payload = {
        "trace_unique_id": span_data["trace_id"],
        "span_unique_id": span_data["span_id"],
        "span_parent_id": span_data.get("parent_id"),
        "span_name": span_name,
        "span_workflow_name": workflow_name,
        "log_type": log_type,
        "start_time": start_time,
        "timestamp": end_time,
        "latency": span_data.get("latency", 0),
        "metadata": metadata,
        "disable_log": False,
    }

    input_data = data.get("haystack.component.input")
    if input_data is None:
        input_data = tags.get("haystack.pipeline.input_data")
    if input_data is not None:
        payload["input"] = serialize_data(data=input_data)

    output_data = data.get("haystack.component.output")
    if output_data is None:
        output_data = tags.get("haystack.pipeline.output_data")

    if output_data is not None:
        if operation_name == "haystack.pipeline.run" and isinstance(output_data, dict):
            for key in ["llm", "generator", "chat"]:
                if key in output_data and isinstance(output_data[key], dict):
                    component_output = output_data[key]
                    if "replies" in component_output:
                        replies = component_output["replies"]
                        if replies and len(replies) > 0:
                            first_reply = replies[0]
                            if hasattr(first_reply, "text"):
                                payload["output"] = first_reply.text
                            elif hasattr(first_reply, "content"):
                                payload["output"] = first_reply.content
                            elif isinstance(first_reply, str):
                                payload["output"] = first_reply
                            break

            if "output" not in payload:
                cleaned_output = {key: value for key, value in output_data.items() if key != "tracer"}
                payload["output"] = serialize_data(data=cleaned_output)

        elif isinstance(output_data, dict):
            if "replies" in output_data:
                replies = output_data["replies"]
                if replies and len(replies) > 0:
                    first_reply = replies[0]
                    if hasattr(first_reply, "text"):
                        payload["output"] = first_reply.text
                        payload["log_type"] = LOG_TYPE_CHAT
                    elif hasattr(first_reply, "content"):
                        payload["output"] = first_reply.content
                        payload["log_type"] = LOG_TYPE_CHAT
                    elif isinstance(first_reply, str):
                        payload["output"] = first_reply
                    else:
                        payload["output"] = str(first_reply)

            if "meta" in output_data:
                meta = output_data["meta"]
                if isinstance(meta, list) and len(meta) > 0:
                    first_meta = meta[0]
                    model_name = first_meta.get("model", "")
                    if model_name:
                        payload["model"] = model_name
                    cost = first_meta.get("cost")
                    if cost is not None:
                        payload["cost"] = cost
                    if "usage" in first_meta:
                        usage = first_meta["usage"]
                        prompt_tokens = usage.get("prompt_tokens")
                        completion_tokens = usage.get("completion_tokens")
                        total_tokens = usage.get("total_tokens")
                        if prompt_tokens is not None:
                            payload["prompt_tokens"] = prompt_tokens
                        if completion_tokens is not None:
                            payload["completion_tokens"] = completion_tokens
                        if total_tokens is not None:
                            payload["total_request_tokens"] = total_tokens

            if "output" not in payload:
                payload["output"] = serialize_data(data=output_data)
        else:
            payload["output"] = serialize_data(data=output_data)

    if "error" in span_data:
        payload["warnings"] = span_data["error"]

    payload["status_code"] = span_data.get("status_code", 200)

    try:
        validated = RespanFullLogParams.model_validate(obj=payload).model_dump(
            mode="json", exclude_none=True
        )
        return validated
    except Exception as e:
        logger.warning(
            "Haystack span payload failed RespanFullLogParams validation: %s", e
        )
        return None
