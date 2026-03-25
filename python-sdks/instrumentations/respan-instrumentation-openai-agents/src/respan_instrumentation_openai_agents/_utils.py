"""Shared utility functions for span data serialization and formatting."""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from respan_sdk.utils.serialization import serialize_value


def _responses_api_item_to_message(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Convert a single Responses API input/output item to a chat message dict."""
    item_type = item.get("type", "")

    if item_type == "message":
        role = item.get("role", "user")
        content_blocks = item.get("content", [])
        if isinstance(content_blocks, str):
            return {"role": role, "content": content_blocks}
        text_parts = []
        for block in content_blocks:
            if isinstance(block, dict):
                bt = block.get("type", "")
                if bt in ("input_text", "output_text", "text"):
                    text_parts.append(block.get("text", ""))
                elif bt == "input_image":
                    text_parts.append("[image]")
                elif bt == "input_file":
                    text_parts.append("[file]")
                else:
                    text_parts.append(block.get("text", str(block)))
            elif isinstance(block, str):
                text_parts.append(block)
        return {"role": role, "content": "\n".join(text_parts)}

    if item_type == "function_call":
        return {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": item.get("call_id", ""),
                "type": "function",
                "function": {
                    "name": item.get("name", ""),
                    "arguments": item.get("arguments", ""),
                },
            }],
        }

    if item_type == "function_call_output":
        return {
            "role": "tool",
            "content": item.get("output", ""),
            "tool_call_id": item.get("call_id", ""),
        }

    return None


def _format_input_messages(raw_input: Any) -> List[Dict[str, Any]]:
    """Wrap raw input into proper ``[{"role": ..., "content": ...}]`` format."""
    serialized = serialize_value(raw_input)
    if serialized is None:
        return []
    if isinstance(serialized, list):
        has_responses_api_items = any(
            isinstance(item, dict) and "type" in item
            for item in serialized
        )
        if has_responses_api_items:
            messages = []
            for item in serialized:
                if not isinstance(item, dict):
                    continue
                if "type" in item:
                    msg = _responses_api_item_to_message(item)
                    if msg is not None:
                        messages.append(msg)
                elif "role" in item:
                    messages.append(item)
            return messages if messages else serialized
        if serialized and isinstance(serialized[0], dict) and "role" in serialized[0]:
            return serialized
        return serialized
    if isinstance(serialized, str):
        return [{"role": "user", "content": serialized}]
    if isinstance(serialized, dict):
        return [{"role": "user", "content": json.dumps(serialized, default=str)}]
    return [{"role": "user", "content": str(serialized)}]


def _format_output(resp_output: Any) -> List[Dict[str, Any]]:
    """Extract response output as a normalized message list."""
    serialized = serialize_value(resp_output)
    if not serialized:
        return [{"role": "assistant", "content": "", "_is_placeholder": True}]

    if isinstance(serialized, str):
        return [{"role": "assistant", "content": serialized}]

    if isinstance(serialized, dict):
        if "role" in serialized:
            return [serialized]
        return [{"role": "assistant", "content": json.dumps(serialized, default=str)}]

    if isinstance(serialized, list):
        messages = []
        for item in serialized:
            if not isinstance(item, dict):
                messages.append({"role": "assistant", "content": str(item)})
                continue
            if "type" in item:
                msg = _responses_api_item_to_message(item)
                if msg is not None:
                    messages.append(msg)
                    continue
                if item.get("type") == "output_text":
                    messages.append({
                        "role": "assistant",
                        "content": item.get("text", ""),
                    })
                    continue
            if "role" in item:
                messages.append(item)
                continue
            messages.append({
                "role": "assistant",
                "content": json.dumps(item, default=str),
            })
        return messages or [{"role": "assistant", "content": "", "_is_placeholder": True}]

    return [{"role": "assistant", "content": str(serialized)}]


def _parse_ts(ts: str) -> datetime:
    """Parse an ISO-8601 timestamp string to datetime."""
    ts = ts.replace("Z", "+00:00")
    return datetime.fromisoformat(ts)
