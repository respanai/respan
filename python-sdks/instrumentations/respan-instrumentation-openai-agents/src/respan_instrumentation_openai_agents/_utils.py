"""Shared utility functions for span data serialization and formatting."""

from datetime import datetime
from typing import Any

from respan_instrumentation_openai_agents._serialization import json_string, json_value


def _responses_api_item_to_message(item: dict[str, Any]) -> dict[str, Any] | None:
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
                    text_parts.append(block.get("text", json_string(block)))
            elif isinstance(block, str):
                text_parts.append(block)
        return {"role": role, "content": "\n".join(text_parts)}

    if item_type.endswith("_call"):
        return {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": item.get("call_id") or item.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": item.get("name", item_type),
                        "arguments": item.get(
                            "arguments",
                            json_string(item.get("action", item.get("operation", {}))),
                        ),
                    },
                }
            ],
        }

    if item_type.endswith("_call_output"):
        return {
            "role": "tool",
            "content": item.get("output", ""),
            "tool_call_id": item.get("call_id") or item.get("id", ""),
        }

    return None


def _format_input_messages(raw_input: Any) -> list[dict[str, Any]]:
    """Wrap raw input into proper ``[{"role": ..., "content": ...}]`` format."""
    serialized = json_value(raw_input)
    if serialized is None:
        return []
    if isinstance(serialized, list):
        has_responses_api_items = any(
            isinstance(item, dict) and "type" in item for item in serialized
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
        return [{"role": "user", "content": json_string(serialized)}]
    return [{"role": "user", "content": json_string(serialized)}]


def _format_output(resp_output: Any) -> str:
    """Extract the text content from response output items.

    Returns the plain text string from message output items.
    Function-call / tool items are intentionally skipped here because
    they are extracted separately via ``_extract_tool_calls`` and stored
    as their own span attribute.
    """
    serialized = json_value(resp_output)
    if not serialized:
        return ""

    if isinstance(serialized, str):
        return serialized

    if isinstance(serialized, dict):
        if (
            str(serialized.get("type", "")).endswith(("_call", "_call_output"))
            or serialized.get("type") == "reasoning"
        ):
            return ""
        if serialized.get("tool_calls") and serialized.get("content") is None:
            return ""
        content = serialized.get("content")
        if content is None:
            return json_string(serialized)
        if isinstance(content, str):
            return content
        return json_string(content)

    if isinstance(serialized, list):
        text_parts: list[str] = []
        for item in serialized:
            if not isinstance(item, dict):
                text_parts.append(json_string(item))
                continue
            item_type = item.get("type", "")
            if (
                item_type.endswith(("_call", "_call_output"))
                or item_type == "reasoning"
            ):
                continue
            if item_type in ("output_text", "text", "input_text"):
                text_parts.append(item.get("text", ""))
                continue
            if item_type == "message":
                content_blocks = item.get("content", [])
                if isinstance(content_blocks, str):
                    text_parts.append(content_blocks)
                elif isinstance(content_blocks, list):
                    for block in content_blocks:
                        if isinstance(block, dict):
                            bt = block.get("type", "")
                            if bt in ("output_text", "text", "input_text"):
                                text_parts.append(block.get("text", ""))
                        elif isinstance(block, str):
                            text_parts.append(block)
                continue
            if "content" in item:
                content = item["content"]
                if content is not None:
                    text_parts.append(
                        content if isinstance(content, str) else json_string(content)
                    )
                continue
            text_parts.append(json_string(item))
        return "\n".join(text_parts) if text_parts else ""

    return json_string(serialized)


def _parse_ts(ts: str) -> datetime:
    """Parse an ISO-8601 timestamp string to datetime."""
    ts = ts.replace("Z", "+00:00")
    return datetime.fromisoformat(ts)
