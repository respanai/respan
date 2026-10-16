"""Convert Stagehand objects into the JSON shapes Respan renders."""

from __future__ import annotations

import base64
import json
from collections.abc import Mapping
from typing import Any

from respan_tracing.utils.agent_instrumentation import (  # noqa: F401 - re-exported
    compact_image_data_url,
    json_string,
    jsonable,
)

from respan_instrumentation_stagehand._constants import (
    IMAGE_PLACEHOLDER,
    MAX_EXPORTED_IMAGE_BASE64,
)

_MAX_DEPTH = 32
_JPEG_MAGIC = b"\xff\xd8\xff"
Block = dict[str, Any]


def image_part(data: str, mime: str, *, keep: bool) -> Block:
    """A data-URI image part (Respan's ingest lifts it into media storage), or a
    size placeholder when images are off or too large."""
    if keep and len(data) <= MAX_EXPORTED_IMAGE_BASE64:
        url = compact_image_data_url(f"data:{mime};base64,{data}")
        return {"type": "image_url", "image_url": {"url": url}}
    return {"type": "text", "text": IMAGE_PLACEHOLDER.format(mime=mime, size=len(data))}


def image_bytes_part(image: bytes, *, keep: bool) -> Block:
    mime = "image/jpeg" if image[:3] == _JPEG_MAGIC else "image/png"
    return image_part(base64.b64encode(image).decode(), mime, keep=keep)


def _blocks(content: Any) -> list[Block]:
    if isinstance(content, list):
        return [block for block in content if isinstance(block, dict)]
    return [content] if isinstance(content, dict) else []


def _parts(blocks: list[Block], *, keep_images: bool) -> list[Block]:
    parts: list[Block] = []
    for block in blocks:
        if block.get("type") == "text":
            parts.append({"type": "text", "text": block.get("text", "")})
        elif block.get("type") == "image":
            parts.append(
                image_part(block.get("data", ""), block.get("mime_type", "image"), keep=keep_images)
            )
    return parts


def _flatten(parts: list[Block]) -> str | list[Block]:
    """A plain string when the content is text only, else OpenAI-style parts."""
    if all(part.get("type") == "text" for part in parts):
        return "\n".join(part.get("text", "") for part in parts)
    return parts


def tool_calls(blocks: list[Block]) -> list[Block]:
    return [
        {
            "id": block.get("id"),
            "type": "function",
            "function": {
                "name": block.get("name"),
                "arguments": json.dumps(block.get("input") or {}, ensure_ascii=False),
            },
        }
        for block in blocks
        if block.get("type") == "tool_use"
    ]


def chat_messages(params: Mapping[str, Any], *, keep_images: bool) -> list[Block]:
    """Stagehand's llm.generate messages (MCP-style content blocks) as
    OpenAI-shaped chat messages, the shape Respan's log view renders."""
    messages: list[Block] = []
    if system := params.get("system_prompt"):
        messages.append({"role": "system", "content": system})
    for message in params.get("messages") or ():
        blocks = _blocks(message.get("content"))
        results = [block for block in blocks if block.get("type") == "tool_result"]
        for result in results:
            structured = result.get("structured_content")
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": result.get("tool_use_id"),
                    "content": json.dumps(structured, ensure_ascii=False)
                    if structured is not None
                    else _flatten(_parts(_blocks(result.get("content")), keep_images=keep_images)),
                }
            )
        rest = [block for block in blocks if block.get("type") != "tool_result"]
        if not rest and results:
            continue
        chat: Block = {
            "role": message.get("role") or "user",
            "content": _flatten(_parts(rest, keep_images=keep_images)),
        }
        if calls := tool_calls(rest):
            chat["tool_calls"] = calls
        messages.append(chat)
    return messages


def completion(result: Mapping[str, Any]) -> Block:
    """The assistant message of an llm.generate result."""
    blocks = _blocks(result.get("content"))
    structured = result.get("structured_content")
    if result.get("output_format") == "json_schema" and structured is not None:
        text = json.dumps(structured, ensure_ascii=False)
    else:
        text = "".join(block.get("text", "") for block in blocks if block.get("type") == "text")
    message: Block = {"role": "assistant", "content": text}
    if calls := tool_calls(blocks):
        message["tool_calls"] = calls
    return message


def functions(tools: Any) -> list[Block]:
    """Stagehand's client tools as OpenAI function definitions."""
    return [
        {
            "name": tool.get("name"),
            "description": tool.get("description"),
            "parameters": tool.get("input_schema"),
        }
        for tool in tools or ()
        if isinstance(tool, dict)
    ]


def content_text(content: Any) -> str:
    """Flatten a message content for the gen_ai.prompt.N.content attribute."""
    if content is None or isinstance(content, str):
        return content or ""
    return json.dumps(content, ensure_ascii=False, default=str)
