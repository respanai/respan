"""Translate Aleph Alpha SDK payloads into canonical span values."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Iterable
from typing import Any

from respan_instrumentation_aleph_alpha._constants import (
    CONTENT_KEY,
    DATA_KEY,
    DIMENSIONS_KEY,
    FINISH_REASON_KEY,
    FUNCTION_KEY,
    INPUT_KEY,
    INSTRUCTION_KEY,
    MESSAGES_KEY,
    NAME_KEY,
    PROMPT_KEY,
    PROMPTS_KEY,
    ROLE_KEY,
    TEXT_TYPE,
    TOOL_CALL_ID_KEY,
    TOOL_CALLS_KEY,
    TOOLS_KEY,
    TYPE_KEY,
)
from respan_sdk.utils.serialization import serialize_value


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _enum_value(value: Any) -> Any:
    enum_value = getattr(value, "value", None)
    return enum_value if enum_value is not None else value


def _sanitize(value: Any) -> Any:
    if isinstance(value, str):
        if value.startswith("data:image/") and len(value) > 128:
            return value[:64] + "...[redacted]"
        return value
    if isinstance(value, dict):
        return {str(key): _sanitize(nested) for key, nested in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(item) for item in value]
    return value


def dump_value(value: Any) -> Any:
    if value is None:
        return None
    value = _enum_value(value)
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {
            str(key): dump_value(nested_value)
            for key, nested_value in value.items()
            if nested_value is not None
        }
    if isinstance(value, (list, tuple, set)):
        return [dump_value(item) for item in value if item is not None]

    to_json = getattr(value, "to_json", None)
    if callable(to_json):
        try:
            return dump_value(to_json())
        except Exception:
            pass

    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return dump_value(model_dump(exclude_none=True, by_alias=False))
        except TypeError:
            return dump_value(model_dump())

    if dataclasses.is_dataclass(value):
        return dump_value(dataclasses.asdict(value))

    return serialize_value(value=value)


def safe_json(value: Any) -> str:
    try:
        return json.dumps(_sanitize(dump_value(value)), default=str)
    except Exception:
        return str(value)


def to_json_attr(value: Any) -> str:
    if isinstance(value, str):
        return value
    return safe_json(value=value)


def request_payload(request: Any) -> dict[str, Any]:
    payload = dump_value(request)
    return payload if isinstance(payload, dict) else {"value": payload}


def prompt_content(prompt: Any) -> Any:
    prompt_value = dump_value(prompt)
    if isinstance(prompt_value, list):
        text_parts: list[str] = []
        for item in prompt_value:
            if not isinstance(item, dict):
                return prompt_value
            if item.get(TYPE_KEY) != TEXT_TYPE or not isinstance(
                item.get(DATA_KEY), str
            ):
                return prompt_value
            text_parts.append(item[DATA_KEY])
        return "\n".join(text_parts)
    return prompt_value


def input_messages_from_completion_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if PROMPT_KEY not in payload:
        return []
    return [{"role": "user", "content": prompt_content(payload.get(PROMPT_KEY))}]


def input_messages_from_chat_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    messages = payload.get(MESSAGES_KEY) or []
    result: list[dict[str, Any]] = []
    if not isinstance(messages, (list, tuple)):
        return result
    for message in messages:
        message_value = dump_value(message)
        if not isinstance(message_value, dict):
            continue
        normalized: dict[str, Any] = {}
        role = message_value.get(ROLE_KEY)
        content = message_value.get(CONTENT_KEY)
        tool_call_id = message_value.get(TOOL_CALL_ID_KEY)
        tool_calls = message_value.get(TOOL_CALLS_KEY)
        if role is not None:
            normalized[ROLE_KEY] = _enum_value(role)
        if content is not None:
            normalized[CONTENT_KEY] = content
        if tool_call_id is not None:
            normalized[TOOL_CALL_ID_KEY] = tool_call_id
        if tool_calls:
            normalized[TOOL_CALLS_KEY] = normalize_tool_calls(tool_calls)
        result.append(normalized)
    return result


def normalize_tool_calls(tool_calls: Any) -> list[dict[str, Any]]:
    tool_call_values = dump_value(tool_calls)
    if not isinstance(tool_call_values, list):
        tool_call_values = [tool_call_values]

    normalized: list[dict[str, Any]] = []
    for tool_call in tool_call_values:
        if not isinstance(tool_call, dict):
            continue
        function = tool_call.get(FUNCTION_KEY) or {}
        if not isinstance(function, dict):
            function = dump_value(function)
        item = {
            key: value
            for key, value in {
                "id": tool_call.get("id"),
                TYPE_KEY: tool_call.get(TYPE_KEY) or "function",
                FUNCTION_KEY: {
                    NAME_KEY: function.get(NAME_KEY),
                    "arguments": to_json_attr(function.get("arguments", "")),
                },
            }.items()
            if value is not None
        }
        normalized.append(item)
    return normalized


def tools_from_payload(payload: dict[str, Any]) -> list[Any]:
    tools = payload.get(TOOLS_KEY)
    if not tools:
        return []
    dumped = dump_value(tools)
    return dumped if isinstance(dumped, list) else [dumped]


def completion_texts(response_or_items: Any) -> list[str]:
    if response_or_items is None:
        return []
    if isinstance(response_or_items, list):
        by_index: dict[int, list[str]] = {}
        for item in response_or_items:
            completion = _field(item, "completion")
            if completion is None:
                continue
            index = _field(item, "index", 0) or 0
            by_index.setdefault(int(index), []).append(str(completion))
        return ["".join(chunks) for _, chunks in sorted(by_index.items())]

    completions = _field(response_or_items, "completions", []) or []
    texts: list[str] = []
    for completion in completions:
        text = _field(completion, "completion")
        if text is None:
            text = _field(completion, "raw_completion")
        if text is not None:
            texts.append(str(text))
    return texts


def chat_output(response_or_items: Any) -> tuple[str, str, list[dict[str, Any]], str | None]:
    if isinstance(response_or_items, list):
        content_parts: list[str] = []
        role = "assistant"
        tool_calls: list[dict[str, Any]] = []
        finish_reason: str | None = None
        for item in response_or_items:
            content = _field(item, CONTENT_KEY)
            if isinstance(content, str):
                content_parts.append(content)
            item_role = _enum_value(_field(item, ROLE_KEY))
            if item_role:
                role = str(item_role)
            if item.__class__.__name__ == "ToolCall" or _field(item, FUNCTION_KEY):
                tool_calls.extend(normalize_tool_calls([item]))
            if item.__class__.__name__ == "FinishReason":
                finish_reason = str(_enum_value(item))
            elif getattr(item, "value", None) in {"stop", "length", "content_filter", "tool_calls"}:
                finish_reason = str(getattr(item, "value"))
        return "".join(content_parts), role, tool_calls, finish_reason

    message = _field(response_or_items, "message")
    role = _enum_value(_field(message, ROLE_KEY, "assistant")) or "assistant"
    content = _field(message, CONTENT_KEY, "") or ""
    tool_calls = normalize_tool_calls(_field(message, TOOL_CALLS_KEY, []) or [])
    finish_reason = _enum_value(_field(response_or_items, FINISH_REASON_KEY))
    return str(content), str(role), tool_calls, str(finish_reason) if finish_reason else None


def usage_from_response(response_or_items: Any) -> dict[str, int]:
    if response_or_items is None:
        return {}

    if isinstance(response_or_items, list):
        result: dict[str, int] = {}
        for item in response_or_items:
            prompt_tokens = _field(item, "prompt_tokens")
            completion_tokens = _field(item, "completion_tokens")
            total_tokens = _field(item, "total_tokens")
            if isinstance(prompt_tokens, int):
                result["prompt_tokens"] = prompt_tokens
            if isinstance(completion_tokens, int):
                result["completion_tokens"] = completion_tokens
            if isinstance(total_tokens, int):
                result["total_tokens"] = total_tokens
        if result:
            return result

        prompt_tokens = None
        completion_tokens = None
        for item in response_or_items:
            if isinstance(_field(item, "num_tokens_prompt_total"), int):
                prompt_tokens = _field(item, "num_tokens_prompt_total")
            if isinstance(_field(item, "num_tokens_generated"), int):
                completion_tokens = _field(item, "num_tokens_generated")
        result = {}
        if isinstance(prompt_tokens, int):
            result["prompt_tokens"] = prompt_tokens
        if isinstance(completion_tokens, int):
            result["completion_tokens"] = completion_tokens
        if result:
            result["total_tokens"] = result.get("prompt_tokens", 0) + result.get(
                "completion_tokens", 0
            )
        return result

    usage = _field(response_or_items, "usage")
    if usage is not None:
        prompt_tokens = _field(usage, "prompt_tokens")
        completion_tokens = _field(usage, "completion_tokens")
        total_tokens = _field(usage, "total_tokens")
    else:
        prompt_tokens = _field(response_or_items, "num_tokens_prompt_total")
        completion_tokens = _field(response_or_items, "num_tokens_generated")
        total_tokens = None

    result: dict[str, int] = {}
    if isinstance(prompt_tokens, int):
        result["prompt_tokens"] = prompt_tokens
    if isinstance(completion_tokens, int):
        result["completion_tokens"] = completion_tokens
    if isinstance(total_tokens, int):
        result["total_tokens"] = total_tokens
    elif result:
        result["total_tokens"] = result.get("prompt_tokens", 0) + result.get(
            "completion_tokens", 0
        )
    return result


def embedding_input(payload: dict[str, Any]) -> Any:
    if PROMPT_KEY in payload:
        return prompt_content(payload.get(PROMPT_KEY))
    if PROMPTS_KEY in payload:
        prompts = payload.get(PROMPTS_KEY) or []
        return [prompt_content(prompt) for prompt in prompts]
    if INPUT_KEY in payload and INSTRUCTION_KEY in payload:
        return {
            INSTRUCTION_KEY: payload.get(INSTRUCTION_KEY),
            INPUT_KEY: prompt_content(payload.get(INPUT_KEY)),
        }
    if INPUT_KEY in payload:
        return payload.get(INPUT_KEY)
    return payload


def _embedding_dimensions(embedding: Any) -> int | None:
    if isinstance(embedding, (list, tuple)):
        return len(embedding)
    return None


def embedding_output_summary(response: Any) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    model = _field(response, "model") or _field(response, "model_version")
    if model:
        summary["model"] = model

    data = _field(response, "data")
    if isinstance(data, list):
        dimensions = _embedding_dimensions(_field(data[0], "embedding")) if data else None
        summary["embedding_count"] = len(data)
        if dimensions is not None:
            summary[DIMENSIONS_KEY] = dimensions
        return summary

    embeddings = _field(response, "embeddings")
    if isinstance(embeddings, dict):
        values = list(embeddings.values())
        dimensions = _embedding_dimensions(values[0]) if values else None
        summary["embedding_count"] = len(values)
        if dimensions is not None:
            summary[DIMENSIONS_KEY] = dimensions
        return summary
    if isinstance(embeddings, list):
        dimensions = _embedding_dimensions(embeddings[0]) if embeddings else None
        summary["embedding_count"] = len(embeddings)
        if dimensions is not None:
            summary[DIMENSIONS_KEY] = dimensions
        return summary

    embedding = _field(response, "embedding")
    dimensions = _embedding_dimensions(embedding)
    if dimensions is not None:
        summary["embedding_count"] = 1
        summary[DIMENSIONS_KEY] = dimensions
    return summary


def first_non_empty(values: Iterable[Any]) -> Any:
    for value in values:
        if value not in (None, "", [], {}):
            return value
    return None
