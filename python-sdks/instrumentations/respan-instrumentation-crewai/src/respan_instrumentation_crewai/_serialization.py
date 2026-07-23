"""Serialization and normalization helpers for CrewAI event payloads."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from typing import Any

from opentelemetry.semconv_ai import SpanAttributes

from respan_sdk.utils.serialization import serialize_value

from respan_instrumentation_crewai._constants import ASSISTANT_ROLE, USER_ROLE


def json_attribute(value: Any) -> str:
    """Return an OTel-safe JSON string for a structured value."""
    try:
        serialized = serialize_value(value=value)
        return json.dumps(
            serialized,
            default=str,
            ensure_ascii=False,
            separators=(",", ":"),
        )
    except Exception:
        return json.dumps(str(value), ensure_ascii=False, separators=(",", ":"))


def attribute_text(value: Any) -> str:
    """Preserve text and JSON-encode every structured value."""
    if isinstance(value, str):
        return value
    return json_attribute(value)


def normalize_messages(
    value: Any, *, default_role: str = USER_ROLE
) -> list[dict[str, Any]]:
    """Normalize CrewAI's provider-neutral message shapes."""
    if value is None:
        return []
    if isinstance(value, str):
        return [{"role": default_role, "content": value}]
    if isinstance(value, Mapping):
        return [dict(value)]
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        messages: list[dict[str, Any]] = []
        for item in value:
            if isinstance(item, Mapping):
                messages.append(dict(item))
            elif isinstance(item, str):
                messages.append({"role": default_role, "content": item})
        return messages
    return [{"role": default_role, "content": attribute_text(value)}]


def _mapping_view(value: Any) -> dict[str, Any] | None:
    """Return a mapping for dictionaries and common Pydantic/object responses."""
    if isinstance(value, Mapping):
        return dict(value)

    for method_name in ("model_dump", "dict"):
        method = getattr(value, method_name, None)
        if not callable(method):
            continue
        try:
            dumped = method()
        except Exception:
            continue
        if isinstance(dumped, Mapping):
            return dict(dumped)

    fields: dict[str, Any] = {}
    for field_name in ("role", "content", "tool_calls", "message", "text", "choices"):
        try:
            field_value = getattr(value, field_name)
        except Exception:
            continue
        if field_value is not None:
            fields[field_name] = field_value
    return fields or None


def _assistant_message_from_mapping(
    response_mapping: Mapping[str, Any],
) -> dict[str, Any]:
    message = dict(response_mapping)
    if any(key in message for key in ("role", "content", "tool_calls")):
        message.setdefault("role", ASSISTANT_ROLE)
        return message
    return {"role": ASSISTANT_ROLE, "content": message}


def _tool_call_completion_message(
    response: Any,
    response_mapping: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if isinstance(response, Sequence) and not isinstance(
        response, (str, bytes, bytearray)
    ):
        return {"role": ASSISTANT_ROLE, "tool_calls": list(response)}
    if response_mapping is None:
        return None

    nested_tool_calls = response_mapping.get("tool_calls")
    if isinstance(nested_tool_calls, Sequence) and not isinstance(
        nested_tool_calls, (str, bytes, bytearray)
    ):
        return {"role": ASSISTANT_ROLE, "tool_calls": list(nested_tool_calls)}

    tool_call_keys = {"id", "function", "name", "arguments", "input", "tool_use_id"}
    if tool_call_keys.intersection(response_mapping):
        return {"role": ASSISTANT_ROLE, "tool_calls": [dict(response_mapping)]}
    return None


def completion_message(
    response: Any,
    *,
    tool_call_response: bool = False,
) -> dict[str, Any]:
    """Extract one assistant message from common provider response shapes."""
    if isinstance(response, str):
        return {"role": ASSISTANT_ROLE, "content": response}

    response_mapping = _mapping_view(response)
    if tool_call_response:
        tool_call_message = _tool_call_completion_message(response, response_mapping)
        if tool_call_message is not None:
            return tool_call_message

    if response_mapping is not None:
        direct_message = _mapping_view(response_mapping.get("message"))
        if direct_message is not None:
            return _assistant_message_from_mapping(direct_message)

        choices = response_mapping.get("choices")
        if isinstance(choices, Sequence) and not isinstance(
            choices, (str, bytes, bytearray)
        ):
            for choice in choices:
                choice_mapping = _mapping_view(choice)
                if choice_mapping is None:
                    continue
                message = _mapping_view(choice_mapping.get("message"))
                if message is not None:
                    return _assistant_message_from_mapping(message)
                if choice_mapping.get("text") is not None:
                    return {
                        "role": ASSISTANT_ROLE,
                        "content": choice_mapping.get("text"),
                    }

        return _assistant_message_from_mapping(response_mapping)

    return {"role": ASSISTANT_ROLE, "content": response}


def set_message_attributes(
    attributes: dict[str, Any],
    *,
    prefix: str,
    messages: list[dict[str, Any]],
) -> None:
    """Write canonical indexed prompt/completion attributes."""
    for index, message in enumerate(messages):
        message_prefix = f"{prefix}.{index}"
        role = message.get("role")
        content = message.get("content")
        tool_calls = message.get("tool_calls")
        if role is not None:
            attributes[f"{message_prefix}.role"] = str(role)
        if content is not None:
            attributes[f"{message_prefix}.content"] = attribute_text(content)
        if tool_calls:
            attributes[f"{message_prefix}.tool_calls"] = json_attribute(tool_calls)


def first_int(mapping: Mapping[str, Any], *keys: str) -> int | None:
    """Return the first real integer-like usage value."""
    for key in keys:
        value = mapping.get(key)
        if value is None or isinstance(value, bool):
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def normalize_token_usage(usage: Mapping[str, Any] | None) -> dict[str, int]:
    """Normalize OpenAI, Anthropic, and Gemini-style token usage dictionaries."""
    if not usage:
        return {}

    normalized_source = dict(usage)
    for nested_key in ("usage", "usage_metadata", "token_usage"):
        nested = normalized_source.get(nested_key)
        if isinstance(nested, Mapping):
            normalized_source.update(nested)

    prompt_tokens = first_int(
        normalized_source,
        "prompt_tokens",
        "prompt_token_count",
        "input_tokens",
        "input_token_count",
        "inputTokens",
        "inputTokenCount",
    )
    completion_tokens = first_int(
        normalized_source,
        "completion_tokens",
        "candidates_token_count",
        "output_tokens",
        "output_token_count",
        "outputTokens",
        "outputTokenCount",
    )
    total_tokens = first_int(
        normalized_source,
        "total_tokens",
        "total_token_count",
        "totalTokens",
        "totalTokenCount",
    )
    cached_tokens = first_int(
        normalized_source,
        "cached_tokens",
        "cached_prompt_tokens",
        "cache_read_input_tokens",
        "cache_read_tokens",
        "cacheReadInputTokenCount",
        "cacheReadInputTokens",
    )
    reasoning_tokens = first_int(
        normalized_source,
        "reasoning_tokens",
        "thoughts_token_count",
        "reasoningTokens",
    )
    cache_creation_tokens = first_int(
        normalized_source,
        "cache_creation_tokens",
        "cache_creation_input_tokens",
        "cacheWriteInputTokenCount",
        "cacheWriteInputTokens",
    )

    for details_key in ("prompt_tokens_details", "input_tokens_details"):
        details = normalized_source.get(details_key)
        if cached_tokens is None and isinstance(details, Mapping):
            cached_tokens = first_int(details, "cached_tokens", "cache_read")

    if total_tokens is None and (
        prompt_tokens is not None or completion_tokens is not None
    ):
        total_tokens = (prompt_tokens or 0) + (completion_tokens or 0)

    values = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "cached_tokens": cached_tokens,
        "reasoning_tokens": reasoning_tokens,
        "cache_creation_tokens": cache_creation_tokens,
    }
    return {key: value for key, value in values.items() if value is not None}


def normalize_provider(provider: Any, model: Any) -> str | None:
    """Return CrewAI's provider as a canonical lowercase GenAI system value."""
    provider_text = str(provider or "").strip().lower()
    model_text = str(model or "").strip().lower()
    if not provider_text and "/" in model_text:
        provider_text = model_text.partition("/")[0]
    if not provider_text:
        return None

    aliases = {
        "gemini": "google",
        "google_genai": "google",
        "amazon": "bedrock",
        "aws": "bedrock",
    }
    return aliases.get(provider_text, provider_text)


def set_llm_message_attributes(
    attributes: dict[str, Any],
    *,
    messages: Any,
    response: Any | None = None,
    tool_call_response: bool = False,
) -> None:
    """Populate canonical prompt and optional completion attributes."""
    prompt_messages = normalize_messages(messages)
    set_message_attributes(
        attributes,
        prefix=SpanAttributes.LLM_PROMPTS,
        messages=prompt_messages,
    )
    if response is not None:
        set_message_attributes(
            attributes,
            prefix=SpanAttributes.LLM_COMPLETIONS,
            messages=[
                completion_message(response, tool_call_response=tool_call_response)
            ],
        )
