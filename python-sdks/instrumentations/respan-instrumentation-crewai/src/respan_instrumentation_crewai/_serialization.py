"""Serialization and normalization helpers for CrewAI event payloads."""

from __future__ import annotations

import ast
import json
import logging
from collections.abc import Mapping, Sequence
from typing import Any

from opentelemetry.semconv_ai import SpanAttributes

from respan_instrumentation_crewai._constants import ASSISTANT_ROLE, USER_ROLE

logger = logging.getLogger(__name__)


def _structured_value(value: Any, *, parse_tool_strings: bool = False) -> Any:
    """Convert provider/Pydantic containers to JSON-native values."""
    if value is None or isinstance(value, (int, float, bool)):
        return value
    if isinstance(value, str):
        if not parse_tool_strings:
            return value
        parsed = _parse_tool_string(value)
        if parsed is value:
            return value
        return _structured_value(parsed, parse_tool_strings=True)
    if isinstance(value, Mapping):
        return {
            str(key): _structured_value(item, parse_tool_strings=parse_tool_strings)
            for key, item in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [
            _structured_value(item, parse_tool_strings=parse_tool_strings)
            for item in value
        ]

    for method_name in ("model_dump", "dict"):
        method = getattr(value, method_name, None)
        if not callable(method):
            continue
        try:
            dumped = method()
        except Exception:
            logger.debug("Could not serialize CrewAI payload field", exc_info=True)
            continue
        if isinstance(dumped, Mapping):
            return _structured_value(
                dumped,
                parse_tool_strings=parse_tool_strings,
            )

    attributes = getattr(value, "__dict__", None)
    if isinstance(attributes, Mapping):
        return _structured_value(
            {
                key: item
                for key, item in attributes.items()
                if not str(key).startswith("_")
            },
            parse_tool_strings=parse_tool_strings,
        )
    return str(value)


def _parse_tool_string(value: str) -> Any:
    stripped = value.strip()
    if len(stripped) < 2:
        return value
    is_container = (stripped[0], stripped[-1]) in {("{", "}"), ("[", "]")}
    is_quoted = stripped[0] == stripped[-1] and stripped[0] in {"'", '"'}
    if not is_container and not is_quoted:
        return value
    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(stripped)
        except (TypeError, ValueError, SyntaxError, json.JSONDecodeError):
            continue
    return value


def json_attribute(value: Any) -> str:
    """Return an OTel-safe JSON string for a structured value."""
    try:
        return json.dumps(
            _structured_value(value),
            default=str,
            ensure_ascii=False,
            separators=(",", ":"),
        )
    except Exception:
        logger.debug("Could not encode CrewAI payload", exc_info=True)
        return json.dumps(str(value), ensure_ascii=False, separators=(",", ":"))


def attribute_text(value: Any) -> str:
    """Preserve text and JSON-encode every structured value."""
    if isinstance(value, str):
        return value
    return json_attribute(value)


def normalize_tool_definitions(value: Any) -> Any:
    """Return tool schemas without Python repr or nested quote artifacts."""
    return _structured_value(value, parse_tool_strings=True)


def _tool_arguments(value: Any) -> str:
    normalized = _structured_value(value, parse_tool_strings=True)
    if isinstance(normalized, str):
        return normalized
    return json.dumps(
        normalized,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _normalize_tool_call(value: Any) -> Any:
    normalized = _structured_value(value, parse_tool_strings=True)
    if not isinstance(normalized, Mapping):
        return normalized

    call = dict(normalized)
    function = call.get("function")
    if isinstance(function, Mapping):
        normalized_function = dict(function)
        if normalized_function.get("arguments") is not None:
            normalized_function["arguments"] = _tool_arguments(
                normalized_function["arguments"]
            )
        call["function"] = normalized_function
    elif call.get("name") is not None:
        normalized_function = {"name": call.pop("name")}
        arguments = call.pop("arguments", call.pop("input", None))
        if arguments is not None:
            normalized_function["arguments"] = _tool_arguments(arguments)
        call["function"] = normalized_function
        call.setdefault("type", "function")

    if call.get("id") is None and call.get("tool_use_id") is not None:
        call["id"] = call.pop("tool_use_id")
    return call


def normalize_tool_calls(value: Any) -> list[Any]:
    """Return OpenAI-shaped tool calls with JSON-string arguments."""
    normalized = _structured_value(value, parse_tool_strings=True)
    if isinstance(normalized, Sequence) and not isinstance(
        normalized,
        (str, bytes, bytearray),
    ):
        return [_normalize_tool_call(item) for item in normalized]
    return [_normalize_tool_call(normalized)]


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
            logger.debug("Could not serialize CrewAI payload field", exc_info=True)
            continue
        if isinstance(dumped, Mapping):
            return dict(dumped)

    fields: dict[str, Any] = {}
    for field_name in ("role", "content", "tool_calls", "message", "text", "choices"):
        try:
            field_value = getattr(value, field_name)
        except Exception:
            logger.debug("Could not serialize CrewAI payload field", exc_info=True)
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
        return {
            "role": ASSISTANT_ROLE,
            "tool_calls": normalize_tool_calls(response),
        }
    if response_mapping is None:
        return None

    nested_tool_calls = response_mapping.get("tool_calls")
    if isinstance(nested_tool_calls, Sequence) and not isinstance(
        nested_tool_calls, (str, bytes, bytearray)
    ):
        return {
            "role": ASSISTANT_ROLE,
            "tool_calls": normalize_tool_calls(nested_tool_calls),
        }

    tool_call_keys = {"id", "function", "name", "arguments", "input", "tool_use_id"}
    if tool_call_keys.intersection(response_mapping):
        return {
            "role": ASSISTANT_ROLE,
            "tool_calls": normalize_tool_calls(response_mapping),
        }
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
            attributes[f"{message_prefix}.tool_calls"] = json_attribute(
                normalize_tool_calls(tool_calls)
            )
        for identity_field in ("tool_call_id", "name"):
            identity = message.get(identity_field)
            if identity is not None:
                attributes[f"{message_prefix}.{identity_field}"] = str(identity)


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

    for details_key in ("completion_tokens_details", "output_tokens_details"):
        details = normalized_source.get(details_key)
        if reasoning_tokens is None and isinstance(details, Mapping):
            reasoning_tokens = first_int(details, "reasoning_tokens")

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
