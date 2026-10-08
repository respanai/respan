"""Canonical translation of AgentScope SDK objects."""

from __future__ import annotations

import inspect
import json
import logging
from collections.abc import Mapping
from typing import Any

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import LLMRequestTypeValues, SpanAttributes
from respan_sdk.constants.llm_logging import (
    LOG_TYPE_AGENT,
    LOG_TYPE_CHAT,
    LOG_TYPE_TOOL,
    LogMethodChoices,
)
from respan_sdk.constants.span_attributes import (
    RESPAN_LOG_METHOD,
    RESPAN_LOG_TYPE,
    RESPAN_METADATA,
)
from respan_sdk.utils.serialization import serialize_value

logger = logging.getLogger(__name__)

NAME_KEY = "name"
ROLE_KEY = "role"
CONTENT_KEY = "content"
ID_KEY = "id"
INPUT_KEY = "input"
STATE_KEY = "state"
TYPE_KEY = "type"
FUNCTION_KEY = "function"
ARGUMENTS_KEY = "arguments"
TOOL_CALLS_KEY = "tool_calls"
ASSISTANT_ROLE = "assistant"
USER_ROLE = "user"
FUNCTION_TYPE = "function"
AGENTSCOPE_PROMPT_PREFIX = f"{SpanAttributes.LLM_PROMPTS}."
AGENTSCOPE_COMPLETION_PREFIX = f"{SpanAttributes.LLM_COMPLETIONS}."
AGENTSCOPE_AGENT_NAME_ATTR = "agentscope.agent.name"
AGENTSCOPE_SESSION_ID_ATTR = "agentscope.session.id"
AGENTSCOPE_REPLY_ID_ATTR = "agentscope.reply.id"
AGENTSCOPE_MODEL_NAME_ATTR = "agentscope.model.name"
AGENTSCOPE_TOOL_NAME_ATTR = "agentscope.tool.name"
AGENTSCOPE_TOOL_STATUS_ATTR = "agentscope.tool.status"


def _first_value(value, *keys):
    for key in keys:
        result = _object_value(value, key)
        if result is not None:
            return result
    return None


def _object_value(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    try:
        return getattr(value, key, default)
    except Exception:
        logger.debug("AgentScope value unavailable", exc_info=True)
        return default


def _object_method(value: Any, key: str) -> Any | None:
    try:
        method = getattr(value, key, None)
    except Exception:
        logger.debug("AgentScope value unavailable", exc_info=True)
        return None
    if callable(method):
        return method
    return None


def _object_has_attr(value: Any, key: str) -> bool:
    try:
        getattr(value, key)
    except Exception:
        logger.debug("AgentScope value unavailable", exc_info=True)
        return False
    return True


def _object_to_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)

    model_dump = _object_method(value=value, key="model_dump")
    if model_dump is not None:
        try:
            converted = model_dump(mode="json")
            if isinstance(converted, Mapping):
                return dict(converted)
        except Exception:
            logger.debug("AgentScope value conversion failed", exc_info=True)

    to_dict = _object_method(value=value, key="to_dict")
    if to_dict is not None:
        try:
            converted = to_dict()
            if isinstance(converted, Mapping):
                return dict(converted)
        except Exception:
            logger.debug("AgentScope value conversion failed", exc_info=True)

    value_dict = _object_value(value=value, key="__dict__")
    if isinstance(value_dict, Mapping):
        return {
            key: item
            for key, item in value_dict.items()
            if not str(key).startswith("_")
        }

    return {"value": value}


def _json_string(value: Any) -> str:
    try:
        return json.dumps(
            serialize_value(value=value),
            default=str,
            separators=(",", ":"),
        )
    except Exception:
        logger.debug("AgentScope value unavailable", exc_info=True)
        return "[unserializable]"


def _attribute_string(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return _json_string(value=value)


def _set_if_present(attributes: dict[str, Any], key: str, value: Any) -> None:
    if value is not None:
        attributes[key] = value


def _coerce_int(value: Any) -> int | None:
    return (
        value
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0
        else None
    )


def _normalize_provider(value: Any) -> str | None:
    if value is None:
        return None
    provider = str(value).strip().lower()
    if not provider:
        return None
    return provider.replace(" ", "_")


def _is_async_iterator(value: Any) -> bool:
    if inspect.isawaitable(value):
        return False
    return _object_has_attr(value=value, key="__aiter__")


def _block_payload(block: Any) -> Any:
    if isinstance(block, str):
        return block

    block_type = _object_value(value=block, key=TYPE_KEY)
    if block_type == "text" or _object_value(value=block, key="text") is not None:
        return _object_value(value=block, key="text")
    if (
        block_type == "thinking"
        or _object_value(value=block, key="thinking") is not None
    ):
        return _object_value(value=block, key="thinking")
    if (
        block_type == "tool_call"
        or _object_value(value=block, key=INPUT_KEY) is not None
    ):
        return {
            ID_KEY: _object_value(value=block, key=ID_KEY),
            NAME_KEY: _object_value(value=block, key=NAME_KEY),
            INPUT_KEY: _object_value(value=block, key=INPUT_KEY),
            STATE_KEY: str(_object_value(value=block, key=STATE_KEY, default="")),
        }
    if (
        block_type == "tool_result"
        or _object_value(value=block, key="output") is not None
    ):
        return {
            ID_KEY: _object_value(value=block, key=ID_KEY),
            NAME_KEY: _object_value(value=block, key=NAME_KEY),
            "output": _object_value(value=block, key="output"),
            STATE_KEY: str(_object_value(value=block, key=STATE_KEY, default="")),
        }
    return _object_to_dict(value=block)


def _content_blocks(value: Any) -> list[Any]:
    if value is None:
        return []

    get_content_blocks = _object_method(value=value, key="get_content_blocks")
    if get_content_blocks is not None:
        try:
            blocks = get_content_blocks()
            if isinstance(blocks, list):
                return blocks
        except Exception:
            logger.debug("AgentScope value conversion failed", exc_info=True)

    content = _object_value(value=value, key=CONTENT_KEY)
    if isinstance(content, list):
        return content
    if content is None:
        return []
    return [content]


def _content_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(_object_value(value, CONTENT_KEY), Mapping):
        return _object_value(value, CONTENT_KEY)
    blocks = _content_blocks(value=value)
    if not blocks:
        return _object_value(value=value, key=CONTENT_KEY, default=value)

    payloads = [_block_payload(block=block) for block in blocks]
    text_parts = [item for item in payloads if isinstance(item, str)]
    if text_parts and len(text_parts) == len(payloads):
        return "".join(text_parts)
    return payloads


def _normalize_message(message: Any) -> dict[str, Any]:
    role = _object_value(value=message, key=ROLE_KEY)
    name = _object_value(value=message, key=NAME_KEY)
    content = _content_value(value=message)
    normalized: dict[str, Any] = {
        ROLE_KEY: str(role or USER_ROLE),
        CONTENT_KEY: content,
    }
    if name:
        normalized[NAME_KEY] = str(name)

    tool_calls = _tool_calls_from_blocks(_content_blocks(value=message))
    if tool_calls:
        normalized[TOOL_CALLS_KEY] = tool_calls
    return normalized


def _normalize_messages(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        value = [value]
    return [_normalize_message(message=item) for item in value]


def _set_message_attributes(
    attributes: dict[str, Any],
    prefix: str,
    messages: list[dict[str, Any]],
) -> None:
    for message_index, message in enumerate(messages):
        role = message.get(ROLE_KEY)
        content = message.get(CONTENT_KEY)
        tool_calls = message.get(TOOL_CALLS_KEY)

        if role is not None:
            attributes[f"{prefix}{message_index}.role"] = str(role)
        if content is not None:
            attributes[f"{prefix}{message_index}.content"] = _attribute_string(
                value=content
            )
        if tool_calls:
            attributes[f"{prefix}{message_index}.{TOOL_CALLS_KEY}"] = _json_string(
                value=tool_calls
            )


def _tool_calls_from_blocks(blocks: list[Any]) -> list[dict[str, Any]]:
    tool_calls: list[dict[str, Any]] = []
    for block in blocks:
        block_type = _object_value(value=block, key=TYPE_KEY)
        tool_name = _object_value(value=block, key=NAME_KEY)
        tool_input = _object_value(value=block, key=INPUT_KEY)
        if block_type != "tool_call" and tool_input is None:
            continue
        if not tool_name:
            continue
        tool_calls.append(
            {
                **(
                    {ID_KEY: str(_object_value(block, ID_KEY))}
                    if _object_value(block, ID_KEY) is not None
                    else {}
                ),
                TYPE_KEY: FUNCTION_TYPE,
                FUNCTION_KEY: {
                    NAME_KEY: str(tool_name),
                    ARGUMENTS_KEY: tool_input
                    if isinstance(tool_input, str)
                    else _json_string(tool_input or {}),
                },
            }
        )
    return tool_calls


def _extract_usage(
    response: Any,
) -> tuple[int | None, int | None, int | None, int | None]:
    usage = _object_value(value=response, key="usage")
    input_tokens = _coerce_int(_first_value(usage, "input_tokens", "prompt_tokens"))
    output_tokens = _coerce_int(
        _first_value(usage, "output_tokens", "completion_tokens")
    )
    total_tokens = _coerce_int(_object_value(value=usage, key="total_tokens"))
    cache_read_tokens = _coerce_int(
        _first_value(usage, "cache_input_tokens", "cache_read_input_tokens")
    )

    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens
    return input_tokens, output_tokens, total_tokens, cache_read_tokens


def _model_name(model: Any) -> str:
    return str(_object_value(value=model, key="model") or type(model).__name__)


def _model_provider(model: Any) -> str | None:
    explicit = _object_value(model, "provider") or _object_value(model, "model_type")
    if explicit:
        return _normalize_provider(explicit)
    name = type(model).__name__
    for suffix in ("ChatModel", "ResponseModel", "EmbeddingModel"):
        if name.endswith(suffix):
            return _normalize_provider(name.removesuffix(suffix))
    return None


def _agent_name(agent: Any) -> str:
    return str(_object_value(value=agent, key=NAME_KEY) or type(agent).__name__)


def _agent_metadata(agent: Any) -> dict[str, Any]:
    state = _object_value(value=agent, key="state")
    metadata: dict[str, Any] = {}
    for public_key, source_key in (
        (AGENTSCOPE_SESSION_ID_ATTR, "session_id"),
        (AGENTSCOPE_REPLY_ID_ATTR, "reply_id"),
    ):
        value = _object_value(value=state, key=source_key)
        if value is not None:
            metadata[public_key] = value
    return metadata


def _agent_input_value(value: Any) -> Any:
    if isinstance(value, list) and len(value) == 1:
        value = value[0]
    if isinstance(value, Mapping) and CONTENT_KEY in value:
        return value.get(CONTENT_KEY)
    content = _content_value(value=value)
    if content is not None and content is not value:
        return content
    return value


def _agent_attributes(
    *,
    agent: Any,
    input_value: Any,
    output_value: Any,
) -> dict[str, Any]:
    entity_name = _agent_name(agent=agent)
    attributes: dict[str, Any] = {
        RESPAN_LOG_METHOD: LogMethodChoices.TRACING_INTEGRATION.value,
        RESPAN_LOG_TYPE: LOG_TYPE_AGENT,
        SpanAttributes.TRACELOOP_ENTITY_NAME: entity_name,
        SpanAttributes.TRACELOOP_ENTITY_PATH: "",
        SpanAttributes.TRACELOOP_WORKFLOW_NAME: entity_name,
        SpanAttributes.TRACELOOP_ENTITY_INPUT: _attribute_string(
            value=_agent_input_value(value=input_value)
        ),
        SpanAttributes.TRACELOOP_ENTITY_OUTPUT: _attribute_string(
            value=_content_value(value=output_value)
        ),
        AGENTSCOPE_AGENT_NAME_ATTR: entity_name,
    }
    for key, value in _agent_metadata(agent=agent).items():
        attributes[key] = str(value)
    metadata = _object_value(value=output_value, key="metadata")
    if metadata:
        attributes[RESPAN_METADATA] = _json_string(value=metadata)
    return attributes


def _model_attributes(
    *,
    model: Any,
    messages: Any,
    tools: Any,
    response: Any,
) -> dict[str, Any]:
    normalized_messages = _normalize_messages(value=messages)
    blocks = _content_blocks(value=response)
    completion_tool_calls = _tool_calls_from_blocks(blocks)
    visible = [
        block for block in blocks if _object_value(block, TYPE_KEY) != "tool_call"
    ]
    completion_content = _content_value(value=response)
    if completion_tool_calls:
        completion_content = _content_value({CONTENT_KEY: visible}) if visible else ""
    completion_message = {
        ROLE_KEY: ASSISTANT_ROLE,
        CONTENT_KEY: completion_content,
    }
    if completion_tool_calls:
        completion_message[TOOL_CALLS_KEY] = completion_tool_calls

    model_name = _model_name(model=model)
    provider = _model_provider(model)
    input_tokens, output_tokens, total_tokens, cache_read_tokens = _extract_usage(
        response=response
    )

    attributes: dict[str, Any] = {
        RESPAN_LOG_METHOD: LogMethodChoices.TRACING_INTEGRATION.value,
        RESPAN_LOG_TYPE: LOG_TYPE_CHAT,
        SpanAttributes.TRACELOOP_ENTITY_NAME: "model_call",
        SpanAttributes.TRACELOOP_ENTITY_PATH: "model_call",
        SpanAttributes.TRACELOOP_ENTITY_INPUT: _json_string(value=normalized_messages),
        SpanAttributes.TRACELOOP_ENTITY_OUTPUT: _attribute_string(
            value=completion_message.get(CONTENT_KEY)
        ),
        SpanAttributes.LLM_REQUEST_TYPE: LLMRequestTypeValues.CHAT.value,
        SpanAttributes.LLM_REQUEST_MODEL: model_name,
        AGENTSCOPE_MODEL_NAME_ATTR: model_name,
    }
    _set_if_present(
        attributes=attributes, key=SpanAttributes.LLM_SYSTEM, value=provider
    )

    if tools:
        attributes[SpanAttributes.LLM_REQUEST_FUNCTIONS] = _json_string(value=tools)

    _set_if_present(
        attributes=attributes,
        key=GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS,
        value=input_tokens,
    )
    _set_if_present(
        attributes=attributes,
        key=GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS,
        value=output_tokens,
    )
    _set_if_present(
        attributes=attributes,
        key=SpanAttributes.LLM_USAGE_PROMPT_TOKENS,
        value=input_tokens,
    )
    _set_if_present(
        attributes=attributes,
        key=SpanAttributes.LLM_USAGE_COMPLETION_TOKENS,
        value=output_tokens,
    )
    _set_if_present(
        attributes=attributes,
        key=SpanAttributes.LLM_USAGE_TOTAL_TOKENS,
        value=total_tokens,
    )
    _set_if_present(
        attributes=attributes,
        key=SpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS,
        value=cache_read_tokens,
    )

    _set_message_attributes(
        attributes=attributes,
        prefix=AGENTSCOPE_PROMPT_PREFIX,
        messages=normalized_messages,
    )
    if response is not None:
        _set_message_attributes(
            attributes=attributes,
            prefix=AGENTSCOPE_COMPLETION_PREFIX,
            messages=[completion_message],
        )
    else:
        attributes.pop(SpanAttributes.TRACELOOP_ENTITY_OUTPUT, None)
    _set_if_present(
        attributes,
        SpanAttributes.LLM_USAGE_CACHE_CREATION_INPUT_TOKENS,
        _coerce_int(
            _object_value(
                _object_value(response, "usage"), "cache_creation_input_tokens"
            )
        ),
    )
    return attributes


def _tool_call_input(tool_call: Any) -> dict[str, Any]:
    return {
        NAME_KEY: _object_value(value=tool_call, key=NAME_KEY),
        ARGUMENTS_KEY: _object_value(value=tool_call, key=INPUT_KEY),
    }


def _tool_output(chunks: list[Any]) -> Any:
    if not chunks:
        return ""
    final = chunks[-1]
    content = _content_value(value=final)
    if content is not None:
        return content
    return _object_to_dict(value=final)


def _tool_attributes(*, tool_call: Any, chunks: list[Any]) -> dict[str, Any]:
    tool_name = str(_object_value(value=tool_call, key=NAME_KEY) or "tool")
    tool_call_id = _object_value(value=tool_call, key=ID_KEY)
    final_state = _object_value(value=chunks[-1], key=STATE_KEY) if chunks else None
    attributes: dict[str, Any] = {
        RESPAN_LOG_METHOD: LogMethodChoices.TRACING_INTEGRATION.value,
        RESPAN_LOG_TYPE: LOG_TYPE_TOOL,
        SpanAttributes.TRACELOOP_ENTITY_NAME: tool_name,
        SpanAttributes.TRACELOOP_ENTITY_PATH: f"tool.{tool_name}",
        SpanAttributes.TRACELOOP_ENTITY_INPUT: _json_string(
            value=_tool_call_input(tool_call=tool_call)
        ),
        SpanAttributes.TRACELOOP_ENTITY_OUTPUT: _attribute_string(
            value=_tool_output(chunks=chunks)
        ),
        AGENTSCOPE_TOOL_NAME_ATTR: tool_name,
    }
    _set_if_present(
        attributes=attributes,
        key=GenAIAttributes.GEN_AI_TOOL_CALL_ID,
        value=tool_call_id,
    )
    _set_if_present(
        attributes=attributes,
        key=AGENTSCOPE_TOOL_STATUS_ATTR,
        value=str(_object_value(value=final_state, key="value", default=final_state)),
    )
    return attributes
