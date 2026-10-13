"""Translate Dify client requests and responses into Respan span attributes."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import LLMRequestTypeValues, SpanAttributes
from respan_sdk.constants.llm_logging import (
    LOG_TYPE_CHAT,
    LOG_TYPE_TASK,
    LOG_TYPE_TEXT,
    LOG_TYPE_WORKFLOW,
    LogMethodChoices,
)
from respan_sdk.constants.span_attributes import (
    RESPAN_CUSTOMER_PARAMS_ID,
    RESPAN_LOG_METHOD,
    RESPAN_LOG_TYPE,
    RESPAN_METADATA,
    RESPAN_SPAN_ATTRIBUTES_MAP,
    RESPAN_THREADS_ID,
    RESPAN_TRACE_GROUP_ID,
)
from respan_sdk.utils.serialization import serialize_value

from respan_instrumentation_dify._constants import (
    ANSWER_KEY,
    CHAT_MESSAGES_ENDPOINT,
    COMPLETION_MESSAGES_ENDPOINT,
    COMPLETION_TOKENS_KEY,
    CONVERSATION_ID_KEY,
    DATA_KEY,
    DIFY_API_SPAN_NAME,
    DIFY_CHAT_SPAN_NAME,
    DIFY_COMPLETION_SPAN_NAME,
    DIFY_WORKFLOW_SPAN_NAME,
    ENDPOINT_KEY,
    ERROR_KEY,
    EVENT_KEY,
    FILES_KEY,
    ID_KEY,
    INPUTS_KEY,
    LATENCY_KEY,
    MESSAGE_ID_KEY,
    METADATA_KEY,
    METHOD_KEY,
    MODE_KEY,
    OFF_CONTRACT_ALIASES,
    OUTPUTS_KEY,
    PROMPT_TOKENS_KEY,
    QUERY_KEY,
    RESPONSE_MODE_KEY,
    STATUS_KEY,
    TASK_ID_KEY,
    TOTAL_TOKENS_KEY,
    USAGE_KEY,
    USER_KEY,
    WORKFLOW_RUN_ID_KEY,
)

_REDACTED = "[REDACTED]"
_SENSITIVE_KEY_FRAGMENTS = (
    "accesskey",
    "apikey",
    "authorization",
    "cookie",
    "credential",
    "password",
    "passwd",
    "passphrase",
    "privatekey",
    "secret",
    "token",
)
_NON_SECRET_TOKEN_KEYS = {
    "cachedtokens",
    "completiontokens",
    "inputtokens",
    "maxtokens",
    "outputtokens",
    "prompttokens",
    "reasoningtokens",
    "tokencount",
    "totaltokens",
}


def _normalized_key(key: Any) -> str:
    return "".join(
        character for character in str(key).casefold() if character.isalnum()
    )


def _is_sensitive_key(key: Any) -> bool:
    normalized = _normalized_key(key)
    if any(normalized.endswith(key) for key in _NON_SECRET_TOKEN_KEYS):
        return False
    return any(fragment in normalized for fragment in _SENSITIVE_KEY_FRAGMENTS)


def _redact_sensitive(value: Any, seen: set[int] | None = None) -> Any:
    seen = seen or set()
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in seen:
            return "[Circular]"
        seen.add(identity)
        return {
            str(key): (
                _REDACTED if _is_sensitive_key(key) else _redact_sensitive(nested, seen)
            )
            for key, nested in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        identity = id(value)
        if identity in seen:
            return "[Circular]"
        seen.add(identity)
        return [_redact_sensitive(nested, seen) for nested in value]
    return value


def safe_json(value: Any) -> str:
    """Serialize arbitrary Dify values into an OTEL-safe JSON string."""
    try:
        serialized = serialize_value(value=value)
        return json.dumps(
            _redact_sensitive(serialized), default=str, separators=(",", ":")
        )
    except Exception:  # noqa: BLE001 -- arbitrary vendor values may fail serialization
        try:
            return json.dumps(
                _redact_sensitive(value),
                default=lambda _: "[Unserializable]",
                separators=(",", ":"),
            )
        except Exception:  # noqa: BLE001 -- never leak or break on vendor values
            return json.dumps("[Unserializable]", separators=(",", ":"))


def _to_mapping(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value
    for method_name in ("model_dump", "to_dict", "dict", "json"):
        method = getattr(value, method_name, None)
        if not callable(method):
            continue
        try:
            converted = method()
        except Exception:  # noqa: BLE001, S112 -- best-effort vendor hook
            continue
        if isinstance(converted, Mapping):
            return converted
        if isinstance(converted, str):
            try:
                parsed = json.loads(converted)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, Mapping):
                return parsed
    value_dict = getattr(value, "__dict__", None)
    if isinstance(value_dict, Mapping):
        return value_dict
    return None


def _get(value: Any, key: str, default: Any = None) -> Any:
    mapping = _to_mapping(value)
    if mapping is not None:
        return mapping.get(key, default)
    return getattr(value, key, default)


def _status_code(response: Any) -> int:
    status_code = getattr(response, "status_code", 200)
    if isinstance(status_code, int):
        return status_code
    return 200


def _response_json(response: Any) -> Mapping[str, Any]:
    if response is None:
        return {}
    json_method = getattr(response, "json", None)
    if callable(json_method):
        try:
            value = json_method()
        except Exception:  # noqa: BLE001 -- vendor response parsing is best effort
            return {}
        mapping = _to_mapping(value)
        return mapping or {}
    mapping = _to_mapping(response)
    return mapping or {}


def _endpoint_kind(endpoint: str) -> str:
    if endpoint == CHAT_MESSAGES_ENDPOINT:
        return "chat"
    if endpoint == COMPLETION_MESSAGES_ENDPOINT:
        return "completion"
    if (
        endpoint.startswith("/workflows/")
        or endpoint == "/workflows"
        or endpoint.endswith("/pipeline/run")
    ):
        return "workflow"
    return "api"


def _default_span_name(endpoint: str) -> str:
    kind = _endpoint_kind(endpoint)
    if kind == "chat":
        return DIFY_CHAT_SPAN_NAME
    if kind == "completion":
        return DIFY_COMPLETION_SPAN_NAME
    if kind == "workflow":
        return DIFY_WORKFLOW_SPAN_NAME
    # Keep endpoint paths (which often contain high-cardinality IDs) in
    # metadata, never in the span name.
    return DIFY_API_SPAN_NAME


def _log_type(endpoint: str) -> str:
    kind = _endpoint_kind(endpoint)
    if kind == "chat":
        return LOG_TYPE_CHAT
    if kind == "completion":
        return LOG_TYPE_TEXT
    if kind == "workflow":
        return LOG_TYPE_WORKFLOW
    return LOG_TYPE_TASK


def _request_input(request_json: Any, request_params: Any, request_data: Any) -> Any:
    if request_json is not None:
        return request_json
    if request_data is not None:
        return request_data
    if request_params is not None:
        return request_params
    return {}


def _string_content(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return safe_json(value=value)


def _completion_prompt(request_json: Mapping[str, Any]) -> str:
    query = request_json.get(QUERY_KEY)
    if query:
        return _string_content(query)
    inputs = request_json.get(INPUTS_KEY)
    if isinstance(inputs, Mapping) and inputs.get(QUERY_KEY):
        return _string_content(inputs.get(QUERY_KEY))
    return _string_content(inputs)


def _response_output(
    response_data: Mapping[str, Any], stream_events: Sequence[Any]
) -> Any:
    if stream_events:
        text = _stream_answer(stream_events=stream_events)
        if text:
            return text
        return stream_events
    if ANSWER_KEY in response_data:
        return response_data.get(ANSWER_KEY)
    data = response_data.get(DATA_KEY)
    if isinstance(data, Mapping):
        if OUTPUTS_KEY in data:
            return data.get(OUTPUTS_KEY)
        if ERROR_KEY in data and data.get(ERROR_KEY):
            return data.get(ERROR_KEY)
    return response_data


def _metadata_usage(response_data: Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = response_data.get(METADATA_KEY)
    usage = _get(metadata, USAGE_KEY)
    mapping = _to_mapping(usage)
    if mapping is not None:
        return mapping

    data = response_data.get(DATA_KEY)
    if isinstance(data, Mapping):
        return data
    return {}


def _response_model(response_data: Mapping[str, Any]) -> str | None:
    """Return a model only when Dify includes one in its response payload."""
    candidates = [response_data.get("model")]
    metadata = response_data.get(METADATA_KEY)
    if isinstance(metadata, Mapping):
        candidates.append(metadata.get("model"))
        usage = metadata.get(USAGE_KEY)
        if isinstance(usage, Mapping):
            candidates.append(usage.get("model"))
    data = response_data.get(DATA_KEY)
    if isinstance(data, Mapping):
        candidates.append(data.get("model"))

    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return None


def _stream_answer(stream_events: Sequence[Any]) -> str:
    parts: list[str] = []
    for event in stream_events:
        mapping = _to_mapping(event)
        if mapping is None:
            if isinstance(event, str) and event and not event.startswith("["):
                parts.append(event)
            continue
        answer = mapping.get(ANSWER_KEY)
        if answer:
            parts.append(str(answer))
            continue
        data = mapping.get(DATA_KEY)
        if isinstance(data, Mapping):
            for key in ("text", ANSWER_KEY):
                value = data.get(key)
                if value:
                    parts.append(str(value))
                    break
    return "".join(parts)


def _stream_usage(stream_events: Sequence[Any]) -> Mapping[str, Any]:
    for event in reversed(stream_events):
        mapping = _to_mapping(event)
        if mapping is None:
            continue
        usage = _metadata_usage(mapping)
        if usage:
            return usage
    return {}


def _stream_response_data(stream_events: Sequence[Any]) -> Mapping[str, Any]:
    merged: dict[str, Any] = {}
    for event in stream_events:
        mapping = _to_mapping(event)
        if mapping is not None:
            merged.update(mapping)
    return merged


def _int_value(mapping: Mapping[str, Any], key: str) -> int | None:
    value = mapping.get(key)
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _set_metadata(attributes: dict[str, Any], key: str, value: Any) -> None:
    if value is None:
        return
    canonical = _metadata_mapping(attributes.get(RESPAN_METADATA))

    serialized_value = serialize_value(value=value)
    canonical[key] = serialized_value
    attributes[RESPAN_METADATA] = safe_json(value=canonical)


def _metadata_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        if isinstance(parsed, dict):
            return parsed
    return {}


def _otel_safe_attribute(value: Any) -> Any:
    serialized = serialize_value(value=value)
    if serialized is None or isinstance(serialized, (str, bool, int, float)):
        return serialized
    if isinstance(serialized, (list, tuple)) and serialized:
        item_types = {type(item) for item in serialized}
        if len(item_types) == 1 and item_types.pop() in {str, bool, int, float}:
            return list(serialized)
    return safe_json(value=serialized)


def _merge_propagated_attributes(
    attributes: dict[str, Any],
    propagated_attributes: Mapping[str, Any],
) -> None:
    canonical_metadata = _metadata_mapping(attributes.get(RESPAN_METADATA))
    metadata_prefix = f"{RESPAN_METADATA}."
    for key, value in propagated_attributes.items():
        if key == RESPAN_METADATA:
            for metadata_key, metadata_value in _metadata_mapping(value).items():
                canonical_metadata.setdefault(metadata_key, metadata_value)
            continue
        if key.startswith(metadata_prefix):
            metadata_key = key.removeprefix(metadata_prefix)
            if metadata_key:
                canonical_metadata.setdefault(
                    metadata_key,
                    serialize_value(value=value),
                )
            continue
        attribute_value = _otel_safe_attribute(value)
        if attribute_value is not None:
            attributes[key] = attribute_value

    if canonical_metadata:
        attributes[RESPAN_METADATA] = safe_json(value=canonical_metadata)


def _apply_response_metadata(
    attributes: dict[str, Any],
    response_data: Mapping[str, Any],
) -> None:
    for key in (
        EVENT_KEY,
        TASK_ID_KEY,
        ID_KEY,
        MESSAGE_ID_KEY,
        CONVERSATION_ID_KEY,
        WORKFLOW_RUN_ID_KEY,
        MODE_KEY,
        STATUS_KEY,
    ):
        _set_metadata(
            attributes=attributes, key=f"dify.{key}", value=response_data.get(key)
        )

    data = response_data.get(DATA_KEY)
    if isinstance(data, Mapping):
        for key in ("workflow_id", "elapsed_time", "total_steps", TOTAL_TOKENS_KEY):
            _set_metadata(attributes=attributes, key=f"dify.{key}", value=data.get(key))


def _apply_request_context(
    attributes: dict[str, Any],
    *,
    request_json: Mapping[str, Any],
    request_params: Mapping[str, Any],
) -> None:
    user = request_json.get(USER_KEY) or request_params.get(USER_KEY)
    if user:
        attributes.setdefault(RESPAN_CUSTOMER_PARAMS_ID, str(user))

    conversation_id = request_json.get(CONVERSATION_ID_KEY) or request_params.get(
        CONVERSATION_ID_KEY
    )
    if conversation_id:
        attributes.setdefault(RESPAN_THREADS_ID, str(conversation_id))

    response_mode = request_json.get(RESPONSE_MODE_KEY)
    if response_mode:
        _set_metadata(
            attributes=attributes,
            key=f"dify.{RESPONSE_MODE_KEY}",
            value=response_mode,
        )
    files = request_json.get(FILES_KEY)
    if isinstance(files, Sequence) and not isinstance(files, (str, bytes, bytearray)):
        _set_metadata(attributes=attributes, key="dify.files_count", value=len(files))


def _apply_usage(
    attributes: dict[str, Any],
    *,
    usage: Mapping[str, Any],
) -> None:
    prompt_tokens = _int_value(mapping=usage, key=PROMPT_TOKENS_KEY)
    completion_tokens = _int_value(mapping=usage, key=COMPLETION_TOKENS_KEY)
    total_tokens = _int_value(mapping=usage, key=TOTAL_TOKENS_KEY)
    if total_tokens is None and (
        prompt_tokens is not None or completion_tokens is not None
    ):
        total_tokens = (prompt_tokens or 0) + (completion_tokens or 0)

    if prompt_tokens is not None:
        attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] = prompt_tokens
        attributes[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] = prompt_tokens
    if completion_tokens is not None:
        attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] = completion_tokens
        attributes[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS] = completion_tokens
    if total_tokens is not None:
        attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] = total_tokens

    _set_metadata(
        attributes=attributes, key=f"dify.{LATENCY_KEY}", value=usage.get(LATENCY_KEY)
    )


def _apply_llm_content(
    attributes: dict[str, Any],
    *,
    endpoint: str,
    request_json: Mapping[str, Any],
    response_data: Mapping[str, Any],
    usage: Mapping[str, Any],
    output: Any,
) -> None:
    kind = _endpoint_kind(endpoint)
    if kind not in {"chat", "completion"}:
        return

    attributes[SpanAttributes.LLM_SYSTEM] = "dify"
    # Respan's backend prompt/completion parser is keyed by llm.request.type=chat
    # even for text-style Dify completion apps, which return answer-shaped
    # message payloads rather than raw completion choices.
    attributes[SpanAttributes.LLM_REQUEST_TYPE] = LLMRequestTypeValues.CHAT.value
    model = _response_model(response_data=response_data)
    if model is None:
        usage_model = usage.get("model")
        if isinstance(usage_model, str) and usage_model.strip():
            model = usage_model.strip()
    if model is None:
        request_model = request_json.get("model")
        if isinstance(request_model, str) and request_model.strip():
            model = request_model.strip()
    if model is not None:
        attributes[SpanAttributes.LLM_REQUEST_MODEL] = model

    prompt_content = (
        _string_content(request_json.get(QUERY_KEY))
        if kind == "chat"
        else _completion_prompt(request_json=request_json)
    )
    if prompt_content:
        attributes[f"{SpanAttributes.LLM_PROMPTS}.0.role"] = "user"
        attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"] = prompt_content

    output_text = _string_content(output)
    if output_text:
        attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.role"] = "assistant"
        attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"] = output_text


def _apply_respan_params(
    *,
    attributes: dict[str, Any],
    respan_params: Mapping[str, Any],
    current_workflow_name: str | None,
    default_span_name: str,
) -> str:
    span_name = str(respan_params.get("span_name") or default_span_name)
    workflow_name = (
        respan_params.get("workflow_name")
        or respan_params.get("span_workflow_name")
        or current_workflow_name
    )
    if workflow_name:
        attributes.setdefault(
            SpanAttributes.TRACELOOP_WORKFLOW_NAME, str(workflow_name)
        )
    if (
        respan_params.get("workflow_name")
        and "trace_group_identifier" not in respan_params
    ):
        attributes.setdefault(
            RESPAN_TRACE_GROUP_ID, str(respan_params["workflow_name"])
        )

    for key, value in respan_params.items():
        if key in {
            "parent_span_id",
            "span_id",
            "span_name",
            "trace_id",
            "trace_name",
            "workflow_name",
            "span_workflow_name",
        }:
            continue
        attr_key = RESPAN_SPAN_ATTRIBUTES_MAP.get(str(key))
        if attr_key is None:
            continue
        if attr_key == RESPAN_METADATA and isinstance(value, Mapping):
            for metadata_key, metadata_value in value.items():
                _set_metadata(
                    attributes=attributes,
                    key=str(metadata_key),
                    value=metadata_value,
                )
        else:
            attribute_value = _otel_safe_attribute(value)
            if attribute_value is not None:
                attributes[attr_key] = attribute_value
    return span_name


def build_dify_span_data(
    *,
    method: str,
    endpoint: str,
    request_json: Any = None,
    request_params: Any = None,
    request_data: Any = None,
    files: Any = None,
    response: Any = None,
    stream_events: Sequence[Any] | None = None,
    error: Exception | None = None,
    include_content: bool = True,
    respan_params: Mapping[str, Any] | None = None,
    propagated_attributes: Mapping[str, Any] | None = None,
    current_workflow_name: str | None = None,
    parent_id: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Build canonical span name and attributes from a Dify client call."""
    endpoint = endpoint or ""
    method = (method or "").upper()
    request_json_mapping = _to_mapping(request_json) or {}
    request_params_mapping = _to_mapping(request_params) or {}
    response_data = _response_json(response)
    stream_events = list(stream_events or [])
    output = (
        str(error)
        if error is not None
        else _response_output(response_data, stream_events)
    )
    default_span_name = _default_span_name(endpoint=endpoint)

    attributes: dict[str, Any] = {
        RESPAN_LOG_METHOD: LogMethodChoices.TRACING_INTEGRATION.value,
        RESPAN_LOG_TYPE: _log_type(endpoint=endpoint),
        SpanAttributes.TRACELOOP_ENTITY_NAME: default_span_name,
        SpanAttributes.TRACELOOP_ENTITY_PATH: "",
    }

    _set_metadata(attributes=attributes, key=f"dify.{METHOD_KEY}", value=method)
    _set_metadata(attributes=attributes, key=f"dify.{ENDPOINT_KEY}", value=endpoint)
    if files:
        _set_metadata(attributes=attributes, key="dify.file_upload", value=True)

    _apply_request_context(
        attributes=attributes,
        request_json=request_json_mapping,
        request_params=request_params_mapping,
    )
    semantic_response_data = dict(response_data)
    if stream_events:
        semantic_response_data.update(_stream_response_data(stream_events))
    _apply_response_metadata(
        attributes=attributes,
        response_data=semantic_response_data,
    )

    usage = (
        _stream_usage(stream_events=stream_events)
        if stream_events
        else _metadata_usage(response_data=response_data)
    )
    _apply_usage(attributes=attributes, usage=usage)

    if include_content:
        request_input = _request_input(
            request_json=request_json,
            request_params=request_params,
            request_data=request_data,
        )
        attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT] = safe_json(
            value=request_input
        )
        attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = _string_content(output)
        _apply_llm_content(
            attributes=attributes,
            endpoint=endpoint,
            request_json=request_json_mapping,
            response_data=semantic_response_data,
            usage=usage,
            output=output,
        )

    span_name = _apply_respan_params(
        attributes=attributes,
        respan_params=respan_params or {},
        current_workflow_name=current_workflow_name,
        default_span_name=default_span_name,
    )
    attributes[SpanAttributes.TRACELOOP_ENTITY_NAME] = span_name

    _merge_propagated_attributes(
        attributes,
        propagated_attributes or {},
    )

    workflow_name = attributes.get(SpanAttributes.TRACELOOP_WORKFLOW_NAME)
    attributes[SpanAttributes.TRACELOOP_ENTITY_PATH] = (
        str(workflow_name) if parent_id and workflow_name else ""
    )

    for alias in OFF_CONTRACT_ALIASES:
        attributes.pop(alias, None)

    return span_name, attributes
