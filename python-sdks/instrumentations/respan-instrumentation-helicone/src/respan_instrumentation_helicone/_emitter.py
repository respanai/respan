"""Translate Helicone manual logs into the canonical Respan span contract."""

from __future__ import annotations

import json
import logging
import math
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import islice
from typing import Any

from opentelemetry import trace
from opentelemetry.semconv._incubating.attributes.error_attributes import (
    ERROR_MESSAGE,
    ERROR_TYPE,
)
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import (
    GEN_AI_PROVIDER_NAME,
    GEN_AI_REQUEST_STREAM,
    GEN_AI_RESPONSE_TIME_TO_FIRST_CHUNK,
    GEN_AI_USAGE_INPUT_TOKENS,
    GEN_AI_USAGE_OUTPUT_TOKENS,
)
from opentelemetry.semconv.attributes.http_attributes import (
    HTTP_RESPONSE_STATUS_CODE,
)
from opentelemetry.semconv_ai import SpanAttributes
from respan_sdk.constants.llm_logging import (
    LOG_TYPE_CHAT,
    LOG_TYPE_EMBEDDING,
    LOG_TYPE_TASK,
    LOG_TYPE_TEXT,
    LOG_TYPE_TOOL,
    LogMethodChoices,
)
from respan_sdk.constants.span_attributes import (
    RESPAN_CUSTOMER_PARAMS_ID,
    RESPAN_INTERNAL_SPAN_NAME_DETAIL,
    RESPAN_INTERNAL_SPAN_NAME_KIND,
    RESPAN_LOG_METHOD,
    RESPAN_LOG_TYPE,
    RESPAN_METADATA,
    RESPAN_SESSION_ID,
)
from respan_sdk.utils.data_processing.id_processing import (
    format_span_id,
    format_trace_id,
)
from respan_tracing.utils.span_factory import (
    build_readable_span,
    inject_span,
    read_propagated_attributes,
)

from respan_instrumentation_helicone._constants import (
    HELICONE_DATA_NAME_KEY,
    HELICONE_OPERATION_KEY,
    HELICONE_TOOL_NAME_KEY,
    HELICONE_TYPE_DATA,
    HELICONE_TYPE_EMBEDDING,
    HELICONE_TYPE_KEY,
    HELICONE_TYPE_TOOL,
    HELICONE_TYPE_VECTOR_DB,
    MAX_COLLECTION_ITEMS,
    MAX_DEPTH,
)
from respan_instrumentation_helicone._serialization import (
    exception_message,
    is_sensitive_key,
    json_dumps,
    parse_json,
    safe_text,
    safe_type_name,
    sanitize,
)

logger = logging.getLogger(__name__)

_PROMPT_PREFIX = f"{SpanAttributes.LLM_PROMPTS}."
_COMPLETION_PREFIX = f"{SpanAttributes.LLM_COMPLETIONS}."
_COMPLETION_TOOL_CALLS = f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls"
_LABEL_RE = re.compile(r"[^a-z0-9_.-]+")


@dataclass(frozen=True)
class HeliconeEmissionContext:
    """Trace and propagated attributes captured when a builder is created."""

    trace_id: str | None
    parent_id: str | None
    propagated_attributes: Mapping[str, Any]


def _label(value: Any, fallback: str) -> str:
    if not isinstance(value, str) or not value.strip():
        return fallback
    normalized = _LABEL_RE.sub("_", value.strip().lower()).strip("_.-")
    return normalized[:128] or fallback


def _display_text(value: Any, fallback: str) -> str:
    if isinstance(value, str):
        return safe_text(value)[:128] or fallback
    if value is None:
        return fallback
    if isinstance(value, bool | int | float):
        return safe_text(str(value))[:128]
    return safe_type_name(value)[:128]


def _canonical_role(value: Any, fallback: str) -> str:
    if not isinstance(value, str):
        return fallback
    normalized = value.strip().lower()
    aliases = {
        "human": "user",
        "ai": "assistant",
        "bot": "assistant",
        "model": "assistant",
        "function": "tool",
        "developer": "system",
    }
    normalized = aliases.get(normalized, normalized)
    return (
        normalized
        if normalized in {"user", "assistant", "tool", "system"}
        else fallback
    )


def _number(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value >= 0:
        return value
    if isinstance(value, float) and value.is_integer() and value >= 0:
        return int(value)
    return None


def _seconds(value: Any, fallback: float) -> float:
    if isinstance(value, bool):
        return fallback
    if (
        isinstance(value, int | float)
        and math.isfinite(float(value))
        and float(value) > 0
    ):
        return float(value)
    return fallback


def _current_ids() -> tuple[str | None, str | None]:
    context = trace.get_current_span().get_span_context()
    trace_id = getattr(context, "trace_id", 0) or 0
    span_id = getattr(context, "span_id", 0) or 0
    if not trace_id or not span_id:
        return None, None
    return format_trace_id(trace_id=trace_id), format_span_id(span_id=span_id)


def _snapshot_propagated_value(value: Any, *, depth: int = 0) -> Any:
    if depth >= MAX_DEPTH:
        return json_dumps(value)
    if isinstance(value, Mapping):
        return {
            key: _snapshot_propagated_value(item, depth=depth + 1)
            for key, item in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return tuple(
            _snapshot_propagated_value(item, depth=depth + 1) for item in value
        )
    return value


def capture_emission_context() -> HeliconeEmissionContext:
    trace_id, parent_id = _current_ids()
    return HeliconeEmissionContext(
        trace_id=trace_id,
        parent_id=parent_id,
        propagated_attributes={
            key: _snapshot_propagated_value(value)
            for key, value in read_propagated_attributes().items()
        },
    )


def _response_payload(response: Any) -> Any:
    parsed = parse_json(response)
    if not isinstance(parsed, str):
        return parsed

    chunks: list[Any] = []
    for line in parsed.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("data:"):
            line = line[5:].strip()
        if line == "[DONE]":
            continue
        try:
            chunks.append(json.loads(line))
        except (TypeError, ValueError):
            continue
    return chunks if chunks else parsed


def _unwrap_response(response: Any) -> Any:
    if not isinstance(response, Mapping):
        return response
    nested = response.get("response")
    if isinstance(nested, Mapping) and any(
        key in nested for key in ("choices", "content", "data", "output")
    ):
        return nested
    return response


def _stream_summary(chunks: Sequence[Any]) -> dict[str, Any]:
    text_parts: list[str] = []
    usage: Mapping[str, Any] | None = None
    model: str | None = None
    tool_slots: dict[int, dict[str, Any]] = {}
    for chunk in chunks:
        if not isinstance(chunk, Mapping):
            continue
        if isinstance(chunk.get("model"), str):
            model = chunk["model"]
        if isinstance(chunk.get("usage"), Mapping):
            usage = chunk["usage"]
        choices = chunk.get("choices")
        if not isinstance(choices, Sequence) or isinstance(choices, str) or not choices:
            continue
        choice = choices[0]
        if not isinstance(choice, Mapping):
            continue
        delta = choice.get("delta")
        if not isinstance(delta, Mapping):
            continue
        content = delta.get("content")
        if isinstance(content, str):
            text_parts.append(content)
        calls = delta.get("tool_calls")
        if not isinstance(calls, Sequence) or isinstance(calls, str):
            continue
        for raw_call in calls:
            if not isinstance(raw_call, Mapping):
                continue
            index = raw_call.get("index", 0)
            index = index if isinstance(index, int) and index >= 0 else 0
            slot = tool_slots.setdefault(
                index,
                {
                    "id": None,
                    "type": "function",
                    "function": {"name": "", "arguments": ""},
                },
            )
            if isinstance(raw_call.get("id"), str):
                slot["id"] = raw_call["id"]
            function = raw_call.get("function")
            if isinstance(function, Mapping):
                if isinstance(function.get("name"), str):
                    slot["function"]["name"] += function["name"]
                if isinstance(function.get("arguments"), str):
                    slot["function"]["arguments"] += function["arguments"]
    message: dict[str, Any] = {
        "role": "assistant",
        "content": "".join(text_parts) or None,
    }
    if tool_slots:
        message["tool_calls"] = [tool_slots[index] for index in sorted(tool_slots)]
    result: dict[str, Any] = {"choices": [{"message": message}]}
    if model:
        result["model"] = model
    if usage:
        result["usage"] = dict(usage)
    return result


def _anthropic_stream_summary(events: Sequence[Any]) -> dict[str, Any]:
    model: str | None = None
    usage: dict[str, Any] = {}
    blocks: dict[int, dict[str, Any]] = {}
    partial_json: dict[int, str] = {}
    for event in events:
        if not isinstance(event, Mapping):
            continue
        event_type = event.get("type")
        if event_type == "message_start":
            message = event.get("message")
            if isinstance(message, Mapping):
                if isinstance(message.get("model"), str):
                    model = message["model"]
                if isinstance(message.get("usage"), Mapping):
                    usage.update(message["usage"])
        if isinstance(event.get("usage"), Mapping):
            usage.update(event["usage"])
        index = event.get("index", 0)
        index = index if isinstance(index, int) and index >= 0 else 0
        if event_type == "content_block_start":
            block = event.get("content_block")
            if isinstance(block, Mapping):
                blocks[index] = dict(block)
        elif event_type == "content_block_delta":
            delta = event.get("delta")
            if not isinstance(delta, Mapping):
                continue
            block = blocks.setdefault(index, {"type": "text", "text": ""})
            if delta.get("type") == "text_delta" and isinstance(delta.get("text"), str):
                block["text"] = str(block.get("text") or "") + delta["text"]
            elif delta.get("type") == "input_json_delta" and isinstance(
                delta.get("partial_json"), str
            ):
                partial_json[index] = (
                    partial_json.get(index, "") + delta["partial_json"]
                )
    for index, value in partial_json.items():
        try:
            blocks.setdefault(index, {"type": "tool_use"})["input"] = json.loads(value)
        except (TypeError, ValueError):
            blocks.setdefault(index, {"type": "tool_use"})["input"] = value
    result: dict[str, Any] = {
        "role": "assistant",
        "content": [blocks[index] for index in sorted(blocks)],
    }
    if model:
        result["model"] = model
    if usage:
        result["usage"] = usage
    return result


def _google_stream_summary(chunks: Sequence[Any]) -> dict[str, Any]:
    model_version: str | None = None
    usage_metadata: dict[str, Any] = {}
    text_parts: list[str] = []
    structured_parts: list[Any] = []
    for chunk in chunks:
        if not isinstance(chunk, Mapping):
            continue
        if isinstance(chunk.get("modelVersion"), str):
            model_version = chunk["modelVersion"]
        if isinstance(chunk.get("usageMetadata"), Mapping):
            usage_metadata.update(chunk["usageMetadata"])
        candidates = chunk.get("candidates")
        if (
            not isinstance(candidates, Sequence)
            or isinstance(candidates, str)
            or not candidates
        ):
            continue
        candidate = candidates[0]
        content = candidate.get("content") if isinstance(candidate, Mapping) else None
        parts = content.get("parts") if isinstance(content, Mapping) else None
        if not isinstance(parts, Sequence) or isinstance(parts, str):
            continue
        for part in parts:
            if isinstance(part, Mapping) and isinstance(part.get("text"), str):
                text_parts.append(part["text"])
            else:
                structured_parts.append(part)
    combined_parts: list[Any] = []
    if text_parts:
        combined_parts.append({"text": "".join(text_parts)})
    combined_parts.extend(structured_parts)
    result: dict[str, Any] = {
        "candidates": [{"content": {"role": "model", "parts": combined_parts}}]
    }
    if model_version:
        result["modelVersion"] = model_version
    if usage_metadata:
        result["usageMetadata"] = usage_metadata
    return result


def _summarize_stream(chunks: Sequence[Any]) -> dict[str, Any]:
    if any(
        isinstance(item, Mapping)
        and isinstance(item.get("type"), str)
        and item["type"].startswith(("message_", "content_block_"))
        for item in chunks
    ):
        return _anthropic_stream_summary(chunks)
    if any(isinstance(item, Mapping) and "candidates" in item for item in chunks):
        return _google_stream_summary(chunks)
    return _stream_summary(chunks)


def _normalized_response(response: Any) -> Any:
    payload = _response_payload(response)
    if isinstance(payload, Mapping):
        raw_chunks = payload.get("chunks")
        if isinstance(raw_chunks, Sequence) and not isinstance(
            raw_chunks, str | bytes | bytearray
        ):
            chunks = [parse_json(item) for item in raw_chunks]
            summary = _summarize_stream(chunks)
            ttft = payload.get("time_to_first_token_ms")
            if isinstance(ttft, int | float) and not isinstance(ttft, bool):
                summary["time_to_first_token_ms"] = float(ttft)
            return summary
    if (
        isinstance(payload, Sequence)
        and not isinstance(payload, str | bytes | bytearray)
        and payload
        and all(isinstance(item, Mapping) for item in payload)
    ):
        return _summarize_stream(payload)
    return _unwrap_response(payload)


def _normalize_input_message(message: Mapping[str, Any]) -> Mapping[str, Any]:
    normalized = dict(message)
    content = message.get("content")
    tool_calls: list[dict[str, Any]] = []
    if isinstance(content, Sequence) and not isinstance(
        content, str | bytes | bytearray
    ):
        normalized["content"] = json_dumps(content)
        for block in content:
            if not isinstance(block, Mapping):
                continue
            if block.get("type") == "tool_use" and isinstance(block.get("name"), str):
                tool_call: dict[str, Any] = {
                    "type": "function",
                    "function": {
                        "name": block["name"],
                        "arguments": json_dumps(block.get("input", {})),
                    },
                }
                if isinstance(block.get("id"), str):
                    tool_call["id"] = block["id"]
                tool_calls.append(tool_call)

    parts = message.get("parts")
    if isinstance(parts, Sequence) and not isinstance(parts, str | bytes | bytearray):
        normalized["content"] = json_dumps(parts)
        if normalized.get("role") == "model":
            normalized["role"] = "assistant"
        for part in parts:
            if not isinstance(part, Mapping):
                continue
            function_call = part.get("functionCall")
            if isinstance(function_call, Mapping) and isinstance(
                function_call.get("name"), str
            ):
                tool_calls.append(
                    {
                        "type": "function",
                        "function": {
                            "name": function_call["name"],
                            "arguments": json_dumps(function_call.get("args", {})),
                        },
                    }
                )
    if tool_calls:
        normalized["tool_calls"] = tool_calls
    return normalized


def _messages(request: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    result: list[Mapping[str, Any]] = []
    system = request.get("system")
    if system is not None:
        result.append(_normalize_input_message({"role": "system", "content": system}))
    system_instruction = request.get("systemInstruction")
    if system_instruction is not None:
        if isinstance(system_instruction, Mapping) and "parts" in system_instruction:
            result.append(
                _normalize_input_message(
                    {"role": "system", "parts": system_instruction["parts"]}
                )
            )
        else:
            result.append(
                _normalize_input_message(
                    {"role": "system", "content": system_instruction}
                )
            )

    value = request.get("messages")
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        result.extend(
            _normalize_input_message(item)
            if isinstance(item, Mapping)
            else {"role": "user", "content": item}
            for item in islice(value, max(0, MAX_COLLECTION_ITEMS - len(result)))
        )
        return result
    contents = request.get("contents")
    if isinstance(contents, Sequence) and not isinstance(
        contents, str | bytes | bytearray
    ):
        result.extend(
            _normalize_input_message(item)
            if isinstance(item, Mapping)
            else {"role": "user", "content": item}
            for item in islice(contents, max(0, MAX_COLLECTION_ITEMS - len(result)))
        )
        return result
    prompt = request.get("prompt")
    if prompt is not None:
        prompts = prompt if isinstance(prompt, list | tuple) else [prompt]
        result.extend(
            {"role": "user", "content": item}
            for item in islice(prompts, max(0, MAX_COLLECTION_ITEMS - len(result)))
        )
        return result
    response_input = request.get("input")
    if isinstance(response_input, str):
        result.append({"role": "user", "content": response_input})
        return result
    if isinstance(response_input, Sequence) and not isinstance(
        response_input, str | bytes | bytearray
    ):
        result.extend(
            _normalize_input_message(item)
            if isinstance(item, Mapping)
            else {"role": "user", "content": item}
            for item in islice(
                response_input, max(0, MAX_COLLECTION_ITEMS - len(result))
            )
        )
    return result


def _output_message(response: Any) -> dict[str, Any]:
    if isinstance(response, str):
        return {"role": "assistant", "content": response}
    if not isinstance(response, Mapping):
        return {"role": "assistant", "content": sanitize(response)}

    choices = response.get("choices")
    if isinstance(choices, Sequence) and not isinstance(choices, str) and choices:
        first = choices[0]
        if isinstance(first, Mapping):
            message = first.get("message")
            if isinstance(message, Mapping):
                return dict(message)
            text = first.get("text")
            if text is not None:
                return {"role": "assistant", "content": text}

    candidates = response.get("candidates")
    if (
        isinstance(candidates, Sequence)
        and not isinstance(candidates, str)
        and candidates
    ):
        first_candidate = candidates[0]
        if isinstance(first_candidate, Mapping):
            candidate_content = first_candidate.get("content")
            if isinstance(candidate_content, Mapping):
                normalized = dict(_normalize_input_message(candidate_content))
                normalized["role"] = "assistant"
                return normalized

    content = response.get("content")
    if isinstance(content, Sequence) and not isinstance(content, str | bytes):
        tool_calls: list[dict[str, Any]] = []
        for item in content:
            if not isinstance(item, Mapping):
                continue
            if item.get("type") == "tool_use" and isinstance(item.get("name"), str):
                tool_call: dict[str, Any] = {
                    "type": "function",
                    "function": {
                        "name": item["name"],
                        "arguments": json_dumps(item.get("input", {})),
                    },
                }
                if isinstance(item.get("id"), str):
                    tool_call["id"] = item["id"]
                tool_calls.append(tool_call)
        message: dict[str, Any] = {
            "role": response.get("role", "assistant"),
            "content": json_dumps(content),
        }
        if tool_calls:
            message["tool_calls"] = tool_calls
        return message
    if isinstance(content, str):
        return {"role": response.get("role", "assistant"), "content": content}

    output_text = response.get("output_text")
    if isinstance(output_text, str):
        return {"role": "assistant", "content": output_text}
    return {"role": "assistant", "content": sanitize(response)}


def _usage(response: Any) -> Mapping[str, Any]:
    if not isinstance(response, Mapping):
        return {}
    usage = response.get("usage")
    if isinstance(usage, Mapping):
        return usage
    usage_metadata = response.get("usageMetadata")
    return usage_metadata if isinstance(usage_metadata, Mapping) else {}


def _request_model(request: Mapping[str, Any]) -> str | None:
    value = request.get("model")
    return safe_text(value)[:128] if isinstance(value, str) else None


def _response_model(response: Any) -> str | None:
    if not isinstance(response, Mapping):
        return None
    for key in ("model", "modelVersion"):
        value = response.get(key)
        if isinstance(value, str):
            return safe_text(value)[:128]
    return None


def _embedding_values(request: Mapping[str, Any], response: Any) -> tuple[Any, Any]:
    input_value = request.get("input")
    vectors: list[Any] = []
    if isinstance(response, Mapping):
        data = response.get("data")
        if isinstance(data, Sequence) and not isinstance(data, str | bytes | bytearray):
            vectors = [
                item["embedding"]
                for item in data
                if isinstance(item, Mapping) and "embedding" in item
            ]
        elif "embedding" in response:
            vectors = [response["embedding"]]
        elif "embeddings" in response and isinstance(response["embeddings"], Sequence):
            vectors = list(response["embeddings"])
    output_value: Any = vectors[0] if len(vectors) == 1 else vectors
    return input_value, output_value


def _error_from_response(
    response: Any,
) -> tuple[str | None, int | None, str | None]:
    if not isinstance(response, Mapping):
        return None, None, None
    status_value = response.get("status_code")
    status = (
        status_value
        if isinstance(status_value, int) and not isinstance(status_value, bool)
        else None
    )
    error = response.get("error")
    if isinstance(error, Mapping):
        message = error.get("message")
        raw_type = error.get("type", error.get("code"))
        error_type = (
            safe_text(raw_type)
            if isinstance(raw_type, str) and raw_type.strip()
            else "HeliconeError"
        )
        return (
            safe_text(message)
            if isinstance(message, str)
            else "Helicone operation failed",
            status,
            error_type,
        )
    if isinstance(error, str):
        raw_type = response.get("type", response.get("error_type"))
        error_type = (
            safe_text(raw_type)
            if isinstance(raw_type, str) and raw_type.strip()
            else "HeliconeError"
        )
        return safe_text(error), status, error_type
    if (
        isinstance(response.get("status"), str)
        and response["status"].lower() == "error"
    ):
        message = response.get("message")
        raw_type = response.get("type", response.get("error_type"))
        error_type = (
            safe_text(raw_type)
            if isinstance(raw_type, str) and raw_type.strip()
            else safe_text(response["status"])
        )
        return (
            safe_text(message)
            if isinstance(message, str)
            else "Helicone operation failed",
            status,
            error_type,
        )
    return None, status, None


def _base_attrs(
    *, log_type: str, entity_name: str, parent_id: str | None
) -> dict[str, Any]:
    return {
        RESPAN_LOG_METHOD: LogMethodChoices.TRACING_INTEGRATION.value,
        RESPAN_LOG_TYPE: log_type,
        SpanAttributes.TRACELOOP_ENTITY_NAME: entity_name,
        SpanAttributes.TRACELOOP_ENTITY_PATH: entity_name if parent_id else "",
    }


def _metadata_dict(value: Any) -> dict[str, Any]:
    parsed = parse_json(value)
    return dict(parsed) if isinstance(parsed, Mapping) else {}


def _merge_metadata(attrs: dict[str, Any], update: Mapping[str, Any]) -> None:
    metadata = _metadata_dict(attrs.get(RESPAN_METADATA))
    for key, value in update.items():
        if (
            key == "helicone"
            and isinstance(value, Mapping)
            and isinstance(metadata.get(key), Mapping)
        ):
            helicone = dict(metadata[key])
            for nested_key, nested_value in value.items():
                if (
                    nested_key == "properties"
                    and isinstance(nested_value, Mapping)
                    and isinstance(helicone.get(nested_key), Mapping)
                ):
                    properties = dict(helicone[nested_key])
                    properties.update(nested_value)
                    helicone[nested_key] = properties
                else:
                    helicone[nested_key] = nested_value
            metadata[key] = helicone
        else:
            metadata[key] = value
    if metadata:
        attrs[RESPAN_METADATA] = json_dumps(metadata)


def _otel_propagated_value(value: Any) -> Any:
    if isinstance(value, bool | str | int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else json_dumps(value)
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        if not value:
            return json_dumps(value)
        scalar_types = {type(item) for item in value}
        if (
            len(scalar_types) == 1
            and scalar_types.pop() in {bool, str, int, float}
            and all(
                not isinstance(item, float) or math.isfinite(item) for item in value
            )
        ):
            return tuple(value)
    return json_dumps(value)


def _apply_propagated_attributes(
    attrs: dict[str, Any], propagated: Mapping[str, Any]
) -> None:
    metadata: dict[str, Any] = {}
    prefix = f"{RESPAN_METADATA}."
    for key, value in propagated.items():
        if key == RESPAN_METADATA:
            metadata.update(_metadata_dict(value))
        elif key.startswith(prefix):
            metadata[key.removeprefix(prefix)] = value
        else:
            attrs.setdefault(key, _otel_propagated_value(value))
    _merge_metadata(attrs, metadata)


def _apply_safe_options(
    attrs: dict[str, Any],
    options: Mapping[str, Any],
    *,
    constructor_headers: Any,
    response: Any,
) -> None:
    merged_headers: dict[str, str] = {}
    if isinstance(constructor_headers, Mapping):
        merged_headers.update(
            (key, value)
            for key, value in constructor_headers.items()
            if isinstance(key, str) and isinstance(value, str)
        )
    additional_headers = options.get("additional_headers")
    if isinstance(additional_headers, Mapping):
        merged_headers.update(
            (key, value)
            for key, value in additional_headers.items()
            if isinstance(key, str) and isinstance(value, str)
        )

    helicone_metadata: dict[str, Any] = {}
    properties: dict[str, str] = {}
    for raw_name, raw_value in merged_headers.items():
        name = raw_name.strip().lower()
        if name == "helicone-session-id":
            attrs[RESPAN_SESSION_ID] = safe_text(raw_value)
        elif name == "helicone-user-id":
            attrs[RESPAN_CUSTOMER_PARAMS_ID] = safe_text(raw_value)
        elif name in {"helicone-session-name", "helicone-session-path"}:
            suffix = name.removeprefix("helicone-").replace("-", "_")
            helicone_metadata[suffix] = safe_text(raw_value)
        elif name.startswith("helicone-property-"):
            property_name = _label(name.removeprefix("helicone-property-"), "property")
            if not is_sensitive_key(property_name):
                properties[property_name] = safe_text(raw_value)
    if properties:
        helicone_metadata["properties"] = properties

    ttft = options.get("time_to_first_token_ms")
    if ttft is None and isinstance(response, Mapping):
        ttft = response.get("time_to_first_token_ms")
    if (
        isinstance(ttft, int | float)
        and not isinstance(ttft, bool)
        and math.isfinite(float(ttft))
        and ttft >= 0
    ):
        ttft_ms = float(ttft)
        attrs[GEN_AI_RESPONSE_TIME_TO_FIRST_CHUNK] = ttft_ms / 1000
        helicone_metadata["time_to_first_token_ms"] = ttft_ms
    if helicone_metadata:
        _merge_metadata(attrs, {"helicone": helicone_metadata})


def _set_usage(attrs: dict[str, Any], response: Any) -> None:
    usage = _usage(response)
    input_tokens = _number(usage.get("input_tokens"))
    if input_tokens is None:
        input_tokens = _number(usage.get("prompt_tokens"))
    if input_tokens is None:
        input_tokens = _number(
            usage.get("promptTokenCount", usage.get("inputTokenCount"))
        )
    output_tokens = _number(usage.get("output_tokens"))
    if output_tokens is None:
        output_tokens = _number(usage.get("completion_tokens"))
    if output_tokens is None:
        output_tokens = _number(
            usage.get("candidatesTokenCount", usage.get("outputTokenCount"))
        )
    total_tokens = _number(usage.get("total_tokens"))
    if total_tokens is None:
        total_tokens = _number(usage.get("totalTokenCount"))
    cache_read_tokens = _number(usage.get("cache_read_input_tokens"))
    if cache_read_tokens is None:
        cache_read_tokens = _number(usage.get("cache_read_tokens"))
    prompt_details = usage.get("prompt_tokens_details")
    if cache_read_tokens is None and isinstance(prompt_details, Mapping):
        cache_read_tokens = _number(prompt_details.get("cached_tokens"))
    if cache_read_tokens is None:
        cache_read_tokens = _number(usage.get("cachedContentTokenCount"))
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens
    if input_tokens is not None:
        attrs[GEN_AI_USAGE_INPUT_TOKENS] = input_tokens
        attrs[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] = input_tokens
    if output_tokens is not None:
        attrs[GEN_AI_USAGE_OUTPUT_TOKENS] = output_tokens
        attrs[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS] = output_tokens
    if total_tokens is not None:
        attrs[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] = total_tokens
    if cache_read_tokens is not None:
        attrs[SpanAttributes.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS] = cache_read_tokens
        attrs[SpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS] = cache_read_tokens


def _llm_attrs(
    request: Mapping[str, Any],
    response: Any,
    *,
    provider: str | None,
    parent_id: str | None,
    capture_content: bool,
    is_streaming_override: bool | None,
) -> tuple[str, dict[str, Any]]:
    is_embedding = request.get(HELICONE_TYPE_KEY) == HELICONE_TYPE_EMBEDDING or (
        "input" in request
        and isinstance(response, Mapping)
        and isinstance(response.get("data"), Sequence)
        and any(
            isinstance(item, Mapping) and "embedding" in item
            for item in response.get("data", [])
        )
    )
    raw_messages = request.get("messages")
    is_chat = isinstance(raw_messages, Sequence) and not isinstance(
        raw_messages, str | bytes | bytearray
    )
    raw_contents = request.get("contents")
    if isinstance(raw_contents, Sequence) and not isinstance(
        raw_contents, str | bytes | bytearray
    ):
        is_chat = True
    if not is_embedding and "input" in request and "prompt" not in request:
        is_chat = True
    log_type = (
        LOG_TYPE_EMBEDDING
        if is_embedding
        else (LOG_TYPE_CHAT if is_chat else LOG_TYPE_TEXT)
    )
    entity_name = {
        LOG_TYPE_EMBEDDING: "helicone.manual.embedding",
        LOG_TYPE_CHAT: "helicone.manual.chat",
        LOG_TYPE_TEXT: "helicone.manual.text",
    }[log_type]
    attrs = _base_attrs(log_type=log_type, entity_name=entity_name, parent_id=parent_id)
    provider_name = _label(provider or request.get("provider"), "custom")
    attrs[SpanAttributes.LLM_SYSTEM] = provider_name
    attrs[GEN_AI_PROVIDER_NAME] = provider_name
    attrs[SpanAttributes.LLM_REQUEST_TYPE] = "embedding" if is_embedding else "chat"
    is_streaming = (
        bool(is_streaming_override)
        if is_streaming_override is not None
        else bool(request.get("stream"))
    )
    attrs[GEN_AI_REQUEST_STREAM] = is_streaming
    attrs[SpanAttributes.GEN_AI_IS_STREAMING] = is_streaming
    request_model = _request_model(request)
    response_model = _response_model(response)
    if request_model:
        attrs[SpanAttributes.LLM_REQUEST_MODEL] = request_model
    if response_model:
        attrs[SpanAttributes.LLM_RESPONSE_MODEL] = response_model
    _set_usage(attrs, response)

    if not capture_content:
        return "embedding" if is_embedding else "llm", attrs

    if is_embedding:
        embedding_input, embedding_output = _embedding_values(request, response)
        attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = json_dumps(
            embedding_input, preserve_large_vectors=True
        )
        attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = json_dumps(
            embedding_output, preserve_large_vectors=True
        )
    else:
        attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = json_dumps(request)
        attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = json_dumps(response)
    for index, message in enumerate(_messages(request)):
        role = message.get("role")
        content = message.get("content")
        attrs[f"{_PROMPT_PREFIX}{index}.role"] = _canonical_role(role, "user")
        if content is not None:
            attrs[f"{_PROMPT_PREFIX}{index}.content"] = (
                safe_text(content) if isinstance(content, str) else json_dumps(content)
            )
        tool_calls = message.get("tool_calls")
        if tool_calls:
            attrs[f"{_PROMPT_PREFIX}{index}.tool_calls"] = json_dumps(tool_calls)
    tools = request.get("tools")
    if isinstance(tools, Sequence) and not isinstance(tools, str | bytes):
        attrs[SpanAttributes.LLM_REQUEST_FUNCTIONS] = json_dumps(tools)
    if not is_embedding:
        output_message = _output_message(response)
        attrs[f"{_COMPLETION_PREFIX}0.role"] = "assistant"
        content = output_message.get("content")
        if content is not None:
            attrs[f"{_COMPLETION_PREFIX}0.content"] = (
                safe_text(content) if isinstance(content, str) else json_dumps(content)
            )
        tool_calls = output_message.get("tool_calls")
        if tool_calls:
            attrs[_COMPLETION_TOOL_CALLS] = json_dumps(tool_calls)
    return "embedding" if is_embedding else "llm", attrs


def _custom_attrs(
    request: Mapping[str, Any],
    response: Any,
    *,
    parent_id: str | None,
    capture_content: bool,
) -> tuple[str, dict[str, Any]]:
    event_type = request.get(HELICONE_TYPE_KEY)
    if event_type == HELICONE_TYPE_TOOL:
        name = _display_text(
            request.get(HELICONE_TOOL_NAME_KEY) or request.get("name"), "tool"
        )
        attrs = _base_attrs(
            log_type=LOG_TYPE_TOOL, entity_name=name, parent_id=parent_id
        )
        attrs[RESPAN_INTERNAL_SPAN_NAME_KIND] = "tool"
        attrs[RESPAN_INTERNAL_SPAN_NAME_DETAIL] = name
        if capture_content:
            arguments = request.get("input", request.get("arguments", request))
            attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = json_dumps(
                {"name": name, "arguments": arguments}
            )
            attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = json_dumps(response)
        return "tool", attrs

    if event_type == HELICONE_TYPE_VECTOR_DB:
        operation = _label(request.get(HELICONE_OPERATION_KEY), "operation")
        attrs = _base_attrs(
            log_type=LOG_TYPE_TASK,
            entity_name=f"vector_db.{operation}",
            parent_id=parent_id,
        )
        if capture_content:
            attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = json_dumps(
                request, preserve_large_vectors=True
            )
            attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = json_dumps(
                response, preserve_large_vectors=True
            )
        return "task", attrs

    name = _display_text(request.get(HELICONE_DATA_NAME_KEY), "custom_data")
    attrs = _base_attrs(log_type=LOG_TYPE_TASK, entity_name=name, parent_id=parent_id)
    if capture_content:
        attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = json_dumps(request)
        attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = json_dumps(response)
    return "task", attrs


def emit_helicone_log(
    *,
    provider: str | None,
    request: Any,
    response: Any,
    options: Any,
    capture_content: bool,
    error: BaseException | None = None,
    status_code: int | None = None,
    is_streaming: bool | None = None,
    context_snapshot: HeliconeEmissionContext | None = None,
    constructor_headers: Any = None,
) -> bool:
    """Emit one Helicone sink call while never changing SDK behavior."""

    try:
        request_map = request if isinstance(request, Mapping) else {"value": request}
        inferred_streaming = is_streaming
        if inferred_streaming is None:
            inferred_streaming = (
                bool(request_map.get("stream"))
                or (
                    isinstance(response, Mapping)
                    and isinstance(response.get("chunks"), Sequence)
                    and not isinstance(response.get("chunks"), str | bytes | bytearray)
                )
                or (
                    isinstance(response, Sequence)
                    and not isinstance(response, str | bytes | bytearray)
                )
            )
        normalized_response = _normalized_response(response)
        effective_context = context_snapshot or capture_emission_context()
        trace_id = effective_context.trace_id
        parent_id = effective_context.parent_id
        event_type = request_map.get(HELICONE_TYPE_KEY)
        if event_type in {
            HELICONE_TYPE_TOOL,
            HELICONE_TYPE_VECTOR_DB,
            HELICONE_TYPE_DATA,
        }:
            semantic_kind, attrs = _custom_attrs(
                request_map,
                normalized_response,
                parent_id=parent_id,
                capture_content=capture_content,
            )
        elif provider is not None or any(
            key in request_map for key in ("model", "messages", "prompt", "input")
        ):
            semantic_kind, attrs = _llm_attrs(
                request_map,
                normalized_response,
                provider=provider,
                parent_id=parent_id,
                capture_content=capture_content,
                is_streaming_override=inferred_streaming,
            )
        else:
            semantic_kind, attrs = _custom_attrs(
                request_map,
                normalized_response,
                parent_id=parent_id,
                capture_content=capture_content,
            )

        options_map = options if isinstance(options, Mapping) else {}
        _apply_safe_options(
            attrs,
            options_map,
            constructor_headers=constructor_headers,
            response=normalized_response,
        )
        _apply_propagated_attributes(attrs, effective_context.propagated_attributes)

        response_error, response_status, response_error_type = _error_from_response(
            normalized_response
        )
        message = exception_message(error) if error is not None else response_error
        error_type = safe_type_name(error) if error is not None else response_error_type
        if status_code is not None:
            resolved_status = status_code
        elif response_status is not None and response_status >= 400:
            resolved_status = response_status
        elif message:
            resolved_status = 500
        else:
            resolved_status = response_status or 200
        attrs[HTTP_RESPONSE_STATUS_CODE] = resolved_status
        if message:
            attrs[ERROR_MESSAGE] = message
            attrs[ERROR_TYPE] = error_type or "HeliconeError"

        now = time.time()
        start = _seconds(options_map.get("start_time"), now)
        end = _seconds(options_map.get("end_time"), now)
        end = max(end, start)
        model = attrs.get(SpanAttributes.LLM_REQUEST_MODEL) or attrs.get(
            SpanAttributes.LLM_RESPONSE_MODEL
        )
        span_name = (
            f"llm.{model}"
            if semantic_kind == "llm" and isinstance(model, str)
            else semantic_kind
        )
        span = build_readable_span(
            name=span_name,
            trace_id=trace_id,
            parent_id=parent_id,
            start_time_ns=int(start * 1_000_000_000),
            end_time_ns=int(end * 1_000_000_000),
            attributes=attrs,
            status_code=resolved_status,
            error_message=message,
            merge_propagated=False,
        )
        return inject_span(span=span)
    except BaseException:
        logger.debug("Failed to emit Helicone manual logger span", exc_info=True)
        return False
