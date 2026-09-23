"""Pure Exa-to-Respan span translation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from opentelemetry.semconv_ai import LLMRequestTypeValues, SpanAttributes
from respan_sdk.constants.llm_logging import (
    LOG_TYPE_AGENT,
    LOG_TYPE_CHAT,
    LOG_TYPE_TASK,
    LOG_TYPE_TOOL,
    LogMethodChoices,
)
from respan_sdk.constants.span_attributes import (
    RESPAN_LOG_METHOD,
    RESPAN_LOG_TYPE,
    RESPAN_METADATA,
)

from respan_instrumentation_exa._constants import (
    EXA_METADATA_NAMESPACE,
    EXA_SYSTEM,
    FAMILY_AGENT,
    FAMILY_CHAT,
    FAMILY_TASK,
    FAMILY_TOOL,
    METADATA_CITATIONS,
    METADATA_COST_TOTAL_USD,
    METADATA_LANGUAGE,
    METADATA_OPERATION,
    METADATA_REQUEST_ID,
    METADATA_RESEARCH_LEGACY,
    METADATA_RESOLVED_SEARCH_TYPE,
    METADATA_RESULT_COUNT,
    METADATA_STREAM,
    METADATA_STREAM_COMPLETED,
    OperationConfig,
)
from respan_instrumentation_exa._serialization import json_dumps, to_jsonable, value_at

_LOG_TYPE_BY_FAMILY = {
    FAMILY_AGENT: LOG_TYPE_AGENT,
    FAMILY_CHAT: LOG_TYPE_CHAT,
    FAMILY_TASK: LOG_TYPE_TASK,
    FAMILY_TOOL: LOG_TYPE_TOOL,
}


def resolve_family(config: OperationConfig, *, streaming: bool) -> str:
    if streaming and config.stream_family:
        return config.stream_family
    return config.family


def build_start_attributes(
    *,
    config: OperationConfig,
    call_input: Mapping[str, Any],
    capture_content: bool,
    streaming: bool,
    has_parent: bool,
) -> dict[str, Any]:
    family = resolve_family(config, streaming=streaming)
    attrs: dict[str, Any] = {
        RESPAN_LOG_METHOD: LogMethodChoices.TRACING_INTEGRATION.value,
        RESPAN_LOG_TYPE: _LOG_TYPE_BY_FAMILY[family],
        SpanAttributes.TRACELOOP_ENTITY_NAME: config.entity_name,
        SpanAttributes.TRACELOOP_ENTITY_PATH: config.entity_name if has_parent else "",
        RESPAN_METADATA: _metadata_json(config=config, streaming=streaming),
    }

    if capture_content:
        payload: Any = dict(call_input)
        if family == FAMILY_TOOL:
            payload = {"name": config.entity_name, "arguments": dict(call_input)}
        attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = json_dumps(payload)

    if family == FAMILY_CHAT:
        _set_chat_request_attributes(
            attrs,
            call_input,
            capture_content=capture_content,
            streaming=streaming,
        )
    return attrs


def build_success_attributes(
    *,
    config: OperationConfig,
    call_input: Mapping[str, Any],
    result: Any,
    capture_content: bool,
    streaming: bool,
    stream_completed: bool = True,
) -> dict[str, Any]:
    serialized = to_jsonable(result)
    metadata = _base_metadata(config=config, streaming=streaming)
    if streaming:
        metadata[METADATA_STREAM_COMPLETED] = stream_completed

    attrs: dict[str, Any] = {}
    if capture_content:
        attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = json_dumps(serialized)

    family = resolve_family(config, streaming=streaming)
    if family == FAMILY_CHAT:
        if capture_content:
            answer = _answer_text(result)
            if answer:
                attrs[f"{SpanAttributes.LLM_COMPLETIONS}.0.role"] = "assistant"
                attrs[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"] = answer
        response_model = _first_value(serialized, "model")
        if response_model is not None:
            attrs[SpanAttributes.LLM_REQUEST_MODEL] = str(response_model)

    result_count = _result_count(serialized)
    if result_count is not None:
        metadata[METADATA_RESULT_COUNT] = result_count
    request_id = _first_value(serialized, "request_id", "requestId")
    if request_id is not None:
        metadata[METADATA_REQUEST_ID] = str(request_id)
    resolved_type = _first_value(
        serialized,
        "resolved_search_type",
        "resolvedSearchType",
    )
    if resolved_type is not None:
        metadata[METADATA_RESOLVED_SEARCH_TYPE] = str(resolved_type)
    cost = _cost_total(serialized)
    if cost is not None:
        metadata[METADATA_COST_TOTAL_USD] = cost
    citations = _first_value(serialized, "citations")
    if (
        capture_content
        and isinstance(citations, Sequence)
        and not isinstance(citations, (str, bytes, bytearray))
    ):
        metadata[METADATA_CITATIONS] = to_jsonable(citations)
    attrs[RESPAN_METADATA] = json_dumps({EXA_METADATA_NAMESPACE: metadata})
    return attrs


def stream_result(chunks: Sequence[Any]) -> dict[str, Any]:
    content_parts: list[str] = []
    citations: list[Any] = []
    serialized_chunks = []
    for chunk in chunks:
        serialized = to_jsonable(chunk)
        serialized_chunks.append(serialized)
        content = value_at(chunk, "content")
        if content:
            content_parts.append(str(content))
        chunk_citations = value_at(chunk, "citations")
        if isinstance(chunk_citations, Sequence) and not isinstance(
            chunk_citations, (str, bytes, bytearray)
        ):
            citations.extend(to_jsonable(chunk_citations))
    return {
        "content": "".join(content_parts),
        "citations": citations,
        "chunks": serialized_chunks,
    }


def _set_chat_request_attributes(
    attrs: dict[str, Any],
    call_input: Mapping[str, Any],
    *,
    capture_content: bool,
    streaming: bool,
) -> None:
    attrs[SpanAttributes.LLM_SYSTEM] = EXA_SYSTEM
    attrs[SpanAttributes.LLM_REQUEST_TYPE] = LLMRequestTypeValues.CHAT.value
    model = call_input.get("model")
    if model is not None:
        attrs[SpanAttributes.LLM_REQUEST_MODEL] = str(model)
    attrs[SpanAttributes.LLM_IS_STREAMING] = streaming
    if not capture_content:
        return
    index = 0
    system_prompt = call_input.get("system_prompt")
    if system_prompt:
        attrs[f"{SpanAttributes.LLM_PROMPTS}.{index}.role"] = "system"
        attrs[f"{SpanAttributes.LLM_PROMPTS}.{index}.content"] = str(system_prompt)
        index += 1
    query = call_input.get("query")
    if query is not None:
        attrs[f"{SpanAttributes.LLM_PROMPTS}.{index}.role"] = "user"
        attrs[f"{SpanAttributes.LLM_PROMPTS}.{index}.content"] = str(query)


def _answer_text(value: Any) -> str | None:
    answer = value_at(value, "answer")
    if answer is None:
        answer = value_at(value, "content")
    if answer is None and isinstance(value, Mapping):
        answer = value.get("output")
    if answer is None:
        return None
    if isinstance(answer, str):
        return answer
    return json_dumps(answer)


def _result_count(value: Any) -> int | None:
    if not isinstance(value, Mapping):
        return None
    results = value.get("results")
    if isinstance(results, Sequence) and not isinstance(
        results, (str, bytes, bytearray)
    ):
        return len(results)
    return None


def _first_value(value: Any, *keys: str) -> Any:
    if not isinstance(value, Mapping):
        return None
    for key in keys:
        if value.get(key) is not None:
            return value[key]
    return None


def _cost_total(value: Any) -> float | None:
    if not isinstance(value, Mapping):
        return None
    cost = value.get("cost_dollars") or value.get("costDollars")
    if isinstance(cost, Mapping):
        total = cost.get("total")
        if isinstance(total, (int, float)):
            return float(total)
    return None


def _base_metadata(
    *,
    config: OperationConfig,
    streaming: bool,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        METADATA_OPERATION: config.operation,
        METADATA_LANGUAGE: "python",
        METADATA_STREAM: streaming,
    }
    if config.legacy_research:
        metadata[METADATA_RESEARCH_LEGACY] = True
    return metadata


def _metadata_json(*, config: OperationConfig, streaming: bool) -> str:
    return json_dumps(
        {EXA_METADATA_NAMESPACE: _base_metadata(config=config, streaming=streaming)}
    )
