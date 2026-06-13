"""Cohere span normalization for the Respan contract."""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

from opentelemetry.context import Context
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import LLMRequestTypeValues, SpanAttributes
from respan_sdk.constants.llm_logging import (
    LOG_TYPE_CHAT,
    LOG_TYPE_EMBEDDING,
    LOG_TYPE_TASK,
    LOG_TYPE_TEXT,
)
from respan_sdk.constants.span_attributes import (
    RESPAN_LOG_TYPE,
    RESPAN_SPAN_HANDOFFS,
    RESPAN_SPAN_TOOL_CALLS,
    RESPAN_SPAN_TOOLS,
)

COHERE_SCOPE_NAME = "opentelemetry.instrumentation.cohere"
COHERE_SPAN_NAMES = {
    "cohere.chat",
    "cohere.completion",
    "cohere.embed",
    "cohere.rerank",
}

_PROMPT_PREFIX = f"{SpanAttributes.LLM_PROMPTS}."
_COMPLETION_PREFIX = f"{SpanAttributes.LLM_COMPLETIONS}."
_FUNCTIONS_ATTR = SpanAttributes.LLM_REQUEST_FUNCTIONS
_FUNCTIONS_PREFIX = f"{_FUNCTIONS_ATTR}."
_TOOL_CALL_PATH = "tool_calls"
_OFF_CONTRACT_ALIASES = {
    "llm.system",
    "model",
    "prompt_tokens",
    "completion_tokens",
    "total_request_tokens",
    "tools",
    "tool_calls",
    "span_tools",
    "has_tool_calls",
    "parallel_tool_calls",
    RESPAN_SPAN_TOOLS,
    RESPAN_SPAN_TOOL_CALLS,
    RESPAN_SPAN_HANDOFFS,
}


def _safe_json_str(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, default=str)


def _parse_json(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return value


def _scope_name(span: ReadableSpan) -> str:
    scope = getattr(span, "instrumentation_scope", None)
    if scope is not None:
        name = getattr(scope, "name", "")
        if name:
            return name
    instrumentation_info = getattr(span, "instrumentation_info", None)
    return getattr(instrumentation_info, "name", "") or ""


def _is_cohere_span(span: ReadableSpan, attrs: dict[str, Any]) -> bool:
    if _scope_name(span) == COHERE_SCOPE_NAME:
        return True
    if getattr(span, "name", "") in COHERE_SPAN_NAMES:
        return True
    system = attrs.get(SpanAttributes.LLM_SYSTEM) or attrs.get("llm.system")
    return isinstance(system, str) and system.lower() == "cohere"


def _request_type_from_span(span: ReadableSpan, attrs: dict[str, Any]) -> str | None:
    request_type = attrs.get(SpanAttributes.LLM_REQUEST_TYPE)
    if isinstance(request_type, str) and request_type:
        return request_type

    span_name = getattr(span, "name", "")
    if span_name == "cohere.chat":
        return LLMRequestTypeValues.CHAT.value
    if span_name == "cohere.completion":
        return LLMRequestTypeValues.COMPLETION.value
    if span_name == "cohere.embed":
        return LLMRequestTypeValues.EMBEDDING.value
    if span_name == "cohere.rerank":
        return LLMRequestTypeValues.RERANK.value
    return None


def _log_type_for_request_type(request_type: str | None) -> str | None:
    if request_type == LLMRequestTypeValues.CHAT.value:
        return LOG_TYPE_CHAT
    if request_type == LLMRequestTypeValues.COMPLETION.value:
        return LOG_TYPE_TEXT
    if request_type == LLMRequestTypeValues.EMBEDDING.value:
        return LOG_TYPE_EMBEDDING
    if request_type == LLMRequestTypeValues.RERANK.value:
        return LOG_TYPE_TASK
    return None


def _set_token_aliases(attrs: dict[str, Any]) -> None:
    prompt_tokens = attrs.get(SpanAttributes.LLM_USAGE_PROMPT_TOKENS)
    input_tokens = attrs.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS)
    if prompt_tokens is None and input_tokens is not None:
        attrs[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] = input_tokens
        prompt_tokens = input_tokens
    elif prompt_tokens is not None:
        attrs.setdefault(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS, prompt_tokens)
        input_tokens = prompt_tokens

    completion_tokens = attrs.get(SpanAttributes.LLM_USAGE_COMPLETION_TOKENS)
    output_tokens = attrs.get(GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS)
    if completion_tokens is None and output_tokens is not None:
        attrs[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS] = output_tokens
        completion_tokens = output_tokens
    elif completion_tokens is not None:
        attrs.setdefault(GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS, completion_tokens)
        output_tokens = completion_tokens

    total_tokens = attrs.get(SpanAttributes.LLM_USAGE_TOTAL_TOKENS)
    if (
        total_tokens is None
        and isinstance(input_tokens, int)
        and isinstance(
            output_tokens,
            int,
        )
    ):
        attrs[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] = input_tokens + output_tokens


def _normalize_tool_definition(raw: dict[str, Any]) -> dict[str, Any]:
    tool: dict[str, Any] = {}
    for field in ("name", "description"):
        if raw.get(field) is not None:
            tool[field] = raw[field]
    if raw.get("parameters") is not None:
        tool["parameters"] = _parse_json(raw["parameters"])
    return tool


def _normalize_indexed_functions(attrs: dict[str, Any]) -> None:
    buckets: dict[int, dict[str, Any]] = defaultdict(dict)
    indexed_keys: list[str] = []
    for key, value in attrs.items():
        if not key.startswith(_FUNCTIONS_PREFIX):
            continue
        rest = key[len(_FUNCTIONS_PREFIX) :]
        parts = rest.split(".", 1)
        if not parts[0].isdigit() or len(parts) == 1:
            continue
        buckets[int(parts[0])][parts[1]] = value
        indexed_keys.append(key)

    if buckets and attrs.get(_FUNCTIONS_ATTR) is None:
        tools = [
            _normalize_tool_definition(buckets[index])
            for index in sorted(buckets)
            if _normalize_tool_definition(buckets[index])
        ]
        if tools:
            attrs[_FUNCTIONS_ATTR] = _safe_json_str(tools)
    elif attrs.get(_FUNCTIONS_ATTR) is not None:
        attrs[_FUNCTIONS_ATTR] = _safe_json_str(attrs[_FUNCTIONS_ATTR])

    for key in indexed_keys:
        attrs.pop(key, None)


def _set_nested_value(target: dict[str, Any], dotted_path: str, value: Any) -> None:
    parts = dotted_path.split(".")
    cursor = target
    for part in parts[:-1]:
        child = cursor.get(part)
        if not isinstance(child, dict):
            child = {}
            cursor[part] = child
        cursor = child
    cursor[parts[-1]] = value


def _cohere_indexed_tool_call(raw: dict[str, Any]) -> dict[str, Any]:
    tool_call: dict[str, Any] = {}
    if raw.get("id") is not None:
        tool_call["id"] = raw["id"]

    function_payload: dict[str, Any] = {}
    function = raw.get("function")
    if isinstance(function, dict):
        if function.get("name") is not None:
            function_payload["name"] = function["name"]
        if function.get("arguments") is not None:
            arguments = function["arguments"]
            function_payload["arguments"] = (
                arguments if isinstance(arguments, str) else _safe_json_str(arguments)
            )

    if raw.get("name") is not None:
        function_payload.setdefault("name", raw["name"])
    if raw.get("arguments") is not None:
        arguments = raw["arguments"]
        function_payload.setdefault(
            "arguments",
            arguments if isinstance(arguments, str) else _safe_json_str(arguments),
        )

    if raw.get("type") is not None:
        tool_call["type"] = raw["type"]
    elif function_payload:
        tool_call["type"] = "function"
    if function_payload:
        tool_call["function"] = function_payload
    return tool_call


def _normalize_indexed_tool_calls(
    attrs: dict[str, Any],
    *,
    message_prefix: str,
) -> None:
    prefix = f"{message_prefix}."
    buckets: dict[int, dict[int, dict[str, Any]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    indexed_keys: list[str] = []

    for key, value in attrs.items():
        if not key.startswith(prefix):
            continue
        rest = key[len(prefix) :]
        parts = rest.split(".", 3)
        if (
            len(parts) < 4
            or not parts[0].isdigit()
            or parts[1] != _TOOL_CALL_PATH
            or not parts[2].isdigit()
        ):
            continue
        _set_nested_value(buckets[int(parts[0])][int(parts[2])], parts[3], value)
        indexed_keys.append(key)

    for message_index, tool_call_buckets in buckets.items():
        target = f"{message_prefix}.{message_index}.{_TOOL_CALL_PATH}"
        if attrs.get(target) is None:
            tool_calls = [
                _cohere_indexed_tool_call(tool_call_buckets[index])
                for index in sorted(tool_call_buckets)
                if _cohere_indexed_tool_call(tool_call_buckets[index])
            ]
            if tool_calls:
                attrs[target] = _safe_json_str(tool_calls)
        else:
            attrs[target] = _safe_json_str(attrs[target])

    for key in indexed_keys:
        attrs.pop(key, None)


def _stringify_structured_canonical_values(attrs: dict[str, Any]) -> None:
    if attrs.get(_FUNCTIONS_ATTR) is not None:
        attrs[_FUNCTIONS_ATTR] = _safe_json_str(attrs[_FUNCTIONS_ATTR])

    for key, value in list(attrs.items()):
        if not (key.startswith(_PROMPT_PREFIX) or key.startswith(_COMPLETION_PREFIX)):
            continue
        if key.endswith(f".{_TOOL_CALL_PATH}"):
            attrs[key] = _safe_json_str(value)
        elif key.endswith(".content") and isinstance(value, (dict, list)):
            attrs[key] = _safe_json_str(value)


def _normalize_cohere_attrs(
    span: ReadableSpan,
    attrs: dict[str, Any],
) -> None:
    request_type = _request_type_from_span(span, attrs)
    if request_type:
        attrs[SpanAttributes.LLM_REQUEST_TYPE] = request_type

    log_type = _log_type_for_request_type(request_type)
    if log_type is not None:
        attrs.setdefault(RESPAN_LOG_TYPE, log_type)

    attrs[SpanAttributes.LLM_SYSTEM] = "cohere"
    _set_token_aliases(attrs)
    _normalize_indexed_functions(attrs)
    _normalize_indexed_tool_calls(attrs, message_prefix=SpanAttributes.LLM_PROMPTS)
    _normalize_indexed_tool_calls(attrs, message_prefix=SpanAttributes.LLM_COMPLETIONS)
    _stringify_structured_canonical_values(attrs)

    attrs.pop(SpanAttributes.TRACELOOP_SPAN_KIND, None)
    for key in _OFF_CONTRACT_ALIASES:
        attrs.pop(key, None)


class CohereSpanProcessor(SpanProcessor):
    """Clean up spans emitted by the Cohere OpenTelemetry instrumentor."""

    def on_start(
        self,
        span: Span,
        parent_context: Context | None = None,
    ) -> None:
        return None

    def on_end(self, span: ReadableSpan) -> None:
        raw_attrs = dict(getattr(span, "attributes", None) or {})
        if not _is_cohere_span(span, raw_attrs):
            return

        attrs = dict(getattr(span, "_attributes", None) or raw_attrs)
        _normalize_cohere_attrs(span, attrs)
        span._attributes = attrs

    def shutdown(self) -> None:
        return None

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


def _is_export_processor(processor: Any) -> bool:
    class_name = processor.__class__.__name__
    module_name = processor.__class__.__module__
    return class_name in {
        "BufferingSpanProcessor",
        "FilteringSpanProcessor",
        "SimpleSpanProcessor",
        "BatchSpanProcessor",
    } or module_name.startswith("respan_tracing.processors")


def insert_span_processor_before_export(
    tracer_provider: Any,
    processor: SpanProcessor,
) -> None:
    active_processor = getattr(tracer_provider, "_active_span_processor", None)
    processors = (
        getattr(active_processor, "_span_processors", None)
        if active_processor is not None
        else None
    )

    if processors is None:
        tracer_provider.add_span_processor(processor)
        return

    if processor in processors:
        return

    insert_index = len(processors)
    for index, existing_processor in enumerate(processors):
        if _is_export_processor(existing_processor):
            insert_index = index
            break

    active_processor._span_processors = (
        *processors[:insert_index],
        processor,
        *processors[insert_index:],
    )


def remove_span_processor(tracer_provider: Any, processor: SpanProcessor) -> None:
    active_processor = getattr(tracer_provider, "_active_span_processor", None)
    processors = (
        getattr(active_processor, "_span_processors", None)
        if active_processor is not None
        else None
    )
    if processors is None:
        return
    active_processor._span_processors = tuple(
        existing_processor
        for existing_processor in processors
        if existing_processor is not processor
    )
