"""Translate OpenInference spans into Respan's canonical span contract."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from typing import Any

from openinference.semconv.trace import (
    EmbeddingAttributes,
    MessageAttributes,
    ToolAttributes,
)
from openinference.semconv.trace import (
    SpanAttributes as OISpanAttributes,
)
from opentelemetry.sdk.trace import Event, ReadableSpan, SpanProcessor
from opentelemetry.semconv._incubating.attributes import gen_ai_attributes as GenAI
from opentelemetry.semconv._incubating.attributes.error_attributes import ERROR_MESSAGE
from opentelemetry.semconv._incubating.attributes.exception_attributes import (
    EXCEPTION_MESSAGE,
    EXCEPTION_STACKTRACE,
)
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import (
    GEN_AI_PROVIDER_NAME,
    GEN_AI_RESPONSE_MODEL,
    GEN_AI_TOOL_CALL_ID,
    GEN_AI_USAGE_INPUT_TOKENS,
    GEN_AI_USAGE_OUTPUT_TOKENS,
)
from opentelemetry.semconv_ai import (
    LLMRequestTypeValues,
)
from opentelemetry.semconv_ai import (
    SpanAttributes as TLSpanAttributes,
)
from opentelemetry.trace import Status
from respan_sdk.constants.llm_logging import (
    LOG_TYPE_AGENT,
    LOG_TYPE_CHAT,
    LOG_TYPE_EMBEDDING,
    LOG_TYPE_GUARDRAIL,
    LOG_TYPE_TASK,
    LOG_TYPE_TOOL,
    LOG_TYPE_WORKFLOW,
)
from respan_sdk.constants.span_attributes import RESPAN_LOG_TYPE

from respan_instrumentation_openinference._policy import ContentPolicy, clear_content
from respan_instrumentation_openinference._serialization import (
    bounded_json,
    bounded_text,
    complete_json,
    complete_value,
    content_value,
    parse_json,
    to_jsonable,
)

logger = logging.getLogger(__name__)

_OI_KIND_TO_LOG_TYPE = {
    "CHAIN": LOG_TYPE_WORKFLOW,
    "LLM": LOG_TYPE_CHAT,
    "TOOL": LOG_TYPE_TOOL,
    "AGENT": LOG_TYPE_AGENT,
    "RETRIEVER": LOG_TYPE_TASK,
    "EMBEDDING": LOG_TYPE_EMBEDDING,
    "RERANKER": LOG_TYPE_TASK,
    "GUARDRAIL": LOG_TYPE_GUARDRAIL,
    "EVALUATOR": LOG_TYPE_TASK,
    "PROMPT": LOG_TYPE_TASK,
    "UNKNOWN": LOG_TYPE_TASK,
    "DECISION": LOG_TYPE_TASK,
}
_LLM_KINDS = {"LLM", "EMBEDDING"}
_INVOCATION_PARAM_MAP = {
    "model": TLSpanAttributes.LLM_REQUEST_MODEL,
    "temperature": TLSpanAttributes.LLM_REQUEST_TEMPERATURE,
    "top_p": TLSpanAttributes.LLM_REQUEST_TOP_P,
    "max_tokens": TLSpanAttributes.LLM_REQUEST_MAX_TOKENS,
    "max_output_tokens": TLSpanAttributes.LLM_REQUEST_MAX_TOKENS,
    "top_k": TLSpanAttributes.LLM_TOP_K,
    "stop_sequences": TLSpanAttributes.LLM_CHAT_STOP_SEQUENCES,
    "stop": TLSpanAttributes.LLM_CHAT_STOP_SEQUENCES,
    "repetition_penalty": TLSpanAttributes.LLM_REQUEST_REPETITION_PENALTY,
    "frequency_penalty": TLSpanAttributes.LLM_FREQUENCY_PENALTY,
    "presence_penalty": TLSpanAttributes.LLM_PRESENCE_PENALTY,
    "stream": TLSpanAttributes.LLM_IS_STREAMING,
}
_OFF_CONTRACT_ALIAS_KEYS = {
    TLSpanAttributes.TRACELOOP_SPAN_KIND,
    "respan.span.tools",
    "respan.span.tool_calls",
    "respan.span.handoffs",
    "tools",
    "tool_calls",
    "model",
    "prompt_tokens",
    "completion_tokens",
    "total_request_tokens",
    "span_tools",
    "has_tool_calls",
    "parallel_tool_calls",
}


def _collect_buckets(attrs: dict[str, Any], prefix: str) -> dict[int, dict[str, Any]]:
    buckets: dict[int, dict[str, Any]] = defaultdict(dict)
    for key, value in attrs.items():
        if not key.startswith(prefix):
            continue
        rest = key[len(prefix) :]
        index, separator, field = rest.partition(".")
        if separator and index.isdigit():
            buckets[int(index)][field] = value
    return buckets


def _set_nested(target: dict[str, Any], dotted_path: str, value: Any) -> None:
    parts = dotted_path.split(".")
    cursor = target
    for part in parts[:-1]:
        child = cursor.get(part)
        if not isinstance(child, dict):
            child = {}
            cursor[part] = child
        cursor = child
    cursor[parts[-1]] = value


def _signature(value: Any) -> str:
    return json.dumps(
        to_jsonable(value, complete=True),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _normalize_tool_call(tool_call: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    call_id = tool_call.get("id")
    if isinstance(call_id, (str, int)):
        normalized["id"] = complete_value(str(call_id))

    function = tool_call.get("function")
    normalized_function: dict[str, Any] = {}
    if isinstance(function, dict):
        name = function.get("name")
        if isinstance(name, str) and name:
            normalized_function["name"] = bounded_text(name)
        if "arguments" in function:
            normalized_function["arguments"] = complete_json(function["arguments"])

    tool_type = tool_call.get("type")
    if isinstance(tool_type, str) and tool_type:
        normalized["type"] = bounded_text(tool_type)
    elif normalized_function:
        normalized["type"] = "function"
    signature = tool_call.get("reasoning_signature")
    if isinstance(signature, str):
        normalized["reasoning_signature"] = complete_value(signature)
    if normalized_function:
        normalized["function"] = normalized_function
    return normalized


def _extract_tool_calls_from_buckets(
    buckets: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    modern_function_signatures: set[str] = set()
    for index in sorted(buckets):
        raw = buckets[index]
        tool_call_buckets: dict[int, dict[str, Any]] = defaultdict(dict)
        for field, value in raw.items():
            if not field.startswith(f"{MessageAttributes.MESSAGE_TOOL_CALLS}."):
                continue
            rest = field[len(f"{MessageAttributes.MESSAGE_TOOL_CALLS}.") :]
            tool_index, separator, tool_field = rest.partition(".")
            if not separator or not tool_index.isdigit():
                continue
            tool_field = tool_field.removeprefix("tool_call.")
            tool_call_buckets[int(tool_index)][tool_field] = value

        for tool_index in sorted(tool_call_buckets):
            reconstructed: dict[str, Any] = {}
            for field, value in tool_call_buckets[tool_index].items():
                _set_nested(reconstructed, field, value)
            tool_call = _normalize_tool_call(reconstructed)
            signature = _signature(tool_call)
            if tool_call and signature not in seen:
                seen.add(signature)
                result.append(tool_call)
                modern_function_signatures.add(
                    _signature(tool_call.get("function", {}))
                )

        legacy_name = raw.get(MessageAttributes.MESSAGE_FUNCTION_CALL_NAME)
        legacy_arguments = raw.get(
            MessageAttributes.MESSAGE_FUNCTION_CALL_ARGUMENTS_JSON
        )
        if legacy_name is None and legacy_arguments is None:
            continue
        legacy = _normalize_tool_call(
            {
                "type": "function",
                "function": {
                    "name": legacy_name,
                    "arguments": legacy_arguments,
                },
            }
        )
        signature = _signature(legacy)
        function_signature = _signature(legacy.get("function", {}))
        if (
            legacy
            and signature not in seen
            and function_signature not in modern_function_signatures
        ):
            seen.add(signature)
            result.append(legacy)
    return result


def _extract_message_content(raw: dict[str, Any]) -> Any:
    if MessageAttributes.MESSAGE_CONTENT in raw:
        return raw[MessageAttributes.MESSAGE_CONTENT]
    content_buckets = _collect_buckets(raw, f"{MessageAttributes.MESSAGE_CONTENTS}.")
    if content_buckets:
        contents = []
        for index in sorted(content_buckets):
            item = {}
            for field, value in content_buckets[index].items():
                _set_nested(
                    item, field.removeprefix("message_content."), parse_json(value)
                )
            if item:
                contents.append(item)
        return contents or None
    indexed = []
    for field, value in raw.items():
        if not field.startswith(f"{MessageAttributes.MESSAGE_CONTENT}."):
            continue
        index = field[len(f"{MessageAttributes.MESSAGE_CONTENT}.") :]
        if index.isdigit():
            indexed.append((int(index), value))
    values = [value for _, value in sorted(indexed)]
    if len(values) == 1:
        return values[0]
    if values and all(isinstance(value, str) for value in values):
        return "\n".join(values)
    return values or None


def _message_payloads(buckets: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for index in sorted(buckets):
        raw = buckets[index]
        message: dict[str, Any] = {}
        role = raw.get(MessageAttributes.MESSAGE_ROLE)
        if isinstance(role, str):
            message["role"] = bounded_text(role)
        content = _extract_message_content(raw)
        if content is not None:
            message["content"] = parse_json(content)
        tool_calls = _extract_tool_calls_from_buckets({index: raw})
        if tool_calls:
            message["tool_calls"] = tool_calls
        finish_reason = raw.get("message.finish_reason")
        if isinstance(finish_reason, str):
            message["finish_reason"] = bounded_text(finish_reason)
        tool_call_id = raw.get(MessageAttributes.MESSAGE_TOOL_CALL_ID)
        if isinstance(tool_call_id, str):
            message["tool_call_id"] = complete_value(tool_call_id)
        if message:
            messages.append(message)
    return messages


def _messages_to_canonical(
    attrs: dict[str, Any],
    buckets: dict[int, dict[str, Any]],
    target_prefix: str,
) -> None:
    for index in sorted(buckets):
        raw = buckets[index]
        target = f"{target_prefix}{index}"
        role = raw.get(MessageAttributes.MESSAGE_ROLE)
        if isinstance(role, str):
            attrs[f"{target}.role"] = bounded_text(role)
        content = _extract_message_content(raw)
        if content is not None:
            attrs[f"{target}.content"] = content_value(content)
        tool_calls = _extract_tool_calls_from_buckets({index: raw})
        if tool_calls:
            attrs[f"{target}.tool_calls"] = complete_json(tool_calls)
        tool_call_id = raw.get(MessageAttributes.MESSAGE_TOOL_CALL_ID)
        if isinstance(tool_call_id, str):
            attrs[f"{target}.tool_call_id"] = complete_value(tool_call_id)
        finish_reason = raw.get("message.finish_reason")
        if isinstance(finish_reason, str):
            attrs[f"{target}.finish_reason"] = bounded_text(finish_reason)


def _normalize_tools(value: Any) -> list[dict[str, Any]]:
    parsed = parse_json(value)
    candidates = parsed if isinstance(parsed, list) else [parsed]
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        normalized = to_jsonable(candidate, complete=True)
        if not isinstance(normalized, dict):
            continue
        signature = _signature(normalized)
        if signature not in seen:
            seen.add(signature)
            result.append(normalized)
    return result


def _indexed_tools(attrs: dict[str, Any]) -> list[dict[str, Any]]:
    buckets = _collect_buckets(attrs, f"{OISpanAttributes.LLM_TOOLS}.")
    tools: list[dict[str, Any]] = []
    for index in sorted(buckets):
        raw = buckets[index]
        reconstructed: dict[str, Any] = {}
        schema = raw.get(ToolAttributes.TOOL_JSON_SCHEMA)
        if schema is not None:
            parsed_schema = parse_json(schema)
            if isinstance(parsed_schema, dict):
                reconstructed.update(parsed_schema)
            else:
                reconstructed["json_schema"] = parsed_schema
        for field, value in raw.items():
            if field == ToolAttributes.TOOL_JSON_SCHEMA:
                continue
            normalized_field = field.removeprefix("tool.")
            _set_nested(reconstructed, normalized_field, parse_json(value))
        if reconstructed:
            tools.extend(_normalize_tools(reconstructed))
    return _normalize_tools(tools)


def _attribute_value(value: Any) -> Any:
    normalized = to_jsonable(value)
    if isinstance(normalized, str):
        return bounded_text(normalized)
    if normalized is None or isinstance(normalized, (bool, int, float, str)):
        return normalized
    if (
        isinstance(normalized, list)
        and all(isinstance(item, (bool, int, float, str)) for item in normalized)
        and len({type(item) for item in normalized}) <= 1
    ):
        return tuple(normalized)
    return bounded_json(normalized)


def _lower_label(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return bounded_text(value.strip().lower())


class OpenInferenceTranslator(SpanProcessor):
    """Normalize real OpenInference spans before export; preserve native behavior."""

    def __init__(self) -> None:
        self._policy = ContentPolicy()

    def on_start(self, span: Any, parent_context: Any = None) -> None:
        self._policy.start(span, parent_context)

    def on_end(self, span: ReadableSpan) -> None:
        allowed = self._policy.end(span)
        if not isinstance(
            getattr(span, "_attributes", {}).get(
                OISpanAttributes.OPENINFERENCE_SPAN_KIND
            ),
            str,
        ):
            return
        try:
            self._translate(span, allowed)
        except Exception:
            # Telemetry processing must not replace application results/errors.
            logger.debug("OpenInference translation failed", exc_info=True)
            attrs = dict(getattr(span, "_attributes", {}))
            clear_content(attrs)
            self._remove_raw_and_alias_attrs(attrs)
            span._attributes = attrs
        if getattr(span, "_attributes", {}).get(RESPAN_LOG_TYPE) is not None:
            # Preserve status/event identities while excluding secret-shaped
            # diagnostics and private exception messages from exported content.
            status = getattr(span, "status", None)
            if status is not None and status.description:
                span._status = Status(
                    status.status_code,
                    bounded_text(status.description, max_chars=16000)
                    if allowed
                    else None,
                )
            diagnostic_keys = {ERROR_MESSAGE, EXCEPTION_MESSAGE, EXCEPTION_STACKTRACE}
            attributes = dict(span._attributes)
            for key in diagnostic_keys & attributes.keys():
                if not allowed:
                    attributes.pop(key, None)
                else:
                    attributes[key] = bounded_text(attributes[key], max_chars=16000)
            span._attributes = attributes
            events = []
            for event in getattr(span, "events", ()):
                attrs = dict(event.attributes or {})
                for key in tuple(attrs):
                    if key in diagnostic_keys:
                        if not allowed:
                            attrs.pop(key, None)
                        else:
                            attrs[key] = bounded_text(attrs[key], max_chars=16000)
                events.append(Event(event.name, attrs, event.timestamp))
            span._events = tuple(events)

    def _translate(self, span: ReadableSpan, allowed: bool) -> None:
        original_attrs = getattr(span, "_attributes", None)
        if original_attrs is None:
            return
        attrs = dict(original_attrs)
        oi_kind = attrs.get(OISpanAttributes.OPENINFERENCE_SPAN_KIND)
        if not isinstance(oi_kind, str) or not oi_kind:
            return

        kind = oi_kind.upper()
        logger.debug("[OI->Respan] Translating %s span: %s", kind, span.name)
        attrs.setdefault(RESPAN_LOG_TYPE, _OI_KIND_TO_LOG_TYPE.get(kind, LOG_TYPE_TASK))

        entity_name = attrs.get(OISpanAttributes.TOOL_NAME) if kind == "TOOL" else None
        entity_name = entity_name or attrs.get(OISpanAttributes.AGENT_NAME) or span.name
        if isinstance(entity_name, str):
            attrs.setdefault(
                TLSpanAttributes.TRACELOOP_ENTITY_NAME, bounded_text(entity_name)
            )
        canonical_name = attrs.get(TLSpanAttributes.TRACELOOP_ENTITY_NAME)
        default_path = (
            ""
            if getattr(span, "parent", None) is None
            else bounded_text(
                canonical_name if isinstance(canonical_name, str) else span.name
            )
        )
        attrs.setdefault(TLSpanAttributes.TRACELOOP_ENTITY_PATH, default_path)

        if not allowed:
            clear_content(attrs)

        input_value = attrs.get(OISpanAttributes.INPUT_VALUE)
        if input_value is None:
            input_value = attrs.get(TLSpanAttributes.TRACELOOP_ENTITY_INPUT)
        output_value = attrs.get(OISpanAttributes.OUTPUT_VALUE)
        if output_value is None:
            output_value = attrs.get(TLSpanAttributes.TRACELOOP_ENTITY_OUTPUT)
        if kind == "TOOL" and allowed:
            arguments = parse_json(input_value) if input_value is not None else {}
            if isinstance(arguments, dict) and set(arguments) == {"name", "arguments"}:
                tool_input = arguments
            else:
                tool_input = {"name": entity_name, "arguments": arguments}
            attrs[TLSpanAttributes.TRACELOOP_ENTITY_INPUT] = complete_json(tool_input)
        elif input_value is not None:
            attrs[TLSpanAttributes.TRACELOOP_ENTITY_INPUT] = bounded_json(input_value)
        if output_value is not None:
            attrs[TLSpanAttributes.TRACELOOP_ENTITY_OUTPUT] = (
                complete_json(output_value)
                if kind in {"TOOL", "EMBEDDING"}
                else bounded_json(output_value)
            )

        model = (
            attrs.get(getattr(OISpanAttributes, "LLM_REQUEST_MODEL_NAME", None))
            or attrs.get(OISpanAttributes.LLM_MODEL_NAME)
            or attrs.get(OISpanAttributes.EMBEDDING_MODEL_NAME)
        )
        response_model = attrs.get(
            getattr(OISpanAttributes, "LLM_RESPONSE_MODEL_NAME", None)
        )
        if isinstance(response_model, str) and response_model:
            attrs.setdefault(GEN_AI_RESPONSE_MODEL, bounded_text(response_model))
        if kind == "TOOL":
            call_id = attrs.get(OISpanAttributes.TOOL_ID)
            if isinstance(call_id, str) and call_id:
                attrs.setdefault(GEN_AI_TOOL_CALL_ID, complete_value(call_id))
        if isinstance(model, str) and model:
            attrs.setdefault(TLSpanAttributes.LLM_REQUEST_MODEL, bounded_text(model))

        system = _lower_label(attrs.get(OISpanAttributes.LLM_SYSTEM))
        provider = _lower_label(attrs.get(OISpanAttributes.LLM_PROVIDER))
        canonical_system = _lower_label(attrs.get(TLSpanAttributes.LLM_SYSTEM))
        canonical_provider = _lower_label(attrs.get(GEN_AI_PROVIDER_NAME))
        if canonical_system or system or provider or canonical_provider:
            attrs[TLSpanAttributes.LLM_SYSTEM] = (
                canonical_system or system or provider or canonical_provider
            )
        if canonical_provider or provider or system or canonical_system:
            attrs[GEN_AI_PROVIDER_NAME] = (
                canonical_provider or provider or system or canonical_system
            )

        if kind in _LLM_KINDS:
            usage = (
                (
                    OISpanAttributes.LLM_TOKEN_COUNT_PROMPT,
                    (
                        TLSpanAttributes.LLM_USAGE_PROMPT_TOKENS,
                        GEN_AI_USAGE_INPUT_TOKENS,
                    ),
                ),
                (
                    OISpanAttributes.LLM_TOKEN_COUNT_COMPLETION,
                    (
                        TLSpanAttributes.LLM_USAGE_COMPLETION_TOKENS,
                        GEN_AI_USAGE_OUTPUT_TOKENS,
                    ),
                ),
                (
                    OISpanAttributes.LLM_TOKEN_COUNT_TOTAL,
                    (TLSpanAttributes.LLM_USAGE_TOTAL_TOKENS,),
                ),
                (
                    OISpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ,
                    (TLSpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS,),
                ),
                (
                    OISpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE,
                    (TLSpanAttributes.LLM_USAGE_CACHE_CREATION_INPUT_TOKENS,),
                ),
                (
                    OISpanAttributes.LLM_TOKEN_COUNT_COMPLETION_DETAILS_REASONING,
                    (TLSpanAttributes.GEN_AI_USAGE_REASONING_TOKENS,),
                ),
            )
            for source, targets in usage:
                value = attrs.get(source)
                if type(value) is int and value >= 0:
                    for target in targets:
                        attrs.setdefault(target, value)
            if TLSpanAttributes.LLM_USAGE_TOTAL_TOKENS not in attrs:
                prompt = attrs.get(GEN_AI_USAGE_INPUT_TOKENS)
                output = attrs.get(GEN_AI_USAGE_OUTPUT_TOKENS)
                if (
                    type(prompt) is int
                    and type(output) is int
                    and prompt >= 0
                    and output >= 0
                ):
                    attrs[TLSpanAttributes.LLM_USAGE_TOTAL_TOKENS] = prompt + output

        if kind in _LLM_KINDS:
            self._translate_llm(attrs, kind)
        if kind == "EMBEDDING":
            self._translate_embedding(attrs)

        if kind not in _LLM_KINDS:
            # Workflow/agent/task/tool spans carry the common contract only;
            # an inherited model hint is not an additional model invocation.
            llm_keys = {
                TLSpanAttributes.LLM_REQUEST_MODEL,
                TLSpanAttributes.LLM_SYSTEM,
                TLSpanAttributes.LLM_REQUEST_TYPE,
                TLSpanAttributes.LLM_REQUEST_FUNCTIONS,
                GEN_AI_PROVIDER_NAME,
                GEN_AI_RESPONSE_MODEL,
                *_INVOCATION_PARAM_MAP.values(),
            }
            for key in tuple(attrs):
                if key in llm_keys or key.startswith(
                    (
                        f"{TLSpanAttributes.LLM_PROMPTS}.",
                        f"{TLSpanAttributes.LLM_COMPLETIONS}.",
                        "gen_ai.usage.",
                        "llm.usage.",
                    )
                ):
                    attrs.pop(key, None)

        self._normalize_canonical_content(attrs, kind)
        if not allowed:
            clear_content(attrs)
        self._remove_raw_and_alias_attrs(attrs)
        span._attributes = attrs

    def _translate_llm(self, attrs: dict[str, Any], kind: str) -> None:
        attrs.setdefault(
            TLSpanAttributes.LLM_REQUEST_TYPE,
            LLMRequestTypeValues.EMBEDDING.value
            if kind == "EMBEDDING"
            else LLMRequestTypeValues.CHAT.value,
        )
        input_buckets = _collect_buckets(
            attrs, f"{OISpanAttributes.LLM_INPUT_MESSAGES}."
        )
        output_buckets = _collect_buckets(
            attrs, f"{OISpanAttributes.LLM_OUTPUT_MESSAGES}."
        )
        _messages_to_canonical(attrs, input_buckets, f"{TLSpanAttributes.LLM_PROMPTS}.")
        _messages_to_canonical(
            attrs, output_buckets, f"{TLSpanAttributes.LLM_COMPLETIONS}."
        )
        if TLSpanAttributes.TRACELOOP_ENTITY_INPUT not in attrs and input_buckets:
            attrs[TLSpanAttributes.TRACELOOP_ENTITY_INPUT] = bounded_json(
                {"messages": _message_payloads(input_buckets)}
            )
        if TLSpanAttributes.TRACELOOP_ENTITY_OUTPUT not in attrs and output_buckets:
            attrs[TLSpanAttributes.TRACELOOP_ENTITY_OUTPUT] = bounded_json(
                {"messages": _message_payloads(output_buckets)}
            )

        invocation = attrs.get(OISpanAttributes.LLM_INVOCATION_PARAMETERS)
        if kind == "EMBEDDING" and invocation is None:
            invocation = attrs.get(OISpanAttributes.EMBEDDING_INVOCATION_PARAMETERS)
        parameters = parse_json(invocation)
        if isinstance(parameters, dict):
            for key, value in parameters.items():
                target = _INVOCATION_PARAM_MAP.get(key)
                if target:
                    attrs.setdefault(target, _attribute_value(value))

        tools = _normalize_tools(attrs.get(OISpanAttributes.LLM_TOOLS))
        if not tools:
            tools = _indexed_tools(attrs)
        if tools and kind == "LLM":
            attrs.setdefault(
                TLSpanAttributes.LLM_REQUEST_FUNCTIONS, complete_json(tools)
            )

    @staticmethod
    def _translate_embedding(attrs: dict[str, Any]) -> None:
        buckets = _collect_buckets(attrs, f"{OISpanAttributes.EMBEDDING_EMBEDDINGS}.")
        texts: list[Any] = []
        vectors: list[Any] = []
        for index in sorted(buckets):
            raw = buckets[index]
            if EmbeddingAttributes.EMBEDDING_TEXT in raw:
                texts.append(raw[EmbeddingAttributes.EMBEDDING_TEXT])
            if EmbeddingAttributes.EMBEDDING_VECTOR in raw:
                vectors.append(raw[EmbeddingAttributes.EMBEDDING_VECTOR])
        if texts and TLSpanAttributes.TRACELOOP_ENTITY_INPUT not in attrs:
            attrs[TLSpanAttributes.TRACELOOP_ENTITY_INPUT] = bounded_json(
                texts[0] if len(texts) == 1 else texts
            )
        if vectors and TLSpanAttributes.TRACELOOP_ENTITY_OUTPUT not in attrs:
            attrs[TLSpanAttributes.TRACELOOP_ENTITY_OUTPUT] = complete_json(
                vectors[0] if len(vectors) == 1 else vectors
            )

    @staticmethod
    def _normalize_canonical_content(attrs: dict[str, Any], kind: str) -> None:
        label_keys = {
            TLSpanAttributes.TRACELOOP_ENTITY_NAME,
            TLSpanAttributes.TRACELOOP_ENTITY_PATH,
            TLSpanAttributes.LLM_SYSTEM,
            GEN_AI_PROVIDER_NAME,
            GEN_AI_RESPONSE_MODEL,
            TLSpanAttributes.LLM_REQUEST_MODEL,
            TLSpanAttributes.LLM_REQUEST_TYPE,
        }
        for key, value in tuple(attrs.items()):
            if (
                key
                in {
                    TLSpanAttributes.TRACELOOP_ENTITY_INPUT,
                    TLSpanAttributes.TRACELOOP_ENTITY_OUTPUT,
                }
                or key == TLSpanAttributes.LLM_REQUEST_FUNCTIONS
                or (
                    key.startswith(
                        (
                            f"{TLSpanAttributes.LLM_PROMPTS}.",
                            f"{TLSpanAttributes.LLM_COMPLETIONS}.",
                        )
                    )
                    and key.endswith(".tool_calls")
                )
            ):
                attrs[key] = (
                    complete_json(value)
                    if key == TLSpanAttributes.LLM_REQUEST_FUNCTIONS
                    or key.endswith(".tool_calls")
                    or (
                        kind == "TOOL"
                        and key
                        in {
                            TLSpanAttributes.TRACELOOP_ENTITY_INPUT,
                            TLSpanAttributes.TRACELOOP_ENTITY_OUTPUT,
                        }
                    )
                    or (
                        kind == "EMBEDDING"
                        and key == TLSpanAttributes.TRACELOOP_ENTITY_OUTPUT
                    )
                    else bounded_json(value)
                )
            elif key.startswith(
                (
                    f"{TLSpanAttributes.LLM_PROMPTS}.",
                    f"{TLSpanAttributes.LLM_COMPLETIONS}.",
                )
            ) and key.endswith(".content"):
                attrs[key] = content_value(value)
            elif key in label_keys or (
                key.startswith(
                    (
                        f"{TLSpanAttributes.LLM_PROMPTS}.",
                        f"{TLSpanAttributes.LLM_COMPLETIONS}.",
                    )
                )
                and key.endswith((".role", ".finish_reason"))
            ):
                attrs[key] = bounded_text(value)

    @staticmethod
    def _remove_raw_and_alias_attrs(attrs: dict[str, Any]) -> None:
        exact_raw_keys = {
            OISpanAttributes.OPENINFERENCE_SPAN_KIND,
            OISpanAttributes.INPUT_VALUE,
            OISpanAttributes.INPUT_MIME_TYPE,
            OISpanAttributes.OUTPUT_VALUE,
            OISpanAttributes.OUTPUT_MIME_TYPE,
            OISpanAttributes.LLM_MODEL_NAME,
            getattr(OISpanAttributes, "LLM_REQUEST_MODEL_NAME", None),
            getattr(OISpanAttributes, "LLM_RESPONSE_MODEL_NAME", None),
            OISpanAttributes.LLM_PROVIDER,
            OISpanAttributes.LLM_SYSTEM,
            OISpanAttributes.LLM_INVOCATION_PARAMETERS,
            OISpanAttributes.LLM_TOKEN_COUNT_PROMPT,
            OISpanAttributes.LLM_TOKEN_COUNT_COMPLETION,
            OISpanAttributes.LLM_TOKEN_COUNT_TOTAL,
            OISpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ,
            OISpanAttributes.LLM_TOOLS,
            OISpanAttributes.AGENT_NAME,
            OISpanAttributes.EMBEDDING_MODEL_NAME,
            OISpanAttributes.EMBEDDING_INVOCATION_PARAMETERS,
            OISpanAttributes.TOOL_NAME,
            *_OFF_CONTRACT_ALIAS_KEYS,
        }
        raw_prefixes = (
            f"{OISpanAttributes.LLM_INPUT_MESSAGES}.",
            f"{OISpanAttributes.LLM_OUTPUT_MESSAGES}.",
            "llm.token_count.",
            f"{OISpanAttributes.LLM_TOOLS}.",
            f"{OISpanAttributes.EMBEDDING_EMBEDDINGS}.",
            "openinference.",
            "llm.cost.",
            "llm.choices",
            "llm.function_call",
            "llm.prompt",
            "tool.",
        )
        exact_raw_keys.update(
            getattr(GenAI, name, None)
            for name in (
                "GEN_AI_INPUT_MESSAGES",
                "GEN_AI_OUTPUT_MESSAGES",
                "GEN_AI_TOOL_DEFINITIONS",
                "GEN_AI_TOOL_CALL_ARGUMENTS",
                "GEN_AI_TOOL_CALL_RESULT",
            )
        )
        for key in exact_raw_keys:
            attrs.pop(key, None)
        for key in tuple(attrs):
            if key.startswith(raw_prefixes) or (
                key.startswith("gen_ai.tool.") and key != GEN_AI_TOOL_CALL_ID
            ):
                attrs.pop(key, None)

    def shutdown(self) -> None:
        self._policy.clear()

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        del timeout_millis
        return True
