"""Translate Pipecat OpenInference spans into Respan tracing attributes."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

from openinference.semconv.trace import (
    MessageAttributes,
    ToolCallAttributes,
)
from openinference.semconv.trace import (
    SpanAttributes as OISpanAttributes,
)
from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv._incubating.attributes.error_attributes import (
    ERROR_MESSAGE,
    ERROR_TYPE,
)
from opentelemetry.semconv._incubating.attributes.http_attributes import (
    HTTP_RESPONSE_STATUS_CODE,
)
from opentelemetry.semconv_ai import (
    LLMRequestTypeValues,
)
from opentelemetry.semconv_ai import (
    SpanAttributes as TLSpanAttributes,
)
from opentelemetry.trace import Status, StatusCode
from respan_sdk.constants.llm_logging import (
    LOG_TYPE_AGENT,
    LOG_TYPE_CHAT,
    LOG_TYPE_EMBEDDING,
    LOG_TYPE_GUARDRAIL,
    LOG_TYPE_SPEECH,
    LOG_TYPE_TASK,
    LOG_TYPE_TOOL,
    LOG_TYPE_TRANSCRIPTION,
    LOG_TYPE_WORKFLOW,
    LogMethodChoices,
)
from respan_sdk.constants.span_attributes import (
    RESPAN_LOG_METHOD,
    RESPAN_LOG_TYPE,
    RESPAN_SESSION_ID,
)

from respan_instrumentation_pipecat._observer_hooks import (
    RESPAN_PIPECAT_ERROR_STATUS,
    RESPAN_PIPECAT_ERROR_TYPE,
    RESPAN_PIPECAT_LLM_COMPLETED,
)
from respan_instrumentation_pipecat._policy import span_key
from respan_instrumentation_pipecat._runtime import is_content
from respan_instrumentation_pipecat._serialization import (
    json_dumps,
    parse_json,
    safe_text,
)

logger = logging.getLogger(__name__)

_TL_LLM_PROMPTS_PREFIX = f"{TLSpanAttributes.LLM_PROMPTS}."
_TL_LLM_COMPLETIONS_PREFIX = f"{TLSpanAttributes.LLM_COMPLETIONS}."
_GEN_AI_COMPLETION_TOOL_CALLS = f"{GenAIAttributes.GEN_AI_COMPLETION}.0.tool_calls"


def _valid_count(value):
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


_PIPECAT_SERVICE_TYPE = "service.type"
_PIPECAT_TOOLS_DEFINITIONS = "tools.definitions"
_PIPECAT_TOOL_RESULT = "tool.result"
_PIPECAT_SERVICE_TYPE_STT = "stt"
_PIPECAT_SERVICE_TYPE_TTS = "tts"
_PIPECAT_SERVICE_TYPE_LLM = "llm"
_PIPECAT_UNKNOWN_MODEL_VALUES = {None, "", "unknown", "None"}

_OI_INPUT_MESSAGES_PREFIX = f"{OISpanAttributes.LLM_INPUT_MESSAGES}."
_OI_OUTPUT_MESSAGES_PREFIX = f"{OISpanAttributes.LLM_OUTPUT_MESSAGES}."
_OI_TOKEN_COUNT_PREFIX = f"{OISpanAttributes.LLM_TOKEN_COUNT_PROMPT.rsplit('.', 1)[0]}."
_OI_MESSAGE_CONTENT_PREFIX = f"{MessageAttributes.MESSAGE_CONTENT}."
_OI_MESSAGE_TOOL_CALLS_PREFIX = f"{MessageAttributes.MESSAGE_TOOL_CALLS}."
_OI_MESSAGE_FINISH_REASON = "message.finish_reason"
_OI_TOOL_CALL_PREFIX = f"{ToolCallAttributes.TOOL_CALL_ID.rsplit('.', 1)[0]}."

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
}


def _safe_json_str(value: Any) -> str:
    return json_dumps(parse_json(value), complete=True)


def _parse_json(value: Any) -> Any:
    return parse_json(value)


def _set_nested_value(target: dict[str, Any], dotted_path: str, value: Any) -> None:
    cursor = target
    parts = dotted_path.split(".")
    for part in parts[:-1]:
        current = cursor.get(part)
        if not isinstance(current, dict):
            current = {}
            cursor[part] = current
        cursor = current
    cursor[parts[-1]] = value


def _collect_message_buckets(
    attrs: dict[str, Any],
    prefix: str,
) -> dict[int, dict[str, Any]]:
    buckets: dict[int, dict[str, Any]] = defaultdict(dict)
    for key, value in attrs.items():
        if not key.startswith(prefix):
            continue
        rest = key[len(prefix) :]
        parts = rest.split(".", 1)
        if not parts[0].isdigit():
            continue
        buckets[int(parts[0])][parts[1] if len(parts) > 1 else ""] = value
    return buckets


def _extract_message_content(raw: dict[str, Any]) -> Any:
    content = raw.get(MessageAttributes.MESSAGE_CONTENT)
    if content is not None:
        return content

    indexed_content: list[tuple[int, Any]] = []
    for key, value in raw.items():
        if not key.startswith(_OI_MESSAGE_CONTENT_PREFIX):
            continue
        index = key[len(_OI_MESSAGE_CONTENT_PREFIX) :]
        if index.isdigit():
            indexed_content.append((int(index), value))

    if not indexed_content:
        return None

    values = [value for _, value in sorted(indexed_content)]
    if len(values) == 1:
        return values[0]
    if all(isinstance(value, str) for value in values):
        return "\n".join(values)
    return values


def _normalize_tool_call(tool_call: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    if tool_call.get("id") is not None:
        normalized["id"] = tool_call["id"]

    function = tool_call.get("function")
    normalized_function: dict[str, Any] = {}
    if isinstance(function, dict):
        if function.get("name") is not None:
            normalized_function["name"] = function["name"]
        if function.get("arguments") is not None:
            normalized_function["arguments"] = function["arguments"]

    tool_type = tool_call.get("type")
    if tool_type is not None:
        normalized["type"] = tool_type
    elif normalized_function:
        normalized["type"] = "function"

    if normalized_function:
        normalized["function"] = normalized_function
    return normalized


def _tool_call_signature(tool_call: dict[str, Any]) -> str:
    return json_dumps(tool_call, complete=True)


def _extract_tool_calls_from_buckets(
    buckets: dict[int, dict[str, Any]],
) -> list[dict[str, Any]] | None:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()

    for message_index in sorted(buckets):
        raw = buckets[message_index]
        tool_call_buckets: dict[int, dict[str, Any]] = defaultdict(dict)

        for key, value in raw.items():
            if not key.startswith(_OI_MESSAGE_TOOL_CALLS_PREFIX):
                continue
            rest = key[len(_OI_MESSAGE_TOOL_CALLS_PREFIX) :]
            parts = rest.split(".", 1)
            if not parts[0].isdigit() or len(parts) == 1:
                continue
            field = parts[1]
            field = field.removeprefix(_OI_TOOL_CALL_PREFIX)
            tool_call_buckets[int(parts[0])][field] = value

        for tool_call_index in sorted(tool_call_buckets):
            tool_call: dict[str, Any] = {}
            for field, value in tool_call_buckets[tool_call_index].items():
                _set_nested_value(tool_call, field, value)
            normalized = _normalize_tool_call(tool_call)
            if not normalized:
                continue
            signature = _tool_call_signature(normalized)
            if signature not in seen:
                seen.add(signature)
                result.append(normalized)

        legacy_name = raw.get(MessageAttributes.MESSAGE_FUNCTION_CALL_NAME)
        legacy_arguments = raw.get(
            MessageAttributes.MESSAGE_FUNCTION_CALL_ARGUMENTS_JSON
        )
        if legacy_name is None and legacy_arguments is None:
            continue
        legacy_tool_call = {"type": "function", "function": {}}
        if legacy_name is not None:
            legacy_tool_call["function"]["name"] = legacy_name
        if legacy_arguments is not None:
            legacy_tool_call["function"]["arguments"] = legacy_arguments
        normalized = _normalize_tool_call(legacy_tool_call)
        signature = _tool_call_signature(normalized)
        if normalized and signature not in seen:
            seen.add(signature)
            result.append(normalized)

    return result or None


def _messages_to_openllmetry(
    attrs: dict[str, Any],
    source_prefix: str,
    target_prefix: str,
) -> None:
    buckets = _collect_message_buckets(attrs=attrs, prefix=source_prefix)
    for index in sorted(buckets)[:8]:
        raw = buckets[index]
        target = f"{target_prefix}.{index}"

        role = raw.get(MessageAttributes.MESSAGE_ROLE)
        if role:
            attrs[f"{target}.role"] = safe_text(str(role))

        content = _extract_message_content(raw)
        if content is not None:
            attrs[f"{target}.content"] = (
                safe_text(content) if isinstance(content, str) else json_dumps(content)
            )

        tool_calls = _extract_tool_calls_from_buckets({index: raw})
        if tool_calls is not None:
            attrs[f"{target}.tool_calls"] = json_dumps(tool_calls, complete=True)

        finish_reason = raw.get(_OI_MESSAGE_FINISH_REASON)
        if finish_reason:
            attrs[f"{target}.finish_reason"] = safe_text(str(finish_reason))


def _normalize_tools(value: Any) -> list[dict[str, Any]] | None:
    parsed = _parse_json(value)
    if isinstance(parsed, list):
        tools = [tool for tool in parsed if isinstance(tool, dict)]
        return tools or None
    if isinstance(parsed, dict):
        return [parsed]
    return None


def _is_unknown_model(value: Any) -> bool:
    return value in _PIPECAT_UNKNOWN_MODEL_VALUES


def _extract_settings_model(attrs: dict[str, Any]) -> Any:
    settings = _parse_json(attrs.get(OISpanAttributes.METADATA))
    if not isinstance(settings, dict):
        return None
    return settings.get("model")


def _pipecat_log_type(attrs: dict[str, Any], oi_kind_upper: str) -> str:
    service_type = str(attrs.get(_PIPECAT_SERVICE_TYPE, "")).lower()
    if service_type == _PIPECAT_SERVICE_TYPE_STT:
        return LOG_TYPE_TRANSCRIPTION
    if service_type == _PIPECAT_SERVICE_TYPE_TTS:
        return LOG_TYPE_SPEECH
    if service_type == _PIPECAT_SERVICE_TYPE_LLM:
        return LOG_TYPE_CHAT
    return _OI_KIND_TO_LOG_TYPE.get(oi_kind_upper, LOG_TYPE_TASK)


def _has_parent(span: ReadableSpan) -> bool:
    parent = getattr(span, "parent", None)
    return bool(parent and getattr(parent, "span_id", 0))


def _status_message(span: ReadableSpan, attrs: dict[str, Any]) -> str | None:
    direct = attrs.get(ERROR_MESSAGE)
    if isinstance(direct, str) and direct:
        return safe_text(direct)
    status = getattr(span, "status", None)
    description = getattr(status, "description", None)
    if isinstance(description, str) and description:
        return safe_text(description)
    for event in getattr(span, "events", ()) or ():
        event_attrs = getattr(event, "attributes", None) or {}
        message = event_attrs.get("exception.message")
        if isinstance(message, str) and message:
            return safe_text(message)
    return None


def _normalize_status(span, attrs, *, is_chat):
    code = attrs.pop(RESPAN_PIPECAT_ERROR_STATUS, None)
    error_type = attrs.pop(RESPAN_PIPECAT_ERROR_TYPE, None)
    if (
        error_type is None
        and getattr(span.status, "status_code", None) is not StatusCode.ERROR
    ):
        return
    if isinstance(code, int) and not isinstance(code, bool) and 400 <= code <= 599:
        attrs[HTTP_RESPONSE_STATUS_CODE] = code
    if isinstance(error_type, str):
        attrs[ERROR_TYPE] = error_type
    attrs.pop("status_code", None)
    span._status = Status(StatusCode.ERROR, attrs.get(ERROR_MESSAGE))


class PipecatOpenInferenceTranslator(SpanProcessor):
    """SpanProcessor for OpenInference spans emitted by Pipecat."""

    def __init__(self, runtime=None):
        self.runtime = runtime

    def on_start(self, span: Any, parent_context: Any = None) -> None:
        pass

    def on_end(self, span: ReadableSpan) -> None:
        try:
            self._translate(span)
        except Exception:  # noqa: BLE001 - telemetry faults preserve native execution.
            if self.runtime is not None and span_key(span) in self.runtime.spans:
                span._attributes = {
                    k: v for k, v in dict(span._attributes).items() if not is_content(k)
                }
                span._events = ()

    def _translate(self, span: ReadableSpan) -> None:
        if self.runtime is not None and span_key(span) not in self.runtime.spans:
            return
        original_attrs = getattr(span, "_attributes", None)
        if original_attrs is None:
            return

        attrs = dict(original_attrs)
        if (
            self.runtime is not None
            and not self.runtime.spans[span_key(span)].allowed()
        ):
            attrs = {k: v for k, v in attrs.items() if not is_content(k)}
            span._events = ()
            if span.status.status_code is StatusCode.ERROR:
                span._status = Status(StatusCode.ERROR)
        oi_kind = attrs.get(OISpanAttributes.OPENINFERENCE_SPAN_KIND)
        if not oi_kind:
            return

        oi_kind_upper = str(oi_kind).upper()
        log_type = _pipecat_log_type(attrs=attrs, oi_kind_upper=oi_kind_upper)
        is_chat = log_type == LOG_TYPE_CHAT
        logger.debug("[Pipecat OI] Translating %s span: %s", oi_kind_upper, span.name)

        entity_name = safe_text(
            str(attrs.get(OISpanAttributes.AGENT_NAME) or span.name or "pipecat")
        )
        attrs[RESPAN_LOG_METHOD] = LogMethodChoices.TRACING_INTEGRATION.value
        attrs[RESPAN_LOG_TYPE] = log_type
        attrs[TLSpanAttributes.TRACELOOP_ENTITY_NAME] = entity_name
        attrs[TLSpanAttributes.TRACELOOP_ENTITY_PATH] = (
            entity_name if _has_parent(span) else ""
        )

        session_id = attrs.get(OISpanAttributes.SESSION_ID)
        if session_id:
            attrs[RESPAN_SESSION_ID] = safe_text(str(session_id))

        input_value = attrs.get(OISpanAttributes.INPUT_VALUE)
        if input_value is not None:
            attrs[TLSpanAttributes.TRACELOOP_ENTITY_INPUT] = _safe_json_str(input_value)

        output_value = attrs.get(OISpanAttributes.OUTPUT_VALUE)
        if output_value is not None:
            attrs[TLSpanAttributes.TRACELOOP_ENTITY_OUTPUT] = _safe_json_str(
                output_value
            )

        if log_type == LOG_TYPE_TOOL:
            call_id = attrs.get(ToolCallAttributes.TOOL_CALL_ID)
            if isinstance(call_id, str):
                attrs[GenAIAttributes.GEN_AI_TOOL_CALL_ID] = call_id
            tool_input = attrs.get(OISpanAttributes.TOOL_PARAMETERS)
            tool_name = attrs.get(OISpanAttributes.TOOL_NAME)
            if tool_input is not None:
                attrs[TLSpanAttributes.TRACELOOP_ENTITY_INPUT] = json_dumps(
                    {
                        "name": tool_name or span.name,
                        "arguments": parse_json(tool_input),
                    },
                    complete=True,
                )
            tool_output = attrs.get(_PIPECAT_TOOL_RESULT)
            if tool_output is not None:
                attrs[TLSpanAttributes.TRACELOOP_ENTITY_OUTPUT] = _safe_json_str(
                    tool_output
                )
            if tool_name:
                tool_entity = safe_text(f"pipecat.tool.{tool_name}")
                attrs[TLSpanAttributes.TRACELOOP_ENTITY_NAME] = tool_entity
                attrs[TLSpanAttributes.TRACELOOP_ENTITY_PATH] = (
                    tool_entity if _has_parent(span) else ""
                )

        if is_chat:
            self._translate_chat(attrs)

        llm_completed = attrs.pop(RESPAN_PIPECAT_LLM_COMPLETED, False) is True
        if (
            llm_completed
            and not attrs.get(RESPAN_PIPECAT_ERROR_STATUS)
            and not attrs.get("conversation.was_interrupted")
            and attrs.get("conversation.end_reason") not in {"cancelled", "interrupted"}
        ):
            attrs["conversation.end_reason"] = "completed"
            attrs["conversation.was_interrupted"] = False

        _normalize_status(span, attrs, is_chat=is_chat)
        self._remove_raw_attrs(attrs)
        attrs.pop(TLSpanAttributes.TRACELOOP_SPAN_KIND, None)
        for key in list(attrs):
            if key.startswith(("frame.", "service.")):
                attrs.pop(key, None)
            elif (
                isinstance(attrs.get(key), str)
                and key
                not in {
                    TLSpanAttributes.TRACELOOP_ENTITY_INPUT,
                    TLSpanAttributes.TRACELOOP_ENTITY_OUTPUT,
                    TLSpanAttributes.LLM_REQUEST_FUNCTIONS,
                    _GEN_AI_COMPLETION_TOOL_CALLS,
                }
                and not key.endswith(".tool_calls")
            ):
                attrs[key] = safe_text(
                    attrs[key],
                    complete=True,
                )
        if (
            self.runtime is not None
            and not self.runtime.spans[span_key(span)].allowed()
        ):
            attrs = {k: v for k, v in attrs.items() if not is_content(k)}
        span._attributes = attrs

    def _translate_chat(self, attrs: dict[str, Any]) -> None:
        attrs[TLSpanAttributes.LLM_REQUEST_TYPE] = LLMRequestTypeValues.CHAT.value

        model = attrs.get(TLSpanAttributes.LLM_REQUEST_MODEL) or attrs.get(
            OISpanAttributes.LLM_MODEL_NAME
        )
        settings_model = _extract_settings_model(attrs)
        if _is_unknown_model(model) and not _is_unknown_model(settings_model):
            model = settings_model
        if not _is_unknown_model(model):
            attrs[TLSpanAttributes.LLM_REQUEST_MODEL] = safe_text(str(model))

        provider = attrs.get(OISpanAttributes.LLM_PROVIDER)
        if provider:
            provider_name = safe_text(str(provider).lower())
            attrs[GenAIAttributes.GEN_AI_PROVIDER_NAME] = provider_name
            attrs[TLSpanAttributes.LLM_SYSTEM] = provider_name
        system = attrs.get(OISpanAttributes.LLM_SYSTEM)
        if system:
            attrs[TLSpanAttributes.LLM_SYSTEM] = safe_text(str(system).lower())

        prompt_tokens = attrs.get(OISpanAttributes.LLM_TOKEN_COUNT_PROMPT)
        if _valid_count(prompt_tokens):
            attrs[TLSpanAttributes.LLM_USAGE_PROMPT_TOKENS] = prompt_tokens
            attrs[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] = prompt_tokens

        completion_tokens = attrs.get(OISpanAttributes.LLM_TOKEN_COUNT_COMPLETION)
        if _valid_count(completion_tokens):
            attrs[TLSpanAttributes.LLM_USAGE_COMPLETION_TOKENS] = completion_tokens
            attrs[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] = completion_tokens

        for source, target in [
            (
                OISpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ,
                TLSpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS,
            ),
            (
                OISpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE,
                TLSpanAttributes.LLM_USAGE_CACHE_CREATION_INPUT_TOKENS,
            ),
            (
                OISpanAttributes.LLM_TOKEN_COUNT_COMPLETION_DETAILS_REASONING,
                TLSpanAttributes.LLM_USAGE_REASONING_TOKENS,
            ),
        ]:
            value = attrs.get(source)
            if _valid_count(value):
                attrs[target] = value
        total_tokens = attrs.get(OISpanAttributes.LLM_TOKEN_COUNT_TOTAL)
        if _valid_count(total_tokens):
            attrs[TLSpanAttributes.LLM_USAGE_TOTAL_TOKENS] = total_tokens

        _messages_to_openllmetry(
            attrs,
            _OI_INPUT_MESSAGES_PREFIX,
            _TL_LLM_PROMPTS_PREFIX.rstrip("."),
        )
        _messages_to_openllmetry(
            attrs,
            _OI_OUTPUT_MESSAGES_PREFIX,
            _TL_LLM_COMPLETIONS_PREFIX.rstrip("."),
        )

        tool_calls = _extract_tool_calls_from_buckets(
            _collect_message_buckets(attrs, _OI_OUTPUT_MESSAGES_PREFIX)
        )
        if tool_calls is not None:
            attrs[_GEN_AI_COMPLETION_TOOL_CALLS] = json_dumps(tool_calls, complete=True)

        tools = _normalize_tools(attrs.get(OISpanAttributes.LLM_TOOLS))
        if tools is None:
            tools = _normalize_tools(attrs.get(_PIPECAT_TOOLS_DEFINITIONS))
        if tools is not None:
            attrs[TLSpanAttributes.LLM_REQUEST_FUNCTIONS] = json_dumps(
                tools, complete=True, tool_definitions=True
            )

        if attrs.get("stream") is True:
            attrs[TLSpanAttributes.LLM_IS_STREAMING] = True
        invocation_parameters = _parse_json(
            attrs.get(OISpanAttributes.LLM_INVOCATION_PARAMETERS)
        )
        if isinstance(invocation_parameters, dict):
            if invocation_parameters.get("model"):
                attrs[TLSpanAttributes.LLM_REQUEST_MODEL] = safe_text(
                    str(invocation_parameters["model"])
                )
            if invocation_parameters.get("stream") is True:
                attrs[TLSpanAttributes.LLM_IS_STREAMING] = True

    @staticmethod
    def _remove_raw_attrs(attrs: dict[str, Any]) -> None:
        keys_to_remove = {
            OISpanAttributes.OPENINFERENCE_SPAN_KIND,
            OISpanAttributes.INPUT_VALUE,
            OISpanAttributes.OUTPUT_VALUE,
            OISpanAttributes.INPUT_MIME_TYPE,
            OISpanAttributes.OUTPUT_MIME_TYPE,
            OISpanAttributes.LLM_MODEL_NAME,
            OISpanAttributes.LLM_PROVIDER,
            OISpanAttributes.LLM_SYSTEM,
            OISpanAttributes.LLM_INVOCATION_PARAMETERS,
            OISpanAttributes.LLM_TOKEN_COUNT_PROMPT,
            OISpanAttributes.LLM_TOKEN_COUNT_COMPLETION,
            OISpanAttributes.LLM_TOKEN_COUNT_TOTAL,
            OISpanAttributes.LLM_TOOLS,
            OISpanAttributes.METADATA,
            OISpanAttributes.AGENT_NAME,
            OISpanAttributes.SESSION_ID,
            OISpanAttributes.TOOL_NAME,
            OISpanAttributes.TOOL_PARAMETERS,
            OISpanAttributes.TOOL_DESCRIPTION,
            _PIPECAT_TOOLS_DEFINITIONS,
            _PIPECAT_TOOL_RESULT,
            "tools",
            "tool_calls",
            "model",
            "prompt_tokens",
            "completion_tokens",
            "total_request_tokens",
            "span_tools",
            "has_tool_calls",
            "parallel_tool_calls",
            "stream",
            "tool_call.id",
            "tool_call.function.name",
            "tool_call.function.arguments",
        }
        prefixes_to_remove = (
            _OI_INPUT_MESSAGES_PREFIX,
            _OI_OUTPUT_MESSAGES_PREFIX,
            _OI_TOKEN_COUNT_PREFIX,
        )

        for key in keys_to_remove:
            attrs.pop(key, None)

        for key in list(attrs):
            if any(key.startswith(prefix) for prefix in prefixes_to_remove):
                attrs.pop(key, None)

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True
