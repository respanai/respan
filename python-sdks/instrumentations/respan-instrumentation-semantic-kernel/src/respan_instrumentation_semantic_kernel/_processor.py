"""Semantic Kernel span normalization for the Respan OTLP pipeline."""

from __future__ import annotations

import json
import logging
import re
from ast import literal_eval
from collections.abc import Callable, Mapping
from typing import Any

from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor
from opentelemetry.semconv._incubating.attributes import (
    error_attributes as incubating_error,
)
from opentelemetry.semconv._incubating.attributes import event_attributes
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv.attributes import error_attributes, exception_attributes
from opentelemetry.semconv_ai import LLMRequestTypeValues, SpanAttributes
from respan_sdk.constants.llm_logging import (
    LOG_TYPE_AGENT,
    LOG_TYPE_CHAT,
    LOG_TYPE_EMBEDDING,
    LOG_TYPE_TASK,
    LOG_TYPE_TEXT,
    LOG_TYPE_TOOL,
    LogMethodChoices,
)
from respan_sdk.constants.span_attributes import (
    RESPAN_LOG_METHOD,
    RESPAN_LOG_TYPE,
    RESPAN_SPAN_HANDOFFS,
    RESPAN_SPAN_TOOL_CALLS,
    RESPAN_SPAN_TOOLS,
)
from semantic_kernel.utils.telemetry.agent_diagnostics import (
    gen_ai_attributes as SKAgentAttributes,
)

from respan_instrumentation_semantic_kernel._constants import (
    SEMANTIC_KERNEL_SCOPE_PREFIX,
    SK_ASSISTANT_MESSAGE_EVENT,
    SK_AUTO_FUNCTION_INVOCATION_SPAN_NAME,
    SK_AVAILABLE_FUNCTIONS_ATTR,
    SK_CHAT_COMPLETION_OPERATION,
    SK_CHAT_MESSAGE_INDEX_ATTR,
    SK_CHAT_OPERATION,
    SK_CHAT_STREAMING_COMPLETION_OPERATION,
    SK_CHOICE_EVENT,
    SK_CONTENT_CAPTURE_ATTRIBUTE,
    SK_CONTENT_COMPLETION_EVENT,
    SK_CONTENT_PROMPT_EVENT,
    SK_PROMPT_EVENT,
    SK_RESPONSE_COMPLETION_TOKENS_ATTR,
    SK_RESPONSE_PROMPT_TOKENS_ATTR,
    SK_SYSTEM_MESSAGE_EVENT,
    SK_TEXT_COMPLETION_OPERATION,
    SK_TEXT_STREAMING_COMPLETION_OPERATION,
    SK_TOOL_MESSAGE_EVENT,
    SK_TOOL_OPERATION,
    SK_USER_MESSAGE_EVENT,
)
from respan_instrumentation_semantic_kernel._native import (
    capture_content as content_allowed,
)
from respan_instrumentation_semantic_kernel._serialization import (
    json_string,
    redact_text,
)

logger = logging.getLogger(__name__)

_CHAT_OPERATIONS = {
    SK_CHAT_OPERATION,
    SK_CHAT_COMPLETION_OPERATION,
    SK_CHAT_STREAMING_COMPLETION_OPERATION,
}
_TEXT_OPERATIONS = {
    SK_TEXT_COMPLETION_OPERATION,
    SK_TEXT_STREAMING_COMPLETION_OPERATION,
}
_PROMPT_MESSAGE_EVENTS = {
    SK_SYSTEM_MESSAGE_EVENT: "system",
    SK_USER_MESSAGE_EVENT: "user",
    SK_ASSISTANT_MESSAGE_EVENT: "assistant",
    SK_TOOL_MESSAGE_EVENT: "tool",
}
_OFF_CONTRACT_ALIASES = {
    RESPAN_SPAN_TOOLS,
    RESPAN_SPAN_TOOL_CALLS,
    RESPAN_SPAN_HANDOFFS,
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
_GENERATED_PROMPT_NAME = re.compile(r"^[A-Za-z0-9_-]{12,32}$")


def _safe_json_loads(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return value


def _json_string(value: Any) -> str | None:
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (TypeError, json.JSONDecodeError):
            return redact_text(value)
        return json_string(parsed)
    return json_string(value)


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value >= 0:
        return value
    return None


def _scope_name(span: ReadableSpan) -> str:
    scope = getattr(span, "instrumentation_scope", None)
    if scope is not None:
        name = getattr(scope, "name", "")
        if name:
            return name
    instrumentation_info = getattr(span, "instrumentation_info", None)
    return getattr(instrumentation_info, "name", "") or ""


def _span_attrs(span: Any) -> dict[str, Any]:
    attrs = getattr(span, "_attributes", None)
    if attrs is not None:
        return dict(attrs)
    return dict(getattr(span, "attributes", None) or {})


def _is_semantic_kernel_span(span: ReadableSpan) -> bool:
    name = _scope_name(span)
    return name == SEMANTIC_KERNEL_SCOPE_PREFIX or name.startswith(
        SEMANTIC_KERNEL_SCOPE_PREFIX + "."
    )


def _normalize_role(role: Any, fallback: str = "user") -> str:
    if hasattr(role, "value"):
        role = role.value
    if not isinstance(role, str) or not role:
        return fallback
    normalized = role.lower()
    if normalized == "model":
        return "assistant"
    return normalized


def _message_content(message: Mapping[str, Any]) -> Any:
    content = message.get("content")
    if content is not None:
        if message.get("tool_call_id") is not None:
            return {"result": content, "tool_call_id": message["tool_call_id"]}
        return content

    items = message.get("items")
    if not isinstance(items, list):
        return None

    text_parts: list[str] = []
    for item in items:
        if not isinstance(item, Mapping):
            continue
        text = item.get("text") or item.get("content")
        if text is not None:
            text_parts.append(redact_text(text) if isinstance(text, str) else "")
    return "\n".join(text_parts) if text_parts else None


def _message_tool_calls(message: Mapping[str, Any]) -> Any:
    tool_calls = message.get("tool_calls")
    if tool_calls is not None:
        return tool_calls

    items = message.get("items")
    if not isinstance(items, list):
        return None

    calls: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, Mapping):
            continue
        if "result" in item:
            continue
        name = item.get("name")
        arguments = item.get("arguments")
        if name is None and arguments is None:
            continue
        function_payload: dict[str, Any] = {}
        if name is not None:
            function_payload["name"] = name
        if arguments is not None:
            function_payload["arguments"] = arguments
        call: dict[str, Any] = {"type": "function", "function": function_payload}
        if item.get("id") is not None:
            call["id"] = item["id"]
        calls.append(call)
    return calls or None


def _set_message_attrs(
    set_attr: Callable[[str, Any], None],
    *,
    prefix: str,
    index: int,
    message: Mapping[str, Any],
    fallback_role: str,
) -> None:
    target = f"{prefix}.{index}"
    set_attr(f"{target}.role", _normalize_role(message.get("role"), fallback_role))

    content = _message_content(message)
    if content is not None:
        set_attr(f"{target}.content", _json_string(content))

    tool_calls = _message_tool_calls(message)
    if tool_calls is not None:
        set_attr(f"{target}.tool_calls", _json_string(tool_calls))


def _promote_messages(
    attrs: dict[str, Any],
    *,
    prefix: str,
    messages: Any,
    fallback_role: str,
) -> None:
    parsed = _safe_json_loads(messages)
    if isinstance(parsed, Mapping) and isinstance(parsed.get("message"), Mapping):
        parsed = parsed["message"]
    if isinstance(parsed, Mapping):
        parsed = [parsed]
    if not isinstance(parsed, list):
        return

    for index, item in enumerate(parsed):
        if isinstance(item, Mapping) and isinstance(item.get("message"), Mapping):
            item = item["message"]
        if not isinstance(item, Mapping):
            continue
        _set_message_attrs(
            attrs.__setitem__,
            prefix=prefix,
            index=index,
            message=item,
            fallback_role=fallback_role,
        )


def _promote_legacy_event_payloads(span: ReadableSpan, attrs: dict[str, Any]) -> None:
    for event in getattr(span, "events", ()) or ():
        event_attrs = dict(getattr(event, "attributes", None) or {})
        event_name = getattr(event, "name", "")
        if (
            event_name == SK_CONTENT_PROMPT_EVENT
            and GenAIAttributes.GEN_AI_PROMPT in event_attrs
        ):
            _promote_messages(
                attrs,
                prefix=SpanAttributes.LLM_PROMPTS,
                messages=event_attrs[GenAIAttributes.GEN_AI_PROMPT],
                fallback_role="user",
            )
        if (
            event_name == SK_CONTENT_COMPLETION_EVENT
            and GenAIAttributes.GEN_AI_COMPLETION in event_attrs
        ):
            _promote_messages(
                attrs,
                prefix=SpanAttributes.LLM_COMPLETIONS,
                messages=event_attrs[GenAIAttributes.GEN_AI_COMPLETION],
                fallback_role="assistant",
            )


def _collect_indexed_messages(
    attrs: Mapping[str, Any], prefix: str
) -> list[dict[str, Any]]:
    buckets: dict[int, dict[str, Any]] = {}
    indexed_prefix = f"{prefix}."
    for key, value in attrs.items():
        if not key.startswith(indexed_prefix):
            continue
        rest = key[len(indexed_prefix) :]
        index_text, _, field = rest.partition(".")
        if not index_text.isdigit() or not field:
            continue
        buckets.setdefault(int(index_text), {})[field] = value

    messages: list[dict[str, Any]] = []
    for index in sorted(buckets):
        raw = buckets[index]
        message: dict[str, Any] = {}
        if raw.get("role") is not None:
            message["role"] = raw["role"]
        if raw.get("content") is not None:
            message["content"] = raw["content"]
        if raw.get("tool_calls") is not None:
            message["tool_calls"] = _safe_json_loads(raw["tool_calls"])
        if message:
            messages.append(message)
    return messages


def _normalize_available_functions(value: Any) -> list[dict[str, Any]] | None:
    parsed = _safe_json_loads(value)
    if isinstance(parsed, str):
        names = [name.strip() for name in parsed.split(",") if name.strip()]
    elif isinstance(parsed, list):
        names = [
            redact_text(name, limit=256)
            for name in parsed
            if isinstance(name, str) and name
        ]
    else:
        return None

    return [
        {
            "type": "function",
            "function": {"name": name},
        }
        for name in names
    ] or None


def _map_usage(attrs: dict[str, Any]) -> None:
    input_tokens = next(
        (
            value
            for key in (
                GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS,
                SK_RESPONSE_PROMPT_TOKENS_ATTR,
                SpanAttributes.LLM_USAGE_PROMPT_TOKENS,
            )
            if (value := _as_int(attrs.get(key))) is not None
        ),
        None,
    )
    output_tokens = next(
        (
            value
            for key in (
                GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS,
                SK_RESPONSE_COMPLETION_TOKENS_ATTR,
                SpanAttributes.LLM_USAGE_COMPLETION_TOKENS,
            )
            if (value := _as_int(attrs.get(key))) is not None
        ),
        None,
    )

    if input_tokens is not None:
        attrs[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] = input_tokens
        attrs[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] = input_tokens
    if output_tokens is not None:
        attrs[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] = output_tokens
        attrs[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS] = output_tokens
    if (
        input_tokens is not None
        and output_tokens is not None
        and SpanAttributes.LLM_USAGE_TOTAL_TOKENS not in attrs
    ):
        attrs[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] = input_tokens + output_tokens


def _map_completion_span(
    span: ReadableSpan,
    attrs: dict[str, Any],
    *,
    log_type: str,
    request_type: str,
) -> None:
    if GenAIAttributes.GEN_AI_PROMPT in attrs:
        _promote_messages(
            attrs,
            prefix=SpanAttributes.LLM_PROMPTS,
            messages=attrs[GenAIAttributes.GEN_AI_PROMPT],
            fallback_role="user",
        )
    if GenAIAttributes.GEN_AI_COMPLETION in attrs:
        _promote_messages(
            attrs,
            prefix=SpanAttributes.LLM_COMPLETIONS,
            messages=attrs[GenAIAttributes.GEN_AI_COMPLETION],
            fallback_role="assistant",
        )
    _promote_legacy_event_payloads(span, attrs)

    attrs[RESPAN_LOG_TYPE] = log_type
    attrs[RESPAN_LOG_METHOD] = LogMethodChoices.TRACING_INTEGRATION.value
    attrs[SpanAttributes.LLM_REQUEST_TYPE] = request_type
    entity_name = attrs.setdefault(
        SpanAttributes.TRACELOOP_ENTITY_NAME,
        redact_text(getattr(span, "name", ""), limit=256),
    )
    attrs.setdefault(
        SpanAttributes.TRACELOOP_ENTITY_PATH,
        "" if getattr(span, "parent", None) is None else entity_name,
    )
    _map_usage(attrs)

    prompt_messages = _collect_indexed_messages(attrs, SpanAttributes.LLM_PROMPTS)
    if prompt_messages:
        attrs.setdefault(
            SpanAttributes.TRACELOOP_ENTITY_INPUT, _json_string(prompt_messages)
        )

    completion_messages = _collect_indexed_messages(
        attrs, SpanAttributes.LLM_COMPLETIONS
    )
    if completion_messages:
        attrs.setdefault(
            SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
            _json_string(completion_messages),
        )

    available_functions = _normalize_available_functions(
        attrs.get(SK_AVAILABLE_FUNCTIONS_ATTR)
    )
    if available_functions is not None:
        attrs.setdefault(
            SpanAttributes.LLM_REQUEST_FUNCTIONS,
            _json_string(available_functions),
        )


def _tool_name_from_span(span: ReadableSpan, attrs: Mapping[str, Any]) -> str:
    value = attrs.get(GenAIAttributes.GEN_AI_TOOL_NAME)
    if isinstance(value, str) and value:
        return value
    name = getattr(span, "name", "")
    prefix = f"{SK_TOOL_OPERATION} "
    if isinstance(name, str) and name.startswith(prefix):
        return name[len(prefix) :]
    return str(name or SK_TOOL_OPERATION)


def _map_tool_span(span: ReadableSpan, attrs: dict[str, Any]) -> None:
    tool_name = _tool_name_from_span(span, attrs)
    attrs[RESPAN_LOG_TYPE] = LOG_TYPE_TOOL
    attrs[RESPAN_LOG_METHOD] = LogMethodChoices.TRACING_INTEGRATION.value
    attrs[SpanAttributes.TRACELOOP_ENTITY_NAME] = tool_name
    attrs[SpanAttributes.TRACELOOP_ENTITY_PATH] = (
        "" if getattr(span, "parent", None) is None else tool_name
    )

    arguments = attrs.get(GenAIAttributes.GEN_AI_TOOL_CALL_ARGUMENTS)
    if arguments is not None:
        parsed_arguments = _safe_json_loads(arguments)
        attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = _json_string(
            {"name": tool_name, "arguments": parsed_arguments}
        )

    result = attrs.get(GenAIAttributes.GEN_AI_TOOL_CALL_RESULT)
    if result is not None:
        attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = json_string(
            _safe_json_loads(result)
        )

    for key in (
        GenAIAttributes.GEN_AI_TOOL_CALL_ARGUMENTS,
        GenAIAttributes.GEN_AI_TOOL_CALL_RESULT,
        GenAIAttributes.GEN_AI_TOOL_DESCRIPTION,
        GenAIAttributes.GEN_AI_TOOL_NAME,
        GenAIAttributes.GEN_AI_TOOL_TYPE,
    ):
        attrs.pop(key, None)


def _is_generated_prompt_wrapper(tool_name: str) -> bool:
    return "-" not in tool_name and bool(_GENERATED_PROMPT_NAME.fullmatch(tool_name))


def _prompt_result_content(result: Any) -> str | None:
    if not isinstance(result, str):
        return None
    if "ChatMessageContent(" not in result:
        return redact_text(result)
    match = re.search(
        r"\bcontent=(?P<literal>'(?:\\.|[^'])*'|\"(?:\\.|[^\"])*\")", result
    )
    if match is None:
        return None
    try:
        value = literal_eval(match.group("literal"))
    except (SyntaxError, ValueError):
        return None
    return redact_text(value) if isinstance(value, str) else None


def _map_prompt_wrapper(span: ReadableSpan, attrs: dict[str, Any]) -> None:
    attrs[RESPAN_LOG_TYPE] = LOG_TYPE_TASK
    attrs[RESPAN_LOG_METHOD] = LogMethodChoices.TRACING_INTEGRATION.value
    attrs[SpanAttributes.TRACELOOP_ENTITY_NAME] = "semantic_kernel.prompt"
    attrs[SpanAttributes.TRACELOOP_ENTITY_PATH] = (
        "" if getattr(span, "parent", None) is None else "semantic_kernel.prompt"
    )
    arguments = _safe_json_loads(attrs.get(GenAIAttributes.GEN_AI_TOOL_CALL_ARGUMENTS))
    attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = _json_string(
        {"operation": "prompt", "arguments": arguments or {}}
    )
    result = attrs.get(GenAIAttributes.GEN_AI_TOOL_CALL_RESULT)
    attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = _json_string(
        {
            "type": "prompt_result",
            "content": _prompt_result_content(result),
        }
    )
    for key in (
        GenAIAttributes.GEN_AI_TOOL_CALL_ARGUMENTS,
        GenAIAttributes.GEN_AI_TOOL_CALL_ID,
        GenAIAttributes.GEN_AI_TOOL_CALL_RESULT,
        GenAIAttributes.GEN_AI_TOOL_DESCRIPTION,
        GenAIAttributes.GEN_AI_TOOL_NAME,
        GenAIAttributes.GEN_AI_TOOL_TYPE,
    ):
        attrs.pop(key, None)


def _map_error(span: ReadableSpan, attrs: dict[str, Any]) -> None:
    status = getattr(span, "status", None)
    status_code = getattr(status, "status_code", None)
    if getattr(status_code, "name", "") != "ERROR":
        return
    error_type = "SemanticKernelError"
    error_message = getattr(status, "description", None)
    for event in getattr(span, "events", ()) or ():
        event_attrs = dict(getattr(event, "attributes", None) or {})
        event_type = event_attrs.get(exception_attributes.EXCEPTION_TYPE)
        event_message = event_attrs.get(exception_attributes.EXCEPTION_MESSAGE)
        if isinstance(event_type, str) and event_type:
            error_type = redact_text(event_type, limit=256)
        if isinstance(event_message, str) and event_message:
            error_message = event_message
            break
    attrs[error_attributes.ERROR_TYPE] = error_type
    attrs[incubating_error.ERROR_MESSAGE] = redact_text(error_message or error_type)
    if attrs.get(RESPAN_LOG_TYPE) not in {
        LOG_TYPE_CHAT,
        LOG_TYPE_TEXT,
        LOG_TYPE_EMBEDDING,
    }:
        attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = _json_string(
            {"error": {"type": error_type, "message": error_message or error_type}}
        )


def _cleanup_attrs(attrs: dict[str, Any]) -> None:
    attrs.pop(SpanAttributes.TRACELOOP_SPAN_KIND, None)
    attrs.pop(GenAIAttributes.GEN_AI_PROMPT, None)
    attrs.pop(GenAIAttributes.GEN_AI_COMPLETION, None)
    attrs.pop(SK_RESPONSE_PROMPT_TOKENS_ATTR, None)
    attrs.pop(SK_RESPONSE_COMPLETION_TOKENS_ATTR, None)
    attrs.pop(SK_AVAILABLE_FUNCTIONS_ATTR, None)
    for key in _OFF_CONTRACT_ALIASES:
        attrs.pop(key, None)


def _sanitize_contract_attrs(attrs: dict[str, Any]) -> None:
    for key in (
        SpanAttributes.LLM_REQUEST_MODEL,
        SpanAttributes.LLM_SYSTEM,
        GenAIAttributes.GEN_AI_PROVIDER_NAME,
        SpanAttributes.TRACELOOP_ENTITY_NAME,
        SpanAttributes.TRACELOOP_ENTITY_PATH,
    ):
        value = attrs.get(key)
        if isinstance(value, str):
            attrs[key] = redact_text(value, limit=256)


def _strip_content(attrs: dict[str, Any]) -> None:
    for key in (
        SK_AVAILABLE_FUNCTIONS_ATTR,
        GenAIAttributes.GEN_AI_TOOL_CALL_ARGUMENTS,
        GenAIAttributes.GEN_AI_TOOL_CALL_RESULT,
        SpanAttributes.LLM_REQUEST_FUNCTIONS,
        SpanAttributes.TRACELOOP_ENTITY_INPUT,
        SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
    ):
        attrs.pop(key, None)
    for key in list(attrs):
        if key.startswith(
            (f"{SpanAttributes.LLM_PROMPTS}.", f"{SpanAttributes.LLM_COMPLETIONS}.")
        ):
            attrs.pop(key, None)


def enrich_semantic_kernel_span(
    span: ReadableSpan, *, capture_content: bool = True
) -> bool:
    """Normalize a Semantic Kernel span in place.

    Returns ``True`` when the span belonged to Semantic Kernel and was handled.
    """

    if not _is_semantic_kernel_span(span):
        return False

    attrs = _span_attrs(span)
    operation = attrs.get(GenAIAttributes.GEN_AI_OPERATION_NAME)

    if operation in _CHAT_OPERATIONS:
        _map_completion_span(
            span,
            attrs,
            log_type=LOG_TYPE_CHAT,
            request_type=LLMRequestTypeValues.CHAT.value,
        )
    elif operation in _TEXT_OPERATIONS:
        _map_completion_span(
            span,
            attrs,
            log_type=LOG_TYPE_TEXT,
            request_type=LLMRequestTypeValues.COMPLETION.value,
        )
    elif operation == "embeddings":
        attrs[RESPAN_LOG_TYPE] = LOG_TYPE_EMBEDDING
        attrs[SpanAttributes.LLM_REQUEST_TYPE] = LLMRequestTypeValues.EMBEDDING.value
        attrs.setdefault(
            SpanAttributes.TRACELOOP_ENTITY_NAME, "semantic_kernel.embeddings"
        )
        attrs.setdefault(SpanAttributes.TRACELOOP_ENTITY_PATH, "")
        _map_usage(attrs)
    elif operation == "invoke_agent":
        attrs[RESPAN_LOG_TYPE] = LOG_TYPE_AGENT
        attrs[SpanAttributes.TRACELOOP_ENTITY_NAME] = attrs.get(
            GenAIAttributes.GEN_AI_AGENT_NAME, "agent"
        )
        attrs.setdefault(SpanAttributes.TRACELOOP_ENTITY_PATH, "")
        for source, destination in (
            (
                GenAIAttributes.GEN_AI_INPUT_MESSAGES,
                SpanAttributes.TRACELOOP_ENTITY_INPUT,
            ),
            (
                GenAIAttributes.GEN_AI_OUTPUT_MESSAGES,
                SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
            ),
            (
                getattr(SKAgentAttributes, "AGENT_INVOCATION_INPUT", None),
                SpanAttributes.TRACELOOP_ENTITY_INPUT,
            ),
            (
                getattr(SKAgentAttributes, "AGENT_INVOCATION_OUTPUT", None),
                SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
            ),
        ):
            if source in attrs:
                attrs[destination] = _json_string(_safe_json_loads(attrs.pop(source)))
        attrs.pop(GenAIAttributes.GEN_AI_TOOL_DEFINITIONS, None)
    elif getattr(span, "name", None) == SK_AUTO_FUNCTION_INVOCATION_SPAN_NAME:
        attrs[RESPAN_LOG_TYPE] = LOG_TYPE_TASK
        attrs[SpanAttributes.TRACELOOP_ENTITY_NAME] = (
            "semantic_kernel.auto_function_invocation"
        )
        attrs.setdefault(SpanAttributes.TRACELOOP_ENTITY_PATH, "")
    elif operation == SK_TOOL_OPERATION:
        tool_name = _tool_name_from_span(span, attrs)
        if attrs.get(RESPAN_LOG_TYPE) == LOG_TYPE_TASK or (
            RESPAN_LOG_TYPE not in attrs and _is_generated_prompt_wrapper(tool_name)
        ):
            _map_prompt_wrapper(span, attrs)
        else:
            _map_tool_span(span, attrs)
    else:
        return False

    _cleanup_attrs(attrs)
    _map_error(span, attrs)
    _sanitize_contract_attrs(attrs)
    if not content_allowed(capture_content):
        _strip_content(attrs)
    span._attributes = attrs
    if hasattr(span, "_events"):
        span._events = ()
    return True


class SemanticKernelLogRecordHandler(logging.Handler):
    """Attach Semantic Kernel diagnostic log payloads to the active span."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            if not content_allowed(True):
                return
            event_name = record.__dict__.get(event_attributes.EVENT_NAME)
            if not isinstance(event_name, str):
                return

            span = trace.get_current_span()
            if (
                span is None
                or not hasattr(span, "set_attribute")
                or not _is_semantic_kernel_span(span)
            ):
                return
            is_recording = getattr(span, "is_recording", None)
            if callable(is_recording) and not is_recording():
                return

            if event_name in _PROMPT_MESSAGE_EVENTS:
                self._record_prompt_message(
                    span,
                    record,
                    fallback_role=_PROMPT_MESSAGE_EVENTS[event_name],
                )
            elif event_name == SK_PROMPT_EVENT:
                self._record_prompt_text(span, record)
            elif event_name == SK_CHOICE_EVENT:
                self._record_completion_choice(span, record)
        except Exception:
            logger.debug("Failed to attach Semantic Kernel log record", exc_info=True)

    @staticmethod
    def _record_prompt_message(
        span: Any,
        record: logging.LogRecord,
        *,
        fallback_role: str,
    ) -> None:
        payload = _safe_json_loads(record.getMessage())
        if not isinstance(payload, Mapping):
            payload = {"role": fallback_role, "content": record.getMessage()}

        index = getattr(record, SK_CHAT_MESSAGE_INDEX_ATTR, 0)
        if not isinstance(index, int):
            index = 0
        _set_message_attrs(
            span.set_attribute,
            prefix=SpanAttributes.LLM_PROMPTS,
            index=index,
            message=payload,
            fallback_role=fallback_role,
        )

    @staticmethod
    def _record_prompt_text(span: Any, record: logging.LogRecord) -> None:
        _set_message_attrs(
            span.set_attribute,
            prefix=SpanAttributes.LLM_PROMPTS,
            index=0,
            message={"role": "user", "content": record.getMessage()},
            fallback_role="user",
        )

    @staticmethod
    def _record_completion_choice(span: Any, record: logging.LogRecord) -> None:
        payload = _safe_json_loads(record.getMessage())
        if not isinstance(payload, Mapping):
            payload = {"message": {"role": "assistant", "content": record.getMessage()}}

        message = payload.get("message")
        if not isinstance(message, Mapping):
            message = payload

        index = payload.get("index", 0)
        if not isinstance(index, int):
            index = 0
        _set_message_attrs(
            span.set_attribute,
            prefix=SpanAttributes.LLM_COMPLETIONS,
            index=index,
            message=message,
            fallback_role="assistant",
        )


class SemanticKernelSpanProcessor(SpanProcessor):
    """Translate Semantic Kernel spans to the Respan span contract."""

    def __init__(self, *, capture_content: bool = True) -> None:
        self.capture_content = capture_content
        self._content_policy: dict[tuple[int, int], bool] = {}

    def on_start(
        self,
        span: Span,
        parent_context: Context | None = None,
    ) -> None:
        if not _is_semantic_kernel_span(span):
            return
        get_context = getattr(span, "get_span_context", None)
        span_context = get_context() if get_context else None
        allowed = content_allowed(self.capture_content)
        setattr(span, SK_CONTENT_CAPTURE_ATTRIBUTE, allowed)
        if span_context is not None:
            self._content_policy[(span_context.trace_id, span_context.span_id)] = (
                allowed
            )
        parent = trace.get_current_span(parent_context)
        parent_attrs = getattr(parent, "attributes", None) or {}
        available_functions = parent_attrs.get(SK_AVAILABLE_FUNCTIONS_ATTR)
        if available_functions is not None:
            span.set_attribute(SK_AVAILABLE_FUNCTIONS_ATTR, available_functions)

    def on_end(self, span: ReadableSpan) -> None:
        get_context = getattr(span, "get_span_context", None)
        span_context = get_context() if get_context else None
        allowed = (
            self._content_policy.pop(
                (span_context.trace_id, span_context.span_id), self.capture_content
            )
            if span_context
            else self.capture_content
        )
        enrich_semantic_kernel_span(span, capture_content=allowed)

    def shutdown(self) -> None:
        self._content_policy.clear()

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
