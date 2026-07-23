"""Build and inject canonical OTEL spans for Watson Orchestrate ADK calls."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Mapping, Sequence
from typing import Any

from opentelemetry import context as context_api
from opentelemetry import trace
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import (
    GEN_AI_USAGE_INPUT_TOKENS,
    GEN_AI_USAGE_OUTPUT_TOKENS,
)
from opentelemetry.semconv_ai import LLMRequestTypeValues
from opentelemetry.semconv_ai import SpanAttributes as TLSpanAttributes
from respan_instrumentation_watson_orchestrate_adk._constants import (
    AGENT_ID_KEY,
    ARGUMENTS_KEY,
    ASSISTANT_ROLE,
    CHAT_LLM_KEY,
    CHAT_MODEL_NAME_KEY,
    CHOICES_KEY,
    COMPLETION_TOKENS_KEY,
    CONTENT_KEY,
    DELTA_KEY,
    EVENT_KEY,
    FUNCTION_KEY,
    FUNCTION_TOOL_TYPE,
    INPUT_KEY,
    INSTRUCTION_KEY,
    MESSAGE_KEY,
    MESSAGES_KEY,
    MODEL_ID_KEY,
    MODEL_KEY,
    NAME_KEY,
    OFF_CONTRACT_ALIASES,
    OUTPUT_TOKENS_KEY,
    PROMPT_TOKENS_KEY,
    ROLE_KEY,
    RUN_ID_KEY,
    STATUS_KEY,
    TEXT_KEY,
    THREAD_ID_KEY,
    TOKEN_USAGE_KEY,
    TOOLS_KEY,
    TOTAL_TOKENS_KEY,
    TYPE_KEY,
    USAGE_KEY,
    USER_MESSAGE_KEY,
    USER_ROLE,
    WATSON_ORCHESTRATE_ADK_SYSTEM_NAME,
    WATSON_ORCHESTRATE_CHAT_SPAN_NAME,
    WATSON_ORCHESTRATE_RUN_SPAN_NAME,
    WATSON_ORCHESTRATE_TOOL_SPAN_NAME,
)
from respan_sdk.constants.llm_logging import (
    LOG_TYPE_AGENT,
    LOG_TYPE_CHAT,
    LOG_TYPE_TOOL,
    LogMethodChoices,
)
from respan_sdk.constants.span_attributes import (
    RESPAN_LOG_ID,
    RESPAN_LOG_METHOD,
    RESPAN_LOG_TYPE,
    RESPAN_THREADS_ID,
)
from respan_sdk.utils.data_processing.id_processing import format_span_id, format_trace_id
from respan_sdk.utils.serialization import serialize_value
from respan_tracing.utils.span_factory import build_readable_span, inject_span

logger = logging.getLogger(__name__)

_PROMPT_PREFIX = f"{TLSpanAttributes.LLM_PROMPTS}."
_COMPLETION_PREFIX = f"{TLSpanAttributes.LLM_COMPLETIONS}."


def _request_type_value(name: str, fallback: str) -> str:
    value = getattr(LLMRequestTypeValues, name, None)
    return getattr(value, "value", fallback)


def _field(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    try:
        return getattr(value, key)
    except Exception:
        return default


def _dump_value(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    try:
        return serialize_value(value=value)
    except Exception:
        return str(value)


def safe_json(value: Any) -> str:
    try:
        return json.dumps(_dump_value(value), default=str, separators=(",", ":"))
    except Exception:
        return json.dumps(str(value), separators=(",", ":"))


def _to_string(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return safe_json(value)


def _as_sequence(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return list(value)
    return [value]


def _message(role: str, content: Any) -> dict[str, Any]:
    return {ROLE_KEY: role, CONTENT_KEY: _dump_value(content)}


def _normalize_messages(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, str):
        return [_message(USER_ROLE, value)]
    if isinstance(value, Mapping):
        return [
            {
                ROLE_KEY: str(value.get(ROLE_KEY) or USER_ROLE),
                CONTENT_KEY: _dump_value(value.get(CONTENT_KEY, value)),
            }
        ]

    messages: list[dict[str, Any]] = []
    for item in _as_sequence(value):
        if isinstance(item, Mapping):
            messages.append(
                {
                    ROLE_KEY: str(item.get(ROLE_KEY) or USER_ROLE),
                    CONTENT_KEY: _dump_value(item.get(CONTENT_KEY, item)),
                }
            )
        else:
            messages.append(_message(USER_ROLE, item))
    return messages


def _messages_from_call(call_kwargs: Mapping[str, Any]) -> list[dict[str, Any]]:
    for key in (MESSAGES_KEY, MESSAGE_KEY):
        value = call_kwargs.get(key)
        if value is not None:
            if key == MESSAGE_KEY and isinstance(value, Mapping):
                return _normalize_messages(value)
            return _normalize_messages(value)

    for key in (USER_MESSAGE_KEY, INPUT_KEY):
        value = call_kwargs.get(key)
        if value is not None:
            return [_message(USER_ROLE, value)]

    instruction = call_kwargs.get(INSTRUCTION_KEY)
    if instruction is not None:
        return [_message(USER_ROLE, instruction)]

    return []


def _set_prompt_attrs(attrs: dict[str, Any], messages: list[dict[str, Any]]) -> None:
    for index, message in enumerate(messages):
        role = message.get(ROLE_KEY)
        content = message.get(CONTENT_KEY)
        if role is not None:
            attrs[f"{_PROMPT_PREFIX}{index}.role"] = str(role)
        if content is not None:
            attrs[f"{_PROMPT_PREFIX}{index}.content"] = _to_string(content)


def _set_completion_attrs(attrs: dict[str, Any], text: str) -> None:
    attrs[f"{_COMPLETION_PREFIX}0.role"] = ASSISTANT_ROLE
    attrs[f"{_COMPLETION_PREFIX}0.content"] = text


def _model_from_call(call_kwargs: Mapping[str, Any], instance: Any = None) -> str | None:
    for key in (
        MODEL_KEY,
        MODEL_ID_KEY,
        CHAT_LLM_KEY,
        CHAT_MODEL_NAME_KEY,
        "llm",
        "selected_model",
    ):
        value = call_kwargs.get(key)
        if value:
            return str(value)
    for key in (MODEL_KEY, MODEL_ID_KEY, "_model", "_model_id"):
        value = _field(instance, key)
        if value:
            return str(value)
    return None


def _first_choice(response: Any) -> Any:
    choices = _field(response, CHOICES_KEY, []) or []
    if isinstance(choices, Sequence) and not isinstance(choices, str | bytes | bytearray):
        return choices[0] if choices else None
    return None


def _content_from_message(message: Any) -> Any:
    if message is None:
        return None
    content = _field(message, CONTENT_KEY)
    if content is not None:
        return content
    text = _field(message, TEXT_KEY)
    if text is not None:
        return text
    return None


def _response_text(response: Any) -> str:
    if response is None:
        return ""
    if isinstance(response, str):
        return response

    choice = _first_choice(response)
    if choice is not None:
        for key in (MESSAGE_KEY, DELTA_KEY):
            content = _content_from_message(_field(choice, key))
            if content is not None:
                return _to_string(content)
        choice_text = _field(choice, TEXT_KEY)
        if choice_text is not None:
            return _to_string(choice_text)

    if isinstance(response, Mapping):
        event = response.get(EVENT_KEY)
        if event and response.get("data") is not None:
            return _response_text(response["data"])

        for key in ("formatted_message", MESSAGE_KEY):
            content = _content_from_message(response.get(key))
            if content is not None:
                return _to_string(content)

        for key in (CONTENT_KEY, TEXT_KEY, "response", "output", "result"):
            value = response.get(key)
            if value is not None:
                return _to_string(value)

    if isinstance(response, Sequence) and not isinstance(response, str | bytes | bytearray):
        for item in reversed(response):
            text = _response_text(item)
            if text:
                return text

    return ""


def _coerce_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _usage_source(value: Any) -> Any:
    usage = _field(value, USAGE_KEY)
    if usage is not None:
        return usage
    token_usage = _field(value, TOKEN_USAGE_KEY)
    if token_usage is not None:
        return token_usage
    return value


def _set_usage_attrs(attrs: dict[str, Any], response: Any) -> None:
    source = _usage_source(response)
    prompt_tokens = _coerce_int(_field(source, PROMPT_TOKENS_KEY))
    if prompt_tokens is None:
        prompt_tokens = _coerce_int(_field(source, INPUT_KEY + "_tokens"))

    completion_tokens = _coerce_int(_field(source, COMPLETION_TOKENS_KEY))
    if completion_tokens is None:
        completion_tokens = _coerce_int(_field(source, OUTPUT_TOKENS_KEY))

    total_tokens = _coerce_int(_field(source, TOTAL_TOKENS_KEY))
    if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
        total_tokens = prompt_tokens + completion_tokens

    if prompt_tokens is not None:
        attrs[TLSpanAttributes.LLM_USAGE_PROMPT_TOKENS] = prompt_tokens
        attrs[GEN_AI_USAGE_INPUT_TOKENS] = prompt_tokens
    if completion_tokens is not None:
        attrs[TLSpanAttributes.LLM_USAGE_COMPLETION_TOKENS] = completion_tokens
        attrs[GEN_AI_USAGE_OUTPUT_TOKENS] = completion_tokens
    if total_tokens is not None:
        attrs[TLSpanAttributes.LLM_USAGE_TOTAL_TOKENS] = total_tokens


def _tool_definitions(value: Any) -> list[dict[str, Any]]:
    tools = []
    for item in _as_sequence(value):
        if item is None:
            continue
        if isinstance(item, Mapping):
            if item.get(TYPE_KEY) == FUNCTION_TOOL_TYPE:
                tools.append(dict(item))
                continue
            if item.get(FUNCTION_KEY):
                tools.append({TYPE_KEY: FUNCTION_TOOL_TYPE, FUNCTION_KEY: item[FUNCTION_KEY]})
                continue
            name = item.get(NAME_KEY)
            if name:
                function = {NAME_KEY: name}
                for key in ("description", "parameters", "input_schema", "schema"):
                    if item.get(key) is not None:
                        function["parameters" if key in {"input_schema", "schema"} else key] = item[key]
                tools.append({TYPE_KEY: FUNCTION_TOOL_TYPE, FUNCTION_KEY: function})
                continue

        spec = _field(item, "__tool_spec__")
        if spec is not None:
            dumped = _dump_value(spec)
            if isinstance(dumped, Mapping) and dumped.get(NAME_KEY):
                function = {NAME_KEY: dumped[NAME_KEY]}
                if dumped.get("description") is not None:
                    function["description"] = dumped["description"]
                if dumped.get("input_schema") is not None:
                    function["parameters"] = dumped["input_schema"]
                tools.append({TYPE_KEY: FUNCTION_TOOL_TYPE, FUNCTION_KEY: function})
    return tools


def _base_attrs(*, span_name: str, log_type: str) -> dict[str, Any]:
    attrs = {
        RESPAN_LOG_METHOD: LogMethodChoices.TRACING_INTEGRATION.value,
        RESPAN_LOG_TYPE: log_type,
        TLSpanAttributes.TRACELOOP_ENTITY_NAME: span_name,
        TLSpanAttributes.TRACELOOP_ENTITY_PATH: span_name,
    }
    workflow_name = context_api.get_value(TLSpanAttributes.TRACELOOP_ENTITY_NAME)
    if workflow_name:
        attrs[TLSpanAttributes.TRACELOOP_WORKFLOW_NAME] = workflow_name
    return attrs


def _clean_attrs(attrs: dict[str, Any]) -> dict[str, Any]:
    cleaned = dict(attrs)
    for alias in OFF_CONTRACT_ALIASES:
        cleaned.pop(alias, None)
    cleaned.pop(TLSpanAttributes.TRACELOOP_SPAN_KIND, None)
    return cleaned


def _current_trace_parent_ids() -> tuple[str | None, str | None]:
    try:
        current_span = trace.get_current_span()
        span_context = current_span.get_span_context()
    except Exception:
        return None, None
    trace_id = getattr(span_context, "trace_id", 0) or 0
    span_id = getattr(span_context, "span_id", 0) or 0
    if trace_id == 0 or span_id == 0:
        return None, None
    return format_trace_id(trace_id=trace_id), format_span_id(span_id=span_id)


def build_tool_attrs(
    *,
    tool_name: str,
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
    response: Any = None,
    error_message: str | None = None,
) -> dict[str, Any]:
    attrs = _base_attrs(span_name=tool_name, log_type=LOG_TYPE_TOOL)
    arguments: dict[str, Any] = dict(kwargs)
    if args:
        arguments["_args"] = list(args)
    attrs[TLSpanAttributes.TRACELOOP_ENTITY_INPUT] = safe_json(
        {NAME_KEY: tool_name, ARGUMENTS_KEY: arguments}
    )

    if error_message is not None:
        attrs[TLSpanAttributes.TRACELOOP_ENTITY_OUTPUT] = error_message
    elif response is not None:
        content = _field(response, CONTENT_KEY, response)
        context_updates = _field(response, "context_updates")
        output = {CONTENT_KEY: content}
        if context_updates:
            output["context_updates"] = context_updates
        attrs[TLSpanAttributes.TRACELOOP_ENTITY_OUTPUT] = safe_json(output)
    return _clean_attrs(attrs)


def build_agent_run_attrs(
    *,
    method_name: str,
    call_kwargs: Mapping[str, Any],
    response: Any = None,
    error_message: str | None = None,
) -> dict[str, Any]:
    agent_name = str(call_kwargs.get(AGENT_ID_KEY) or method_name)
    attrs = _base_attrs(span_name=agent_name, log_type=LOG_TYPE_AGENT)
    attrs[TLSpanAttributes.TRACELOOP_ENTITY_INPUT] = safe_json(
        {"method": method_name, "request": dict(call_kwargs)}
    )

    thread_id = call_kwargs.get(THREAD_ID_KEY) or _field(response, THREAD_ID_KEY)
    if thread_id:
        attrs[RESPAN_THREADS_ID] = str(thread_id)
    run_id = _field(response, RUN_ID_KEY)
    if run_id:
        attrs[RESPAN_LOG_ID] = str(run_id)

    if error_message is not None:
        attrs[TLSpanAttributes.TRACELOOP_ENTITY_OUTPUT] = error_message
    elif response is not None:
        attrs[TLSpanAttributes.TRACELOOP_ENTITY_OUTPUT] = safe_json(response)
    return _clean_attrs(attrs)


def build_chat_attrs(
    *,
    method_name: str,
    call_kwargs: Mapping[str, Any],
    response: Any = None,
    error_message: str | None = None,
    instance: Any = None,
) -> dict[str, Any]:
    attrs = _base_attrs(span_name=WATSON_ORCHESTRATE_CHAT_SPAN_NAME, log_type=LOG_TYPE_CHAT)
    attrs[TLSpanAttributes.LLM_SYSTEM] = WATSON_ORCHESTRATE_ADK_SYSTEM_NAME
    attrs[TLSpanAttributes.LLM_REQUEST_TYPE] = _request_type_value("CHAT", "chat")

    model = _model_from_call(call_kwargs=call_kwargs, instance=instance)
    if model:
        attrs[TLSpanAttributes.LLM_REQUEST_MODEL] = model

    messages = _messages_from_call(call_kwargs)
    if messages:
        _set_prompt_attrs(attrs=attrs, messages=messages)

    tools = _tool_definitions(call_kwargs.get(TOOLS_KEY))
    if tools:
        attrs[TLSpanAttributes.LLM_REQUEST_FUNCTIONS] = safe_json(tools)

    attrs[TLSpanAttributes.TRACELOOP_ENTITY_INPUT] = safe_json(
        {"method": method_name, "request": dict(call_kwargs)}
    )

    if error_message is not None:
        attrs[TLSpanAttributes.TRACELOOP_ENTITY_OUTPUT] = error_message
        _set_completion_attrs(attrs=attrs, text=error_message)
    elif response is not None:
        response_text = _response_text(response)
        attrs[TLSpanAttributes.TRACELOOP_ENTITY_OUTPUT] = (
            response_text if response_text else safe_json(response)
        )
        if response_text:
            _set_completion_attrs(attrs=attrs, text=response_text)
        _set_usage_attrs(attrs=attrs, response=response)
    return _clean_attrs(attrs)


def emit_span(
    *,
    span_name: str,
    attrs: dict[str, Any],
    start_ns: int,
    error_message: str | None = None,
    status_code: int = 200,
) -> None:
    try:
        clean_attrs = _clean_attrs(attrs)
        if error_message is not None:
            clean_attrs["error.message"] = error_message
            clean_attrs["status_code"] = status_code if status_code >= 400 else 500

        trace_id, parent_id = _current_trace_parent_ids()
        span = build_readable_span(
            name=span_name,
            trace_id=trace_id,
            parent_id=parent_id,
            start_time_ns=start_ns,
            end_time_ns=time.time_ns(),
            attributes=clean_attrs,
            error_message=error_message,
            status_code=status_code,
        )
        inject_span(span=span)
    except Exception:
        logger.debug("Failed to emit Watson Orchestrate ADK span", exc_info=True)


def emit_tool_span(
    *,
    tool_name: str,
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
    start_ns: int,
    response: Any = None,
    error_message: str | None = None,
) -> None:
    emit_span(
        span_name=WATSON_ORCHESTRATE_TOOL_SPAN_NAME,
        attrs=build_tool_attrs(
            tool_name=tool_name,
            args=args,
            kwargs=kwargs,
            response=response,
            error_message=error_message,
        ),
        start_ns=start_ns,
        error_message=error_message,
        status_code=500 if error_message else 200,
    )


def emit_agent_run_span(
    *,
    method_name: str,
    call_kwargs: Mapping[str, Any],
    start_ns: int,
    response: Any = None,
    error_message: str | None = None,
) -> None:
    emit_span(
        span_name=WATSON_ORCHESTRATE_RUN_SPAN_NAME,
        attrs=build_agent_run_attrs(
            method_name=method_name,
            call_kwargs=call_kwargs,
            response=response,
            error_message=error_message,
        ),
        start_ns=start_ns,
        error_message=error_message,
        status_code=500 if error_message else 200,
    )


def emit_chat_span(
    *,
    method_name: str,
    call_kwargs: Mapping[str, Any],
    start_ns: int,
    response: Any = None,
    error_message: str | None = None,
    instance: Any = None,
) -> None:
    emit_span(
        span_name=WATSON_ORCHESTRATE_CHAT_SPAN_NAME,
        attrs=build_chat_attrs(
            method_name=method_name,
            call_kwargs=call_kwargs,
            response=response,
            error_message=error_message,
            instance=instance,
        ),
        start_ns=start_ns,
        error_message=error_message,
        status_code=500 if error_message else 200,
    )
