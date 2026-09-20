"""Google ADK-specific OpenInference span normalization."""

from __future__ import annotations

import json
import logging
import threading
from collections import defaultdict
from typing import Any

from openinference.semconv.trace import (
    MessageAttributes as OIMessageAttributes,
    MessageContentAttributes as OIMessageContentAttributes,
    SpanAttributes as OISpanAttributes,
    ToolCallAttributes as OIToolCallAttributes,
)
from opentelemetry.context import Context
from opentelemetry.sdk.trace import ReadableSpan, Span
from opentelemetry.sdk.trace import SpanProcessor
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import (
    GEN_AI_PROVIDER_NAME,
    GEN_AI_SYSTEM,
    GEN_AI_TOOL_NAME,
    GEN_AI_USAGE_INPUT_TOKENS,
    GEN_AI_USAGE_OUTPUT_TOKENS,
)
from opentelemetry.semconv_ai import SpanAttributes as TLSpanAttributes
from respan_instrumentation_openinference._translator import OpenInferenceTranslator
from respan_sdk.constants.llm_logging import (
    LOG_TYPE_AGENT,
    LOG_TYPE_CHAT,
    LOG_TYPE_TOOL,
    LOG_TYPE_WORKFLOW,
)
from respan_sdk.constants.span_attributes import (
    RESPAN_LOG_TYPE,
    RESPAN_SESSION_ID,
    RESPAN_SPAN_HANDOFFS,
    RESPAN_SPAN_TOOL_CALLS,
    RESPAN_SPAN_TOOLS,
)

logger = logging.getLogger(__name__)

GOOGLE_ADK_SCOPE_NAME = "openinference.instrumentation.google_adk"

_OI_INPUT_MESSAGES_PREFIX = f"{OISpanAttributes.LLM_INPUT_MESSAGES}."
_OI_OUTPUT_MESSAGES_PREFIX = f"{OISpanAttributes.LLM_OUTPUT_MESSAGES}."
_OI_TOKEN_COUNT_PREFIX = f"{OISpanAttributes.LLM_TOKEN_COUNT_TOTAL.rsplit('.', 1)[0]}."
_OI_TOOLS_PREFIX = f"{OISpanAttributes.LLM_TOOLS}."
_OI_MESSAGE_CONTENT_PREFIX = f"{OIMessageAttributes.MESSAGE_CONTENT}."
_OI_MESSAGE_CONTENTS_PREFIX = f"{OIMessageAttributes.MESSAGE_CONTENTS}."
_OI_MESSAGE_CONTENT_BLOCK_PREFIX = (
    f"{OIMessageContentAttributes.MESSAGE_CONTENT_TEXT.rsplit('.', 1)[0]}."
)
_OI_MESSAGE_TOOL_CALLS_PREFIX = f"{OIMessageAttributes.MESSAGE_TOOL_CALLS}."
# OpenInference exposes llm.finish_reason, but not this nested message field.
_OI_MESSAGE_FINISH_REASON = "message.finish_reason"
_OI_TOOL_CALL_PREFIX = f"{OIToolCallAttributes.TOOL_CALL_ID.split('.', 1)[0]}."

_OI_TOOL_PREFIX = f"{OISpanAttributes.TOOL_NAME.rsplit('.', 1)[0]}."
_GEN_AI_TOOL_PREFIX = f"{GEN_AI_TOOL_NAME.rsplit('.', 1)[0]}."

# Google ADK raw payload keys are SDK-specific translator inputs.
_GOOGLE_ADK_PREFIX = "gcp.vertex.agent."
_GOOGLE_ADK_LLM_REQUEST = "gcp.vertex.agent.llm_request"
_GOOGLE_ADK_LLM_RESPONSE = "gcp.vertex.agent.llm_response"
_GOOGLE_ADK_TOOL_CALL_ARGS = "gcp.vertex.agent.tool_call_args"
_GOOGLE_ADK_TOOL_RESPONSE = "gcp.vertex.agent.tool_response"
_SESSION_ID_ATTRS = (
    "session.id",
    "gen_ai.conversation.id",
)

_OFF_CONTRACT_ALIASES = {
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
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, default=str)


def _span_trace_id(span: ReadableSpan) -> int | None:
    context = getattr(span, "context", None)
    if context is None:
        get_span_context = getattr(span, "get_span_context", None)
        if callable(get_span_context):
            context = get_span_context()
    trace_id = getattr(context, "trace_id", None)
    return trace_id if isinstance(trace_id, int) and trace_id else None


def _first_user_prompt(attrs: dict[str, Any]) -> str | None:
    prefix = f"{TLSpanAttributes.LLM_PROMPTS}."
    indexes = sorted(
        {
            int(parts[0])
            for key in attrs
            if key.startswith(prefix)
            and len((parts := key[len(prefix) :].split(".", 1))) == 2
            and parts[0].isdigit()
        }
    )
    for index in indexes:
        role = attrs.get(f"{prefix}{index}.role")
        content = attrs.get(f"{prefix}{index}.content")
        if role == "user" and content not in (None, ""):
            return str(content)
    return None


def _session_id(raw_attrs: dict[str, Any]) -> str | None:
    for key in _SESSION_ID_ATTRS:
        value = raw_attrs.get(key)
        if value not in (None, ""):
            return str(value)
    return None


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


def _is_google_adk_span(span: ReadableSpan, attrs: dict[str, Any]) -> bool:
    if _scope_name(span) == GOOGLE_ADK_SCOPE_NAME:
        return True
    if attrs.get(GEN_AI_SYSTEM) == "gcp.vertex.agent":
        return True
    return any(key.startswith(_GOOGLE_ADK_PREFIX) for key in attrs)


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
        if not parts[0].isdigit() or len(parts) == 1:
            continue
        buckets[int(parts[0])][parts[1]] = value
    return buckets


def _set_nested_value(target: dict[str, Any], dotted_path: str, value: Any) -> None:
    cursor = target
    parts = dotted_path.split(".")
    for part in parts[:-1]:
        child = cursor.get(part)
        if not isinstance(child, dict):
            child = {}
            cursor[part] = child
        cursor = child
    cursor[parts[-1]] = value


def _normalize_tool_call(tool_call: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    if tool_call.get("id") is not None:
        normalized["id"] = tool_call["id"]

    function = tool_call.get("function")
    function_payload: dict[str, Any] = {}
    if isinstance(function, dict):
        if function.get("name") is not None:
            function_payload["name"] = function["name"]
        if function.get("arguments") is not None:
            function_payload["arguments"] = function["arguments"]

    if tool_call.get("type") is not None:
        normalized["type"] = tool_call["type"]
    elif function_payload:
        normalized["type"] = "function"
    if function_payload:
        normalized["function"] = function_payload
    return normalized


def _tool_call_signature(tool_call: dict[str, Any]) -> str:
    normalized = _normalize_tool_call(tool_call)
    function = normalized.get("function")
    if isinstance(function, dict) and "arguments" in function:
        function["arguments"] = _parse_json(function["arguments"])
    return json.dumps(normalized, default=str, sort_keys=True, separators=(",", ":"))


def _extract_tool_calls_from_message(
    raw: dict[str, Any],
) -> list[dict[str, Any]] | None:
    tool_call_buckets: dict[int, dict[str, Any]] = defaultdict(dict)
    for field_key, field_val in raw.items():
        if not field_key.startswith(_OI_MESSAGE_TOOL_CALLS_PREFIX):
            continue
        rest = field_key[len(_OI_MESSAGE_TOOL_CALLS_PREFIX) :]
        parts = rest.split(".", 1)
        if not parts[0].isdigit() or len(parts) == 1:
            continue
        tc_field = parts[1]
        if tc_field.startswith(_OI_TOOL_CALL_PREFIX):
            tc_field = tc_field[len(_OI_TOOL_CALL_PREFIX) :]
        tool_call_buckets[int(parts[0])][tc_field] = field_val

    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for idx in sorted(tool_call_buckets):
        tool_call: dict[str, Any] = {}
        for field_key, field_val in tool_call_buckets[idx].items():
            _set_nested_value(tool_call, field_key, field_val)
        normalized = _normalize_tool_call(tool_call)
        if not normalized:
            continue
        signature = _tool_call_signature(normalized)
        if signature not in seen:
            seen.add(signature)
            result.append(normalized)

    legacy_name = raw.get(OIMessageAttributes.MESSAGE_FUNCTION_CALL_NAME)
    legacy_args = raw.get(OIMessageAttributes.MESSAGE_FUNCTION_CALL_ARGUMENTS_JSON)
    if legacy_name is not None or legacy_args is not None:
        legacy_tool_call = {
            "type": "function",
            "function": {},
        }
        if legacy_name is not None:
            legacy_tool_call["function"]["name"] = legacy_name
        if legacy_args is not None:
            legacy_tool_call["function"]["arguments"] = legacy_args
        normalized = _normalize_tool_call(legacy_tool_call)
        signature = _tool_call_signature(normalized)
        if signature not in seen:
            result.append(normalized)

    return result or None


def _extract_message_content(raw: dict[str, Any]) -> Any:
    content = raw.get(OIMessageAttributes.MESSAGE_CONTENT)
    if content is not None:
        return content

    indexed_content: list[tuple[int, Any]] = []
    content_blocks: dict[int, dict[str, Any]] = defaultdict(dict)
    for field_key, field_val in raw.items():
        if field_key.startswith(_OI_MESSAGE_CONTENT_PREFIX):
            idx_str = field_key[len(_OI_MESSAGE_CONTENT_PREFIX) :]
            if idx_str.isdigit():
                indexed_content.append((int(idx_str), field_val))
            continue
        if not field_key.startswith(_OI_MESSAGE_CONTENTS_PREFIX):
            continue
        rest = field_key[len(_OI_MESSAGE_CONTENTS_PREFIX) :]
        parts = rest.split(".", 1)
        if not parts[0].isdigit() or len(parts) == 1:
            continue
        block_field = parts[1]
        if block_field.startswith(_OI_MESSAGE_CONTENT_BLOCK_PREFIX):
            block_field = block_field[len(_OI_MESSAGE_CONTENT_BLOCK_PREFIX) :]
        _set_nested_value(content_blocks[int(parts[0])], block_field, field_val)

    if content_blocks:
        text_blocks = [
            block.get("text")
            for _, block in sorted(content_blocks.items())
            if block.get("text") is not None
        ]
        if len(text_blocks) == 1:
            return text_blocks[0]
        if text_blocks and all(isinstance(value, str) for value in text_blocks):
            return "\n".join(text_blocks)
        return _safe_json_str([block for _, block in sorted(content_blocks.items())])

    if not indexed_content:
        return None
    ordered_values = [value for _, value in sorted(indexed_content)]
    if len(ordered_values) == 1:
        return ordered_values[0]
    if all(isinstance(value, str) for value in ordered_values):
        return "\n".join(ordered_values)
    return ordered_values


def _normalize_adk_role(role: Any) -> str:
    if role == "model":
        return "assistant"
    if isinstance(role, str) and role:
        return role
    return "user"


def _promote_message_attrs(
    raw_attrs: dict[str, Any],
    attrs: dict[str, Any],
    *,
    source_prefix: str,
    target_prefix: str,
) -> None:
    for index, raw_message in sorted(
        _collect_message_buckets(raw_attrs, source_prefix).items()
    ):
        target = f"{target_prefix}.{index}"
        role = raw_message.get(OIMessageAttributes.MESSAGE_ROLE)
        if role is not None:
            attrs[f"{target}.role"] = _normalize_adk_role(role)

        content = _extract_message_content(raw_message)
        if content is not None:
            attrs[f"{target}.content"] = content

        tool_calls = _extract_tool_calls_from_message(raw_message)
        if tool_calls is not None:
            attrs[f"{target}.tool_calls"] = _safe_json_str(tool_calls)

        finish_reason = raw_message.get(_OI_MESSAGE_FINISH_REASON)
        if finish_reason and source_prefix == _OI_OUTPUT_MESSAGES_PREFIX:
            attrs[f"{target}.finish_reason"] = finish_reason


def _adk_part_text_tool_calls(
    parts: Any,
) -> tuple[str | None, list[dict[str, Any]] | None, str | None]:
    if not isinstance(parts, list):
        return None, None, None

    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    tool_response: str | None = None

    for part in parts:
        if not isinstance(part, dict):
            continue
        text = part.get("text")
        if isinstance(text, str):
            text_parts.append(text)

        function_call = part.get("function_call")
        if isinstance(function_call, dict):
            function_payload: dict[str, Any] = {}
            if function_call.get("name") is not None:
                function_payload["name"] = function_call["name"]
            if function_call.get("args") is not None:
                function_payload["arguments"] = _safe_json_str(function_call["args"])
            tool_call = {
                "type": "function",
                "function": function_payload,
            }
            if function_call.get("id") is not None:
                tool_call["id"] = function_call["id"]
            normalized = _normalize_tool_call(tool_call)
            if normalized:
                tool_calls.append(normalized)

        function_response = part.get("function_response")
        if isinstance(function_response, dict):
            response_payload = {
                key: function_response[key]
                for key in ("id", "name", "response")
                if function_response.get(key) is not None
            }
            if response_payload:
                tool_response = _safe_json_str(response_payload)

    text_value = "\n".join(text_parts) if text_parts else None
    return text_value, tool_calls or None, tool_response


def _set_adk_content_message(
    attrs: dict[str, Any],
    *,
    target_prefix: str,
    index: int,
    content: Any,
) -> None:
    if not isinstance(content, dict):
        return

    text, tool_calls, tool_response = _adk_part_text_tool_calls(content.get("parts"))
    role = _normalize_adk_role(content.get("role"))
    if tool_response is not None:
        role = "tool"
        text = tool_response

    target = f"{target_prefix}.{index}"
    if tool_response is not None:
        attrs[f"{target}.role"] = role
    else:
        attrs.setdefault(f"{target}.role", role)
    if attrs.get(f"{target}.role") == "model":
        attrs[f"{target}.role"] = "assistant"
    if text is not None:
        if tool_response is not None:
            attrs[f"{target}.content"] = text
        else:
            attrs.setdefault(f"{target}.content", text)
    if tool_calls is not None:
        attrs.setdefault(f"{target}.tool_calls", _safe_json_str(tool_calls))


def _extract_adk_tools(config: Any) -> list[dict[str, Any]] | None:
    if not isinstance(config, dict):
        return None
    tools = config.get("tools")
    if not isinstance(tools, list):
        return None

    normalized_tools: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        declarations = tool.get("function_declarations")
        if not isinstance(declarations, list):
            continue
        for declaration in declarations:
            if isinstance(declaration, dict):
                normalized_tools.append(declaration)
    return normalized_tools or None


def _apply_google_adk_payload_fallbacks(
    raw_attrs: dict[str, Any],
    attrs: dict[str, Any],
) -> None:
    oi_kind = str(raw_attrs.get(OISpanAttributes.OPENINFERENCE_SPAN_KIND, "")).upper()

    if oi_kind == "LLM" or attrs.get(RESPAN_LOG_TYPE) == LOG_TYPE_CHAT:
        request = _parse_json(raw_attrs.get(_GOOGLE_ADK_LLM_REQUEST))
        if not isinstance(request, dict):
            request = _parse_json(raw_attrs.get(OISpanAttributes.INPUT_VALUE))
        if isinstance(request, dict):
            model = request.get("model")
            if model:
                attrs.setdefault(TLSpanAttributes.LLM_REQUEST_MODEL, model)

            inserted_system_prompt = False
            config = request.get("config")
            if isinstance(config, dict):
                system_instruction = config.get("system_instruction")
                if isinstance(system_instruction, str) and system_instruction:
                    system_prompt = f"{TLSpanAttributes.LLM_PROMPTS}.0"
                    system_content_key = f"{system_prompt}.content"
                    if system_content_key not in attrs:
                        attrs[f"{system_prompt}.role"] = "system"
                        attrs[system_content_key] = system_instruction
                        inserted_system_prompt = True

                tools = _extract_adk_tools(config)
                if tools is not None:
                    attrs.setdefault(
                        TLSpanAttributes.LLM_REQUEST_FUNCTIONS,
                        _safe_json_str(tools),
                    )

            contents = request.get("contents")
            if isinstance(contents, list):
                first_prompt_role = attrs.get(f"{TLSpanAttributes.LLM_PROMPTS}.0.role")
                start_index = (
                    1 if inserted_system_prompt or first_prompt_role == "system" else 0
                )
                for offset, content in enumerate(contents):
                    _set_adk_content_message(
                        attrs,
                        target_prefix=TLSpanAttributes.LLM_PROMPTS,
                        index=start_index + offset,
                        content=content,
                    )

        response = _parse_json(raw_attrs.get(_GOOGLE_ADK_LLM_RESPONSE))
        if not isinstance(response, dict):
            response = _parse_json(raw_attrs.get(OISpanAttributes.OUTPUT_VALUE))
        if isinstance(response, dict):
            usage = response.get("usage_metadata")
            if isinstance(usage, dict):
                prompt_tokens = usage.get("prompt_token_count")
                if isinstance(prompt_tokens, int):
                    attrs[TLSpanAttributes.LLM_USAGE_PROMPT_TOKENS] = prompt_tokens
                    attrs[GEN_AI_USAGE_INPUT_TOKENS] = prompt_tokens

                completion_tokens = usage.get("candidates_token_count")
                thoughts_tokens = usage.get("thoughts_token_count")
                if isinstance(completion_tokens, int):
                    if isinstance(thoughts_tokens, int):
                        completion_tokens += thoughts_tokens
                    # ADK 1.5 writes total_token_count into the modern output
                    # field. The response usage is authoritative, including 0.
                    attrs[TLSpanAttributes.LLM_USAGE_COMPLETION_TOKENS] = completion_tokens
                    attrs[GEN_AI_USAGE_OUTPUT_TOKENS] = completion_tokens

                total_tokens = usage.get("total_token_count")
                if isinstance(total_tokens, int):
                    attrs[TLSpanAttributes.LLM_USAGE_TOTAL_TOKENS] = total_tokens

            content = response.get("content")
            if isinstance(content, dict):
                _set_adk_content_message(
                    attrs,
                    target_prefix=TLSpanAttributes.LLM_COMPLETIONS,
                    index=0,
                    content=content,
                )

    if oi_kind == "TOOL" or attrs.get(RESPAN_LOG_TYPE) == LOG_TYPE_TOOL:
        tool_args = raw_attrs.get(_GOOGLE_ADK_TOOL_CALL_ARGS)
        if tool_args is not None:
            tool_name = raw_attrs.get(OISpanAttributes.TOOL_NAME) or raw_attrs.get(
                GEN_AI_TOOL_NAME
            )
            attrs[TLSpanAttributes.TRACELOOP_ENTITY_INPUT] = _safe_json_str(
                {"name": tool_name, "arguments": _parse_json(tool_args)}
            )
        tool_response = raw_attrs.get(_GOOGLE_ADK_TOOL_RESPONSE)
        if tool_response is not None:
            attrs.setdefault(
                TLSpanAttributes.TRACELOOP_ENTITY_OUTPUT,
                _safe_json_str(tool_response),
            )


def _is_indexed_structured_message_attr(key: str) -> bool:
    for prefix in (TLSpanAttributes.LLM_PROMPTS, TLSpanAttributes.LLM_COMPLETIONS):
        prefixed = f"{prefix}."
        if not key.startswith(prefixed):
            continue
        parts = key[len(prefixed) :].split(".")
        if len(parts) < 3 or not parts[0].isdigit():
            return False
        return parts[1] in {"tool_calls", "function_call"}
    return False


def _stringify_structured_message_values(attrs: dict[str, Any]) -> None:
    for key, value in list(attrs.items()):
        if (
            key.startswith(f"{TLSpanAttributes.LLM_PROMPTS}.")
            or key.startswith(f"{TLSpanAttributes.LLM_COMPLETIONS}.")
        ) and key.endswith(".tool_calls"):
            attrs[key] = _safe_json_str(value)
        if key.endswith(".role") and value == "model":
            attrs[key] = "assistant"


def _cleanup_google_adk_attrs(attrs: dict[str, Any], *, is_chat_span: bool) -> None:
    if is_chat_span and attrs.get(GEN_AI_SYSTEM) in (None, "", "gcp.vertex.agent"):
        provider = attrs.get(GEN_AI_PROVIDER_NAME)
        attrs[GEN_AI_SYSTEM] = str(provider).lower() if provider else "google"
    elif not is_chat_span:
        attrs.pop(GEN_AI_SYSTEM, None)

    _stringify_structured_message_values(attrs)

    for key in (TLSpanAttributes.TRACELOOP_SPAN_KIND, *_OFF_CONTRACT_ALIASES):
        attrs.pop(key, None)
    for key in _SESSION_ID_ATTRS:
        attrs.pop(key, None)

    prefixes_to_remove = (
        _GOOGLE_ADK_PREFIX,
        _OI_TOOL_PREFIX,
        _GEN_AI_TOOL_PREFIX,
        _OI_INPUT_MESSAGES_PREFIX,
        _OI_OUTPUT_MESSAGES_PREFIX,
        _OI_TOKEN_COUNT_PREFIX,
        _OI_TOOLS_PREFIX,
    )
    for key in list(attrs.keys()):
        if any(key.startswith(prefix) for prefix in prefixes_to_remove):
            attrs.pop(key, None)
            continue
        if _is_indexed_structured_message_attr(key):
            attrs.pop(key, None)


def _normalize_google_adk_attrs(
    raw_attrs: dict[str, Any],
    attrs: dict[str, Any],
) -> None:
    oi_kind = str(raw_attrs.get(OISpanAttributes.OPENINFERENCE_SPAN_KIND, "")).upper()
    is_chat_span = oi_kind == "LLM" or attrs.get(RESPAN_LOG_TYPE) == LOG_TYPE_CHAT
    _promote_message_attrs(
        raw_attrs,
        attrs,
        source_prefix=_OI_INPUT_MESSAGES_PREFIX,
        target_prefix=TLSpanAttributes.LLM_PROMPTS,
    )
    _promote_message_attrs(
        raw_attrs,
        attrs,
        source_prefix=_OI_OUTPUT_MESSAGES_PREFIX,
        target_prefix=TLSpanAttributes.LLM_COMPLETIONS,
    )
    _apply_google_adk_payload_fallbacks(raw_attrs, attrs)
    session_id = _session_id(raw_attrs)
    if session_id is not None:
        attrs.setdefault(RESPAN_SESSION_ID, session_id)
    _cleanup_google_adk_attrs(attrs, is_chat_span=is_chat_span)


class GoogleADKSpanProcessor(SpanProcessor):
    """Translate and clean up spans emitted by the Google ADK OpenInference hook."""

    def __init__(self) -> None:
        self._translator = OpenInferenceTranslator()
        self._trace_context: dict[int, dict[str, str]] = {}
        self._context_lock = threading.Lock()

    def on_start(
        self,
        span: Span,
        parent_context: Context | None = None,
    ) -> None:
        return None

    def on_end(self, span: ReadableSpan) -> None:
        raw_attrs = dict(getattr(span, "attributes", None) or {})
        if not _is_google_adk_span(span, raw_attrs):
            return

        self._translator.on_end(span)
        translated_attrs = dict(getattr(span, "_attributes", None) or {})
        _normalize_google_adk_attrs(raw_attrs, translated_attrs)

        trace_id = _span_trace_id(span)
        log_type = translated_attrs.get(RESPAN_LOG_TYPE)
        if trace_id is not None and log_type == LOG_TYPE_CHAT:
            prompt = _first_user_prompt(translated_attrs)
            session_id = translated_attrs.get(RESPAN_SESSION_ID)
            if prompt is not None or session_id not in (None, ""):
                with self._context_lock:
                    context = self._trace_context.setdefault(trace_id, {})
                    if prompt is not None:
                        context.setdefault("prompt", prompt)
                    if session_id not in (None, ""):
                        context.setdefault("session_id", str(session_id))
        elif trace_id is not None and log_type in (LOG_TYPE_AGENT, LOG_TYPE_WORKFLOW):
            with self._context_lock:
                context = self._trace_context.pop(trace_id, {})
            if _parse_json(translated_attrs.get(TLSpanAttributes.TRACELOOP_ENTITY_INPUT)) in (
                None,
                "",
            ) and context.get("prompt"):
                translated_attrs[TLSpanAttributes.TRACELOOP_ENTITY_INPUT] = (
                    _safe_json_str({"prompt": context["prompt"]})
                )
            if translated_attrs.get(RESPAN_SESSION_ID) in (None, "") and context.get(
                "session_id"
            ):
                translated_attrs[RESPAN_SESSION_ID] = context["session_id"]
        span._attributes = translated_attrs

    def shutdown(self) -> None:
        with self._context_lock:
            self._trace_context.clear()
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
