"""Normalize native OpenLIT spans into the Respan span contract."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from openlit.semcov import SemanticConvention
from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor
from opentelemetry.sdk.util import BoundedList
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import (
    GEN_AI_AGENT_NAME,
    GEN_AI_INPUT_MESSAGES,
    GEN_AI_OPERATION_NAME,
    GEN_AI_OUTPUT_MESSAGES,
    GEN_AI_PROVIDER_NAME,
    GEN_AI_SYSTEM_INSTRUCTIONS,
    GEN_AI_TOOL_CALL_ID,
    GEN_AI_TOOL_NAME,
    GEN_AI_USAGE_INPUT_TOKENS,
    GEN_AI_USAGE_OUTPUT_TOKENS,
)
from opentelemetry.semconv.attributes.db_attributes import (
    DB_OPERATION_NAME,
    DB_QUERY_TEXT,
    DB_SYSTEM_NAME,
)
from opentelemetry.semconv.attributes.error_attributes import ERROR_TYPE
from opentelemetry.semconv_ai import LLMRequestTypeValues
from opentelemetry.semconv_ai import SpanAttributes as TLSpanAttributes
from opentelemetry.trace import Status, StatusCode
from respan_sdk.constants.llm_logging import LogMethodChoices
from respan_sdk.constants.span_attributes import (
    RESPAN_LOG_METHOD,
    RESPAN_LOG_TYPE,
    RESPAN_METADATA,
)

from respan_instrumentation_openlit._constants import (
    OFF_CONTRACT_ALIASES,
    OPENLIT_INSTRUMENTATION_NAME,
    OPENLIT_OPERATION_LOG_TYPES,
    OPENLIT_PROVIDER_USAGE,
    OPENLIT_REQUEST_PROVIDER,
    OPENLIT_RESPONSE_TOOL_CALLS,
    OPENLIT_SCOPE_PREFIX,
    OPENLIT_SOURCE_INPUT,
    OPENLIT_SOURCE_OUTPUT,
    OPENLIT_SOURCE_TOOLS,
    OPENLIT_SOURCE_USAGE,
    OPENLIT_TOOL_ARGS,
    OPENLIT_TOOL_INPUT,
    OPENLIT_TOOL_OUTPUT,
    OPENLIT_USAGE_TOTAL_TOKENS,
    OPENLIT_WORKFLOW_INPUT,
    OPENLIT_WORKFLOW_OUTPUT,
    STANDARD_DB_ATTRIBUTES,
    STANDARD_GEN_AI_ATTRIBUTES,
)
from respan_instrumentation_openlit._serialization import (
    MAX_ATTRIBUTE_BYTES,
    MAX_STRING_CHARS,
    safe_text,
    safe_url,
)
from respan_instrumentation_openlit._serialization import (
    json_string as _bounded_json_string,
)
from respan_instrumentation_openlit._serialization import (
    json_value as _bounded_json_value,
)

from ._policy import content_allowed

_PROMPT_PREFIX = f"{TLSpanAttributes.LLM_PROMPTS}."
_COMPLETION_PREFIX = f"{TLSpanAttributes.LLM_COMPLETIONS}."


def _json_value(value: Any, *, complete: bool = False, schema: bool = False) -> Any:
    if isinstance(value, str):
        if value.lstrip().startswith(("{", "[", '"')) and (
            complete or len(value.encode("utf-8")) <= MAX_ATTRIBUTE_BYTES
        ):
            try:
                return _bounded_json_value(
                    json.loads(value), complete=complete, schema=schema
                )
            except (json.JSONDecodeError, TypeError):
                pass
        return safe_text(value, limit=None if complete else MAX_STRING_CHARS)
    return _bounded_json_value(value, complete=complete, schema=schema)


def _json_string(value: Any, *, complete: bool = False, schema: bool = False) -> str:
    return _bounded_json_string(value, complete=complete, schema=schema)


def _sequence(value: Any) -> list[Any]:
    parsed = _json_value(value, complete=True)
    if isinstance(parsed, list):
        return parsed
    if parsed is None:
        return []
    return [parsed]


def _is_truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "on", "true", "yes"}
    return False


def _part_payload(part: Any) -> Any:
    if not isinstance(part, Mapping):
        return part
    for key in ("content", "text", "value"):
        if part.get(key) is not None:
            return part[key]
    return dict(part)


def _message_content(message: Mapping[str, Any]) -> Any:
    content = message.get("content")
    if content is not None:
        return content
    parts = message.get("parts")
    if isinstance(parts, Sequence) and not isinstance(parts, str | bytes):
        values = [
            _part_payload(part)
            for part in parts
            if not (isinstance(part, Mapping) and part.get("type") == "tool_call")
        ]
        if len(values) == 1:
            return values[0]
        if values:
            return values
    return None


def _message_tool_calls(message: Mapping[str, Any]) -> list[dict[str, Any]]:
    direct = _json_value(message.get("tool_calls"), complete=True)
    if isinstance(direct, list):
        return [dict(item) for item in direct if isinstance(item, Mapping)]

    calls: list[dict[str, Any]] = []
    parts = message.get("parts")
    if not isinstance(parts, Sequence) or isinstance(parts, str | bytes):
        return calls
    for part in parts:
        if not isinstance(part, Mapping) or part.get("type") not in {
            "tool_call",
            "function_call",
        }:
            continue
        calls.append(
            {
                "id": part.get("id") or part.get("tool_call_id"),
                "type": "function",
                "function": {
                    "name": part.get("name"),
                    "arguments": part.get("arguments", part.get("args")),
                },
            }
        )
    return calls


def _set_message_attributes(
    attrs: dict[str, Any],
    *,
    messages: list[Any],
    target_prefix: str,
) -> None:
    for index, raw_message in enumerate(messages):
        if isinstance(raw_message, str):
            message: Mapping[str, Any] = {
                "role": "assistant" if target_prefix == _COMPLETION_PREFIX else "user",
                "content": raw_message,
            }
        elif isinstance(raw_message, Mapping):
            message = raw_message
        else:
            message = {"role": "user", "content": raw_message}

        role = message.get("role") or (
            "assistant" if target_prefix == _COMPLETION_PREFIX else "user"
        )
        attrs[f"{target_prefix}{index}.role"] = safe_text(
            role, default="user", limit=64
        )
        content = _message_content(message)
        if content is not None:
            attrs[f"{target_prefix}{index}.content"] = (
                safe_text(
                    content,
                    limit=None if message.get("tool_call_id") else MAX_STRING_CHARS,
                )
                if isinstance(content, str)
                else _json_string(content, complete=bool(message.get("tool_call_id")))
            )
        if message.get("tool_call_id") is not None:
            attrs[f"{target_prefix}{index}.tool_call_id"] = safe_text(
                message["tool_call_id"], limit=None
            )
        tool_calls = _message_tool_calls(message)
        if tool_calls:
            for call in tool_calls:
                function = call.get("function")
                if (
                    isinstance(function, dict)
                    and function.get("arguments") is not None
                    and not isinstance(function["arguments"], str)
                ):
                    function["arguments"] = _json_string(
                        function["arguments"], complete=True
                    )
            attrs[f"{target_prefix}{index}.tool_calls"] = _json_string(
                tool_calls, complete=True
            )


def _canonical_definition(value):
    if not isinstance(value, dict) or isinstance(value.get("function"), dict):
        return value
    if value.get("type") != "function" or not value.get("name"):
        return value
    return {
        "type": "function",
        "function": {k: v for k, v in value.items() if k != "type"},
    }


def _scope_name(span: ReadableSpan) -> str:
    scope = getattr(span, "instrumentation_scope", None) or getattr(
        span, "instrumentation_info", None
    )
    return safe_text(getattr(scope, "name", ""), limit=256)


def _has_parent(span: ReadableSpan) -> bool:
    parent = getattr(span, "parent", None)
    if parent is None:
        return False
    is_valid = getattr(parent, "is_valid", None)
    if is_valid is not None:
        return bool(is_valid)
    span_id = getattr(parent, "span_id", None)
    return bool(span_id) if span_id is not None else True


def _is_openlit_span(span: ReadableSpan, attrs: Mapping[str, Any]) -> bool:
    scope_name = _scope_name(span)
    if scope_name == OPENLIT_SCOPE_PREFIX or scope_name.startswith(
        f"{OPENLIT_SCOPE_PREFIX}."
    ):
        return True
    return bool(
        attrs.get("gen_ai.sdk.version")
        and (attrs.get(GEN_AI_OPERATION_NAME) or attrs.get(DB_SYSTEM_NAME))
    )


def _operation_log_type(attrs: Mapping[str, Any]) -> str:
    operation = safe_text(attrs.get(GEN_AI_OPERATION_NAME), limit=128).lower()
    if attrs.get(DB_SYSTEM_NAME):
        return "task"
    return OPENLIT_OPERATION_LOG_TYPES.get(operation, "task")


def _entity_name(span: ReadableSpan, attrs: Mapping[str, Any], log_type: str) -> str:
    candidates: tuple[Any, ...]
    if log_type == "tool":
        candidates = (attrs.get(GEN_AI_TOOL_NAME), attrs.get(DB_OPERATION_NAME))
    elif log_type == "agent":
        candidates = (attrs.get(GEN_AI_AGENT_NAME),)
    elif log_type == "workflow":
        candidates = (attrs.get(SemanticConvention.GEN_AI_WORKFLOW_NAME),)
    elif log_type == "task" and attrs.get(DB_SYSTEM_NAME):
        candidates = (
            ".".join(
                filter(
                    None,
                    [
                        safe_text(attrs.get(DB_SYSTEM_NAME), limit=128),
                        safe_text(attrs.get(DB_OPERATION_NAME), limit=128),
                    ],
                )
            ),
        )
    else:
        candidates = ()
    for candidate in candidates:
        if candidate:
            return safe_text(candidate, default=OPENLIT_INSTRUMENTATION_NAME, limit=256)
    return safe_text(
        getattr(span, "name", None),
        default=OPENLIT_INSTRUMENTATION_NAME,
        limit=256,
    )


def _set_usage(attrs):
    pairs = (
        (GEN_AI_USAGE_INPUT_TOKENS, TLSpanAttributes.LLM_USAGE_PROMPT_TOKENS),
        (GEN_AI_USAGE_OUTPUT_TOKENS, TLSpanAttributes.LLM_USAGE_COMPLETION_TOKENS),
        (
            SemanticConvention.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
            TLSpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS,
        ),
        (
            SemanticConvention.GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS,
            TLSpanAttributes.LLM_USAGE_CACHE_CREATION_INPUT_TOKENS,
        ),
    )
    for source, target in pairs:
        value = attrs.get(source)
        if type(value) is int and value >= 0:
            attrs[target] = value
        elif value is not None:
            attrs.pop(source, None)
    total = attrs.pop(
        OPENLIT_USAGE_TOTAL_TOKENS, attrs.get(TLSpanAttributes.LLM_USAGE_TOTAL_TOKENS)
    )
    if total is None:
        i, o = (
            attrs.get(GEN_AI_USAGE_INPUT_TOKENS),
            attrs.get(GEN_AI_USAGE_OUTPUT_TOKENS),
        )
        if type(i) is int and i >= 0 and type(o) is int and o >= 0:
            total = i + o
    if type(total) is int and total >= 0:
        attrs[TLSpanAttributes.LLM_USAGE_TOTAL_TOKENS] = total
    else:
        attrs.pop(TLSpanAttributes.LLM_USAGE_TOTAL_TOKENS, None)


def _clear_synthetic_usage(attrs):
    for key in list(attrs):
        if key.startswith(("gen_ai.usage.", "llm.usage.")):
            attrs.pop(key, None)


def _strip_content(attrs: dict[str, Any]) -> None:
    for key in (
        GEN_AI_INPUT_MESSAGES,
        GEN_AI_OUTPUT_MESSAGES,
        GEN_AI_SYSTEM_INSTRUCTIONS,
        SemanticConvention.GEN_AI_TOOL_DEFINITIONS,
        SemanticConvention.GEN_AI_TOOL_CALL_ARGUMENTS,
        SemanticConvention.GEN_AI_TOOL_CALL_RESULT,
        "gen_ai.prompt",
        "gen_ai.completion",
        "gen_ai.retrieval.query.text",
        "gen_ai.retrieval.documents",
        OPENLIT_RESPONSE_TOOL_CALLS,
        OPENLIT_TOOL_ARGS,
        OPENLIT_TOOL_INPUT,
        OPENLIT_TOOL_OUTPUT,
        OPENLIT_WORKFLOW_INPUT,
        OPENLIT_WORKFLOW_OUTPUT,
        DB_QUERY_TEXT,
        "db.query.parameter",
        TLSpanAttributes.LLM_REQUEST_FUNCTIONS,
        TLSpanAttributes.TRACELOOP_ENTITY_INPUT,
        TLSpanAttributes.TRACELOOP_ENTITY_OUTPUT,
    ):
        attrs.pop(key, None)
    for key in list(attrs):
        if key.startswith(
            (
                _PROMPT_PREFIX,
                _COMPLETION_PREFIX,
                "gen_ai.retrieval.",
                "db.query.parameter.",
            )
        ):
            attrs.pop(key, None)


def _strip_openlit_vendor_attributes(attrs: dict[str, Any]) -> None:
    """Drop OpenLIT-only fields while retaining standard span semantics."""

    for key in list(attrs):
        if (
            key.startswith("openlit.")
            or (
                key.startswith("gen_ai.")
                and key not in STANDARD_GEN_AI_ATTRIBUTES
                and not key.startswith(("gen_ai.prompt.", "gen_ai.completion."))
            )
            or key.startswith("db.")
            and key not in STANDARD_DB_ATTRIBUTES
        ):
            attrs.pop(key, None)


def _sanitize_url_attributes(attrs: dict[str, Any]) -> None:
    for key in list(attrs):
        normalized = key.lower()
        if not (
            normalized in {"http.url", "url.full", "url.original"}
            or normalized.endswith((".url", "_url", ".base_url", "_base_url"))
        ):
            continue
        sanitized = safe_url(attrs[key])
        if sanitized:
            attrs[key] = sanitized
        else:
            attrs.pop(key, None)


def _sanitize_error(span, attrs, capture_content):
    status = getattr(span, "status", None)
    if getattr(status, "status_code", None) is StatusCode.ERROR:
        attrs.pop(TLSpanAttributes.TRACELOOP_ENTITY_OUTPUT, None)
        for key in list(attrs):
            if key.startswith(_COMPLETION_PREFIX):
                attrs.pop(key, None)
        description = (
            safe_text(getattr(status, "description", None), limit=1024)
            if capture_content
            else None
        )
        span._status = Status(StatusCode.ERROR, description)
    attrs.pop("status_code", None)
    attrs.pop("error.message", None)


def _remove_openlit_events(span: ReadableSpan) -> None:
    events = getattr(span, "_events", None)
    if events is None:
        return
    span._events = ()


def translate_openlit_span(span: ReadableSpan, *, capture_content: bool) -> bool:
    """Translate a completed native OpenLIT span in place.

    The processor edits the mutable SDK ReadableSpan before downstream
    exporters observe it. It returns False for spans not owned by OpenLIT.
    """

    original = getattr(span, "_attributes", None)
    if original is None:
        original = getattr(span, "attributes", None)
    attrs = dict(original or {})
    if not _is_openlit_span(span, attrs):
        return False

    log_type = _operation_log_type(attrs)
    entity_name = _entity_name(span, attrs, log_type)
    attrs[RESPAN_LOG_METHOD] = LogMethodChoices.TRACING_INTEGRATION.value
    attrs[RESPAN_LOG_TYPE] = log_type
    attrs[TLSpanAttributes.TRACELOOP_ENTITY_NAME] = entity_name
    attrs[TLSpanAttributes.TRACELOOP_ENTITY_PATH] = (
        entity_name if _has_parent(span) else ""
    )

    provider = attrs.get(GEN_AI_PROVIDER_NAME) or attrs.pop(
        OPENLIT_REQUEST_PROVIDER, None
    )
    if provider is not None:
        attrs[TLSpanAttributes.LLM_SYSTEM] = safe_text(provider, limit=128).lower()

    if _is_truthy(attrs.get("gen_ai.request.stream")):
        attrs[TLSpanAttributes.LLM_IS_STREAMING] = True

    operation = safe_text(attrs.get(GEN_AI_OPERATION_NAME), limit=128).lower()
    if log_type in {"chat", "text"}:
        attrs[TLSpanAttributes.LLM_REQUEST_TYPE] = LLMRequestTypeValues.CHAT.value
    elif operation in {"embeddings", "embedding"}:
        attrs[TLSpanAttributes.LLM_REQUEST_TYPE] = LLMRequestTypeValues.EMBEDDING.value

    provider_usage = attrs.pop(OPENLIT_PROVIDER_USAGE, None)
    if provider_usage is False:
        _clear_synthetic_usage(attrs)
    else:
        _set_usage(attrs)

    source_input = attrs.pop(OPENLIT_SOURCE_INPUT, None)
    if source_input is not None:
        attrs[GEN_AI_INPUT_MESSAGES] = source_input
    source_tools = attrs.pop(OPENLIT_SOURCE_TOOLS, None)
    if source_tools is not None:
        attrs[SemanticConvention.GEN_AI_TOOL_DEFINITIONS] = source_tools
    source_output = attrs.pop(OPENLIT_SOURCE_OUTPUT, None)
    if source_output is not None:
        attrs[GEN_AI_OUTPUT_MESSAGES] = source_output
    source_usage = attrs.pop(OPENLIT_SOURCE_USAGE, None)
    if source_usage is not None:
        actual = _json_value(source_usage)
        _clear_synthetic_usage(attrs)
        for source, target in {
            "input": GEN_AI_USAGE_INPUT_TOKENS,
            "output": GEN_AI_USAGE_OUTPUT_TOKENS,
            "total": TLSpanAttributes.LLM_USAGE_TOTAL_TOKENS,
            "cache_read": TLSpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS,
            "cache_creation": TLSpanAttributes.LLM_USAGE_CACHE_CREATION_INPUT_TOKENS,
            "reasoning": TLSpanAttributes.LLM_USAGE_REASONING_TOKENS,
        }.items():
            value = actual.get(source) if isinstance(actual, dict) else None
            if type(value) is int and value >= 0:
                attrs[target] = value
        _set_usage(attrs)

    if capture_content:
        input_messages = _sequence(attrs.get(GEN_AI_INPUT_MESSAGES))
        output_messages = _sequence(attrs.get(GEN_AI_OUTPUT_MESSAGES))
        if input_messages:
            if log_type == "embedding":
                if TLSpanAttributes.TRACELOOP_ENTITY_INPUT not in attrs:
                    embedding_input = [
                        _message_content(message)
                        if isinstance(message, Mapping)
                        else message
                        for message in input_messages
                    ]
                    attrs[TLSpanAttributes.TRACELOOP_ENTITY_INPUT] = _json_string(
                        embedding_input
                    )
                attrs.pop(GEN_AI_INPUT_MESSAGES, None)
                for key in list(attrs):
                    if key.startswith(_PROMPT_PREFIX):
                        attrs.pop(key, None)
            else:
                attrs[TLSpanAttributes.TRACELOOP_ENTITY_INPUT] = _json_string(
                    input_messages
                )
                _set_message_attributes(
                    attrs, messages=input_messages, target_prefix=_PROMPT_PREFIX
                )
        if output_messages:
            attrs[TLSpanAttributes.TRACELOOP_ENTITY_OUTPUT] = _json_string(
                output_messages
            )
            _set_message_attributes(
                attrs, messages=output_messages, target_prefix=_COMPLETION_PREFIX
            )

        tool_definitions = _json_value(
            attrs.get(SemanticConvention.GEN_AI_TOOL_DEFINITIONS),
            complete=True,
            schema=True,
        )
        if tool_definitions:
            attrs[TLSpanAttributes.LLM_REQUEST_FUNCTIONS] = _json_string(
                [_canonical_definition(d) for d in tool_definitions]
                if isinstance(tool_definitions, list)
                else tool_definitions,
                complete=True,
                schema=True,
            )
        response_tool_calls = _json_value(
            attrs.pop(OPENLIT_RESPONSE_TOOL_CALLS, None), complete=True
        )
        if response_tool_calls and source_output is None:
            attrs[f"{_COMPLETION_PREFIX}0.tool_calls"] = _json_string(
                response_tool_calls, complete=True
            )

        if log_type == "tool":
            tool_input = attrs.get(SemanticConvention.GEN_AI_TOOL_CALL_ARGUMENTS)
            if tool_input is None:
                tool_input = attrs.get(OPENLIT_TOOL_ARGS, attrs.get(OPENLIT_TOOL_INPUT))
            tool_output = attrs.get(
                SemanticConvention.GEN_AI_TOOL_CALL_RESULT,
                attrs.get(OPENLIT_TOOL_OUTPUT),
            )
            if tool_input is not None:
                attrs[TLSpanAttributes.TRACELOOP_ENTITY_INPUT] = _json_string(
                    {
                        "name": entity_name,
                        "arguments": _json_value(tool_input, complete=True),
                    },
                    complete=True,
                )
            if tool_output is not None:
                attrs[TLSpanAttributes.TRACELOOP_ENTITY_OUTPUT] = _json_string(
                    _json_value(tool_output, complete=True), complete=True
                )
            for key in list(attrs):
                if key.startswith("gen_ai.tool.") and not (
                    log_type == "tool" and key == GEN_AI_TOOL_CALL_ID
                ):
                    attrs.pop(key, None)
        elif log_type == "workflow":
            workflow_input = attrs.get(OPENLIT_WORKFLOW_INPUT)
            workflow_output = attrs.get(OPENLIT_WORKFLOW_OUTPUT)
            if workflow_input is not None:
                attrs[TLSpanAttributes.TRACELOOP_ENTITY_INPUT] = _json_string(
                    _json_value(workflow_input)
                )
            if workflow_output is not None:
                attrs[TLSpanAttributes.TRACELOOP_ENTITY_OUTPUT] = _json_string(
                    _json_value(workflow_output)
                )
        elif attrs.get(DB_QUERY_TEXT) is not None:
            attrs[TLSpanAttributes.TRACELOOP_ENTITY_INPUT] = _json_string(
                {"query": attrs.get(DB_QUERY_TEXT)}
            )
    else:
        _strip_content(attrs)
        attrs = _private_attributes(attrs)

    for key in list(attrs):
        if key.startswith("gen_ai.tool.") and not (
            log_type == "tool" and key == GEN_AI_TOOL_CALL_ID
        ):
            attrs.pop(key, None)

    _sanitize_error(span, attrs, capture_content)
    _remove_openlit_events(span)
    _sanitize_url_attributes(attrs)
    for key in OFF_CONTRACT_ALIASES:
        attrs.pop(key, None)
    _strip_openlit_vendor_attributes(attrs)
    for key, value in list(attrs.items()):
        if isinstance(value, str):
            attrs[key] = safe_text(value, limit=None)
    span._attributes = attrs
    return True


def _private_attributes(attrs):
    structural = {
        RESPAN_LOG_TYPE,
        RESPAN_LOG_METHOD,
        TLSpanAttributes.TRACELOOP_ENTITY_NAME,
        TLSpanAttributes.TRACELOOP_ENTITY_PATH,
        TLSpanAttributes.LLM_REQUEST_TYPE,
        TLSpanAttributes.LLM_SYSTEM,
        TLSpanAttributes.LLM_REQUEST_MODEL,
        TLSpanAttributes.LLM_RESPONSE_MODEL,
        TLSpanAttributes.LLM_IS_STREAMING,
        GEN_AI_OPERATION_NAME,
        GEN_AI_PROVIDER_NAME,
        GEN_AI_TOOL_CALL_ID,
        ERROR_TYPE,
        DB_SYSTEM_NAME,
        DB_OPERATION_NAME,
        "server.address",
        "server.port",
        GEN_AI_USAGE_INPUT_TOKENS,
        GEN_AI_USAGE_OUTPUT_TOKENS,
        TLSpanAttributes.LLM_USAGE_PROMPT_TOKENS,
        TLSpanAttributes.LLM_USAGE_COMPLETION_TOKENS,
        TLSpanAttributes.LLM_USAGE_TOTAL_TOKENS,
        TLSpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS,
        TLSpanAttributes.LLM_USAGE_CACHE_CREATION_INPUT_TOKENS,
        TLSpanAttributes.LLM_USAGE_REASONING_TOKENS,
        "http.response.status_code",
    }
    kept = {
        k: v
        for k, v in attrs.items()
        if k in structural
        or k.startswith(("respan.threads.", "respan.trace.", "respan.customer_params."))
    }
    markers = {
        "run_id",
        "example_run_id",
        "framework",
        "example",
        "scenario",
        "example_set",
        "workflow_name",
    }
    try:
        metadata = _json_value(attrs.get(RESPAN_METADATA))
    except Exception:  # noqa: BLE001 - content stays excluded if serialization fails.
        metadata = None
    if isinstance(metadata, dict):
        metadata = {k: v for k, v in metadata.items() if k in markers}
        if metadata:
            kept[RESPAN_METADATA] = _json_string(metadata)
    for key, value in attrs.items():
        if (
            key.startswith(RESPAN_METADATA + ".")
            and key[len(RESPAN_METADATA) + 1 :] in markers
        ):
            kept[key] = value
    return kept


def _span_key(span):
    c = span.get_span_context()
    return c.trace_id, c.span_id


class OpenLITSpanProcessor(SpanProcessor):
    """Owned native spans retain their initial capture bound and observed vetoes."""

    def __init__(self, *, capture_content=True):
        self.capture_content = capture_content
        self.active = True
        self._spans = {}
        self._capture = {}
        self._parents = {}
        self._closed = {}
        self._closing_provider = None
        self._detach_hook = None
        import threading

        self._lock = threading.RLock()

    def on_start(self, span, parent_context=None):
        if not self.active:
            return
        span._respan_openlit_processor = self
        sid = _span_key(span)
        parent_span = trace.get_current_span(parent_context)
        parent = (
            _span_key(parent_span) if parent_span.get_span_context().is_valid else None
        )
        with self._lock:
            if (
                parent is not None
                and parent not in self._capture
                and parent not in self._closed
            ):
                self._spans[parent] = parent_span
                self._capture[parent] = content_allowed(
                    self.capture_content, parent_context
                )
                ancestor = getattr(parent_span, "parent", None)
                self._parents[parent] = (
                    (ancestor.trace_id, ancestor.span_id)
                    if ancestor is not None and ancestor.is_valid
                    else None
                )
            permitted = content_allowed(
                self.capture_content, parent_context
            ) and self._capture.get(parent, self._closed.get(parent, True))
            self._spans[sid] = span
            self._capture[sid] = permitted
            self._parents[sid] = parent
            if not permitted:
                self.veto(span)

    def observe_detach(self, span):
        sid = _span_key(span)
        with self._lock:
            if sid not in self._spans and sid in self._parents.values():
                self._spans[sid] = span
                self._capture[sid] = True
                ancestor = getattr(span, "parent", None)
                self._parents[sid] = (
                    (ancestor.trace_id, ancestor.span_id)
                    if ancestor is not None and ancestor.is_valid
                    else None
                )
            if sid in self._spans and (
                not content_allowed(self.capture_content)
                or not self._ancestors_allowed(sid)
            ):
                self.veto(span)

    @staticmethod
    def _clear_live_content(current):
        if current.is_recording() and _is_openlit_span(
            current, current.attributes or {}
        ):
            kept = _private_attributes(dict(current.attributes or {}))
            current._attributes.clear()
            current._attributes.update(kept)
            current._events = BoundedList(0)
            if current.status.status_code is StatusCode.ERROR:
                current._status = Status(StatusCode.ERROR)

    def veto(self, span):
        sid = _span_key(span)
        with self._lock:
            seen = set()
            while sid and sid not in seen:
                seen.add(sid)
                if sid in self._spans:
                    self._capture[sid] = False
                if sid in self._closed:
                    self._closed[sid] = False
                current = self._spans.get(sid)
                if current is not None:
                    self._clear_live_content(current)
                sid = self._parents.get(sid)
            for key, current in self._spans.items():
                if not self._ancestors_allowed(key):
                    self._capture[key] = False
                    self._clear_live_content(current)

    def _ancestors_allowed(self, sid):
        seen = set()
        while sid and sid not in seen:
            seen.add(sid)
            if not self._capture.get(sid, self._closed.get(sid, True)):
                return False
            sid = self._parents.get(sid)
        return True

    def allowed(self, span):
        if not getattr(span, "is_recording", lambda: False)():
            return False
        with self._lock:
            if not content_allowed(self.capture_content) or not self._ancestors_allowed(
                _span_key(span)
            ):
                self.veto(span)
            sid = _span_key(span)
            return self.active and self._capture.get(sid, False)

    def on_end(self, span):
        sid = _span_key(span)
        with self._lock:
            if sid not in self._spans:
                return
            if not content_allowed(self.capture_content) or not self._ancestors_allowed(
                _span_key(span)
            ):
                self.veto(span)
            capture = self._capture.pop(sid, False)
            self._closed[sid] = capture
            self._spans.pop(sid, None)
            if len(self._closed) > 4096:
                old = next(iter(self._closed))
                self._closed.pop(old, None)
                self._parents.pop(old, None)
        try:
            translate_openlit_span(span, capture_content=capture)
        except Exception:  # noqa: BLE001 - telemetry must preserve native outcomes.
            if _is_openlit_span(span, span.attributes or {}):
                span._attributes = _private_attributes(dict(span.attributes or {}))
                span._events = ()
                if span.status.status_code is StatusCode.ERROR:
                    span._status = Status(StatusCode.ERROR)

        self._finish_retirement()

    def retire(self, provider):
        self.active = False
        self._closing_provider = provider
        self._finish_retirement()

    def _finish_retirement(self):
        if self._closing_provider is not None and not any(
            _is_openlit_span(s, getattr(s, "attributes", {}) or {})
            for s in self._spans.values()
        ):
            from ._instrumentation import _unregister

            _unregister(self._closing_provider, self)
            self._closing_provider = None
            self.shutdown()

    def shutdown(self):
        from ._guard import remove_context_guard

        remove_context_guard(self._detach_hook)
        self._detach_hook = None
        self.active = False
        with self._lock:
            self._spans.clear()
            self._capture.clear()
            self._parents.clear()
            self._closed.clear()

    def force_flush(self, timeout_millis=30000):
        return True
