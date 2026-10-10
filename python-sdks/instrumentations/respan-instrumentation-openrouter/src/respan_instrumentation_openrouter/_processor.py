"""OpenRouter span normalization for the Respan OTLP pipeline."""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor
from opentelemetry.sdk.util.instrumentation import InstrumentationScope
from opentelemetry.semconv._incubating.attributes.error_attributes import ERROR_MESSAGE
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import (
    GEN_AI_OUTPUT_MESSAGES,
    GEN_AI_PROVIDER_NAME,
    GEN_AI_REQUEST_STREAM,
    GEN_AI_RESPONSE_ID,
    GEN_AI_TOOL_DEFINITIONS,
    GEN_AI_USAGE_INPUT_TOKENS,
    GEN_AI_USAGE_OUTPUT_TOKENS,
)
from opentelemetry.semconv.attributes.http_attributes import HTTP_RESPONSE_STATUS_CODE
from opentelemetry.semconv_ai import LLMRequestTypeValues
from opentelemetry.semconv_ai import SpanAttributes as TLSpanAttributes
from opentelemetry.trace import SpanContext, Status, StatusCode
from respan_sdk.constants.llm_logging import (
    LOG_TYPE_CHAT,
    LOG_TYPE_EMBEDDING,
    LOG_TYPE_TEXT,
)
from respan_sdk.constants.span_attributes import (
    RESPAN_LOG_TYPE,
    RESPAN_SPAN_HANDOFFS,
    RESPAN_SPAN_TOOL_CALLS,
    RESPAN_SPAN_TOOLS,
)

from respan_instrumentation_openrouter._constants import (
    MAX_ATTRIBUTE_BYTES,
    MAX_COLLECTION_ITEMS,
    MAX_ERROR_BYTES,
    OPENAI_INSTRUMENTATION_SCOPE_FRAGMENT,
    OPENROUTER_INSTRUMENTATION_SCOPE,
    OPENROUTER_SYSTEM_NAME,
    OPENROUTER_URL_ATTRIBUTE_KEYS,
    SENSITIVE_KEY_NAMES,
)

OTEL_SCOPE_NAME_ATTR = "otel.scope.name"

GEN_AI_PROMPT_PREFIX = f"{TLSpanAttributes.LLM_PROMPTS}."
GEN_AI_COMPLETION_PREFIX = f"{TLSpanAttributes.LLM_COMPLETIONS}."
GEN_AI_TOOL_CALLS_SUFFIX = ".tool_calls"
GEN_AI_TOOL_CALLS_INDEX_FRAGMENT = ".tool_calls."
GEN_AI_COMPLETION_TOOL_CALLS_ATTR = f"{TLSpanAttributes.LLM_COMPLETIONS}.0.tool_calls"
_PROVIDER_CONTROLLED_IDENTITY_KEYS = (
    TLSpanAttributes.LLM_REQUEST_MODEL,
    TLSpanAttributes.LLM_RESPONSE_MODEL,
    GEN_AI_RESPONSE_ID,
    TLSpanAttributes.LLM_OPENAI_RESPONSE_SYSTEM_FINGERPRINT,
)

_OFF_CONTRACT_ALIAS_ATTRIBUTES = frozenset(
    {
        "status_code",
        "completion_tokens",
        "has_tool_calls",
        "model",
        "parallel_tool_calls",
        "prompt_tokens",
        "span_tools",
        "tool_calls",
        "tools",
        "total_request_tokens",
        RESPAN_SPAN_HANDOFFS,
        RESPAN_SPAN_TOOL_CALLS,
        RESPAN_SPAN_TOOLS,
        TLSpanAttributes.TRACELOOP_SPAN_KIND,
    }
)

_OPENAI_OMIT_VALUE_PREFIX = "<openai.Omit object"

_BEARER_SECRET = re.compile(r"(?i)\bBearer\s+(?:\[REDACTED\]|[^\s,;\"'{}\[\]\\]+)")
_OPENAI_STYLE_SECRET = re.compile(r"\bsk-(?:or-v1-)?[A-Za-z0-9_-]{8,}\b")
_SECRET_FIELD_PATTERN = (
    r"(?:api[_-]?key|authorization|access[_-]?token|refresh[_-]?token|"
    r"auth[_-]?token|client[_-]?secret|db[_-]?password|password|credential|"
    r"secret|token)"
)
_KEY_VALUE_SECRET = re.compile(
    rf"(?P<prefix>['\"]?{_SECRET_FIELD_PATTERN}['\"]?\s*[:=]\s*)"
    rf"(?:(?P<quote>['\"])(?P<quoted>(?:\\.|(?!(?P=quote)|\\).)*)(?P=quote)|"
    rf"(?P<bare>\[REDACTED\]|[^\s,;&?#}}\]\[\"']+))",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class OpenRouterEmissionContext:
    """Request facts lost by the delegated OpenAI ReadableSpan builder."""

    kind: str
    stream: bool
    error_message: str | None
    status_code: int | None


_CURRENT_EMISSION: ContextVar[OpenRouterEmissionContext | None] = ContextVar(
    "respan_openrouter_emission",
    default=None,
)


@contextmanager
def openrouter_emission_context(
    *,
    kind: str,
    request_kwargs: Mapping[str, Any],
    error_message: str | None,
    status_code: int,
):
    """Expose delegate-only request/error facts to the synchronous processor."""

    resolved_status = (
        status_code if type(status_code) is int and 100 <= status_code <= 599 else None
    )
    token = _CURRENT_EMISSION.set(
        OpenRouterEmissionContext(
            kind=kind,
            stream=bool(request_kwargs.get("stream")),
            error_message=error_message,
            status_code=resolved_status,
        )
    )
    try:
        yield
    finally:
        _CURRENT_EMISSION.reset(token)


def _span_scope_name(span: ReadableSpan, attrs: dict[str, Any]) -> str | None:
    scope = getattr(span, "instrumentation_scope", None) or getattr(
        span,
        "_instrumentation_scope",
        None,
    )
    scope_name = getattr(scope, "name", None)
    if scope_name:
        return scope_name
    attr_scope_name = attrs.get(OTEL_SCOPE_NAME_ATTR)
    return attr_scope_name if isinstance(attr_scope_name, str) else None


def _is_openai_span(span: ReadableSpan, attrs: dict[str, Any]) -> bool:
    scope_name = _span_scope_name(span, attrs)
    if (
        isinstance(scope_name, str)
        and OPENAI_INSTRUMENTATION_SCOPE_FRAGMENT in scope_name.lower()
    ):
        return True

    system = attrs.get(TLSpanAttributes.LLM_SYSTEM) or attrs.get("llm.system")
    return isinstance(system, str) and system.lower() == "openai"


def _has_openrouter_url_marker(attrs: dict[str, Any]) -> bool:
    for key in OPENROUTER_URL_ATTRIBUTE_KEYS:
        value = attrs.get(key)
        if not isinstance(value, str):
            continue
        parsed = urlsplit(value if "://" in value else "//" + value)
        host = parsed.hostname
        if host == "openrouter.ai" or (host and host.endswith(".openrouter.ai")):
            return True
    return False


def _drop_attribute(attrs: dict[str, Any], key: str) -> None:
    attrs.pop(key, None)


def _json_string(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return json.dumps(
        _safe_json_value(value),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _parse_json_if_string(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return value


def _is_openai_omit(value: Any) -> bool:
    if isinstance(value, str) and value.startswith(_OPENAI_OMIT_VALUE_PREFIX):
        return True
    value_type = type(value)
    return value_type.__name__ == "Omit" and value_type.__module__.startswith("openai")


def _truncate_utf8(value: str, *, limit: int) -> str:
    if limit <= 0:
        return ""
    candidate = value[:limit]
    encoded = candidate.encode("utf-8")
    if len(value) <= limit and len(encoded) <= limit:
        return value
    suffix = "...[truncated]"
    suffix_bytes = suffix.encode("utf-8")
    if limit <= len(suffix_bytes):
        return suffix_bytes[: max(0, limit)].decode("utf-8", errors="ignore")
    prefix = encoded[: max(0, limit - len(suffix_bytes))]
    return prefix.decode("utf-8", errors="ignore") + suffix


def _redact_text(value: str, *, limit: int = MAX_ATTRIBUTE_BYTES) -> str:
    candidate = _truncate_utf8(value, limit=max(limit * 2, limit))
    redacted = _BEARER_SECRET.sub("Bearer [REDACTED]", candidate)
    redacted = _OPENAI_STYLE_SECRET.sub("[REDACTED]", redacted)

    def replace_secret(match: re.Match[str]) -> str:
        quote = match.group("quote") or ""
        return f"{match.group('prefix')}{quote}[REDACTED]{quote}"

    redacted = _KEY_VALUE_SECRET.sub(replace_secret, redacted)
    return _truncate_utf8(redacted, limit=limit)


def _is_sensitive_key(value: str) -> bool:
    normalized = value.lower().replace("-", "_")
    collapsed = normalized.replace("_", "")
    collapsed_sensitive_names = {
        sensitive_name.replace("_", "") for sensitive_name in SENSITIVE_KEY_NAMES
    }
    return (
        normalized in SENSITIVE_KEY_NAMES
        or collapsed in collapsed_sensitive_names
        or normalized.endswith(
            (
                "_api_key",
                "_authorization",
                "_credential",
                "_password",
                "_secret",
                "_token",
            )
        )
    )


def _redact_url(value: str) -> str:
    """Redact credentials and sensitive query values without losing URL identity."""

    safe_value = _redact_text(value)
    try:
        parts = urlsplit(safe_value)
    except ValueError:
        return safe_value
    if not parts.scheme or not parts.netloc:
        return safe_value

    netloc = parts.netloc.rsplit("@", 1)[-1]
    try:
        query_items = parse_qsl(
            parts.query,
            keep_blank_values=True,
            max_num_fields=MAX_COLLECTION_ITEMS,
        )
    except ValueError:
        query = "truncated_query=%5BREDACTED%5D"
    else:
        query = urlencode(
            [
                (
                    _redact_text(key, limit=256),
                    "[REDACTED]" if _is_sensitive_key(key) else _redact_text(item),
                )
                for key, item in query_items
            ]
        )
    return _truncate_utf8(
        urlunsplit(
            (
                parts.scheme,
                netloc,
                parts.path,
                query,
                _redact_text(parts.fragment),
            )
        ),
        limit=MAX_ATTRIBUTE_BYTES,
    )


def _safe_json_value(value: Any, *, depth: int = 0, budget: Any = None) -> Any:
    if depth > 40:
        return None
    if value is None or type(value) in (bool, int):
        return value
    if type(value) is float:
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        return _redact_text(value, limit=len(value.encode("utf-8")) + 100)
    if type(value) is dict:
        return {
            key: "[REDACTED]"
            if _is_sensitive_key(key)
            else _safe_json_value(item, depth=depth + 1)
            for key, item in value.items()
            if isinstance(key, str)
        }
    if type(value) in (list, tuple):
        return [_safe_json_value(item, depth=depth + 1) for item in value]
    return None


@lru_cache(maxsize=1)
def _package_version() -> str | None:
    try:
        return version("respan-instrumentation-openrouter")
    except PackageNotFoundError:
        return None


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


def _is_gen_ai_message_tool_call_key(key: str) -> bool:
    return (
        key.startswith((GEN_AI_PROMPT_PREFIX, GEN_AI_COMPLETION_PREFIX))
        and GEN_AI_TOOL_CALLS_INDEX_FRAGMENT in key
    )


def _is_gen_ai_message_tool_call_aggregate_key(key: str) -> bool:
    return key.startswith(
        (GEN_AI_PROMPT_PREFIX, GEN_AI_COMPLETION_PREFIX)
    ) and key.endswith(GEN_AI_TOOL_CALLS_SUFFIX)


def _tool_calls_content(value: Any) -> str | None:
    tool_calls = _parse_json_if_string(value)
    if not isinstance(tool_calls, list) or not tool_calls:
        return None

    descriptions: list[str] = []
    for tool_call in tool_calls[:MAX_COLLECTION_ITEMS]:
        if not isinstance(tool_call, dict):
            continue
        function = tool_call.get("function")
        if not isinstance(function, dict):
            continue
        name = function.get("name")
        if not isinstance(name, str) or not name:
            continue
        arguments = function.get("arguments")
        if arguments is None or arguments == "":
            descriptions.append(name)
        elif isinstance(arguments, str):
            descriptions.append(f"{name}({arguments})")
        else:
            safe_arguments = _json_string(arguments) or "{}"
            descriptions.append(f"{name}({safe_arguments})")

    if not descriptions:
        return None
    prefix = "Tool call" if len(descriptions) == 1 else "Tool calls"
    return _redact_text(f"{prefix}: {', '.join(descriptions)}")


def _normalize_gen_ai_tool_calls(attrs: dict[str, Any]) -> None:
    indexed_calls: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)

    for key in tuple(attrs):
        if not _is_gen_ai_message_tool_call_key(key):
            continue
        message_key, rest = key.split(GEN_AI_TOOL_CALLS_INDEX_FRAGMENT, 1)
        aggregate_key = f"{message_key}{GEN_AI_TOOL_CALLS_SUFFIX}"
        parts = rest.split(".", 1)
        if not parts[0].isdigit() or len(parts) == 1:
            continue
        tool_call_index = int(parts[0])
        tool_call = indexed_calls[aggregate_key].setdefault(tool_call_index, {})
        _set_nested_value(tool_call, parts[1], attrs[key])

    for aggregate_key, tool_calls_by_index in indexed_calls.items():
        if aggregate_key not in attrs:
            attrs[aggregate_key] = [
                tool_calls_by_index[index] for index in sorted(tool_calls_by_index)
            ]

    for key in tuple(attrs):
        if _is_gen_ai_message_tool_call_aggregate_key(key):
            structured_tool_calls = _parse_json_if_string(attrs[key])
            attrs[key] = _json_string(structured_tool_calls) or "[]"
            message_key = key[: -len(GEN_AI_TOOL_CALLS_SUFFIX)]
            content_key = f"{message_key}.content"
            content_value = attrs.get(content_key)
            if content_value is None or content_value == "":
                attrs[content_key] = _tool_calls_content(structured_tool_calls) or ""
            role_key = f"{message_key}.role"
            if attrs.get(role_key) is None:
                attrs[role_key] = "assistant"
        elif _is_gen_ai_message_tool_call_key(key):
            _drop_attribute(attrs, key)


def _canonical_tool_definition(tool_definition: Any) -> Any:
    if not isinstance(tool_definition, dict):
        return tool_definition
    if tool_definition.get("type") != "function":
        return tool_definition
    if "function" in tool_definition:
        return tool_definition

    name = tool_definition.get("name")
    if not isinstance(name, str) or not name:
        # An incomplete upstream definition is still useful diagnostic data.
        # Preserve it verbatim instead of fabricating a callable named null.
        return tool_definition

    function = {"name": name}
    if tool_definition.get("description") is not None:
        function["description"] = tool_definition.get("description")
    if tool_definition.get("parameters") is not None:
        function["parameters"] = tool_definition.get("parameters")
    return {"type": "function", "function": function}


def _canonical_tool_definitions(value: Any) -> Any:
    tool_definitions = _parse_json_if_string(value)
    if not isinstance(tool_definitions, list):
        return tool_definitions
    return [
        _canonical_tool_definition(tool)
        for tool in tool_definitions[:MAX_COLLECTION_ITEMS]
    ]


def _tool_call_from_output_part(part: dict[str, Any]) -> dict[str, Any] | None:
    name = part.get("name")
    if not isinstance(name, str) or not name:
        return None
    arguments = part.get("arguments")
    if not isinstance(arguments, str):
        arguments = _json_string(arguments if arguments is not None else {})
        if arguments is None:
            arguments = "{}"
    tool_call: dict[str, Any] = {
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }
    tool_call_id = part.get("id")
    if isinstance(tool_call_id, str) and tool_call_id:
        tool_call["id"] = tool_call_id
    return tool_call


def _normalize_gen_ai_output_messages(attrs: dict[str, Any]) -> None:
    output_messages = _parse_json_if_string(attrs.get(GEN_AI_OUTPUT_MESSAGES))
    if not isinstance(output_messages, list):
        return

    for message_index, message in enumerate(output_messages[:MAX_COLLECTION_ITEMS]):
        if not isinstance(message, dict):
            continue
        prefix = f"{TLSpanAttributes.LLM_COMPLETIONS}.{message_index}"
        role = message.get("role")
        if isinstance(role, str) and role:
            attrs.setdefault(f"{prefix}.role", role)

        parts = message.get("parts") or []
        if not isinstance(parts, list):
            continue

        content = ""
        tool_calls: list[dict[str, Any]] = []
        for part in parts[:MAX_COLLECTION_ITEMS]:
            if not isinstance(part, dict):
                continue
            part_type = part.get("type")
            if part_type == "text" and isinstance(part.get("content"), str):
                remaining = MAX_ATTRIBUTE_BYTES - len(content.encode("utf-8"))
                if remaining > 0:
                    content += _redact_text(part["content"], limit=remaining)
            elif part_type == "tool_call":
                tool_call = _tool_call_from_output_part(part)
                if tool_call is not None:
                    tool_calls.append(tool_call)

        if content:
            attrs.setdefault(f"{prefix}.content", content)
        if tool_calls:
            attrs.setdefault(f"{prefix}.tool_calls", json.dumps(tool_calls))
            if not attrs.get(f"{prefix}.content"):
                attrs[f"{prefix}.content"] = _tool_calls_content(tool_calls) or ""


def _normalize_structured_contract_attrs(attrs: dict[str, Any]) -> None:
    tools_value = attrs.get(TLSpanAttributes.LLM_REQUEST_FUNCTIONS)
    if tools_value is None:
        tools_value = attrs.get(RESPAN_SPAN_TOOLS)
    if tools_value is None:
        tools_value = attrs.get(GEN_AI_TOOL_DEFINITIONS)
    tools_json = _json_string(_canonical_tool_definitions(tools_value))
    if tools_json:
        attrs[TLSpanAttributes.LLM_REQUEST_FUNCTIONS] = tools_json

    tool_calls_value = attrs.get(GEN_AI_COMPLETION_TOOL_CALLS_ATTR)
    if tool_calls_value is None:
        tool_calls_value = attrs.get(RESPAN_SPAN_TOOL_CALLS) or attrs.get("tool_calls")
    tool_calls_json = _json_string(_parse_json_if_string(tool_calls_value))
    if tool_calls_json:
        attrs[GEN_AI_COMPLETION_TOOL_CALLS_ATTR] = tool_calls_json


def _has_llm_message_attrs(attrs: dict[str, Any]) -> bool:
    return any(
        key.startswith((GEN_AI_PROMPT_PREFIX, GEN_AI_COMPLETION_PREFIX))
        for key in attrs
    )


def _normalize_usage(attrs: dict[str, Any]) -> None:
    prompt_tokens = attrs.get(TLSpanAttributes.LLM_USAGE_PROMPT_TOKENS)
    completion_tokens = attrs.get(TLSpanAttributes.LLM_USAGE_COMPLETION_TOKENS)
    total_tokens = attrs.get(TLSpanAttributes.LLM_USAGE_TOTAL_TOKENS)

    if prompt_tokens is not None:
        attrs[GEN_AI_USAGE_INPUT_TOKENS] = prompt_tokens
    if completion_tokens is not None:
        attrs[GEN_AI_USAGE_OUTPUT_TOKENS] = completion_tokens
    if total_tokens is not None:
        attrs[TLSpanAttributes.GEN_AI_USAGE_TOTAL_TOKENS] = total_tokens


def _normalize_stream(
    attrs: dict[str, Any], emission: OpenRouterEmissionContext | None
) -> None:
    is_stream = bool(
        (emission is not None and emission.stream)
        or attrs.get(GEN_AI_REQUEST_STREAM)
        or attrs.get(TLSpanAttributes.LLM_IS_STREAMING)
        or attrs.get(TLSpanAttributes.GEN_AI_IS_STREAMING)
    )
    if not is_stream:
        return
    attrs[GEN_AI_REQUEST_STREAM] = True
    attrs[TLSpanAttributes.LLM_IS_STREAMING] = True


def _error_message(
    span: ReadableSpan, emission: OpenRouterEmissionContext | None
) -> str | None:
    if emission is not None and emission.error_message:
        return emission.error_message
    try:
        status = getattr(span, "status", None)
        status_code = getattr(status, "status_code", None)
    except Exception:  # noqa: BLE001 - span/status properties may be hostile
        return None
    if status_code is StatusCode.ERROR:
        try:
            description = getattr(status, "description", None)
        except Exception:  # noqa: BLE001 - status descriptions may be hostile
            description = None
        if isinstance(description, str) and description:
            return description
        return "OpenRouter request failed"
    return None


def _normalize_error(
    span: ReadableSpan,
    attrs: dict[str, Any],
    emission: OpenRouterEmissionContext | None,
) -> None:
    message = _error_message(span, emission)
    if message is None:
        return

    safe_message = _redact_text(message, limit=MAX_ERROR_BYTES)
    attrs[ERROR_MESSAGE] = safe_message
    if (
        emission is not None
        and type(emission.status_code) is int
        and 100 <= emission.status_code <= 599
    ):
        attrs[HTTP_RESPONSE_STATUS_CODE] = emission.status_code
    if hasattr(span, "_status"):
        span._status = Status(StatusCode.ERROR)


def _drop_content(attrs: dict[str, Any]) -> None:
    keep = {
        TLSpanAttributes.LLM_SYSTEM,
        GEN_AI_PROVIDER_NAME,
        TLSpanAttributes.LLM_REQUEST_MODEL,
        TLSpanAttributes.LLM_RESPONSE_MODEL,
        GEN_AI_RESPONSE_ID,
        TLSpanAttributes.LLM_REQUEST_TYPE,
        TLSpanAttributes.GEN_AI_IS_STREAMING,
        GEN_AI_REQUEST_STREAM,
        RESPAN_LOG_TYPE,
        TLSpanAttributes.TRACELOOP_ENTITY_NAME,
        TLSpanAttributes.TRACELOOP_ENTITY_PATH,
        HTTP_RESPONSE_STATUS_CODE,
    }
    for key in tuple(attrs):
        if key not in keep and not key.startswith(("gen_ai.usage.", "llm.usage.")):
            attrs.pop(key, None)


def _canonical_operation_kind(
    attrs: dict[str, Any], emission: OpenRouterEmissionContext | None
) -> str:
    if emission is not None and emission.kind in {
        "chat",
        "completion",
        "embedding",
        "response",
    }:
        return emission.kind

    request_type = attrs.get(TLSpanAttributes.LLM_REQUEST_TYPE)
    if request_type == LLMRequestTypeValues.EMBEDDING.value:
        return "embedding"
    log_type = attrs.get(RESPAN_LOG_TYPE)
    if log_type in {"completion", LOG_TYPE_TEXT}:
        return "completion"
    if log_type == "response":
        return "response"
    entity_name = attrs.get(TLSpanAttributes.TRACELOOP_ENTITY_NAME)
    if isinstance(entity_name, str):
        suffix = entity_name.rsplit(".", 1)[-1]
        if suffix in {"chat", "completion", "response"}:
            return suffix
        if suffix in {"embedding", "embeddings"}:
            return "embedding"
    return "chat"


def _normalize_common_contract(
    span: ReadableSpan,
    attrs: dict[str, Any],
    emission: OpenRouterEmissionContext | None,
) -> None:
    kind = _canonical_operation_kind(attrs, emission)
    if kind == "embedding":
        attrs[TLSpanAttributes.LLM_REQUEST_TYPE] = LLMRequestTypeValues.EMBEDDING.value
        attrs[RESPAN_LOG_TYPE] = LOG_TYPE_EMBEDDING
        detail = "embeddings"
    elif kind == "completion":
        attrs[TLSpanAttributes.LLM_REQUEST_TYPE] = LLMRequestTypeValues.CHAT.value
        attrs[RESPAN_LOG_TYPE] = LOG_TYPE_TEXT
        detail = "completion"
    else:
        attrs[TLSpanAttributes.LLM_REQUEST_TYPE] = LLMRequestTypeValues.CHAT.value
        attrs[RESPAN_LOG_TYPE] = LOG_TYPE_CHAT
        detail = "response" if kind == "response" else "chat"

    entity_name = f"openrouter.{detail}"
    attrs[TLSpanAttributes.TRACELOOP_ENTITY_NAME] = entity_name
    try:
        parent = getattr(span, "parent", None)
    except Exception:  # noqa: BLE001 - parent properties may be hostile
        parent = None
    is_nested = parent is not None
    if isinstance(parent, SpanContext):
        is_nested = parent.is_valid
    attrs[TLSpanAttributes.TRACELOOP_ENTITY_PATH] = entity_name if is_nested else ""


def _sanitize_url_attributes(attrs: dict[str, Any]) -> None:
    for key in OPENROUTER_URL_ATTRIBUTE_KEYS:
        value = attrs.get(key)
        if isinstance(value, str):
            attrs[key] = _redact_url(value)


def _sanitize_identity_attributes(attrs: dict[str, Any]) -> None:
    """Bound retained provider identifiers without coercing arbitrary values."""

    for key in _PROVIDER_CONTROLLED_IDENTITY_KEYS:
        value = attrs.get(key)
        if isinstance(value, str):
            attrs[key] = _redact_text(value)


class OpenRouterSpanProcessor(SpanProcessor):
    """Normalize OpenAI-compatible OpenRouter spans before Respan export."""

    def __init__(
        self,
        *,
        normalize_all_openai_spans: bool = True,
        capture_content: bool = True,
    ) -> None:
        self._normalize_all_openai_spans = normalize_all_openai_spans
        self._capture_content = capture_content
        from respan_instrumentation_openrouter._policy import Policy

        self._policy = Policy(capture_content)

    def on_start(self, span: Any, parent_context: Any = None) -> None:
        self._policy.start(span, parent_context)

    def _is_openrouter_span(self, span: ReadableSpan, attrs: dict[str, Any]) -> bool:
        system = attrs.get(TLSpanAttributes.LLM_SYSTEM)
        if isinstance(system, str) and system.lower() == OPENROUTER_SYSTEM_NAME:
            return True
        if not _is_openai_span(span, attrs):
            return False
        return self._normalize_all_openai_spans or _has_openrouter_url_marker(attrs)

    def on_end(self, span: ReadableSpan) -> None:
        allowed = self._policy.finish(span)
        if (
            getattr(getattr(span, "instrumentation_scope", None), "name", None)
            == "respan.instrumentation.openrouter"
        ):
            return
        original_attrs = getattr(span, "_attributes", None)
        if original_attrs is None:
            return

        attrs = dict(original_attrs)
        if not self._is_openrouter_span(span, attrs):
            return

        attrs[TLSpanAttributes.LLM_SYSTEM] = OPENROUTER_SYSTEM_NAME
        attrs[GEN_AI_PROVIDER_NAME] = OPENROUTER_SYSTEM_NAME

        emission = _CURRENT_EMISSION.get()
        _normalize_common_contract(span, attrs, emission)

        _normalize_gen_ai_output_messages(attrs)
        _normalize_gen_ai_tool_calls(attrs)
        _normalize_structured_contract_attrs(attrs)
        _normalize_usage(attrs)

        _normalize_stream(attrs, emission)
        _normalize_error(span, attrs, emission)
        _sanitize_identity_attributes(attrs)
        _sanitize_url_attributes(attrs)

        for key, value in list(attrs.items()):
            if key in _OFF_CONTRACT_ALIAS_ATTRIBUTES or _is_openai_omit(value):
                attrs.pop(key, None)

        if self._capture_content and allowed:
            for key, value in list(attrs.items()):
                is_content = key in {
                    TLSpanAttributes.TRACELOOP_ENTITY_INPUT,
                    TLSpanAttributes.TRACELOOP_ENTITY_OUTPUT,
                    TLSpanAttributes.LLM_REQUEST_FUNCTIONS,
                    GEN_AI_OUTPUT_MESSAGES,
                    GEN_AI_TOOL_DEFINITIONS,
                } or key.startswith((GEN_AI_PROMPT_PREFIX, GEN_AI_COMPLETION_PREFIX))
                if is_content and isinstance(value, str):
                    parsed = _parse_json_if_string(value)
                    attrs[key] = (
                        _json_string(parsed)
                        if parsed is not value
                        else _redact_text(value, limit=len(value.encode("utf-8")) + 100)
                    )

        else:
            _drop_content(attrs)

        span._attributes = attrs
        if hasattr(span, "_instrumentation_scope"):
            span._instrumentation_scope = InstrumentationScope(
                name=OPENROUTER_INSTRUMENTATION_SCOPE,
                version=_package_version(),
            )

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True
