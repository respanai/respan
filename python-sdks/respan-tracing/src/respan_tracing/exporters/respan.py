import base64
import hashlib
import importlib.metadata
import json
import os
import re
from collections.abc import Mapping
from types import SimpleNamespace
from typing import Dict, Optional, Sequence, List, Any

import requests
from opentelemetry.context import attach, detach, set_value
from opentelemetry.context import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.sdk.trace.export import SpanExportResult
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.trace import StatusCode

from respan_sdk.constants import RESPAN_DOGFOOD_HEADER
from respan_sdk.utils.data_processing.id_processing import format_trace_id, format_span_id
from respan_sdk.constants.otlp_constants import (
    OTLP_BOOL_VALUE,
    OTLP_INT_VALUE,
    OTLP_DOUBLE_VALUE,
    OTLP_STRING_VALUE,
    OTLP_BYTES_VALUE,
    OTLP_ARRAY_VALUE,
    OTLP_ARRAY_VALUES_KEY,
    OTLP_KVLIST_VALUE,
    OTLP_ATTR_KEY,
    OTLP_ATTR_VALUE,
    OTLP_TRACE_ID_KEY,
    OTLP_SPAN_ID_KEY,
    OTLP_PARENT_SPAN_ID_KEY,
    OTLP_NAME_KEY,
    OTLP_KIND_KEY,
    OTLP_START_TIME_KEY,
    OTLP_END_TIME_KEY,
    OTLP_ATTRIBUTES_KEY,
    OTLP_STATUS_KEY,
    OTLP_EVENTS_KEY,
    OTLP_LINKS_KEY,
    OTLP_FLAGS_KEY,
    OTLP_TRACE_STATE_KEY,
    OTLP_DROPPED_ATTRIBUTES_COUNT_KEY,
    OTLP_REMOTE_LINK_FLAG,
    OTLP_RESOURCE_SPANS_KEY,
    OTLP_SCOPE_SPANS_KEY,
    OTLP_RESOURCE_KEY,
    OTLP_SCOPE_KEY,
    OTLP_SPANS_KEY,
    OTLP_VERSION_KEY,
    OTEL_STATUS_CODE_UNSET,
    OTEL_STATUS_CODE_OK,
    OTEL_STATUS_CODE_ERROR,
    OTEL_STATUS_CODE_KEY,
    OTEL_STATUS_MESSAGE_KEY,
    OTEL_SPAN_PARENT_FIELD,
    OTEL_SPAN_ATTRIBUTES_FIELD,
)

from opentelemetry.semconv_ai import SpanAttributes, LLMRequestTypeValues

from respan_sdk.constants.llm_logging import (
    LOG_TYPE_AGENT,
    LOG_TYPE_CHAT,
    LOG_TYPE_COMPLETION,
    LOG_TYPE_EMBEDDING,
    LOG_TYPE_FUNCTION,
    LOG_TYPE_GENERATION,
    LOG_TYPE_GUARDRAIL,
    LOG_TYPE_HANDOFF,
    LOG_TYPE_RESPONSE,
    LOG_TYPE_TASK,
    LOG_TYPE_TEXT,
    LOG_TYPE_TOOL,
    LOG_TYPE_WORKFLOW,
)
from respan_sdk.constants.span_attributes import (
    GEN_AI_AGENT_NAME,
    GEN_AI_OPERATION_NAME,
    GEN_AI_SYSTEM,
    GEN_AI_TOOL_NAME,
    LLM_REQUEST_MODEL,
    LLM_REQUEST_TYPE,
    RESPAN_INTERNAL_DROP_SPAN,
    RESPAN_INTERNAL_EXPORT_PARENT,
    RESPAN_INTERNAL_SPAN_NAME_DETAIL,
    RESPAN_INTERNAL_SPAN_NAME_KIND,
    RESPAN_LOG_TYPE,
    RESPAN_METADATA_FROM_AGENT,
    RESPAN_METADATA_GUARDRAIL_NAME,
    RESPAN_METADATA_INTERNAL_TRACING_SDK_VERSION,
    RESPAN_METADATA_TO_AGENT,
    RESPAN_SPAN_TOOL_CALLS,
    RESPAN_SPAN_TOOLS,
)
from respan_tracing.utils.logging import get_respan_logger, build_spans_export_preview
from respan_tracing.constants.generic_constants import LOGGER_NAME_EXPORTER

logger = get_respan_logger(LOGGER_NAME_EXPORTER)

try:
    _RESPAN_TRACING_SDK_VERSION = importlib.metadata.version("respan-tracing")
except importlib.metadata.PackageNotFoundError:
    _RESPAN_TRACING_SDK_VERSION = ""


def _resolve_traces_endpoint(endpoint: str) -> str:
    """Normalize a Respan base URL or full traces URL to /v2/traces."""
    normalized_endpoint = endpoint.rstrip("/")
    if normalized_endpoint.endswith("/v2/traces"):
        return normalized_endpoint
    return f"{normalized_endpoint}/v2/traces"


class ModifiedSpan:
    """A proxy wrapper that forwards the original span with optional overrides."""

    def __init__(
        self,
        original_span: ReadableSpan,
        overrides: Optional[Dict[str, Any]] = None,
    ):
        self._original_span = original_span
        self._overrides = overrides or {}

    def __getattr__(self, name):
        """Forward all attribute access to the original span unless overridden."""
        if name in self._overrides:
            return self._overrides[name]
        return getattr(self._original_span, name)


class SyntheticSpan:
    """A lightweight ReadableSpan-compatible object for exporter-only spans."""

    def __init__(
        self,
        *,
        name: str,
        trace_id: int,
        span_id: int,
        parent: Any,
        attributes: Dict[str, Any],
        start_time: Optional[int],
        end_time: Optional[int],
        status: Any,
        kind: Any,
        resource: Any,
        instrumentation_scope: Any,
    ) -> None:
        self.name = name
        self.parent = parent
        self._parent = parent
        self.attributes = attributes
        self.kind = kind
        self.start_time = start_time
        self.end_time = end_time
        self.status = status
        self.events = []
        self.links = ()
        self.resource = resource
        self.instrumentation_scope = instrumentation_scope
        self._span_context = SimpleNamespace(trace_id=trace_id, span_id=span_id)

    def get_span_context(self) -> Any:
        return self._span_context


_CLAUDE_AGENT_SCOPE_NAME = "openinference.instrumentation.claude_agent_sdk"
_CLAUDE_AGENT_RESPONSE_SPAN_NAMES = frozenset({
    "ClaudeAgentSDK.query",
    "ClaudeAgentSDK.ClaudeSDKClient.receive_response",
})
_ASSISTANT_MESSAGE_SPAN_NAME = "assistant_message"
_GEN_AI_PROMPT_PREFIX = f"{SpanAttributes.LLM_PROMPTS}."
_GEN_AI_COMPLETION_PREFIX = f"{SpanAttributes.LLM_COMPLETIONS}."
_COMPLETION_TOOL_CALLS_ATTR = f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls"


def _derive_synthetic_span_id(*parts: Any) -> int:
    """Generate a deterministic non-zero OTLP span ID for exporter-only spans."""
    digest = hashlib.sha256(
        "|".join(str(part) for part in parts).encode("utf-8")
    ).digest()
    span_id = int.from_bytes(digest[:8], byteorder="big", signed=False)
    if span_id == 0:
        return 1
    return span_id


def _is_claude_agent_response_span(span: ReadableSpan) -> bool:
    """Return whether this span is a Claude Agent SDK response-turn parent."""
    scope = getattr(span, "instrumentation_scope", None)
    scope_name = getattr(scope, "name", None)
    return (
        scope_name == _CLAUDE_AGENT_SCOPE_NAME
        and span.name in _CLAUDE_AGENT_RESPONSE_SPAN_NAMES
    )


def _build_claude_agent_final_chat_span(
    span: ReadableSpan,
) -> Optional[ReadableSpan]:
    """Synthesize the missing final child chat span for Claude Agent tool turns."""
    if not _is_claude_agent_response_span(span):
        return None

    attrs = span.attributes or {}
    tool_calls = _parse_structured_json_attr(attrs.get(_COMPLETION_TOOL_CALLS_ATTR))
    if not isinstance(tool_calls, list) or not tool_calls:
        tool_calls = _parse_structured_json_attr(attrs.get(RESPAN_SPAN_TOOL_CALLS))
    if not isinstance(tool_calls, list) or not tool_calls:
        return None

    primary_completion_message = _select_primary_completion_from_attrs(attrs)
    completion_text = _extract_text_from_message(primary_completion_message)
    if completion_text in {None, ""}:
        return None

    span_context = span.get_span_context()
    if span_context is None:
        return None

    child_attributes: Dict[str, Any] = {
        RESPAN_LOG_TYPE: LOG_TYPE_CHAT,
        SpanAttributes.LLM_REQUEST_TYPE: LLMRequestTypeValues.CHAT.value,
        SpanAttributes.TRACELOOP_ENTITY_NAME: _ASSISTANT_MESSAGE_SPAN_NAME,
        f"{SpanAttributes.LLM_COMPLETIONS}.0.role": "assistant",
        f"{SpanAttributes.LLM_COMPLETIONS}.0.content": completion_text,
        _COMPLETION_TOOL_CALLS_ATTR: tool_calls,
        SpanAttributes.TRACELOOP_ENTITY_OUTPUT: json.dumps(
            primary_completion_message,
            default=str,
        ),
    }

    input_value = attrs.get(SpanAttributes.TRACELOOP_ENTITY_INPUT)
    if input_value is not None:
        child_attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT] = input_value

    model = attrs.get(SpanAttributes.LLM_REQUEST_MODEL)
    if model is not None:
        child_attributes[SpanAttributes.LLM_REQUEST_MODEL] = model

    system = attrs.get(SpanAttributes.LLM_SYSTEM)
    if system is not None:
        child_attributes[SpanAttributes.LLM_SYSTEM] = system

    child_attributes.update({
        key: value
        for key, value in attrs.items()
        if key.startswith(_GEN_AI_PROMPT_PREFIX)
    })

    child_end_time = span.end_time
    child_start_time = span.start_time
    if child_end_time and child_start_time:
        child_start_time = max(child_start_time, child_end_time - 1_000_000)

    child_span_id = _derive_synthetic_span_id(
        span_context.trace_id,
        span_context.span_id,
        _ASSISTANT_MESSAGE_SPAN_NAME,
    )
    if child_span_id == span_context.span_id:
        child_span_id = (child_span_id + 1) % (1 << 64) or 1

    return SyntheticSpan(
        name=_ASSISTANT_MESSAGE_SPAN_NAME,
        trace_id=span_context.trace_id,
        span_id=child_span_id,
        parent=span_context,
        attributes=child_attributes,
        start_time=child_start_time,
        end_time=child_end_time,
        status=getattr(span, "status", None),
        kind=getattr(span, "kind", None),
        resource=getattr(span, "resource", None),
        instrumentation_scope=getattr(span, "instrumentation_scope", None),
    )


def _prepare_spans_for_export(spans: Sequence[ReadableSpan]) -> List[ReadableSpan]:
    prepared_spans: List[ReadableSpan] = []

    for span in spans:
        overrides: Dict[str, Any] = {}

        extra_attrs = _get_enrichment_attrs(span)
        if extra_attrs:
            logger.debug("Enriching span with %s: %s", list(extra_attrs), span.name)
            merged_attrs = dict(span.attributes or {})
            merged_attrs.update(extra_attrs)
            overrides[OTEL_SPAN_ATTRIBUTES_FIELD] = merged_attrs

        prepared_span = (
            ModifiedSpan(original_span=span, overrides=overrides)
            if overrides
            else span
        )
        prepared_spans.append(prepared_span)

        synthetic_child = _build_claude_agent_final_chat_span(prepared_span)
        if synthetic_child is not None:
            prepared_spans.append(synthetic_child)

    return prepared_spans


def _convert_attribute_value(value: Any) -> Optional[Dict[str, Any]]:
    """Convert a Python attribute value to OTLP JSON typed wrapper."""
    if value is None:
        return None
    if isinstance(value, bool):
        return {OTLP_BOOL_VALUE: value}
    if isinstance(value, int):
        return {OTLP_INT_VALUE: str(value)}
    if isinstance(value, float):
        return {OTLP_DOUBLE_VALUE: value}
    if isinstance(value, str):
        return {OTLP_STRING_VALUE: value}
    if isinstance(value, bytes):
        return {OTLP_BYTES_VALUE: base64.b64encode(value).decode("ascii")}
    if isinstance(value, Mapping):
        converted_items = []
        for item_key, item_value in value.items():
            converted_value = _convert_attribute_value(item_value)
            if converted_value is not None:
                converted_items.append(
                    {OTLP_ATTR_KEY: str(item_key), OTLP_ATTR_VALUE: converted_value}
                )
        return {OTLP_KVLIST_VALUE: {OTLP_ARRAY_VALUES_KEY: converted_items}}
    if isinstance(value, (list, tuple)):
        converted = []
        for item in value:
            v = _convert_attribute_value(item)
            if v is not None:
                converted.append(v)
        return {OTLP_ARRAY_VALUE: {OTLP_ARRAY_VALUES_KEY: converted}}
    # Fallback: stringify
    return {OTLP_STRING_VALUE: str(value)}


# Attributes that duplicate data already captured in child spans.
# pydantic_ai.all_messages — full conversation history on the parent "agent run"
#   span; the same content is already in gen_ai.input/output.messages on each
#   "chat <model>" child span.
# logfire.json_schema — Pydantic-AI/Logfire internal metadata, not useful in Respan.
_STRIPPED_ATTRIBUTES = frozenset({
    "pydantic_ai.all_messages",
    "logfire.json_schema",
    RESPAN_SPAN_TOOL_CALLS,
    RESPAN_SPAN_TOOLS,
    # Internal naming hints — exporter-only, never part of the export contract.
    RESPAN_INTERNAL_SPAN_NAME_KIND,
    RESPAN_INTERNAL_SPAN_NAME_DETAIL,
    RESPAN_INTERNAL_DROP_SPAN,
    RESPAN_INTERNAL_EXPORT_PARENT,
})

_LLM_SPAN_NAME_PREFIX = "llm"
_SPAN_NAME_PREFIXES = frozenset({
    LOG_TYPE_AGENT,
    LOG_TYPE_EMBEDDING,
    LOG_TYPE_GUARDRAIL,
    LOG_TYPE_HANDOFF,
    _LLM_SPAN_NAME_PREFIX,
    LOG_TYPE_TASK,
    LOG_TYPE_TOOL,
    LOG_TYPE_WORKFLOW,
})
_LEGACY_SPAN_NAME_PREFIX_ALIASES = {
    LOG_TYPE_CHAT: _LLM_SPAN_NAME_PREFIX,
    LOG_TYPE_COMPLETION: _LLM_SPAN_NAME_PREFIX,
    "generate": _LLM_SPAN_NAME_PREFIX,
    LOG_TYPE_GENERATION: _LLM_SPAN_NAME_PREFIX,
    LOG_TYPE_RESPONSE: _LLM_SPAN_NAME_PREFIX,
    "responses": _LLM_SPAN_NAME_PREFIX,
    LOG_TYPE_TEXT: _LLM_SPAN_NAME_PREFIX,
}
_SPAN_NAME_PREFIX_ALIASES = {
    **{prefix: prefix for prefix in _SPAN_NAME_PREFIXES},
    **_LEGACY_SPAN_NAME_PREFIX_ALIASES,
}
_SUFFIXED_SPAN_NAME_PREFIXES = frozenset({
    LOG_TYPE_AGENT,
    LOG_TYPE_HANDOFF,
    _LLM_SPAN_NAME_PREFIX,
    LOG_TYPE_TOOL,
})
_LOG_TYPE_TO_SPAN_NAME_PREFIX = {
    LOG_TYPE_AGENT: LOG_TYPE_AGENT,
    LOG_TYPE_CHAT: _LLM_SPAN_NAME_PREFIX,
    LOG_TYPE_COMPLETION: _LLM_SPAN_NAME_PREFIX,
    LOG_TYPE_EMBEDDING: LOG_TYPE_EMBEDDING,
    LOG_TYPE_FUNCTION: LOG_TYPE_TOOL,
    LOG_TYPE_GENERATION: _LLM_SPAN_NAME_PREFIX,
    LOG_TYPE_GUARDRAIL: LOG_TYPE_GUARDRAIL,
    LOG_TYPE_HANDOFF: LOG_TYPE_HANDOFF,
    LOG_TYPE_RESPONSE: _LLM_SPAN_NAME_PREFIX,
    LOG_TYPE_TASK: LOG_TYPE_TASK,
    LOG_TYPE_TEXT: _LLM_SPAN_NAME_PREFIX,
    LOG_TYPE_TOOL: LOG_TYPE_TOOL,
    LOG_TYPE_WORKFLOW: LOG_TYPE_WORKFLOW,
}
_REQUEST_TYPE_TO_SPAN_NAME_PREFIX = {
    "chat": _LLM_SPAN_NAME_PREFIX,
    "completion": _LLM_SPAN_NAME_PREFIX,
    "embedding": LOG_TYPE_EMBEDDING,
}
_ENTITY_DETAIL_ATTRS_BY_PREFIX = {
    LOG_TYPE_AGENT: (GEN_AI_AGENT_NAME, SpanAttributes.TRACELOOP_ENTITY_NAME),
    LOG_TYPE_GUARDRAIL: (
        RESPAN_METADATA_GUARDRAIL_NAME,
        SpanAttributes.TRACELOOP_ENTITY_NAME,
    ),
    LOG_TYPE_HANDOFF: (SpanAttributes.TRACELOOP_ENTITY_NAME,),
    LOG_TYPE_TASK: (SpanAttributes.TRACELOOP_ENTITY_NAME, GEN_AI_OPERATION_NAME),
    LOG_TYPE_TOOL: (
        GEN_AI_TOOL_NAME,
        SpanAttributes.TRACELOOP_ENTITY_NAME,
        GEN_AI_OPERATION_NAME,
    ),
    LOG_TYPE_WORKFLOW: (SpanAttributes.TRACELOOP_ENTITY_NAME,),
}
_LLM_DETAIL_ATTRS = (
    LLM_REQUEST_MODEL,
    "llm.model_name",
    "model",
    "ai.model.id",
    "ai.response.model",
    SpanAttributes.LLM_REQUEST_MODEL,
)
_CHAT_NAME_MARKERS = ("chat", "response", "responses")
_GENERATION_NAME_MARKERS = ("generate", "generation", "completion")
_EMBEDDING_NAME_MARKERS = ("embedding", "embeddings", "embed")

# Operation/structural tokens that must not survive as a semantic-name suffix
# (e.g. "handoff.task" or "llm.doGenerate" — the suffix carries no identity).
_GENERIC_SPAN_NAME_DETAILS = frozenset(
    set(_SPAN_NAME_PREFIX_ALIASES)
    | {"completions", "dogenerate", "dostream", "doembed"}
)

_SPAN_NAME_STYLE_ENV_VAR = "RESPAN_SPAN_NAME_STYLE"
_SPAN_NAME_STYLE_LEGACY = "legacy"
_SPAN_NAME_STYLE_SEMANTIC = "semantic"

_SPAN_NAME_ARROW_RE = re.compile(r"\s*(?:→|->)\s*")
_SPAN_NAME_INVALID_CHARS_RE = re.compile(r"[^\w.-]+")
_SPAN_NAME_EDGE_PUNCT_RE = re.compile(r"^[_\-.]+|[_\-.]+$")


def _resolve_span_name_style() -> str:
    """Resolve the export span-name style.

    Semantic renaming is the default (matching released behavior);
    RESPAN_SPAN_NAME_STYLE=legacy preserves original instrumentation names.
    """
    value = (os.environ.get(_SPAN_NAME_STYLE_ENV_VAR) or "").strip().lower()
    if value == _SPAN_NAME_STYLE_LEGACY:
        return _SPAN_NAME_STYLE_LEGACY
    return _SPAN_NAME_STYLE_SEMANTIC


def _sanitize_span_name_part(value: str) -> str:
    """Mirror the JS SDK sanitizer: arrows/whitespace/punctuation → underscores."""
    sanitized = _SPAN_NAME_ARROW_RE.sub("_", value.strip())
    sanitized = _SPAN_NAME_INVALID_CHARS_RE.sub("_", sanitized)
    return _SPAN_NAME_EDGE_PUNCT_RE.sub("", sanitized)


def _clean_span_name_part(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _first_present_attr(attrs: Mapping[str, Any], keys: Sequence[str]) -> Optional[str]:
    for key in keys:
        value = _clean_span_name_part(attrs.get(key))
        if value:
            return value
    return None


def _display_span_name_prefix(prefix: str) -> str:
    return prefix.lower()


def _normalize_existing_semantic_span_name(name: str) -> Optional[str]:
    lower_name = name.lower()
    for candidate, prefix in _SPAN_NAME_PREFIX_ALIASES.items():
        display_prefix = _display_span_name_prefix(prefix)
        if lower_name == candidate:
            return display_prefix
        if lower_name.startswith(f"{candidate}."):
            if prefix not in _SUFFIXED_SPAN_NAME_PREFIXES:
                return display_prefix
            detail = _sanitize_span_name_part(name[len(candidate) + 1 :])
            return f"{display_prefix}.{detail}" if detail else display_prefix
    return None


def _strip_token_from_span_name(name: str, tokens: Sequence[str]) -> Optional[str]:
    normalized = name.strip()
    lower_name = normalized.lower()

    for token in tokens:
        lower_token = token.lower()
        dotted_prefix = f"{lower_token}."
        if lower_name.startswith(dotted_prefix):
            return normalized[len(dotted_prefix):].strip() or None

        spaced_prefix = f"{lower_token} "
        if lower_name.startswith(spaced_prefix):
            return normalized[len(spaced_prefix):].strip() or None

        dotted_suffix = f".{lower_token}"
        if lower_name.endswith(dotted_suffix):
            return normalized[: -len(dotted_suffix)].strip() or None

        spaced_suffix = f" {lower_token}"
        if lower_name.endswith(spaced_suffix):
            return normalized[: -len(spaced_suffix)].strip() or None

    return None


def _resolve_hinted_span_name_prefix(hinted_kind: str) -> Optional[str]:
    """Map an instrumentation-provided kind hint to a semantic prefix."""
    normalized = hinted_kind.lower()
    prefix = (
        _LOG_TYPE_TO_SPAN_NAME_PREFIX.get(normalized)
        or _SPAN_NAME_PREFIX_ALIASES.get(normalized)
    )
    if prefix:
        return prefix
    return _sanitize_span_name_part(normalized) or None


def _infer_span_name_prefix(name: str, attrs: Mapping[str, Any]) -> Optional[str]:
    hinted_kind = _clean_span_name_part(attrs.get(RESPAN_INTERNAL_SPAN_NAME_KIND))
    if hinted_kind:
        prefix = _resolve_hinted_span_name_prefix(hinted_kind)
        if prefix:
            return prefix

    log_type = _clean_span_name_part(attrs.get(RESPAN_LOG_TYPE))
    if log_type:
        prefix = _LOG_TYPE_TO_SPAN_NAME_PREFIX.get(log_type.lower())
        if prefix:
            return prefix

    span_kind = _clean_span_name_part(attrs.get(SpanAttributes.TRACELOOP_SPAN_KIND))
    if span_kind:
        prefix = _LOG_TYPE_TO_SPAN_NAME_PREFIX.get(span_kind.lower())
        if prefix:
            return prefix

    request_type = _clean_span_name_part(attrs.get(LLM_REQUEST_TYPE))
    if request_type:
        prefix = _REQUEST_TYPE_TO_SPAN_NAME_PREFIX.get(request_type.lower())
        if prefix:
            return prefix

    operation_name = _clean_span_name_part(attrs.get(GEN_AI_OPERATION_NAME))
    if operation_name:
        lower_operation = operation_name.lower()
        if any(marker in lower_operation for marker in _EMBEDDING_NAME_MARKERS):
            return LOG_TYPE_EMBEDDING
        if any(marker in lower_operation for marker in _CHAT_NAME_MARKERS):
            return _LLM_SPAN_NAME_PREFIX
        if any(marker in lower_operation for marker in _GENERATION_NAME_MARKERS):
            return _LLM_SPAN_NAME_PREFIX

    if attrs.get(GEN_AI_SYSTEM):
        return _LLM_SPAN_NAME_PREFIX

    if _first_present_attr(attrs, _LLM_DETAIL_ATTRS):
        return _LLM_SPAN_NAME_PREFIX

    lower_name = name.lower()
    for candidate, prefix in _SPAN_NAME_PREFIX_ALIASES.items():
        if lower_name.startswith(f"{candidate} ") or lower_name.endswith(f".{candidate}"):
            return prefix
        if lower_name.endswith(f" {candidate}"):
            return prefix

    if any(marker in lower_name for marker in _EMBEDDING_NAME_MARKERS):
        return LOG_TYPE_EMBEDDING
    if any(marker in lower_name for marker in _CHAT_NAME_MARKERS):
        return _LLM_SPAN_NAME_PREFIX
    if any(marker in lower_name for marker in _GENERATION_NAME_MARKERS):
        return _LLM_SPAN_NAME_PREFIX

    return None


def _detail_tokens_for_prefix(prefix: str) -> Sequence[str]:
    if prefix == _LLM_SPAN_NAME_PREFIX:
        return (
            _LLM_SPAN_NAME_PREFIX,
            LOG_TYPE_CHAT,
            LOG_TYPE_COMPLETION,
            "completion",
            "generate",
            LOG_TYPE_GENERATION,
            LOG_TYPE_RESPONSE,
            "responses",
            LOG_TYPE_TEXT,
        )
    if prefix == LOG_TYPE_EMBEDDING:
        return (LOG_TYPE_EMBEDDING, "embeddings", "embed")
    return (prefix,)


def _span_name_detail(prefix: str, name: str, attrs: Mapping[str, Any]) -> Optional[str]:
    if prefix == _LLM_SPAN_NAME_PREFIX:
        return _first_present_attr(attrs, _LLM_DETAIL_ATTRS)

    hinted_detail = _clean_span_name_part(attrs.get(RESPAN_INTERNAL_SPAN_NAME_DETAIL))
    if hinted_detail:
        return hinted_detail

    if prefix == LOG_TYPE_HANDOFF:
        from_agent = _clean_span_name_part(attrs.get(RESPAN_METADATA_FROM_AGENT))
        to_agent = _clean_span_name_part(attrs.get(RESPAN_METADATA_TO_AGENT))
        if from_agent and to_agent:
            return f"{from_agent}_to_{to_agent}"

    attr_detail = _first_present_attr(
        attrs,
        _ENTITY_DETAIL_ATTRS_BY_PREFIX.get(prefix, ()),
    )
    if attr_detail:
        return attr_detail

    stripped_detail = _strip_token_from_span_name(
        name,
        _detail_tokens_for_prefix(prefix),
    )
    if stripped_detail and stripped_detail.lower() not in _GENERIC_SPAN_NAME_DETAILS:
        return stripped_detail

    if prefix == LOG_TYPE_EMBEDDING:
        llm_detail = _first_present_attr(attrs, _LLM_DETAIL_ATTRS)
        if llm_detail:
            return llm_detail

    fallback = _clean_span_name_part(name)
    if fallback and fallback.lower() in _GENERIC_SPAN_NAME_DETAILS:
        return None
    return fallback


def _export_span_name(span: ReadableSpan) -> str:
    if _resolve_span_name_style() == _SPAN_NAME_STYLE_LEGACY:
        return getattr(span, "name", "")

    original_name = _clean_span_name_part(getattr(span, "name", None))
    if not original_name:
        return getattr(span, "name", "")

    attrs = span.attributes or {}
    has_name_hints = (
        _clean_span_name_part(attrs.get(RESPAN_INTERNAL_SPAN_NAME_KIND)) is not None
        or _clean_span_name_part(attrs.get(RESPAN_INTERNAL_SPAN_NAME_DETAIL)) is not None
    )

    # Names that already look semantic pass through — unless an instrumentation
    # hint overrides them, or the suffix is a structural token ("handoff.task",
    # "llm.doGenerate") that must be recomputed from attributes.
    if not has_name_hints:
        existing_semantic_name = _normalize_existing_semantic_span_name(original_name)
        if existing_semantic_name:
            prefix_part, _, detail_part = existing_semantic_name.partition(".")
            has_generic_detail = (
                bool(detail_part) and detail_part.lower() in _GENERIC_SPAN_NAME_DETAILS
            )
            if prefix_part == _LLM_SPAN_NAME_PREFIX:
                if not _first_present_attr(attrs, _LLM_DETAIL_ATTRS):
                    if has_generic_detail:
                        return _LLM_SPAN_NAME_PREFIX
                    return existing_semantic_name
            elif not has_generic_detail:
                return existing_semantic_name

    prefix = _infer_span_name_prefix(original_name, attrs)
    if not prefix:
        return original_name

    display_prefix = _display_span_name_prefix(prefix)
    if prefix not in _SUFFIXED_SPAN_NAME_PREFIXES:
        return display_prefix

    detail = _span_name_detail(prefix, original_name, attrs)
    if detail:
        detail = _sanitize_span_name_part(detail)
    if not detail or detail.lower() == display_prefix:
        return display_prefix

    return f"{display_prefix}.{detail}"


def _parse_structured_json_attr(value: Any) -> Any:
    """Decode JSON-string helper attrs when instrumentors store structured data safely."""
    if value is None:
        return None
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (TypeError, json.JSONDecodeError):
            return None
    return value


def _parse_json_like(value: Any) -> Any:
    """Parse JSON strings when possible, otherwise return the original value."""
    if value is None:
        return None
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (TypeError, json.JSONDecodeError):
            return value
    return value


def _set_nested_value(target: Dict[str, Any], dotted_path: str, value: Any) -> None:
    """Assign a nested dict field using a dotted path."""
    parts = dotted_path.split(".")
    cursor = target
    for part in parts[:-1]:
        current = cursor.get(part)
        if not isinstance(current, dict):
            current = {}
            cursor[part] = current
        cursor = current
    cursor[parts[-1]] = value


def _collect_indexed_attrs(
    attrs: Mapping[str, Any],
    prefix: str,
) -> Dict[int, Dict[str, Any]]:
    """Group indexed dotted attributes by message index."""
    buckets: Dict[int, Dict[str, Any]] = {}
    for key, value in attrs.items():
        if not key.startswith(prefix):
            continue
        rest = key[len(prefix):]
        parts = rest.split(".", 1)
        if not parts[0].isdigit():
            continue
        idx = int(parts[0])
        field = parts[1] if len(parts) > 1 else ""
        buckets.setdefault(idx, {})[field] = value
    return buckets


def _build_messages_from_indexed_attrs(
    attrs: Mapping[str, Any],
    prefix: str,
) -> Optional[List[Dict[str, Any]]]:
    """Rebuild prompt/completion messages from indexed gen_ai attributes."""
    buckets = _collect_indexed_attrs(attrs=attrs, prefix=prefix)
    messages: List[Dict[str, Any]] = []

    for idx in sorted(buckets):
        message: Dict[str, Any] = {}
        for field_key, field_value in buckets[idx].items():
            if not field_key:
                continue
            _set_nested_value(
                target=message,
                dotted_path=field_key,
                value=_parse_json_like(field_value),
            )
        if message:
            messages.append(message)

    return messages or None


def _extract_text_from_content(content: Any) -> Optional[str]:
    """Best-effort text extraction from string or block-based message content."""
    if content is None:
        return None
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return None

    text_segments: List[str] = []
    for item in content:
        if isinstance(item, str):
            text_segments.append(item)
            continue
        if not isinstance(item, Mapping):
            continue
        text_value = item.get("text")
        if isinstance(text_value, str):
            text_segments.append(text_value)
            continue
        content_value = item.get("content")
        if isinstance(content_value, str):
            text_segments.append(content_value)

    if not text_segments:
        return ""
    return "\n".join(text_segments)


def _extract_text_from_message(message: Any) -> Optional[str]:
    """Return human-readable text for a single message-like dict."""
    if not isinstance(message, Mapping):
        return None
    return _extract_text_from_content(message.get("content"))


def _coerce_raw_output_to_completion_message(
    raw_output_payload: Any,
) -> Optional[Dict[str, Any]]:
    """Normalize raw output payloads into an assistant message when possible."""
    if isinstance(raw_output_payload, str):
        return {
            "role": "assistant",
            "content": raw_output_payload,
        }

    if isinstance(raw_output_payload, Mapping):
        role = raw_output_payload.get("role")
        content = raw_output_payload.get("content")
        tool_calls = raw_output_payload.get("tool_calls")
        if role is not None or content is not None or tool_calls is not None:
            return dict(raw_output_payload)

    if isinstance(raw_output_payload, list):
        candidates = [
            candidate
            for item in raw_output_payload
            if (candidate := _coerce_raw_output_to_completion_message(item)) is not None
        ]
        if not candidates:
            return None
        for candidate in reversed(candidates):
            candidate_text = _extract_text_from_message(candidate)
            if candidate_text not in {None, ""}:
                return candidate
        return candidates[-1]

    return None


def _select_primary_completion_message(
    *,
    completion_messages: Optional[List[Dict[str, Any]]],
    raw_output_payload: Any,
) -> Optional[Dict[str, Any]]:
    """Choose the assistant completion that best represents the final answer."""
    raw_output_message = _coerce_raw_output_to_completion_message(raw_output_payload)

    if not completion_messages:
        return raw_output_message

    for message in reversed(completion_messages):
        message_text = _extract_text_from_message(message)
        if message_text not in {None, ""}:
            return message

    raw_output_text = _extract_text_from_message(raw_output_message)
    if raw_output_text not in {None, ""}:
        return raw_output_message

    return completion_messages[-1]


def _select_primary_completion_from_attrs(
    attrs: Mapping[str, Any],
) -> Optional[Dict[str, Any]]:
    """Choose the best completion message using traced attrs plus raw output."""
    return _select_primary_completion_message(
        completion_messages=_build_messages_from_indexed_attrs(
            attrs=attrs,
            prefix=_GEN_AI_COMPLETION_PREFIX,
        ),
        raw_output_payload=_parse_json_like(
            attrs.get(SpanAttributes.TRACELOOP_ENTITY_OUTPUT)
        ),
    )


def _convert_attributes(attributes: Any) -> List[Dict[str, Any]]:
    """Convert a mapping of attributes to OTLP JSON key-value list."""
    if not attributes:
        return []
    result = []
    for key, value in attributes.items():
        if key in _STRIPPED_ATTRIBUTES:
            continue
        converted = _convert_attribute_value(value)
        if converted is not None:
            result.append({OTLP_ATTR_KEY: str(key), OTLP_ATTR_VALUE: converted})
    return result


def _span_to_otlp_json(span: ReadableSpan) -> Dict[str, Any]:
    """Convert a ReadableSpan (or ModifiedSpan) to OTLP JSON span dict."""
    ctx = span.get_span_context()

    trace_id = format_trace_id(ctx.trace_id) if ctx else ""
    span_id = format_span_id(ctx.span_id) if ctx else ""

    # Parent span ID
    parent_span_id = ""
    parent = getattr(span, OTEL_SPAN_PARENT_FIELD, None)
    if parent is not None:
        parent_sid = getattr(parent, "span_id", None)
        if parent_sid:
            parent_span_id = format_span_id(parent_sid)

    # Timestamps as nanosecond strings
    start_time_ns = str(span.start_time) if span.start_time else "0"
    end_time_ns = str(span.end_time) if span.end_time else "0"

    # Span kind mapping: OTel Python SpanKind enum is 0-4 (INTERNAL=0, SERVER=1, ...)
    # but OTLP wire format is 1-5 (UNSPECIFIED=0, INTERNAL=1, SERVER=2, ...)
    kind_value = 0  # SPAN_KIND_UNSPECIFIED
    if span.kind is not None:
        raw = span.kind.value if hasattr(span.kind, "value") else int(span.kind)
        kind_value = raw + 1

    # Status
    status_dict = {}
    if span.status is not None:
        code = OTEL_STATUS_CODE_UNSET
        if span.status.status_code == StatusCode.OK:
            code = OTEL_STATUS_CODE_OK
        elif span.status.status_code == StatusCode.ERROR:
            code = OTEL_STATUS_CODE_ERROR
        status_dict[OTEL_STATUS_CODE_KEY] = code
        if span.status.description:
            status_dict[OTEL_STATUS_MESSAGE_KEY] = span.status.description

    # Events
    events = []
    for event in span.events or []:
        event_dict = {
            OTLP_NAME_KEY: event.name,
            "timeUnixNano": str(event.timestamp) if event.timestamp else "0",
        }
        event_attrs = _convert_attributes(event.attributes)
        if event_attrs:
            event_dict[OTLP_ATTRIBUTES_KEY] = event_attrs
        events.append(event_dict)

    links = []
    for link in getattr(span, "links", ()) or ():
        link_ctx = getattr(link, "context", None)
        if link_ctx is None:
            continue

        link_dict = {
            OTLP_TRACE_ID_KEY: format_trace_id(link_ctx.trace_id),
            OTLP_SPAN_ID_KEY: format_span_id(link_ctx.span_id),
            OTLP_ATTRIBUTES_KEY: _convert_attributes(getattr(link, "attributes", None)),
            OTLP_FLAGS_KEY: int(link_ctx.trace_flags) | (
                OTLP_REMOTE_LINK_FLAG if getattr(link_ctx, "is_remote", False) else 0
            ),
        }

        trace_state = getattr(link_ctx, "trace_state", None)
        if trace_state:
            trace_state_header = trace_state.to_header()
            if trace_state_header:
                link_dict[OTLP_TRACE_STATE_KEY] = trace_state_header

        dropped_attributes = getattr(link, "dropped_attributes", 0) or 0
        if dropped_attributes:
            link_dict[OTLP_DROPPED_ATTRIBUTES_COUNT_KEY] = dropped_attributes

        links.append(link_dict)

    result = {
        OTLP_TRACE_ID_KEY: trace_id,
        OTLP_SPAN_ID_KEY: span_id,
        OTLP_NAME_KEY: _export_span_name(span),
        OTLP_KIND_KEY: kind_value,
        OTLP_START_TIME_KEY: start_time_ns,
        OTLP_END_TIME_KEY: end_time_ns,
        OTLP_ATTRIBUTES_KEY: _convert_attributes(span.attributes),
    }

    if parent_span_id:
        result[OTLP_PARENT_SPAN_ID_KEY] = parent_span_id
    if status_dict:
        result[OTLP_STATUS_KEY] = status_dict
    if events:
        result[OTLP_EVENTS_KEY] = events
    if links:
        result[OTLP_LINKS_KEY] = links

    return result


def _get_resource_key(span: ReadableSpan) -> str:
    """Build a hashable key for grouping spans by resource."""
    resource = getattr(span, "resource", None)
    if not resource or not resource.attributes:
        return ""
    # Sort for deterministic keys
    return json.dumps(dict(sorted(resource.attributes.items())), sort_keys=True, default=str)


def _get_scope_key(span: ReadableSpan) -> str:
    """Build a hashable key for grouping spans by instrumentation scope."""
    scope = getattr(span, "instrumentation_scope", None)
    if not scope:
        return ""
    return f"{scope.name or ''}|{scope.version or ''}"


def _build_otlp_payload(spans: Sequence[ReadableSpan]) -> Dict[str, Any]:
    """
    Group spans by resource and scope, then build OTLP JSON payload.

    Structure: { resourceSpans: [ { resource, scopeSpans: [ { scope, spans } ] } ] }
    """
    # Group: resource_key -> scope_key -> list of span dicts
    resource_groups: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    resource_attrs_map: Dict[str, Any] = {}
    scope_info_map: Dict[str, Any] = {}

    for span in spans:
        r_key = _get_resource_key(span)
        s_key = _get_scope_key(span)

        if r_key not in resource_groups:
            resource_groups[r_key] = {}
            resource = getattr(span, "resource", None)
            resource_attrs_map[r_key] = resource.attributes if resource else {}

        if s_key not in resource_groups[r_key]:
            resource_groups[r_key][s_key] = []
            scope = getattr(span, "instrumentation_scope", None)
            scope_info_map[s_key] = scope

        resource_groups[r_key][s_key].append(_span_to_otlp_json(span))

    # Build OTLP JSON
    resource_spans = []
    for r_key, scope_groups in resource_groups.items():
        scope_spans = []
        for s_key, span_dicts in scope_groups.items():
            scope_entry = {OTLP_SPANS_KEY: span_dicts}
            scope = scope_info_map.get(s_key)
            if scope:
                scope_dict = {}
                if scope.name:
                    scope_dict[OTLP_NAME_KEY] = scope.name
                if scope.version:
                    scope_dict[OTLP_VERSION_KEY] = scope.version
                scope_entry[OTLP_SCOPE_KEY] = scope_dict
            scope_spans.append(scope_entry)

        rs_entry = {OTLP_SCOPE_SPANS_KEY: scope_spans}
        r_attrs = resource_attrs_map.get(r_key, {})
        if r_attrs:
            rs_entry[OTLP_RESOURCE_KEY] = {OTLP_ATTRIBUTES_KEY: _convert_attributes(r_attrs)}
        resource_spans.append(rs_entry)

    return {OTLP_RESOURCE_SPANS_KEY: resource_spans}


def _get_enrichment_attrs(span: ReadableSpan) -> Dict[str, Any]:
    """Return extra attributes to inject into a span before export.

    Handles GenAI spans (e.g. ``openai.response``) that carry ``gen_ai.system``
    but lack ``llm.request.type``.  The Respan backend uses ``llm.request.type``
    to trigger prompt/completion/model/token parsing, so we inject ``"chat"``
    to ensure the backend processes these spans.

    Note: OpenInference → Traceloop translation is handled earlier in the
    pipeline by ``OpenInferenceTranslator`` (a SpanProcessor).
    """
    attrs = span.attributes or {}
    extra: Dict[str, Any] = {}

    if _RESPAN_TRACING_SDK_VERSION and not attrs.get(
        RESPAN_METADATA_INTERNAL_TRACING_SDK_VERSION
    ):
        extra[RESPAN_METADATA_INTERNAL_TRACING_SDK_VERSION] = (
            _RESPAN_TRACING_SDK_VERSION
        )

    if attrs.get(SpanAttributes.LLM_SYSTEM) and not attrs.get(SpanAttributes.LLM_REQUEST_TYPE):
        extra[SpanAttributes.LLM_REQUEST_TYPE] = LLMRequestTypeValues.CHAT.value

    cache_read = attrs.get(SpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS)
    if cache_read not in {None, ""} and "prompt_cache_hit_tokens" not in attrs:
        extra["prompt_cache_hit_tokens"] = cache_read

    cache_creation = attrs.get(SpanAttributes.LLM_USAGE_CACHE_CREATION_INPUT_TOKENS)
    if (
        cache_creation not in {None, ""}
        and "prompt_cache_creation_tokens" not in attrs
    ):
        extra["prompt_cache_creation_tokens"] = cache_creation

    tool_calls = _parse_structured_json_attr(attrs.get(_COMPLETION_TOOL_CALLS_ATTR))
    if not isinstance(tool_calls, list) or not tool_calls:
        tool_calls = _parse_structured_json_attr(attrs.get(RESPAN_SPAN_TOOL_CALLS))
    if isinstance(tool_calls, list) and tool_calls:
        if _COMPLETION_TOOL_CALLS_ATTR not in attrs:
            extra[_COMPLETION_TOOL_CALLS_ATTR] = tool_calls
        if f"{SpanAttributes.LLM_COMPLETIONS}.0.role" not in attrs:
            extra[f"{SpanAttributes.LLM_COMPLETIONS}.0.role"] = "assistant"
        existing_completion_content = attrs.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.content")
        if existing_completion_content in {None, ""}:
            primary_completion_message = _select_primary_completion_from_attrs(attrs)
            completion_text = _extract_text_from_message(primary_completion_message)
            if completion_text not in {None, ""}:
                extra[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"] = completion_text
            elif f"{SpanAttributes.LLM_COMPLETIONS}.0.content" not in attrs:
                extra[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"] = ""

    return extra


class RespanSpanExporter:
    """
    Custom span exporter for Respan that serializes spans as OTLP JSON
    and POSTs them to the /v2/traces endpoint.

    Anti-recursion: Uses OpenTelemetry's suppress_instrumentation context
    to prevent auto-instrumented HTTP libraries (requests, urllib3) from
    creating spans during export. This ensures no infinite trace loops
    even when the ingest endpoint is itself traced.
    """

    def __init__(
        self,
        endpoint: str,
        api_key: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: int = 30,
    ):
        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self._is_shutdown = False

        # Persistent session for TCP connection reuse across export() calls.
        # At 1% prod sampling with 3-5 traces per request, connection overhead matters.
        self._session = requests.Session()
        self._session.headers.update({
            "Content-Type": "application/json",
            # Anti-recursion marker: tells the server "don't emit new traces
            # while processing this request" — but still ingest the payload.
            # Prevents infinite loops when the ingest endpoint is itself observed.
            RESPAN_DOGFOOD_HEADER: "1",
        })
        if headers:
            self._session.headers.update(headers)
        if api_key:
            self._session.headers["Authorization"] = f"Bearer {api_key}"

        self._traces_url = _resolve_traces_endpoint(endpoint=self.endpoint)
        logger.debug("OTLP JSON traces endpoint: %s", self._traces_url)

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        """Export spans as OTLP JSON to /v2/traces."""
        if self._is_shutdown:
            return SpanExportResult.FAILURE

        modified_spans = _prepare_spans_for_export(spans=spans)
        payload = _build_otlp_payload(modified_spans)

        # Debug preview
        try:
            if logger.isEnabledFor(10):  # logging.DEBUG
                preview = build_spans_export_preview(modified_spans)
                logger.debug("Export preview (sanitized): %s", preview)
        except Exception:
            pass

        # Suppress OTel instrumentation during export to prevent recursion.
        # Without this, auto-instrumented `requests` would create spans for
        # the export POST, which would be exported, creating more spans, etc.
        token = attach(set_value(_SUPPRESS_INSTRUMENTATION_KEY, True))
        try:
            response = self._session.post(
                url=self._traces_url,
                data=json.dumps(payload, default=str),
                timeout=self.timeout,
            )
            if response.status_code < 400:
                logger.debug(
                    "Exported %d spans successfully (HTTP %d)",
                    len(modified_spans),
                    response.status_code,
                )
                return SpanExportResult.SUCCESS
            logger.warning(
                "Failed to export spans: HTTP %d — %s",
                response.status_code,
                response.text[:500],
            )
            return SpanExportResult.FAILURE
        except Exception as e:
            logger.warning("Failed to export spans: %s", e)
            return SpanExportResult.FAILURE
        finally:
            detach(token)

    def shutdown(self):
        """Shutdown the exporter and close the HTTP session."""
        self._is_shutdown = True
        self._session.close()

    def force_flush(self, timeout_millis: int = 30000):
        """Force flush — no-op for HTTP JSON exporter (each export is synchronous)."""
        return True
