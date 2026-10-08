"""Normalize native AgentOps spans before downstream processors."""

from __future__ import annotations

import json
import re
from threading import RLock

from agentops.semconv.core import CoreAttributes
from agentops.semconv.span_attributes import SpanAttributes as AgentOpsSpanAttributes
from opentelemetry.sdk.trace import Event, SpanProcessor
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import (
    GEN_AI_PROVIDER_NAME,
    GEN_AI_TOOL_CALL_ID,
    GEN_AI_USAGE_INPUT_TOKENS,
    GEN_AI_USAGE_OUTPUT_TOKENS,
)
from opentelemetry.semconv.attributes.exception_attributes import (
    EXCEPTION_MESSAGE,
    EXCEPTION_STACKTRACE,
    EXCEPTION_TYPE,
)
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.trace import Status, StatusCode
from respan_sdk.constants.llm_logging import LogMethodChoices
from respan_sdk.constants.span_attributes import (
    RESPAN_LOG_METHOD,
    RESPAN_LOG_TYPE,
    RESPAN_METADATA,
    RESPAN_SPAN_ATTRIBUTES_MAP,
)

from ._constants import (
    AGENTOPS_KIND_LOG_TYPES,
    AGENTOPS_METADATA_PREFIX,
    OFF_CONTRACT_ALIASES,
)
from ._policy import content_allowed, suppressed
from ._serialization import data, json_value, text

_MARKERS = {
    "run_id",
    "example_run_id",
    "framework",
    "integration",
    "example",
    "workflow_name",
    "example_name",
}


def _structural_field(key, value):
    if key in {
        RESPAN_LOG_TYPE,
        RESPAN_LOG_METHOD,
        RESPAN_METADATA,
        SpanAttributes.TRACELOOP_ENTITY_NAME,
        SpanAttributes.TRACELOOP_ENTITY_PATH,
        SpanAttributes.LLM_SYSTEM,
        GEN_AI_PROVIDER_NAME,
        SpanAttributes.LLM_REQUEST_MODEL,
        SpanAttributes.LLM_RESPONSE_MODEL,
        SpanAttributes.LLM_REQUEST_TYPE,
        SpanAttributes.LLM_IS_STREAMING,
        GEN_AI_TOOL_CALL_ID,
        AgentOpsSpanAttributes.AGENTOPS_SPAN_KIND,
        AgentOpsSpanAttributes.OPERATION_NAME,
        AgentOpsSpanAttributes.AGENTOPS_ENTITY_NAME,
    }:
        return True
    if key.startswith(f"{RESPAN_METADATA}."):
        return key.removeprefix(f"{RESPAN_METADATA}.") in _MARKERS
    if key.startswith("respan.") and key in RESPAN_SPAN_ATTRIBUTES_MAP.values():
        return True
    return (
        key.startswith(("gen_ai.usage.", "llm.usage."))
        and type(value) is int
        and value >= 0
    )


def _scope(span):
    scope = getattr(span, "instrumentation_scope", None) or getattr(
        span, "instrumentation_info", None
    )
    return getattr(scope, "name", "")


def _owned(span, attrs):
    return (
        AgentOpsSpanAttributes.AGENTOPS_SPAN_KIND in attrs
        or _scope(span) == "agentops"
        or _scope(span).startswith("agentops.")
    )


def _decode(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (ValueError, TypeError):
            return value
    return value


def _kind(attrs):
    request = attrs.get(AgentOpsSpanAttributes.LLM_REQUEST_TYPE)
    if request in {"chat", "completion", "text"}:
        return "text" if request in {"completion", "text"} else "llm"
    return (
        "embedding"
        if request == "embedding"
        else str(attrs.get(AgentOpsSpanAttributes.AGENTOPS_SPAN_KIND) or "task").lower()
    )


def _name(span, attrs):
    for key in (
        AgentOpsSpanAttributes.AGENTOPS_ENTITY_NAME,
        AgentOpsSpanAttributes.OPERATION_NAME,
    ):
        value = attrs.get(key)
        if isinstance(value, str) and value:
            return text(value, 512)
    return text(span.name, 512)


def _private_field(key):
    return (
        key
        in (
            SpanAttributes.TRACELOOP_ENTITY_INPUT,
            SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
            SpanAttributes.LLM_REQUEST_FUNCTIONS,
            AgentOpsSpanAttributes.LLM_REQUEST_FUNCTIONS,
            EXCEPTION_MESSAGE,
            EXCEPTION_STACKTRACE,
        )
        or key.startswith(
            (
                f"{SpanAttributes.LLM_PROMPTS}.",
                f"{SpanAttributes.LLM_COMPLETIONS}.",
                "error.",
            )
        )
        or "headers" in key.lower()
        or key
        in (
            AgentOpsSpanAttributes.HTTP_REQUEST_BODY,
            AgentOpsSpanAttributes.HTTP_RESPONSE_BODY,
        )
    )


def _messages(attrs, prefix):
    messages = {}
    calls = {}
    for key, value in list(attrs.items()):
        match = re.fullmatch(
            re.escape(prefix) + r"\.(\d+)\.(role|content|tool_call_id|tool_calls)", key
        )
        if match:
            index = int(match[1])
            field = match[2]
            structured = (
                field == "tool_calls"
                or field == "content"
                and isinstance(value, str)
                and value.lstrip().startswith(("{", "["))
            )
            messages.setdefault(index, {})[field] = (
                _decode(value) if structured else value
            )
        match = re.fullmatch(
            re.escape(prefix) + r"\.(\d+)\.tool_calls\.(\d+)\.(id|name|arguments)", key
        )
        if match:
            message_index, call_index = int(match[1]), int(match[2])
            calls.setdefault(message_index, {}).setdefault(call_index, {})[match[3]] = (
                value
            )
            attrs.pop(key, None)
    for index, items in calls.items():
        if isinstance(attrs.get(f"{prefix}.{index}.tool_calls"), str):
            continue
        value = []
        for item in items.values():
            if not isinstance(item.get("name"), str):
                continue
            arguments = item.get("arguments")
            call = {
                "type": "function",
                "function": {
                    "name": item["name"],
                    "arguments": arguments
                    if isinstance(arguments, str)
                    else json_value(arguments, complete=True),
                },
            }
            if isinstance(item.get("id"), str):
                call["id"] = item["id"]
            value.append(call)
        messages.setdefault(index, {})["tool_calls"] = value
        attrs[f"{prefix}.{index}.tool_calls"] = json_value(value, complete=True)
    return [
        data(message, complete=True)
        for _, message in sorted(messages.items())
        if "role" in message or "content" in message or "tool_calls" in message
    ]


def translate_agentops_span(span, *, capture_content):
    attrs = dict(span.attributes or {})
    if not _owned(span, attrs):
        return False
    kind = _kind(attrs)
    name = _name(span, attrs)
    attrs[RESPAN_LOG_TYPE] = AGENTOPS_KIND_LOG_TYPES.get(kind, "task")
    attrs[RESPAN_LOG_METHOD] = LogMethodChoices.TRACING_INTEGRATION.value
    attrs[SpanAttributes.TRACELOOP_ENTITY_NAME] = name
    attrs[SpanAttributes.TRACELOOP_ENTITY_PATH] = "" if span.parent is None else name
    attrs.pop(SpanAttributes.TRACELOOP_SPAN_KIND, None)
    for direction, target in (
        ("input", SpanAttributes.TRACELOOP_ENTITY_INPUT),
        ("output", SpanAttributes.TRACELOOP_ENTITY_OUTPUT),
    ):
        source = attrs.get(
            AgentOpsSpanAttributes.AGENTOPS_DECORATOR_INPUT.format(entity_kind=kind)
            if direction == "input"
            else AgentOpsSpanAttributes.AGENTOPS_DECORATOR_OUTPUT.format(
                entity_kind=kind
            ),
            attrs.get(
                AgentOpsSpanAttributes.AGENTOPS_ENTITY_INPUT
                if direction == "input"
                else AgentOpsSpanAttributes.AGENTOPS_ENTITY_OUTPUT
            ),
        )
        if (
            capture_content
            and source is not None
            and not (
                direction == "output" and span.status.status_code == StatusCode.ERROR
            )
        ):
            value = _decode(source)
            if kind == "tool" and direction == "input":
                value = {"name": name, "arguments": value}
            attrs[target] = json_value(value, complete=kind in {"tool", "embedding"})
    if span.status.status_code == StatusCode.ERROR:
        attrs.pop(SpanAttributes.TRACELOOP_ENTITY_OUTPUT, None)
    if kind in {"llm", "text", "embedding"}:
        attrs[SpanAttributes.LLM_REQUEST_TYPE] = (
            "embedding" if kind == "embedding" else "chat"
        )
        provider = attrs.get(SpanAttributes.LLM_SYSTEM)
        if isinstance(provider, str):
            attrs[SpanAttributes.LLM_SYSTEM] = text(provider, 64).lower()
            attrs[GEN_AI_PROVIDER_NAME] = text(provider, 64).lower()
        for source, targets in (
            (
                SpanAttributes.LLM_USAGE_PROMPT_TOKENS,
                (SpanAttributes.LLM_USAGE_PROMPT_TOKENS, GEN_AI_USAGE_INPUT_TOKENS),
            ),
            (
                SpanAttributes.LLM_USAGE_COMPLETION_TOKENS,
                (
                    SpanAttributes.LLM_USAGE_COMPLETION_TOKENS,
                    GEN_AI_USAGE_OUTPUT_TOKENS,
                ),
            ),
            (
                AgentOpsSpanAttributes.LLM_USAGE_TOTAL_TOKENS,
                (SpanAttributes.LLM_USAGE_TOTAL_TOKENS,),
            ),
            (
                AgentOpsSpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS,
                (SpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS,),
            ),
            (
                AgentOpsSpanAttributes.LLM_USAGE_CACHE_CREATION_INPUT_TOKENS,
                (SpanAttributes.LLM_USAGE_CACHE_CREATION_INPUT_TOKENS,),
            ),
            (
                AgentOpsSpanAttributes.LLM_USAGE_REASONING_TOKENS,
                (
                    SpanAttributes.GEN_AI_USAGE_REASONING_TOKENS,
                    SpanAttributes.LLM_USAGE_REASONING_TOKENS,
                ),
            ),
        ):
            value = attrs.get(source)
            if type(value) is int and value >= 0:
                for target in targets:
                    attrs[target] = value
            else:
                attrs.pop(source, None)
        functions = attrs.get(AgentOpsSpanAttributes.LLM_REQUEST_FUNCTIONS)
        if capture_content and functions is not None:
            attrs[SpanAttributes.LLM_REQUEST_FUNCTIONS] = json_value(
                _decode(functions), complete=True
            )
        if attrs.get(AgentOpsSpanAttributes.LLM_REQUEST_STREAMING) is True:
            attrs[SpanAttributes.LLM_IS_STREAMING] = True
        if capture_content:
            prompts = _messages(attrs, SpanAttributes.LLM_PROMPTS)
            completions = _messages(attrs, SpanAttributes.LLM_COMPLETIONS)
            if prompts:
                attrs.setdefault(
                    SpanAttributes.TRACELOOP_ENTITY_INPUT,
                    json_value(prompts, complete=True),
                )
            if completions and span.status.status_code != StatusCode.ERROR:
                attrs.setdefault(
                    SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
                    json_value({"messages": completions}, complete=True),
                )
    metadata = _decode(attrs.get(RESPAN_METADATA))
    metadata = metadata if isinstance(metadata, dict) else {}
    metadata = {
        k: data(v) for k, v in metadata.items() if capture_content or k in _MARKERS
    }
    details = {"kind": kind}
    if capture_content:
        native_metadata = {
            key.removeprefix(AGENTOPS_METADATA_PREFIX): data(value)
            for key, value in attrs.items()
            if key.startswith(AGENTOPS_METADATA_PREFIX)
        }
        if native_metadata:
            details["trace_metadata"] = native_metadata
    spec = attrs.get(
        AgentOpsSpanAttributes.AGENTOPS_DECORATOR_SPEC.format(entity_kind=kind)
    )
    if kind == "guardrail" and spec in {"input", "output"}:
        details["spec"] = spec
    state = attrs.get(AgentOpsSpanAttributes.AGENTOPS_SESSION_END_STATE)
    if isinstance(state, str):
        details["end_state"] = text(state, 64)
        if state.lower() in {"error", "failed", "failure", "statuscode.error"}:
            span._status = Status(StatusCode.ERROR)
    version = attrs.get(AgentOpsSpanAttributes.OPERATION_VERSION)
    if capture_content and isinstance(version, (str, int, float)):
        details["operation_version"] = data(version)
    tags = attrs.get(CoreAttributes.TAGS)
    if capture_content and tags is not None:
        details["tags"] = data(tags)
    metadata["agentops"] = details
    attrs[RESPAN_METADATA] = json_value(metadata)
    for key in tuple(attrs):
        if (
            key in OFF_CONTRACT_ALIASES
            or key.startswith(("agentops.", AGENTOPS_METADATA_PREFIX))
            or key
            in (
                AgentOpsSpanAttributes.OPERATION_NAME,
                AgentOpsSpanAttributes.OPERATION_VERSION,
                AgentOpsSpanAttributes.LLM_REQUEST_TYPE,
                AgentOpsSpanAttributes.LLM_REQUEST_FUNCTIONS,
                AgentOpsSpanAttributes.LLM_REQUEST_STREAMING,
                AgentOpsSpanAttributes.LLM_USAGE_TOTAL_TOKENS,
            )
            or not capture_content
            and not _structural_field(key, attrs[key])
            or "headers" in key.lower()
            or not capture_content
            and (
                _private_field(key)
                or key.startswith(f"{RESPAN_METADATA}.")
                and key.removeprefix(f"{RESPAN_METADATA}.") not in _MARKERS
            )
        ):
            attrs.pop(key, None)
        elif isinstance(attrs[key], str):
            attrs[key] = data(
                attrs[key], complete=_private_field(key) or key == GEN_AI_TOOL_CALL_ID
            )
    if not capture_content:
        span._events = tuple(
            Event(
                e.name,
                {k: v for k, v in e.attributes.items() if k == EXCEPTION_TYPE},
                e.timestamp,
            )
            for e in span.events
        )
        if span.status.description:
            span._status = Status(span.status.status_code)
    else:
        span._events = tuple(
            Event(
                e.name,
                {
                    k: data(v) if isinstance(v, str) else v
                    for k, v in e.attributes.items()
                },
                e.timestamp,
            )
            for e in span.events
        )
        if span.status.description:
            span._status = Status(
                span.status.status_code, text(span.status.description)
            )
    if span.status.status_code == StatusCode.ERROR:
        attrs.pop(SpanAttributes.TRACELOOP_ENTITY_OUTPUT, None)
    span._attributes = attrs
    return True


class AgentOpsSpanProcessor(SpanProcessor):
    def __init__(self, *, capture_content=True):
        self.capture_content = capture_content
        self.active = True
        self._captures = {}
        self._parents = {}
        self._links = {}
        self._source_usage = {}
        self._current_calls = set()
        self._lock = RLock()

    def on_start(self, span, parent_context=None):
        if not self.active or not _owned(span, span.attributes or {}):
            return
        parent = span.parent.span_id if span.parent else None
        with self._lock:
            self._links[span.context.span_id] = parent
            if not content_allowed(self.capture_content):
                self._veto(parent)
            bound = self._captures.get(parent, self._parents.get(parent, True))
            self._captures[span.context.span_id] = bool(
                bound and content_allowed(self.capture_content) and not suppressed()
            )

    def _veto(self, sid):
        seen = set()
        while sid and sid not in seen:
            seen.add(sid)
            if sid in self._captures:
                self._captures[sid] = False
            if sid in self._parents:
                self._parents[sid] = False
            sid = self._links.get(sid)

    def has_current_calls(self, span):
        return span.get_span_context().span_id in self._current_calls

    def mark_current_calls(self, span):
        self._current_calls.add(span.get_span_context().span_id)

    def source_usage(self, span, usage):
        self._source_usage[span.get_span_context().span_id] = usage

    def allowed(self, span):
        if not self.active or not span.is_recording() or suppressed():
            return False
        with self._lock:
            sid = span.get_span_context().span_id
            allowed = self._captures.get(sid, False)
            if not content_allowed(self.capture_content):
                self._veto(sid)
                allowed = False
            return allowed

    def on_end(self, span):
        if not self.active or not _owned(span, span.attributes or {}):
            return
        with self._lock:
            if not content_allowed(self.capture_content):
                self._veto(span.context.span_id)
            cap = self._captures.pop(span.context.span_id, False) and content_allowed(
                self.capture_content
            )
            self._parents[span.context.span_id] = cap
            while len(self._parents) > 4096:
                oldest = next(iter(self._parents))
                self._parents.pop(oldest)
                self._links.pop(oldest, None)
        try:
            usage = self._source_usage.pop(span.context.span_id, None)
            self._current_calls.discard(span.context.span_id)
            if usage is not None:
                attrs = dict(span.attributes or {})
                for field, key in (
                    ("prompt_tokens", SpanAttributes.LLM_USAGE_PROMPT_TOKENS),
                    ("completion_tokens", SpanAttributes.LLM_USAGE_COMPLETION_TOKENS),
                    ("total_tokens", AgentOpsSpanAttributes.LLM_USAGE_TOTAL_TOKENS),
                ):
                    value = usage.get(field)
                    if type(value) is int and value >= 0:
                        attrs[key] = value
                    else:
                        attrs.pop(key, None)
                for field, source, key in (
                    (
                        "prompt_tokens_details",
                        "cached_tokens",
                        AgentOpsSpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS,
                    ),
                    (
                        "completion_tokens_details",
                        "reasoning_tokens",
                        AgentOpsSpanAttributes.LLM_USAGE_REASONING_TOKENS,
                    ),
                ):
                    details = usage.get(field)
                    details = (
                        details
                        if isinstance(details, dict)
                        else vars(details)
                        if type(details).__module__.startswith("openai.")
                        else {}
                    )
                    value = details.get(source)
                    if type(value) is int and value >= 0:
                        attrs[key] = value
                span._attributes = attrs
            translate_agentops_span(span, capture_content=cap)
        except Exception:  # noqa: BLE001 - failed translation must not change SDK execution
            span._events = tuple(
                Event(
                    e.name,
                    {k: v for k, v in e.attributes.items() if k == EXCEPTION_TYPE},
                    e.timestamp,
                )
                for e in span.events
            )
            if span.status.description:
                span._status = Status(span.status.status_code)
            span._attributes = {
                k: v
                for k, v in span.attributes.items()
                if _structural_field(k, v)
                and not _private_field(k)
                and not k.startswith("agentops.")
            }

    def shutdown(self):
        self.active = False
        self._captures.clear()
        self._parents.clear()
        self._links.clear()
        self._source_usage.clear()
        self._current_calls.clear()

    def force_flush(self, timeout_millis=30000):
        return True
