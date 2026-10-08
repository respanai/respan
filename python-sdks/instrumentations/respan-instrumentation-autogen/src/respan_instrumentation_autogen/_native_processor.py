"""Remove AutoGen runtime noise and repair AutoGen OpenInference payloads."""

from __future__ import annotations

import json
import re
from threading import Lock
from typing import Any

from openinference.semconv.trace import (
    SpanAttributes as OISpanAttributes,
)
from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor
from opentelemetry.semconv_ai import SpanAttributes as TLSpanAttributes

AUTOGEN_CORE_SCOPE_NAME = "autogen-core"
AUTOGEN_RUNTIME_SCOPE_PREFIX = "autogen "
AUTOGEN_OPENINFERENCE_SCOPE_NAME = "openinference.instrumentation.autogen_agentchat"
AUTOGEN_OPERATION_INVOKE_AGENT = "invoke_agent"
AUTOGEN_OPERATION_EXECUTE_TOOL = "execute_tool"

_OI_INPUT_MESSAGE_PREFIX = f"{OISpanAttributes.LLM_INPUT_MESSAGES}."
_FUNCTION_RESULT_RE = re.compile(
    r"^llm\.input_messages\.(\d+)\.(?:message\.)?function\.(\d+)$"
)


def _get_scope_name(span: Any) -> str | None:
    scope = getattr(span, "instrumentation_scope", None)
    return getattr(scope, "name", None)


def _is_native_scope(scope_name: str | None) -> bool:
    return scope_name == AUTOGEN_CORE_SCOPE_NAME or bool(
        scope_name and scope_name.startswith(AUTOGEN_RUNTIME_SCOPE_PREFIX)
    )


def _get_span_id(span: Any) -> int | None:
    get_context = getattr(span, "get_span_context", None)
    context = get_context() if callable(get_context) else getattr(span, "context", None)
    span_id = getattr(context, "span_id", None)
    return span_id if isinstance(span_id, int) and span_id else None


def _get_parent(span: Any) -> Any:
    return getattr(span, "parent", None)


def _parse_result(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return dict(value)
    if not isinstance(value, str):
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return None
    return dict(parsed) if isinstance(parsed, dict) else None


def _normalize_function_result_messages(attrs: dict[str, Any]) -> None:
    """Promote AutoGen's vendor-only function result fields to messages.

    OpenInference AutoGen 0.1.11 stores a function-result input as
    ``message.function.N`` while emitting only ``role=function``. The generic
    translator does not understand that vendor extension, so Respan receives
    an empty history entry. Preserve the complete result as canonical indexed
    message fields before the generic translator removes the raw OI keys.
    """

    results_by_message: dict[int, list[tuple[int, dict[str, Any]]]] = {}
    raw_result_keys: list[str] = []
    for key, value in attrs.items():
        match = _FUNCTION_RESULT_RE.match(key)
        if match is None:
            continue
        result = _parse_result(value)
        if result is None:
            continue
        message_index = int(match.group(1))
        result_index = int(match.group(2))
        results_by_message.setdefault(message_index, []).append((result_index, result))
        raw_result_keys.append(key)

    for message_index, indexed_results in results_by_message.items():
        results = [result for _, result in sorted(indexed_results)]
        prefix = f"{_OI_INPUT_MESSAGE_PREFIX}{message_index}.message"
        attrs[f"{prefix}.role"] = "tool"
        attrs.pop(f"{prefix}.message.role", None)
        attrs[f"{prefix}.content"] = json.dumps(
            results[0] if len(results) == 1 else results,
            default=str,
            separators=(",", ":"),
        )

        if len(results) == 1:
            result = results[0]
            call_id = result.get("call_id")
            name = result.get("name")
            canonical_prefix = f"{TLSpanAttributes.LLM_PROMPTS}.{message_index}"
            if call_id not in (None, ""):
                attrs[f"{canonical_prefix}.tool_call_id"] = str(call_id)
            if name not in (None, ""):
                attrs[f"{canonical_prefix}.name"] = str(name)

    for key in raw_result_keys:
        attrs.pop(key, None)


class AutoGenNativeSpanProcessor(SpanProcessor):
    """Keep only the meaningful OpenInference AutoGen operation tree.

    AutoGen Core emits internal runtime spans for agent creation, message bus
    publish/process/ack operations, native agent invocations, and native tool
    execution. OpenInference AgentChat already emits content-complete logical
    agent, LLM, and tool spans for the same work. Native spans are therefore
    made unprocessable, and any meaningful child is reparented to the nearest
    non-native ancestor before its ``ReadableSpan`` is created.
    """

    def __init__(self) -> None:
        self._native_export_parents: dict[int, Any] = {}
        self.enabled = True
        self._lock = Lock()

    def on_start(self, span: Any, parent_context: Any = None) -> None:
        if not self.enabled:
            return
        parent = _get_parent(span)
        parent_id = getattr(parent, "span_id", None)

        with self._lock:
            if parent_id in self._native_export_parents:
                export_parent = self._native_export_parents[parent_id]
                span._parent = export_parent
            else:
                export_parent = parent

            if _is_native_scope(_get_scope_name(span)):
                span_id = _get_span_id(span)
                if span_id is not None:
                    self._native_export_parents[span_id] = export_parent

    def on_end(self, span: ReadableSpan) -> None:
        if not self.enabled:
            return
        scope_name = _get_scope_name(span)
        if scope_name == AUTOGEN_OPENINFERENCE_SCOPE_NAME:
            attrs = dict(getattr(span, "_attributes", None) or {})
            _normalize_function_result_messages(attrs)
            span._attributes = attrs
            return

        if not _is_native_scope(scope_name):
            return

        span_id = _get_span_id(span)
        if span_id is not None:
            with self._lock:
                self._native_export_parents.pop(span_id, None)

        # The downstream Respan filter rejects an attribute-free runtime span.
        # Reassigning also works for ended spans whose BoundedAttributes froze.
        span._attributes = {}

    def shutdown(self) -> None:
        with self._lock:
            self._native_export_parents.clear()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


class AutoGenPrivacyProcessor(SpanProcessor):
    """Apply the call's content policy after the shared OI translator."""

    def __init__(self):
        self._content = {}
        self._lock = Lock()

    def on_start(self, span, parent_context=None):
        if _get_scope_name(span) not in (
            AUTOGEN_OPENINFERENCE_SCOPE_NAME,
            "openinference.instrumentation.ag2",
        ):
            return
        from respan_instrumentation_autogen._modern import capture_content

        with self._lock:
            self._content[_get_span_id(span)] = capture_content(parent_context)

    def on_end(self, span):
        with self._lock:
            allowed = self._content.pop(_get_span_id(span), True)
        if allowed:
            return
        attrs = dict(span.attributes)
        for key in tuple(attrs):
            if key in (
                TLSpanAttributes.TRACELOOP_ENTITY_INPUT,
                TLSpanAttributes.TRACELOOP_ENTITY_OUTPUT,
                TLSpanAttributes.LLM_REQUEST_FUNCTIONS,
            ) or key.startswith(
                (
                    TLSpanAttributes.LLM_PROMPTS + ".",
                    TLSpanAttributes.LLM_COMPLETIONS + ".",
                )
            ):
                attrs.pop(key, None)
        span._attributes = attrs

    def shutdown(self):
        with self._lock:
            self._content.clear()

    def force_flush(self, timeout_millis=30000):
        return True
