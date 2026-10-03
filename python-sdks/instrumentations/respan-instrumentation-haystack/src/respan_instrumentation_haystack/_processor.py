"""Normalize Haystack content and preserve pipeline parent relationships."""

import json
from typing import Any

from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor
from opentelemetry.semconv_ai import SpanAttributes
from respan_sdk.utils.data_processing.id_processing import format_span_id
from respan_tracing.decorators.base import _should_send_prompts

from ._compat import apply_embedding_capture
from ._constants import (
    HAYSTACK_NATIVE_PROCESSING_ATTRIBUTES,
    HAYSTACK_NATIVE_SPAN_NAMES,
    HAYSTACK_PIPELINE_SPAN_NAMES,
)
from ._content import without_content
from ._context import (
    _CURRENT_COMPONENT_RUN_CONTEXT,
    _CURRENT_PIPELINE_RUN_CONTEXT,
    _HaystackComponentRunContext,
)


def _get_span_id(span: Any) -> str | None:
    get_span_context = getattr(span, "get_span_context", None)
    if get_span_context is None:
        return None

    span_context = get_span_context()
    span_id = getattr(span_context, "span_id", None)
    if not span_id:
        return None
    return format_span_id(span_id)


def _get_parent_span_id(span: Any) -> str | None:
    parent = getattr(span, "parent", None)
    span_id = getattr(parent, "span_id", None)
    if not span_id:
        return None
    return format_span_id(span_id)


def _is_haystack_native_span(span: Any) -> bool:
    scope = getattr(span, "instrumentation_scope", None)
    scope_name = getattr(scope, "name", None)
    return (
        scope_name is None
        or scope_name == "haystack"
        or scope_name.startswith("haystack.")
    ) and getattr(span, "name", None) in HAYSTACK_NATIVE_SPAN_NAMES


def _suppress_haystack_native_span_export(span: Any) -> None:
    attributes = getattr(span, "_attributes", None)
    if attributes is None:
        return

    # ReadableSpan stores ended-span attributes in an immutable
    # BoundedAttributes instance on newer OpenTelemetry releases. Replace the
    # private snapshot instead of mutating it so native Haystack spans remain
    # unprocessable without raising during processor shutdown.
    span._attributes = {
        name: value
        for name, value in attributes.items()
        if name not in HAYSTACK_NATIVE_PROCESSING_ATTRIBUTES
    }


def _parse_json_attr(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return value


def _json_attr(value: Any) -> str:
    return json.dumps(value, default=str, separators=(",", ":"))


def _message_content(value: Any) -> str:
    if isinstance(value, str):
        return value

    if isinstance(value, list):
        text_parts: list[str] = []
        for item in value:
            if isinstance(item, dict):
                text = item.get("text")
                if text is None:
                    text = item.get("content")
                if text is not None:
                    text_parts.append(str(text))
            elif item is not None:
                text_parts.append(str(item))
        return "\n".join(text_parts)

    if value is None:
        return ""
    return str(value)


def _normalize_haystack_message(
    value: Any, *, fallback_role: str
) -> dict[str, Any] | None:
    if isinstance(value, str):
        return {"role": fallback_role, "content": value}

    if not isinstance(value, dict):
        return None

    role = value.get("role") or fallback_role
    content = _message_content(value.get("content"))
    message: dict[str, Any] = {"role": str(role), "content": content}

    name = value.get("name")
    if name is not None:
        message["name"] = name

    return message


def _haystack_input_messages(attrs: dict[str, Any]) -> list[dict[str, Any]]:
    payload = _parse_json_attr(attrs.get("haystack.component.input"))
    if not isinstance(payload, dict):
        return []

    messages = payload.get("messages")
    if isinstance(messages, list):
        normalized = [
            message
            for item in messages
            if (
                message := _normalize_haystack_message(
                    item,
                    fallback_role="user",
                )
            )
            is not None
        ]
        if normalized:
            return normalized

    for key in ("query", "question", "prompt"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            return [{"role": "user", "content": value}]

    return []


def _haystack_completion_message(attrs: dict[str, Any]) -> dict[str, Any] | None:
    payload = _parse_json_attr(attrs.get("haystack.component.output"))
    if not isinstance(payload, dict):
        return None

    replies = payload.get("replies")
    if isinstance(replies, list):
        for item in reversed(replies):
            message = _normalize_haystack_message(item, fallback_role="assistant")
            if message is not None and message.get("content"):
                return message
        for item in reversed(replies):
            message = _normalize_haystack_message(item, fallback_role="assistant")
            if message is not None:
                return message

    answers = payload.get("answers")
    if isinstance(answers, list):
        for item in reversed(answers):
            if isinstance(item, dict):
                data = item.get("data") or item.get("answer")
                if data is not None:
                    return {"role": "assistant", "content": str(data)}
            elif isinstance(item, str):
                return {"role": "assistant", "content": item}

    return None


def _set_indexed_messages(
    attrs: dict[str, Any],
    *,
    prefix: str,
    messages: list[dict[str, Any]],
) -> None:
    for index, message in enumerate(messages):
        role = message.get("role")
        if role is not None:
            attrs[f"{prefix}.{index}.role"] = str(role)
        content = message.get("content")
        if content is not None:
            attrs[f"{prefix}.{index}.content"] = str(content)


def _enrich_haystack_io_attrs(attrs: dict[str, Any]) -> None:
    input_messages = _haystack_input_messages(attrs)
    if input_messages:
        attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = _json_attr(input_messages)
        _set_indexed_messages(
            attrs,
            prefix=SpanAttributes.LLM_PROMPTS,
            messages=input_messages,
        )

    completion_message = _haystack_completion_message(attrs)
    if completion_message is not None:
        attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = _json_attr(completion_message)
        _set_indexed_messages(
            attrs,
            prefix=SpanAttributes.LLM_COMPLETIONS,
            messages=[completion_message],
        )


class _HaystackParentSpanProcessor(SpanProcessor):
    """Suppress native Haystack spans while preserving parent remapping.

    Haystack creates native pipeline/component spans around OpenInference spans.
    Those native spans are not useful Respan log rows, but their IDs are needed
    so exported child spans do not point at missing parents.
    """

    def __init__(self) -> None:
        self._parent_by_span_id: dict[str, str | None] = {}
        self._context_by_span_id: dict[str, Any] = {}
        self._component_context_by_span_id: dict[str, _HaystackComponentRunContext] = {}
        self._native_span_ids: set[str] = set()
        self._span_ids_by_trace: dict[int, set[str]] = {}
        self._active_by_trace: dict[int, int] = {}
        self._content_allowed_by_span: dict[str, bool] = {}

    def on_start(self, span: Any, parent_context: Any = None) -> None:
        span_id = _get_span_id(span)
        if span_id is None:
            return

        trace_id = getattr(span.get_span_context(), "trace_id", None)
        if trace_id is not None:
            self._span_ids_by_trace.setdefault(trace_id, set()).add(span_id)
            self._active_by_trace[trace_id] = self._active_by_trace.get(trace_id, 0) + 1
        self._content_allowed_by_span[span_id] = _should_send_prompts()
        self._parent_by_span_id[span_id] = _get_parent_span_id(span)
        self._context_by_span_id[span_id] = span.get_span_context()
        if _is_haystack_native_span(span):
            self._native_span_ids.add(span_id)

        pipeline_context = _CURRENT_PIPELINE_RUN_CONTEXT.get()
        if (
            pipeline_context is not None
            and pipeline_context.pipeline_span_id is None
            and getattr(span, "name", None) in HAYSTACK_PIPELINE_SPAN_NAMES
        ):
            pipeline_context.pipeline_span_id = span_id

        component_context = _CURRENT_COMPONENT_RUN_CONTEXT.get()
        if component_context is not None and component_context.component_name:
            self._component_context_by_span_id[span_id] = component_context

    def on_end(self, span: ReadableSpan) -> None:
        try:
            self._process_end(span)
            scope = getattr(getattr(span, "instrumentation_scope", None), "name", "")
            allowed = self._content_allowed_by_span.get(_get_span_id(span), True)
            if scope == "openinference.instrumentation.haystack" and (
                not allowed or not _should_send_prompts()
            ):
                span._attributes = without_content(dict(span.attributes or {}))
        finally:
            trace_id = getattr(span.get_span_context(), "trace_id", None)
            if trace_id in self._active_by_trace:
                self._active_by_trace[trace_id] -= 1
                if not self._active_by_trace[trace_id]:
                    del self._active_by_trace[trace_id]
                    for span_id in self._span_ids_by_trace.pop(trace_id, ()):
                        self._parent_by_span_id.pop(span_id, None)
                        self._context_by_span_id.pop(span_id, None)
                        self._component_context_by_span_id.pop(span_id, None)
                        self._native_span_ids.discard(span_id)
                        self._content_allowed_by_span.pop(span_id, None)

    def _process_end(self, span: ReadableSpan) -> None:
        span_id = _get_span_id(span)
        if _is_haystack_native_span(span):
            _suppress_haystack_native_span_export(span)
            return

        attributes = getattr(span, "_attributes", None)
        if attributes is not None:
            _enrich_haystack_io_attrs(attributes)

        apply_embedding_capture(span)
        parent_id = _get_parent_span_id(span)
        component_context = (
            self._component_context_by_span_id.get(span_id)
            if span_id is not None
            else None
        )
        if self._is_pipeline_component_span(
            parent_id=parent_id,
            component_context=component_context,
        ):
            exported_parent_id = self._graph_parent_span_id(component_context)
            if exported_parent_id is not None:
                exported_parent_context = self._context_by_span_id.get(
                    exported_parent_id
                )
                if exported_parent_context is None:
                    return
                span._parent = exported_parent_context
            elif parent_id in self._native_span_ids:
                exported_parent_id = self._nearest_exported_parent_id(parent_id)
                if exported_parent_id is None:
                    return
                exported_parent_context = self._context_by_span_id.get(
                    exported_parent_id
                )
                if exported_parent_context is None:
                    return
                span._parent = exported_parent_context

            if (
                span_id is not None
                and component_context is not None
                and component_context.pipeline_context is not None
            ):
                component_context.pipeline_context.record_completion(
                    component_context.component_name,
                    span_id,
                )
            return

        if parent_id is None or parent_id not in self._native_span_ids:
            return

        exported_parent_id = self._nearest_exported_parent_id(parent_id)
        if exported_parent_id is None:
            return

        exported_parent_context = self._context_by_span_id.get(exported_parent_id)
        if exported_parent_context is None:
            return

        span._parent = exported_parent_context

    def shutdown(self) -> None:
        self._parent_by_span_id.clear()
        self._context_by_span_id.clear()
        self._component_context_by_span_id.clear()
        self._native_span_ids.clear()
        self._span_ids_by_trace.clear()
        self._active_by_trace.clear()
        self._content_allowed_by_span.clear()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True

    def _nearest_exported_parent_id(self, span_id: str) -> str | None:
        seen: set[str] = set()
        current_id: str | None = span_id

        while current_id is not None:
            if current_id in seen:
                return None
            seen.add(current_id)

            parent_id = self._parent_by_span_id.get(current_id)
            if parent_id is None:
                return None
            if parent_id not in self._native_span_ids:
                return parent_id
            current_id = parent_id

        return None

    def _graph_parent_span_id(
        self,
        component_context: _HaystackComponentRunContext | None,
    ) -> str | None:
        if component_context is None or component_context.pipeline_context is None:
            return None

        pipeline_context = component_context.pipeline_context
        graph = pipeline_context.graph
        if graph is None:
            return None

        try:
            predecessors = tuple(graph.predecessors(component_context.component_name))
        except Exception:  # noqa: BLE001
            return None

        candidates: list[tuple[int, str]] = []
        for predecessor in predecessors:
            completed_span_id = pipeline_context.completed_span_id_by_component.get(
                predecessor
            )
            if completed_span_id is None:
                continue
            candidates.append(
                (
                    pipeline_context.completion_order_by_component.get(
                        predecessor,
                        0,
                    ),
                    completed_span_id,
                )
            )

        if not candidates:
            return None
        _, span_id = max(candidates)
        return span_id

    def _is_pipeline_component_span(
        self,
        *,
        parent_id: str | None,
        component_context: _HaystackComponentRunContext | None,
    ) -> bool:
        if (
            parent_id is None
            or component_context is None
            or component_context.pipeline_context is None
        ):
            return False

        if parent_id in self._native_span_ids:
            return True

        return parent_id == component_context.pipeline_context.pipeline_span_id
