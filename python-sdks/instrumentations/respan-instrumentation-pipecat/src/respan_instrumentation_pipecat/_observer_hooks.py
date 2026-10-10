"""Small Pipecat observer compatibility hooks owned by the Respan adapter."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from openinference.semconv.trace import SpanAttributes as OI
from opentelemetry.semconv._incubating.attributes.error_attributes import ERROR_MESSAGE
from opentelemetry.trace import Status, StatusCode

from respan_instrumentation_pipecat._runtime import _SCOPE
from respan_instrumentation_pipecat._serialization import (
    exception_message,
    exception_status_code,
    safe_text,
    safe_type_name,
)

RESPAN_PIPECAT_ERROR_STATUS = "pipecat.respan.error_status_code"
RESPAN_PIPECAT_ERROR_TYPE = "pipecat.respan.error_type"
RESPAN_PIPECAT_LLM_COMPLETED = "pipecat.respan.llm_completed"


@dataclass
class ObserverHook:
    observer_class: type
    original: Callable[..., Awaitable[None]]
    wrapper: Callable[..., Awaitable[None]]
    active: bool = True


def _safe_getattr(value: Any, name: str, default: Any = None) -> Any:
    try:
        return getattr(value, name, default)
    except Exception:  # noqa: BLE001 - hostile provider descriptors are ignored
        return default


def _frame_error(frame: Any) -> tuple[str, str, int | None] | None:
    if type(frame).__name__ != "ErrorFrame":
        return None
    exception = _safe_getattr(frame, "exception")
    if isinstance(exception, BaseException):
        return (
            exception_message(exception)
            or (safe_text(frame.error) if isinstance(frame.error, str) else None),
            safe_type_name(exception),
            exception_status_code(exception),
        )
    raw_error = _safe_getattr(frame, "error")
    message = safe_text(raw_error) if isinstance(raw_error, str) else None
    return message, type(frame).__module__ + "." + type(frame).__name__, None


def _mark_span(span: Any, *, message: str, error_type: str, status_code: int) -> None:
    if span is None:
        return
    current = _SCOPE.get()
    if isinstance(message, str) and (not current or current[0].allowed(current[1])):
        span.set_attribute(ERROR_MESSAGE, message)
    if status_code is not None:
        span.set_attribute(RESPAN_PIPECAT_ERROR_STATUS, status_code)
    span.set_attribute(RESPAN_PIPECAT_ERROR_TYPE, error_type)
    span.set_status(Status(StatusCode.ERROR))


def _record_error_frame(observer: Any, data: Any) -> None:
    frame = _safe_getattr(data, "frame")
    error = _frame_error(frame)
    if error is None:
        return
    message, error_type, status_code = error
    candidates = (
        _safe_getattr(frame, "processor"),
        _safe_getattr(data, "source"),
        _safe_getattr(data, "destination"),
    )
    active_spans = _safe_getattr(observer, "_active_spans", {})
    marked: set[int] = set()
    if isinstance(active_spans, dict):
        for candidate in candidates:
            span_info = (
                active_spans.get(id(candidate)) if candidate is not None else None
            )
            if isinstance(span_info, dict):
                span = span_info.get("span")
                if span is not None and id(span) not in marked:
                    _mark_span(
                        span,
                        message=message,
                        error_type=error_type,
                        status_code=status_code,
                    )
                    marked.add(id(span))
        if not marked:
            for span_info in active_spans.values():
                if (
                    not isinstance(span_info, dict)
                    or span_info.get("service_type") != "llm"
                ):
                    continue
                span = span_info.get("span")
                if span is not None:
                    _mark_span(
                        span,
                        message=message,
                        error_type=error_type,
                        status_code=status_code,
                    )
    _mark_span(
        _safe_getattr(observer, "_turn_span"),
        message=message,
        error_type=error_type,
        status_code=status_code,
    )


def _message_text(message: Any) -> str | None:
    if not isinstance(message, dict) or message.get("role") != "user":
        return None
    content = message.get("content")
    if isinstance(content, str):
        return safe_text(content)
    return None


def _record_semantic_frame(observer: Any, data: Any) -> None:
    current = _SCOPE.get()
    if current and not current[0].allowed(current[1]):
        return
    frame = _safe_getattr(data, "frame")
    frame_name = type(frame).__name__
    turn_span = _safe_getattr(observer, "_turn_span")
    if turn_span is None:
        return
    if frame_name == "LLMContextFrame":
        if current and observer._config.hide_inputs:
            return
        observer._respan_llm_text_chunks = []
        observer._respan_seen_llm_frame_ids = set()
        context = _safe_getattr(frame, "context")
        messages = _safe_getattr(context, "_messages") or _safe_getattr(
            context, "messages", []
        )
        try:
            user_text = [text for item in messages if (text := _message_text(item))]
        except Exception:  # noqa: BLE001 - hostile context collections are ignored
            user_text = []
        if user_text:
            existing = _safe_getattr(observer, "_turn_user_text", [])
            if not existing:
                observer._turn_user_text = list(user_text)
            turn_span.set_attribute(OI.INPUT_VALUE, " ".join(user_text))
    elif frame_name == "LLMFullResponseEndFrame":
        turn_span.set_attribute(RESPAN_PIPECAT_LLM_COMPLETED, True)
    elif frame_name == "LLMTextFrame":
        if current and observer._config.hide_outputs:
            return
        text = _safe_getattr(frame, "text")
        if isinstance(text, str) and text:
            frame_id = _safe_getattr(frame, "id", id(frame))
            seen = _safe_getattr(observer, "_respan_seen_llm_frame_ids", set())
            if not isinstance(seen, set):
                seen = set()
            if frame_id in seen:
                return
            seen.add(frame_id)
            observer._respan_seen_llm_frame_ids = seen
            chunks = _safe_getattr(observer, "_respan_llm_text_chunks", [])
            if isinstance(chunks, list):
                chunks.append(safe_text(text))
                observer._respan_llm_text_chunks = chunks


def _prepare_turn_end(observer: Any, data: Any) -> None:
    current = _SCOPE.get()
    if current and not current[0].allowed(current[1]):
        return
    frame = _safe_getattr(data, "frame")
    if type(frame).__name__ not in {"EndFrame", "CancelFrame"}:
        return
    chunks = _safe_getattr(observer, "_respan_llm_text_chunks", [])
    if isinstance(chunks, list) and chunks:
        observer._turn_bot_text = [safe_text("".join(chunks))]


def install_observer_hook(observer_module: Any) -> ObserverHook:
    observer_class = observer_module.OpenInferenceObserver
    original = observer_class.on_push_frame
    hook: ObserverHook

    async def wrapped(instance: Any, data: Any) -> None:
        current = _SCOPE.get()
        owned = current is not None and current[1] is instance
        if hook.active and owned:
            _record_error_frame(instance, data)
            _prepare_turn_end(instance, data)
        await original(instance, data)
        if hook.active and owned:
            _record_semantic_frame(instance, data)

    hook = ObserverHook(
        observer_class=observer_class, original=original, wrapper=wrapped
    )
    observer_class.on_push_frame = wrapped
    return hook


def remove_observer_hook(hook: ObserverHook | None) -> None:
    if hook is None:
        return
    hook.active = False
    if hook.observer_class.on_push_frame is hook.wrapper:
        hook.observer_class.on_push_frame = hook.original
