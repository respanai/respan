"""Instrumentation for Mirascope 2.x model and toolkit execution surfaces."""

from __future__ import annotations

import functools
import importlib
import logging
import threading
import weakref
from dataclasses import dataclass
from typing import Any, Callable

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from opentelemetry.semconv_ai import LLMRequestTypeValues
from opentelemetry.semconv_ai import SpanAttributes

from respan_instrumentation_mirascope._serialization import json_string, json_value
from respan_sdk.constants import ERROR_MESSAGE_ATTR
from respan_sdk.constants.llm_logging import LogMethodChoices
from respan_sdk.constants.span_attributes import RESPAN_LOG_METHOD, RESPAN_LOG_TYPE
from respan_tracing.core.tracer import RespanTracer

logger = logging.getLogger(__name__)

MIRASCOPE_INSTRUMENTATION_NAME = "mirascope"
_LOCK = threading.RLock()
_REFCOUNT = 0
_CAPTURE_CONTENT = True
_PATCHES: list["_Patch"] = []

_MODERN_INPUT_USAGE = getattr(
    SpanAttributes, "LLM_USAGE_INPUT_TOKENS", "gen_ai.usage.input_tokens"
)
_MODERN_OUTPUT_USAGE = getattr(
    SpanAttributes, "LLM_USAGE_OUTPUT_TOKENS", "gen_ai.usage.output_tokens"
)
_CACHE_READ_USAGE = getattr(
    SpanAttributes,
    "LLM_USAGE_CACHE_READ_INPUT_TOKENS",
    "llm.usage.cache_read_input_tokens",
)
_CACHE_WRITE_USAGE = getattr(
    SpanAttributes,
    "LLM_USAGE_CACHE_WRITE_INPUT_TOKENS",
    "llm.usage.cache_write_input_tokens",
)


@dataclass
class _Patch:
    owner: Any
    name: str
    original: Any
    replacement: Any


def _is_respan_tracing_enabled() -> bool:
    tracer = getattr(RespanTracer, "_instance", None)
    if tracer is None:
        return True
    return bool(getattr(tracer, "is_enabled", True))


def _model_identity(model: Any, response: Any = None) -> tuple[str, str]:
    raw_model = str(
        getattr(response, "model_id", None)
        or getattr(model, "model_id", None)
        or "unknown"
    )
    if "/" in raw_model:
        inferred_provider, model_name = raw_model.split("/", 1)
    else:
        inferred_provider, model_name = "unknown", raw_model
    provider = str(getattr(response, "provider_id", None) or inferred_provider)
    return provider, model_name


def _message_parts(value: Any) -> list[tuple[str, Any]]:
    if isinstance(value, str):
        return [("user", value)]
    values = value if isinstance(value, (list, tuple)) else [value]
    messages: list[tuple[str, Any]] = []
    for item in values:
        if isinstance(item, str):
            messages.append(("user", item))
            continue
        role = getattr(item, "role", None)
        content = getattr(item, "content", None)
        if isinstance(item, dict):
            role = item.get("role", role)
            content = item.get("content", content)
        messages.append((str(role or "user"), content if content is not None else item))
    return messages


def _set_messages(span: Any, *, prefix: str, value: Any) -> None:
    for index, (role, content) in enumerate(_message_parts(value)):
        span.set_attribute(f"{prefix}.{index}.role", role)
        span.set_attribute(
            f"{prefix}.{index}.content",
            content if isinstance(content, str) else json_string(content),
        )


def _tool_definitions(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    tools = value if isinstance(value, (list, tuple)) else [value]
    definitions: list[dict[str, Any]] = []
    for tool in tools:
        definitions.append(
            {
                "name": getattr(tool, "name", None)
                or getattr(tool, "__name__", tool.__class__.__name__),
                "description": getattr(tool, "description", None)
                or getattr(tool, "__doc__", None),
                "parameters": json_value(
                    getattr(tool, "schema", None)
                    or getattr(tool, "parameters", None)
                    or {}
                ),
            }
        )
    return definitions


def _prepare_chat_span(
    span: Any,
    *,
    model: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> None:
    provider, model_name = _model_identity(model)
    entity_name = f"mirascope.{model_name}"
    span.set_attribute(RESPAN_LOG_METHOD, LogMethodChoices.TRACING_INTEGRATION.value)
    span.set_attribute(RESPAN_LOG_TYPE, "chat")
    span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_NAME, entity_name)
    span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_PATH, entity_name)
    span.set_attribute(SpanAttributes.LLM_SYSTEM, provider)
    span.set_attribute(SpanAttributes.LLM_REQUEST_MODEL, model_name)
    span.set_attribute(SpanAttributes.LLM_REQUEST_TYPE, LLMRequestTypeValues.CHAT.value)
    content = args[1] if len(args) > 1 else kwargs.get("content")
    if _CAPTURE_CONTENT:
        span.set_attribute(
            SpanAttributes.TRACELOOP_ENTITY_INPUT,
            json_string({"content": content, "tools": kwargs.get("tools")}),
        )
        _set_messages(span, prefix=SpanAttributes.LLM_PROMPTS, value=content)
        definitions = _tool_definitions(kwargs.get("tools"))
        if definitions:
            span.set_attribute(
                SpanAttributes.LLM_REQUEST_FUNCTIONS, json_string(definitions)
            )


def _set_usage(span: Any, usage: Any) -> None:
    if usage is None:
        return
    input_tokens = getattr(usage, "input_tokens", None)
    output_tokens = getattr(usage, "output_tokens", None)
    cache_read = getattr(usage, "cache_read_tokens", None)
    cache_write = getattr(usage, "cache_write_tokens", None)
    if input_tokens is not None:
        span.set_attribute(SpanAttributes.LLM_USAGE_PROMPT_TOKENS, input_tokens)
        span.set_attribute(_MODERN_INPUT_USAGE, input_tokens)
    if output_tokens is not None:
        span.set_attribute(SpanAttributes.LLM_USAGE_COMPLETION_TOKENS, output_tokens)
        span.set_attribute(_MODERN_OUTPUT_USAGE, output_tokens)
    if input_tokens is not None and output_tokens is not None:
        span.set_attribute(
            SpanAttributes.LLM_USAGE_TOTAL_TOKENS,
            int(input_tokens) + int(output_tokens),
        )
    if cache_read is not None:
        span.set_attribute(_CACHE_READ_USAGE, cache_read)
    if cache_write is not None:
        span.set_attribute(_CACHE_WRITE_USAGE, cache_write)


def _status_code(value: Any, *, default: int) -> int:
    candidates = [
        value,
        getattr(value, "response", None),
        getattr(value, "raw_response", None),
    ]
    for candidate in candidates:
        for name in ("status_code", "status"):
            try:
                code = getattr(candidate, name, None)
                if isinstance(code, int):
                    return code
            except Exception:
                continue
    return default


def _finish_chat_span(span: Any, *, model: Any, response: Any) -> None:
    span.set_attribute("status_code", _status_code(response, default=200))
    _, model_name = _model_identity(model, response)
    span.set_attribute(SpanAttributes.LLM_RESPONSE_MODEL, model_name)
    _set_usage(span, getattr(response, "usage", None))
    if not _CAPTURE_CONTENT or response is None:
        return
    content = getattr(response, "content", None)
    text_value = getattr(response, "text", None)
    if callable(text_value):
        try:
            text_value = text_value()
        except Exception:
            text_value = None
    output = text_value if text_value is not None else content
    tool_calls = getattr(response, "tool_calls", None)
    span.set_attribute(
        SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
        json_string({"content": content, "tool_calls": tool_calls}),
    )
    span.set_attribute(f"{SpanAttributes.LLM_COMPLETIONS}.0.role", "assistant")
    if output is not None:
        span.set_attribute(
            f"{SpanAttributes.LLM_COMPLETIONS}.0.content",
            output if isinstance(output, str) else json_string(output),
        )
    if tool_calls:
        span.set_attribute(
            f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls",
            json_string(tool_calls),
        )


def _record_error(span: Any, exc: BaseException) -> None:
    status_code = _status_code(exc, default=500)
    if status_code < 400:
        status_code = 500
    message = str(exc)
    span.set_attribute("status_code", status_code)
    span.set_attribute(ERROR_MESSAGE_ATTR, message)
    span.set_attribute(
        SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
        json_string(
            {
                "status": "error",
                "error": type(exc).__name__,
                "message": message,
            }
        ),
    )
    span.record_exception(exc)
    span.set_status(Status(StatusCode.ERROR, message))


class _StreamSpanState:
    def __init__(self, span: Any, model: Any) -> None:
        self.span = span
        self.model = model
        self._lock = threading.Lock()
        self._ended = False
        self.finalizer: weakref.finalize | None = None

    def finish(self, response: Any = None) -> None:
        with self._lock:
            if self._ended:
                return
            self._ended = True
            if response is not None:
                _finish_chat_span(self.span, model=self.model, response=response)
            self.span.end()
            if self.finalizer is not None and self.finalizer.alive:
                self.finalizer.detach()

    def fail(self, exc: BaseException) -> None:
        _record_error(self.span, exc)
        self.finish()


def _wrap_sync_iterator(source: Any, response: Any, state: _StreamSpanState) -> Any:
    def iterator() -> Any:
        try:
            while True:
                try:
                    with trace.use_span(state.span, end_on_exit=False):
                        chunk = next(source)
                except StopIteration:
                    break
                yield chunk
        except BaseException as exc:
            state.fail(exc)
            raise
        else:
            state.finish(response)

    return iterator()


def _wrap_async_iterator(source: Any, response: Any, state: _StreamSpanState) -> Any:
    async def iterator() -> Any:
        try:
            while True:
                try:
                    with trace.use_span(state.span, end_on_exit=False):
                        chunk = await source.__anext__()
                except StopAsyncIteration:
                    break
                yield chunk
        except BaseException as exc:
            state.fail(exc)
            raise
        else:
            state.finish(response)

    return iterator()


def _call_wrapper(original: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(original)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        model = args[0]
        span = trace.get_tracer(MIRASCOPE_INSTRUMENTATION_NAME).start_span(
            f"mirascope.{_model_identity(model)[1]}.chat"
        )
        _prepare_chat_span(span, model=model, args=args, kwargs=kwargs)
        try:
            with trace.use_span(span, end_on_exit=False):
                response = original(*args, **kwargs)
            _finish_chat_span(span, model=model, response=response)
            return response
        except BaseException as exc:
            _record_error(span, exc)
            raise
        finally:
            span.end()

    return wrapper


def _async_call_wrapper(original: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(original)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        model = args[0]
        span = trace.get_tracer(MIRASCOPE_INSTRUMENTATION_NAME).start_span(
            f"mirascope.{_model_identity(model)[1]}.chat"
        )
        _prepare_chat_span(span, model=model, args=args, kwargs=kwargs)
        try:
            with trace.use_span(span, end_on_exit=False):
                response = await original(*args, **kwargs)
            _finish_chat_span(span, model=model, response=response)
            return response
        except BaseException as exc:
            _record_error(span, exc)
            raise
        finally:
            span.end()

    return wrapper


def _stream_wrapper(original: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(original)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        model = args[0]
        span = trace.get_tracer(MIRASCOPE_INSTRUMENTATION_NAME).start_span(
            f"mirascope.{_model_identity(model)[1]}.chat"
        )
        _prepare_chat_span(span, model=model, args=args, kwargs=kwargs)
        try:
            with trace.use_span(span, end_on_exit=False):
                response = original(*args, **kwargs)
        except BaseException as exc:
            _record_error(span, exc)
            span.end()
            raise
        state = _StreamSpanState(span, model)
        source = getattr(response, "_chunk_iterator", None)
        if source is None:
            state.finish(response)
            return response
        response._chunk_iterator = _wrap_sync_iterator(source, response, state)
        try:
            state.finalizer = weakref.finalize(response, state.finish)
        except TypeError:
            pass
        return response

    return wrapper


def _async_stream_wrapper(original: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(original)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        model = args[0]
        span = trace.get_tracer(MIRASCOPE_INSTRUMENTATION_NAME).start_span(
            f"mirascope.{_model_identity(model)[1]}.chat"
        )
        _prepare_chat_span(span, model=model, args=args, kwargs=kwargs)
        try:
            with trace.use_span(span, end_on_exit=False):
                response = await original(*args, **kwargs)
        except BaseException as exc:
            _record_error(span, exc)
            span.end()
            raise
        state = _StreamSpanState(span, model)
        source = getattr(response, "_chunk_iterator", None)
        if source is None:
            state.finish(response)
            return response
        response._chunk_iterator = _wrap_async_iterator(source, response, state)
        try:
            state.finalizer = weakref.finalize(response, state.finish)
        except TypeError:
            pass
        return response

    return wrapper


def _tool_call(args: tuple[Any, ...]) -> Any:
    return args[-1] if len(args) > 1 else None


def _tool_name(tool_call: Any) -> str:
    function = getattr(tool_call, "function", None)
    return str(
        getattr(tool_call, "name", None)
        or getattr(function, "name", None)
        or "mirascope.tool"
    )


def _prepare_tool_span(span: Any, tool_call: Any) -> None:
    name = _tool_name(tool_call)
    span.set_attribute(RESPAN_LOG_METHOD, LogMethodChoices.TRACING_INTEGRATION.value)
    span.set_attribute(RESPAN_LOG_TYPE, "tool")
    span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_NAME, name)
    span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_PATH, name)
    if _CAPTURE_CONTENT:
        value = getattr(tool_call, "args", None)
        if value is None:
            value = getattr(tool_call, "arguments", tool_call)
        span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_INPUT, json_string(value))


def _tool_wrapper(original: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(original)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        tool_call = kwargs.get("tool_call") or _tool_call(args)
        name = _tool_name(tool_call)
        tracer = trace.get_tracer(MIRASCOPE_INSTRUMENTATION_NAME)
        with tracer.start_as_current_span(f"{name}.tool") as span:
            _prepare_tool_span(span, tool_call)
            try:
                result = original(*args, **kwargs)
            except BaseException as exc:
                _record_error(span, exc)
                raise
            span.set_attribute("status_code", _status_code(result, default=200))
            if _CAPTURE_CONTENT:
                span.set_attribute(
                    SpanAttributes.TRACELOOP_ENTITY_OUTPUT, json_string(result)
                )
            return result

    return wrapper


def _async_tool_wrapper(original: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(original)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        tool_call = kwargs.get("tool_call") or _tool_call(args)
        name = _tool_name(tool_call)
        tracer = trace.get_tracer(MIRASCOPE_INSTRUMENTATION_NAME)
        with tracer.start_as_current_span(f"{name}.tool") as span:
            _prepare_tool_span(span, tool_call)
            try:
                result = await original(*args, **kwargs)
            except BaseException as exc:
                _record_error(span, exc)
                raise
            span.set_attribute("status_code", _status_code(result, default=200))
            if _CAPTURE_CONTENT:
                span.set_attribute(
                    SpanAttributes.TRACELOOP_ENTITY_OUTPUT, json_string(result)
                )
            return result

    return wrapper


def _patch(owner: Any, name: str, factory: Callable[[Any], Any]) -> None:
    original = getattr(owner, name, None)
    if original is None or getattr(original, "__respan_mirascope_wrapper__", False):
        return
    replacement = factory(original)
    replacement.__respan_mirascope_wrapper__ = True
    setattr(owner, name, replacement)
    _PATCHES.append(_Patch(owner, name, original, replacement))


def _install_patches() -> None:
    models = importlib.import_module("mirascope.llm.models.models")
    model = getattr(models, "Model")
    for name in ("call", "context_call"):
        _patch(model, name, _call_wrapper)
    for name in ("call_async", "context_call_async"):
        _patch(model, name, _async_call_wrapper)
    for name in ("stream", "context_stream"):
        _patch(model, name, _stream_wrapper)
    for name in ("stream_async", "context_stream_async"):
        _patch(model, name, _async_stream_wrapper)

    toolkit_module = importlib.import_module("mirascope.llm.tools.toolkit")
    for class_name in ("Toolkit", "ContextToolkit"):
        owner = getattr(toolkit_module, class_name, None)
        if owner is not None:
            _patch(owner, "execute", _tool_wrapper)
    for class_name in ("AsyncToolkit", "AsyncContextToolkit"):
        owner = getattr(toolkit_module, class_name, None)
        if owner is not None:
            _patch(owner, "execute", _async_tool_wrapper)


def _remove_patches() -> None:
    for patch in reversed(_PATCHES):
        if getattr(patch.owner, patch.name, None) is patch.replacement:
            setattr(patch.owner, patch.name, patch.original)
    _PATCHES.clear()


class MirascopeInstrumentor:
    """Instrument Mirascope Model and Toolkit execution surfaces."""

    name = MIRASCOPE_INSTRUMENTATION_NAME

    def __init__(self, *, capture_content: bool = True) -> None:
        self._capture_content = capture_content
        self._is_instrumented = False

    def activate(self) -> None:
        global _CAPTURE_CONTENT, _REFCOUNT

        if self._is_instrumented or not _is_respan_tracing_enabled():
            return
        try:
            importlib.import_module("mirascope")
        except ImportError as exc:
            logger.warning("Mirascope instrumentation unavailable: %s", exc)
            return
        with _LOCK:
            if _REFCOUNT == 0:
                _CAPTURE_CONTENT = self._capture_content
                _install_patches()
            elif _CAPTURE_CONTENT != self._capture_content:
                logger.warning(
                    "Mirascope is already instrumented; the first capture_content setting wins"
                )
            _REFCOUNT += 1
            self._is_instrumented = True

    def deactivate(self) -> None:
        global _REFCOUNT

        if not self._is_instrumented:
            return
        with _LOCK:
            self._is_instrumented = False
            _REFCOUNT = max(0, _REFCOUNT - 1)
            if _REFCOUNT == 0:
                _remove_patches()
