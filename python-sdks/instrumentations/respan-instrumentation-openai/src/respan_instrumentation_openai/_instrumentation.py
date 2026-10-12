"""Native, current-OpenAI-SDK instrumentation for Respan."""

from __future__ import annotations

import asyncio
import importlib
import inspect
import logging
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from typing import Any, Self

from respan_sdk.constants.tracing_constants import RESPAN_LOG_ID_RESPONSE_HEADER
from respan_tracing.core.tracer import RespanTracer

from respan_instrumentation_openai import _otel_emitter as emitter
from respan_instrumentation_openai._constants import (
    ASYNC_BASE_CLIENT_CLASS,
    ASYNC_CHAT_CLASS,
    ASYNC_COMPLETIONS_CLASS,
    ASYNC_EMBEDDINGS_CLASS,
    ASYNC_RESPONSES_CLASS,
    BASE_CLIENT_MODULE,
    CHAT_MODULE,
    COMPLETIONS_MODULE,
    CREATE_METHOD,
    EMBEDDINGS_MODULE,
    PARSE_METHOD,
    PROCESS_RESPONSE_METHOD,
    RESPONSES_MODULE,
    SYNC_BASE_CLIENT_CLASS,
    SYNC_CHAT_CLASS,
    SYNC_COMPLETIONS_CLASS,
    SYNC_EMBEDDINGS_CLASS,
    SYNC_RESPONSES_CLASS,
)
from respan_instrumentation_openai._serialization import (
    MAX_ERROR_LENGTH,
    error_message,
    redact_text,
)

logger = logging.getLogger(__name__)

_ORIGINAL_METHODS: dict[tuple[type[Any], str], Any] = {}
_INSTALLED_METHODS: dict[tuple[type[Any], str], Any] = {}
_original_methods = _ORIGINAL_METHODS  # compatibility for existing callers/tests
_LIFECYCLE_LOCK = threading.RLock()
_REFCOUNT = 0

_SUPPRESS_OPENAI_INSTRUMENTATION: ContextVar[bool] = ContextVar(
    "respan_suppress_openai_instrumentation", default=False
)


# Set by each instrumented call for its duration; the patched
# ``_process_response`` writes the gateway log id of the response into it.
_GATEWAY_LOG_ID_SLOT: ContextVar[list[str] | None] = ContextVar(
    "respan_openai_gateway_log_id_slot", default=None
)


def _header_log_id(response: Any) -> str | None:
    try:
        value = response.headers.get(RESPAN_LOG_ID_RESPONSE_HEADER)
    except BaseException:  # noqa: BLE001 - header capture must never break a call
        return None
    return value if isinstance(value, str) and value else None


def _record_gateway_log_id(response: Any) -> None:
    slot = _GATEWAY_LOG_ID_SLOT.get()
    if slot is None:
        return
    log_id = _header_log_id(response)
    if log_id:
        slot[:] = [log_id]


def _gateway_log_id(slot: list[str], exc: BaseException | None = None) -> str | None:
    """The Respan gateway log id of this call's response, if it went through it."""
    if slot:
        return slot[0]
    if exc is None:
        return None
    try:
        response = getattr(exc, "response", None)
    except BaseException:  # noqa: BLE001 - never mask the provider error
        return None
    return _header_log_id(response) if response is not None else None


def _is_openai_instrumentation_suppressed() -> bool:
    return bool(_SUPPRESS_OPENAI_INSTRUMENTATION.get())


@contextmanager
def suppress_openai_instrumentation() -> Iterator[None]:
    token = _SUPPRESS_OPENAI_INSTRUMENTATION.set(True)
    try:
        yield
    finally:
        _SUPPRESS_OPENAI_INSTRUMENTATION.reset(token)


def _request_kwargs_from_call(
    original: Callable[..., Any],
    instance: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    try:
        bound = inspect.signature(original).bind_partial(instance, *args, **kwargs)
    except (TypeError, ValueError):
        return dict(kwargs)
    return {
        key: value
        for key, value in bound.arguments.items()
        if key not in {"self", "cls"}
    }


def _exception_status_code(exc: BaseException) -> int:
    try:
        response = getattr(exc, "response", None)
    except BaseException:  # noqa: BLE001 - never mask the provider error
        response = None
    candidates = (exc, response)
    for candidate in candidates:
        try:
            value = getattr(candidate, "status_code", None)
        except BaseException:  # noqa: BLE001, S112 - never mask provider errors
            continue
        if (
            isinstance(value, int)
            and not isinstance(value, bool)
            and 400 <= value <= 599
        ):
            return value
    if isinstance(exc, asyncio.CancelledError):
        return 499
    return 500


def _error_kwargs(exc: BaseException) -> dict[str, Any]:
    return {
        "error_message": error_message(exc),
        "error_type": type(exc).__name__,
        "status_code": _exception_status_code(exc),
    }


class _StreamAccumulator:
    _TEXT_LIMIT = 12_000
    _TOOL_LIMIT = 50
    _ARGUMENT_LIMIT = 8_000

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self.parts: list[str] = []
        self.text_length = 0
        self.tool_calls: dict[int, dict[str, Any]] = {}
        self.usage = None
        self.model = None
        self.response_id = None
        self.final_response = None
        self.error_kwargs: dict[str, Any] = {}

    def _append_text(self, value: Any) -> None:
        if (
            not isinstance(value, str)
            or not value
            or self.text_length >= self._TEXT_LIMIT
        ):
            return
        piece = value[: self._TEXT_LIMIT - self.text_length]
        self.parts.append(piece)
        self.text_length += len(piece)

    def add(self, item: Any) -> None:
        model = getattr(item, "model", None)
        response_id = getattr(item, "id", None)
        usage = getattr(item, "usage", None)
        if isinstance(model, str) and model:
            self.model = model
        if isinstance(response_id, str) and response_id:
            self.response_id = response_id
        if usage is not None:
            self.usage = usage

        if self.kind == "response":
            response = getattr(item, "response", None)
            event_type = getattr(item, "type", None)
            if response is not None:
                self.model = getattr(response, "model", None) or self.model
                self.response_id = getattr(response, "id", None) or self.response_id
                self.usage = getattr(response, "usage", None) or self.usage
                if event_type in {
                    "response.completed",
                    "response.failed",
                    "response.incomplete",
                }:
                    self.final_response = response
                if event_type == "response.failed":
                    error = getattr(response, "error", None)
                    self.error_kwargs = {
                        "error_message": redact_text(
                            getattr(error, "message", None) or "OpenAI response failed",
                            limit=MAX_ERROR_LENGTH,
                        ),
                        "error_type": getattr(error, "code", None) or event_type,
                        "status_code": 500,
                    }
            if event_type == "response.output_text.delta":
                self._append_text(getattr(item, "delta", None))
            elif event_type == "response.output_item.added":
                output = getattr(item, "item", None)
                index = getattr(item, "output_index", 0)
                if (
                    getattr(output, "type", None) == "function_call"
                    and len(self.tool_calls) < self._TOOL_LIMIT
                ):
                    self.tool_calls[index] = {
                        "type": "function_call",
                        "id": getattr(output, "id", None),
                        "call_id": getattr(output, "call_id", None),
                        "name": getattr(output, "name", None),
                        "arguments": (getattr(output, "arguments", None) or "")[
                            : self._ARGUMENT_LIMIT
                        ],
                    }
            elif event_type == "response.function_call_arguments.delta":
                slot = self.tool_calls.get(getattr(item, "output_index", 0))
                delta = getattr(item, "delta", None)
                if slot is not None and isinstance(delta, str):
                    slot["arguments"] = (slot["arguments"] + delta)[
                        : self._ARGUMENT_LIMIT
                    ]
            return

        choices = getattr(item, "choices", None) or []
        if not choices:
            return
        choice = choices[0]
        if self.kind == "completion":
            self._append_text(getattr(choice, "text", None))
            return
        delta = getattr(choice, "delta", None)
        if delta is None:
            return
        self._append_text(getattr(delta, "content", None))
        for tool_delta in getattr(delta, "tool_calls", None) or []:
            index = getattr(tool_delta, "index", 0) or 0
            if (
                index not in self.tool_calls
                and len(self.tool_calls) >= self._TOOL_LIMIT
            ):
                continue
            slot = self.tool_calls.setdefault(
                index,
                {
                    "id": None,
                    "type": "function",
                    "function": {"name": "", "arguments": ""},
                },
            )
            call_id = getattr(tool_delta, "id", None)
            if isinstance(call_id, str) and call_id:
                slot["id"] = call_id
            function = getattr(tool_delta, "function", None)
            if function is None:
                continue
            name = getattr(function, "name", None)
            if isinstance(name, str) and name:
                slot["function"]["name"] = name
            arguments = getattr(function, "arguments", None)
            if isinstance(arguments, str) and arguments:
                current = slot["function"]["arguments"]
                slot["function"]["arguments"] = (current + arguments)[
                    : self._ARGUMENT_LIMIT
                ]

    def response(self) -> Any:
        text = "".join(self.parts)
        if self.kind == "response":
            return self.final_response or {
                "model": self.model,
                "id": self.response_id,
                "usage": self.usage,
                "output_text": text,
                "output": list(self.tool_calls.values()),
            }
        if self.kind == "completion":
            return {
                "model": self.model,
                "id": self.response_id,
                "usage": self.usage,
                "choices": [{"text": text}],
            }
        message: dict[str, Any] = {"role": "assistant", "content": text or None}
        if self.tool_calls:
            message["tool_calls"] = [
                self.tool_calls[index] for index in sorted(self.tool_calls)
            ]
        return {
            "model": self.model,
            "id": self.response_id,
            "usage": self.usage,
            "choices": [{"message": message}],
        }


def _aggregate_chat(chunks: list[Any]) -> Any:
    accumulator = _StreamAccumulator("chat")
    for chunk in chunks:
        accumulator.add(chunk)
    return accumulator.response()


def _aggregate_completion(chunks: list[Any]) -> Any:
    accumulator = _StreamAccumulator("completion")
    for chunk in chunks:
        accumulator.add(chunk)
    return accumulator.response()


def _aggregate_response(events: list[Any]) -> Any:
    accumulator = _StreamAccumulator("response")
    for event in events:
        accumulator.add(event)
    return accumulator.response()


_KINDS: dict[str, tuple[Callable[..., None], Callable[[list[Any]], Any] | None]] = {
    "chat": (emitter.emit_chat_span, _aggregate_chat),
    "completion": (emitter.emit_completion_span, _aggregate_completion),
    "response": (emitter.emit_response_span, _aggregate_response),
    "embedding": (emitter.emit_embedding_span, None),
}


def _is_stream(value: Any) -> bool:
    try:
        from openai import AsyncStream, Stream

        if isinstance(value, Stream | AsyncStream):
            return True
    except (ImportError, TypeError):
        pass
    return hasattr(value, "__aiter__")


class _SyncStreamWrapper:
    def __init__(
        self,
        iterator: Any,
        *,
        kind: str,
        request_kwargs: dict[str, Any],
        start_ns: int,
        trace_id: str | None,
        parent_id: str | None,
        gateway_log_id: str | None = None,
    ) -> None:
        self._iterator = iterator
        self._kind = kind
        self._request_kwargs = request_kwargs
        self._start_ns = start_ns
        self._trace_id = trace_id
        self._parent_id = parent_id
        self._gateway_log_id = gateway_log_id
        self._accumulator = _StreamAccumulator(kind)
        self._finished = False
        self._source_closed = False

    def __iter__(self) -> _SyncStreamWrapper:
        return self

    def __next__(self) -> Any:
        try:
            item = next(self._iterator)
        except StopIteration:
            self._finish()
            raise
        except BaseException as exc:
            self._close_source(suppress=True)
            self._finish(exc)
            raise
        self._accumulator.add(item)
        return item

    def _finish(self, exc: BaseException | None = None) -> None:
        if self._finished:
            return
        self._finished = True
        emit_fn, _ = _KINDS[self._kind]
        kwargs = dict(self._accumulator.error_kwargs)
        if exc is not None and not isinstance(exc, GeneratorExit):
            kwargs.update(_error_kwargs(exc))
        emit_fn(
            request_kwargs=self._request_kwargs,
            response=self._accumulator.response(),
            start_ns=self._start_ns,
            trace_id=self._trace_id,
            parent_id=self._parent_id,
            gateway_log_id=self._gateway_log_id,
            **kwargs,
        )

    def _close_source(self, *, suppress: bool) -> BaseException | None:
        if self._source_closed:
            return None
        self._source_closed = True
        close = getattr(self._iterator, "close", None)
        if not callable(close):
            return None
        try:
            close()
        except BaseException as exc:
            if not suppress:
                raise
            return exc
        return None

    def close(self) -> None:
        error = self._close_source(suppress=True)
        self._finish(error)
        if error is not None:
            raise error

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        if exc is not None:
            self._finish(exc)
            self._close_source(suppress=True)
        else:
            self.close()
        return False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._iterator, name)

    def __del__(self) -> None:
        try:
            self._close_source(suppress=True)
            self._finish()
        except BaseException:
            logger.debug("Failed to finalize abandoned OpenAI stream", exc_info=True)


class _AsyncStreamWrapper:
    def __init__(
        self,
        iterator: Any,
        *,
        kind: str,
        request_kwargs: dict[str, Any],
        start_ns: int,
        trace_id: str | None,
        parent_id: str | None,
        gateway_log_id: str | None = None,
    ) -> None:
        self._iterator = iterator
        self._kind = kind
        self._request_kwargs = request_kwargs
        self._start_ns = start_ns
        self._trace_id = trace_id
        self._parent_id = parent_id
        self._gateway_log_id = gateway_log_id
        self._accumulator = _StreamAccumulator(kind)
        self._finished = False
        self._source_closed = False

    def __aiter__(self) -> _AsyncStreamWrapper:
        return self

    async def __anext__(self) -> Any:
        try:
            item = await self._iterator.__anext__()
        except StopAsyncIteration:
            self._finish()
            raise
        except BaseException as exc:
            await self._close_source(suppress=True)
            self._finish(exc)
            raise
        self._accumulator.add(item)
        return item

    def _finish(self, exc: BaseException | None = None) -> None:
        if self._finished:
            return
        self._finished = True
        emit_fn, _ = _KINDS[self._kind]
        kwargs = dict(self._accumulator.error_kwargs)
        if exc is not None and not isinstance(exc, GeneratorExit):
            kwargs.update(_error_kwargs(exc))
        emit_fn(
            request_kwargs=self._request_kwargs,
            response=self._accumulator.response(),
            start_ns=self._start_ns,
            trace_id=self._trace_id,
            parent_id=self._parent_id,
            gateway_log_id=self._gateway_log_id,
            **kwargs,
        )

    async def _close_source(self, *, suppress: bool) -> BaseException | None:
        if self._source_closed:
            return None
        self._source_closed = True
        close = getattr(self._iterator, "aclose", None)
        if not callable(close):
            close = getattr(self._iterator, "close", None)
        if not callable(close):
            return None
        try:
            result = close()
            if inspect.isawaitable(result):
                await result
        except BaseException as exc:
            if not suppress:
                raise
            return exc
        return None

    async def close(self) -> None:
        error = await self._close_source(suppress=True)
        self._finish(error)
        if error is not None:
            raise error

    async def aclose(self) -> None:
        await self.close()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> bool:
        if exc is not None:
            self._finish(exc)
            await self._close_source(suppress=True)
        else:
            await self.close()
        return False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._iterator, name)


def _make_sync_wrapper(original: Any, *, kind: str) -> Any:
    emit_fn, _ = _KINDS[kind]

    @wraps(original)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        if _is_openai_instrumentation_suppressed():
            return original(self, *args, **kwargs)
        request_kwargs = _request_kwargs_from_call(original, self, args, kwargs)
        start_ns = time.time_ns()
        trace_id, parent_id = emitter.current_trace_parent_ids()
        slot: list[str] = []
        token = _GATEWAY_LOG_ID_SLOT.set(slot)
        try:
            response = original(self, *args, **kwargs)
        except BaseException as exc:
            emit_fn(
                request_kwargs=request_kwargs,
                start_ns=start_ns,
                trace_id=trace_id,
                parent_id=parent_id,
                gateway_log_id=_gateway_log_id(slot, exc),
                **_error_kwargs(exc),
            )
            raise
        finally:
            _GATEWAY_LOG_ID_SLOT.reset(token)
        if kind != "embedding" and _is_stream(response):
            return _SyncStreamWrapper(
                response,
                kind=kind,
                request_kwargs=request_kwargs,
                start_ns=start_ns,
                trace_id=trace_id,
                parent_id=parent_id,
                gateway_log_id=_gateway_log_id(slot),
            )
        emit_fn(
            request_kwargs=request_kwargs,
            response=response,
            start_ns=start_ns,
            trace_id=trace_id,
            parent_id=parent_id,
            gateway_log_id=_gateway_log_id(slot),
        )
        return response

    wrapper.__respan_openai_wrapper__ = True
    return wrapper


def _make_async_wrapper(original: Any, *, kind: str) -> Any:
    emit_fn, _ = _KINDS[kind]

    @wraps(original)
    async def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        if _is_openai_instrumentation_suppressed():
            pending = original(self, *args, **kwargs)
            return await pending if inspect.isawaitable(pending) else pending
        request_kwargs = _request_kwargs_from_call(original, self, args, kwargs)
        start_ns = time.time_ns()
        trace_id, parent_id = emitter.current_trace_parent_ids()
        slot: list[str] = []
        token = _GATEWAY_LOG_ID_SLOT.set(slot)
        try:
            pending = original(self, *args, **kwargs)
            response = await pending if inspect.isawaitable(pending) else pending
        except BaseException as exc:
            emit_fn(
                request_kwargs=request_kwargs,
                start_ns=start_ns,
                trace_id=trace_id,
                parent_id=parent_id,
                gateway_log_id=_gateway_log_id(slot, exc),
                **_error_kwargs(exc),
            )
            raise
        finally:
            _GATEWAY_LOG_ID_SLOT.reset(token)
        if kind != "embedding" and _is_stream(response):
            return _AsyncStreamWrapper(
                response,
                kind=kind,
                request_kwargs=request_kwargs,
                start_ns=start_ns,
                trace_id=trace_id,
                parent_id=parent_id,
                gateway_log_id=_gateway_log_id(slot),
            )
        emit_fn(
            request_kwargs=request_kwargs,
            response=response,
            start_ns=start_ns,
            trace_id=trace_id,
            parent_id=parent_id,
            gateway_log_id=_gateway_log_id(slot),
        )
        return response

    wrapper.__respan_openai_wrapper__ = True
    return wrapper


def _make_sync_process_response_wrapper(original: Any, **_kwargs: Any) -> Any:
    @wraps(original)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        _record_gateway_log_id(kwargs.get("response"))
        return original(self, *args, **kwargs)

    return wrapper


def _make_async_process_response_wrapper(original: Any, **_kwargs: Any) -> Any:
    @wraps(original)
    async def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        _record_gateway_log_id(kwargs.get("response"))
        return await original(self, *args, **kwargs)

    return wrapper


def _make_sync_close_wrapper(original: Any, **_kwargs: Any) -> Any:
    @wraps(original)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        raw = getattr(self, "_raw_stream", None)
        if not isinstance(raw, _SyncStreamWrapper):
            return original(self, *args, **kwargs)
        # SDK stream helpers close the HTTP response directly, bypassing the
        # raw stream's close(). Finalize at that same boundary without closing
        # the underlying response twice or retaining a callback cycle.
        if raw._source_closed:
            return None
        raw._source_closed = True
        try:
            result = original(self, *args, **kwargs)
        except BaseException as exc:
            raw._finish(exc)
            raise
        raw._finish()
        return result

    return wrapper


def _make_async_close_wrapper(original: Any, **_kwargs: Any) -> Any:
    @wraps(original)
    async def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        raw = getattr(self, "_raw_stream", None)
        if not isinstance(raw, _AsyncStreamWrapper):
            return await original(self, *args, **kwargs)
        if raw._source_closed:
            return None
        raw._source_closed = True
        try:
            result = await original(self, *args, **kwargs)
        except BaseException as exc:
            raw._finish(exc)
            raise
        raw._finish()
        return result

    return wrapper


def _load_class(module_path: str, class_name: str) -> type[Any] | None:
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        logger.debug("OpenAI module %s unavailable: %s", module_path, exc)
        return None
    return getattr(module, class_name, None)


def _patch(
    target_class: type[Any] | None,
    *,
    method_name: str,
    kind: str,
    is_async: bool,
) -> bool:
    if target_class is None:
        return False
    original = getattr(target_class, method_name, None)
    if original is None:
        return False
    key = (target_class, method_name)
    if key in _ORIGINAL_METHODS:
        return True
    _ORIGINAL_METHODS[key] = original
    if kind == "stream_close":
        factory = _make_async_close_wrapper if is_async else _make_sync_close_wrapper
    elif kind == "process_response":
        factory = (
            _make_async_process_response_wrapper
            if is_async
            else _make_sync_process_response_wrapper
        )
    else:
        factory = _make_async_wrapper if is_async else _make_sync_wrapper
    try:
        wrapper = factory(original, kind=kind)
        setattr(target_class, method_name, wrapper)
    except BaseException:
        _ORIGINAL_METHODS.pop(key, None)
        _INSTALLED_METHODS.pop(key, None)
        raise
    _INSTALLED_METHODS[key] = wrapper
    return True


_TARGETS = [
    (CHAT_MODULE, SYNC_CHAT_CLASS, CREATE_METHOD, "chat", False),
    (CHAT_MODULE, ASYNC_CHAT_CLASS, CREATE_METHOD, "chat", True),
    (CHAT_MODULE, SYNC_CHAT_CLASS, PARSE_METHOD, "chat", False),
    (CHAT_MODULE, ASYNC_CHAT_CLASS, PARSE_METHOD, "chat", True),
    (COMPLETIONS_MODULE, SYNC_COMPLETIONS_CLASS, CREATE_METHOD, "completion", False),
    (COMPLETIONS_MODULE, ASYNC_COMPLETIONS_CLASS, CREATE_METHOD, "completion", True),
    (RESPONSES_MODULE, SYNC_RESPONSES_CLASS, CREATE_METHOD, "response", False),
    (RESPONSES_MODULE, ASYNC_RESPONSES_CLASS, CREATE_METHOD, "response", True),
    (RESPONSES_MODULE, SYNC_RESPONSES_CLASS, PARSE_METHOD, "response", False),
    (RESPONSES_MODULE, ASYNC_RESPONSES_CLASS, PARSE_METHOD, "response", True),
    (EMBEDDINGS_MODULE, SYNC_EMBEDDINGS_CLASS, CREATE_METHOD, "embedding", False),
    (EMBEDDINGS_MODULE, ASYNC_EMBEDDINGS_CLASS, CREATE_METHOD, "embedding", True),
    (
        BASE_CLIENT_MODULE,
        SYNC_BASE_CLIENT_CLASS,
        PROCESS_RESPONSE_METHOD,
        "process_response",
        False,
    ),
    (
        BASE_CLIENT_MODULE,
        ASYNC_BASE_CLIENT_CLASS,
        PROCESS_RESPONSE_METHOD,
        "process_response",
        True,
    ),
    (
        "openai.lib.streaming.chat",
        "ChatCompletionStream",
        "close",
        "stream_close",
        False,
    ),
    (
        "openai.lib.streaming.chat",
        "AsyncChatCompletionStream",
        "close",
        "stream_close",
        True,
    ),
    (
        "openai.lib.streaming.responses",
        "ResponseStream",
        "close",
        "stream_close",
        False,
    ),
    (
        "openai.lib.streaming.responses",
        "AsyncResponseStream",
        "close",
        "stream_close",
        True,
    ),
]


def _install_patches() -> bool:
    installed_before = set(_ORIGINAL_METHODS)
    patched_any = False
    try:
        for module_path, class_name, method_name, kind, is_async in _TARGETS:
            patched_any = (
                _patch(
                    _load_class(module_path, class_name),
                    method_name=method_name,
                    kind=kind,
                    is_async=is_async,
                )
                or patched_any
            )
    except BaseException:
        for key in set(_ORIGINAL_METHODS) - installed_before:
            target_class, method_name = key
            original = _ORIGINAL_METHODS[key]
            installed = _INSTALLED_METHODS.get(key)
            try:
                current = getattr(target_class, method_name)
            except BaseException:
                logger.exception("Failed to inspect OpenAI patch %s", method_name)
                continue
            if installed is not None and current is not installed:
                _ORIGINAL_METHODS.pop(key, None)
                _INSTALLED_METHODS.pop(key, None)
                continue
            try:
                setattr(target_class, method_name, original)
            except BaseException:
                logger.exception("Failed to roll back OpenAI patch %s", method_name)
            else:
                _ORIGINAL_METHODS.pop(key, None)
                _INSTALLED_METHODS.pop(key, None)
        raise
    return patched_any


def _remove_patches() -> None:
    for key, original in list(_ORIGINAL_METHODS.items()):
        target_class, method_name = key
        installed = _INSTALLED_METHODS.get(key)
        try:
            current = getattr(target_class, method_name)
        except BaseException:
            logger.debug(
                "Failed to inspect %s.%s", target_class, method_name, exc_info=True
            )
            continue
        if installed is not None and current is not installed:
            _ORIGINAL_METHODS.pop(key, None)
            _INSTALLED_METHODS.pop(key, None)
            continue
        try:
            setattr(target_class, method_name, original)
        except BaseException:
            logger.debug(
                "Failed to restore %s.%s", target_class, method_name, exc_info=True
            )
        else:
            _ORIGINAL_METHODS.pop(key, None)
            _INSTALLED_METHODS.pop(key, None)


class OpenAIInstrumentor:
    """Reference-counted instrumentation for OpenAI 3.x resources."""

    name = "openai"

    def __init__(self) -> None:
        self._is_instrumented = False

    @staticmethod
    def _is_respan_tracing_enabled() -> bool:
        tracer = getattr(RespanTracer, "_instance", None)
        if tracer is None:
            return True
        return bool(getattr(tracer, "is_enabled", True))

    def activate(self) -> None:
        global _REFCOUNT
        with _LIFECYCLE_LOCK:
            if self._is_instrumented:
                return
            if not self._is_respan_tracing_enabled():
                logger.info(
                    "OpenAI instrumentation skipped because Respan tracing is disabled"
                )
                return
            try:
                import openai  # noqa: F401
            except ImportError as exc:
                logger.debug("OpenAI instrumentation inactive: %s", exc)
                return
            if _REFCOUNT == 0:
                try:
                    if not _install_patches():
                        logger.warning(
                            "openai is installed but no supported API surface was found"
                        )
                        return
                except BaseException:
                    logger.exception("Failed to activate OpenAI instrumentation")
                    return
            _REFCOUNT += 1
            self._is_instrumented = True
            logger.info("OpenAI SDK instrumentation activated")

    def deactivate(self) -> None:
        global _REFCOUNT
        with _LIFECYCLE_LOCK:
            if not self._is_instrumented:
                return
            self._is_instrumented = False
            _REFCOUNT = max(0, _REFCOUNT - 1)
            if _REFCOUNT == 0:
                _remove_patches()
            logger.info("OpenAI SDK instrumentation deactivated")
