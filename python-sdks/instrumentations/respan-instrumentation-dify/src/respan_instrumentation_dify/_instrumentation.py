"""Dify SDK instrumentation plugin for Respan."""

from __future__ import annotations

import importlib
import json
import logging
from collections.abc import Mapping
from functools import wraps
from types import TracebackType
from typing import Any, Self

from respan_tracing.core.tracer import RespanTracer

from respan_instrumentation_dify._constants import (
    DIFY_ASYNC_CLIENT_MODULE,
    DIFY_CLIENT_MODULE,
    DIFY_INSTRUMENTATION_NAME,
    RESPONSE_MODE_KEY,
    STREAMING_RESPONSE_MODE,
)
from respan_instrumentation_dify._context import use_respan_params
from respan_instrumentation_dify._otel_emitter import (
    capture_call_context,
    emit_dify_span,
)

logger = logging.getLogger(__name__)

_PATCHED_METHODS: dict[tuple[type[Any], str], tuple[Any, Any]] = {}
_IS_PATCHED = False
_ACTIVE_INSTANCES = 0
_INCLUDE_CONTENT = True


def _arg(
    args: tuple[Any, ...],
    index: int,
    kwargs: dict[str, Any],
    name: str,
    default: Any = None,
) -> Any:
    if len(args) > index:
        return args[index]
    return kwargs.get(name, default)


def _pop_respan_params(kwargs: dict[str, Any]) -> Any:
    return kwargs.pop("respan_params", None)


def _parse_sse_event(line: Any) -> Any:
    if isinstance(line, bytes):
        text = line.decode("utf-8", errors="replace")
    else:
        text = str(line)

    text = text.strip()
    if not text:
        return None
    if text.startswith("data:"):
        text = text.split("data:", maxsplit=1)[1].strip()
    if text == "[DONE]":
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _parse_sse_chunks(chunks: list[Any]) -> list[Any]:
    if not chunks:
        return []
    if all(isinstance(chunk, (bytes, bytearray)) for chunk in chunks):
        text = b"".join(bytes(chunk) for chunk in chunks).decode(
            "utf-8", errors="replace"
        )
    else:
        text = "".join(
            chunk.decode("utf-8", errors="replace")
            if isinstance(chunk, (bytes, bytearray))
            else str(chunk)
            for chunk in chunks
        )
    events: list[Any] = []
    for line in text.replace("\r\n", "\n").split("\n"):
        event = _parse_sse_event(line)
        if event is not None:
            events.append(event)
    return events


def _request_uses_streaming(*, stream: bool, request_json: Any) -> bool:
    if stream:
        return True
    if isinstance(request_json, Mapping):
        return request_json.get(RESPONSE_MODE_KEY) == STREAMING_RESPONSE_MODE
    return getattr(request_json, RESPONSE_MODE_KEY, None) == STREAMING_RESPONSE_MODE


class _InstrumentedStreamingResponse:
    def __init__(self, *, response: Any, call_context: Any) -> None:
        self._response = response
        self._call_context = call_context
        self._events: list[Any] = []
        self._chunks: list[Any] = []
        self._emitted = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    def __enter__(self) -> Self:
        enter = getattr(self._response, "__enter__", None)
        if callable(enter):
            enter()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> Any:
        if exc is not None:
            self._emit_once(
                error=exc if isinstance(exc, Exception) else Exception(str(exc))
            )
        else:
            self._emit_once(error=None)
        exit_method = getattr(self._response, "__exit__", None)
        if callable(exit_method):
            return exit_method(exc_type, exc, tb)
        return None

    def close(self) -> Any:
        self._emit_once(error=None)
        return self._response.close()

    def __iter__(self) -> Any:
        return self._iter_chunks("__iter__")

    def iter_lines(self, *args: Any, **kwargs: Any) -> Any:
        try:
            for line in self._response.iter_lines(*args, **kwargs):
                event = _parse_sse_event(line)
                if event is not None:
                    self._events.append(event)
                yield line
        except Exception as exc:
            self._emit_once(error=exc)
            raise
        finally:
            self._emit_once(error=None)

    def iter_content(self, *args: Any, **kwargs: Any) -> Any:
        return self._iter_chunks("iter_content", *args, **kwargs)

    def iter_bytes(self, *args: Any, **kwargs: Any) -> Any:
        return self._iter_chunks("iter_bytes", *args, **kwargs)

    def iter_text(self, *args: Any, **kwargs: Any) -> Any:
        return self._iter_chunks("iter_text", *args, **kwargs)

    def iter_raw(self, *args: Any, **kwargs: Any) -> Any:
        return self._iter_chunks("iter_raw", *args, **kwargs)

    def _iter_chunks(self, method_name: str, *args: Any, **kwargs: Any) -> Any:
        iterator = (
            iter(self._response)
            if method_name == "__iter__"
            else getattr(self._response, method_name)(*args, **kwargs)
        )
        try:
            for chunk in iterator:
                self._chunks.append(chunk)
                yield chunk
        except Exception as exc:
            self._emit_once(error=exc)
            raise
        finally:
            self._emit_once(error=None)

    def _emit_once(self, *, error: Exception | None) -> None:
        if self._emitted:
            return
        self._emitted = True
        emit_dify_span(
            call_context=self._call_context,
            response=self._response,
            stream_events=self._events or _parse_sse_chunks(self._chunks),
            error=error,
            include_content=_INCLUDE_CONTENT,
        )


class _InstrumentedAsyncStreamingResponse:
    def __init__(self, *, response: Any, call_context: Any) -> None:
        self._response = response
        self._call_context = call_context
        self._events: list[Any] = []
        self._chunks: list[Any] = []
        self._emitted = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    async def __aenter__(self) -> Self:
        enter = getattr(self._response, "__aenter__", None)
        if callable(enter):
            await enter()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> Any:
        if exc is not None:
            self._emit_once(
                error=exc if isinstance(exc, Exception) else Exception(str(exc))
            )
        else:
            self._emit_once(error=None)
        exit_method = getattr(self._response, "__aexit__", None)
        if callable(exit_method):
            return await exit_method(exc_type, exc, tb)
        return None

    async def aclose(self) -> Any:
        self._emit_once(error=None)
        close = getattr(self._response, "aclose", None)
        if callable(close):
            return await close()
        return None

    def aiter_lines(self, *args: Any, **kwargs: Any) -> Any:
        return self._aiter_lines(*args, **kwargs)

    async def _aiter_lines(self, *args: Any, **kwargs: Any) -> Any:
        try:
            async for line in self._response.aiter_lines(*args, **kwargs):
                event = _parse_sse_event(line)
                if event is not None:
                    self._events.append(event)
                yield line
        except Exception as exc:
            self._emit_once(error=exc)
            raise
        finally:
            self._emit_once(error=None)

    def aiter_bytes(self, *args: Any, **kwargs: Any) -> Any:
        return self._aiter_chunks("aiter_bytes", *args, **kwargs)

    def aiter_text(self, *args: Any, **kwargs: Any) -> Any:
        return self._aiter_chunks("aiter_text", *args, **kwargs)

    def aiter_raw(self, *args: Any, **kwargs: Any) -> Any:
        return self._aiter_chunks("aiter_raw", *args, **kwargs)

    async def _aiter_chunks(self, method_name: str, *args: Any, **kwargs: Any) -> Any:
        try:
            async for chunk in getattr(self._response, method_name)(*args, **kwargs):
                self._chunks.append(chunk)
                yield chunk
        except Exception as exc:
            self._emit_once(error=exc)
            raise
        finally:
            self._emit_once(error=None)

    def _emit_once(self, *, error: Exception | None) -> None:
        if self._emitted:
            return
        self._emitted = True
        emit_dify_span(
            call_context=self._call_context,
            response=self._response,
            stream_events=self._events or _parse_sse_chunks(self._chunks),
            error=error,
            include_content=_INCLUDE_CONTENT,
        )


def _wrap_send_request(original: Any) -> Any:
    @wraps(original)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        respan_params = _pop_respan_params(kwargs=kwargs)
        with use_respan_params(respan_params):
            method = _arg(args, 0, kwargs, "method", "")
            endpoint = _arg(args, 1, kwargs, "endpoint", "")
            request_json = _arg(args, 2, kwargs, "json")
            request_params = _arg(args, 3, kwargs, "params")
            stream = _request_uses_streaming(
                stream=bool(_arg(args, 4, kwargs, "stream", False)),
                request_json=request_json,
            )
            call_context = capture_call_context(
                method=str(method),
                endpoint=str(endpoint),
                request_json=request_json,
                request_params=request_params,
                stream=stream,
            )
            try:
                response = original(self, *args, **kwargs)
            except Exception as exc:
                emit_dify_span(
                    call_context=call_context,
                    error=exc,
                    include_content=_INCLUDE_CONTENT,
                )
                raise

            if stream:
                return _InstrumentedStreamingResponse(
                    response=response,
                    call_context=call_context,
                )
            emit_dify_span(
                call_context=call_context,
                response=response,
                include_content=_INCLUDE_CONTENT,
            )
            return response

    return wrapper


def _wrap_send_request_with_files(original: Any) -> Any:
    @wraps(original)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        respan_params = _pop_respan_params(kwargs=kwargs)
        with use_respan_params(respan_params):
            method = _arg(args, 0, kwargs, "method", "")
            endpoint = _arg(args, 1, kwargs, "endpoint", "")
            request_data = _arg(args, 2, kwargs, "data")
            files = _arg(args, 3, kwargs, "files")
            call_context = capture_call_context(
                method=str(method),
                endpoint=str(endpoint),
                request_data=request_data,
                files=files,
            )
            try:
                response = original(self, *args, **kwargs)
            except Exception as exc:
                emit_dify_span(
                    call_context=call_context,
                    error=exc,
                    include_content=_INCLUDE_CONTENT,
                )
                raise
            emit_dify_span(
                call_context=call_context,
                response=response,
                include_content=_INCLUDE_CONTENT,
            )
            return response

    return wrapper


def _wrap_high_level_method(original: Any) -> Any:
    @wraps(original)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        respan_params = _pop_respan_params(kwargs=kwargs)
        with use_respan_params(respan_params):
            return original(self, *args, **kwargs)

    return wrapper


def _wrap_async_send_request(original: Any) -> Any:
    @wraps(original)
    async def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        respan_params = _pop_respan_params(kwargs=kwargs)
        with use_respan_params(respan_params):
            method = _arg(args, 0, kwargs, "method", "")
            endpoint = _arg(args, 1, kwargs, "endpoint", "")
            request_json = _arg(args, 2, kwargs, "json")
            request_params = _arg(args, 3, kwargs, "params")
            stream = _request_uses_streaming(
                stream=bool(_arg(args, 4, kwargs, "stream", False)),
                request_json=request_json,
            )
            call_context = capture_call_context(
                method=str(method),
                endpoint=str(endpoint),
                request_json=request_json,
                request_params=request_params,
                stream=stream,
            )
            try:
                response = await original(self, *args, **kwargs)
            except Exception as exc:
                emit_dify_span(
                    call_context=call_context,
                    error=exc,
                    include_content=_INCLUDE_CONTENT,
                )
                raise

            if stream:
                return _InstrumentedAsyncStreamingResponse(
                    response=response,
                    call_context=call_context,
                )
            emit_dify_span(
                call_context=call_context,
                response=response,
                include_content=_INCLUDE_CONTENT,
            )
            return response

    return wrapper


def _wrap_async_send_request_with_files(original: Any) -> Any:
    @wraps(original)
    async def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        respan_params = _pop_respan_params(kwargs=kwargs)
        with use_respan_params(respan_params):
            method = _arg(args, 0, kwargs, "method", "")
            endpoint = _arg(args, 1, kwargs, "endpoint", "")
            request_data = _arg(args, 2, kwargs, "data")
            files = _arg(args, 3, kwargs, "files")
            call_context = capture_call_context(
                method=str(method),
                endpoint=str(endpoint),
                request_data=request_data,
                files=files,
            )
            try:
                response = await original(self, *args, **kwargs)
            except Exception as exc:
                emit_dify_span(
                    call_context=call_context,
                    error=exc,
                    include_content=_INCLUDE_CONTENT,
                )
                raise
            emit_dify_span(
                call_context=call_context,
                response=response,
                include_content=_INCLUDE_CONTENT,
            )
            return response

    return wrapper


def _wrap_async_high_level_method(original: Any) -> Any:
    @wraps(original)
    async def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        respan_params = _pop_respan_params(kwargs=kwargs)
        with use_respan_params(respan_params):
            return await original(self, *args, **kwargs)

    return wrapper


def _patch_method(cls: type[Any], method_name: str, wrapper_factory: Any) -> None:
    key = (cls, method_name)
    if key in _PATCHED_METHODS:
        return
    original = getattr(cls, method_name, None)
    if original is None:
        return
    replacement = wrapper_factory(original)
    _PATCHED_METHODS[key] = (original, replacement)
    setattr(cls, method_name, replacement)


class DifyInstrumentor:
    """Respan instrumentor for the Dify Python Service API client."""

    name = DIFY_INSTRUMENTATION_NAME

    def __init__(self, *, include_content: bool = True) -> None:
        self._include_content = include_content
        self._is_instrumented = False

    @staticmethod
    def _is_respan_tracing_enabled() -> bool:
        tracer = getattr(RespanTracer, "_instance", None)
        if tracer is None:
            return True
        return bool(getattr(tracer, "is_enabled", True))

    def activate(self) -> None:
        """Monkey-patch the Dify client."""
        global _ACTIVE_INSTANCES, _IS_PATCHED, _INCLUDE_CONTENT
        if self._is_instrumented:
            return
        if not self._is_respan_tracing_enabled():
            logger.info(
                "Dify instrumentation skipped because Respan tracing is disabled"
            )
            return

        try:
            module = importlib.import_module(DIFY_CLIENT_MODULE)
            dify_client_cls = module.DifyClient
            chat_client_cls = module.ChatClient
            completion_client_cls = module.CompletionClient
        except (AttributeError, ImportError) as exc:
            logger.warning(
                "Failed to activate Dify instrumentation - missing dependency: %s",
                exc,
            )
            return

        try:
            async_module = importlib.import_module(DIFY_ASYNC_CLIENT_MODULE)
        except ImportError:
            # PyPI 0.1.10 is sync-only. Async clients exist in the refreshed
            # 0.1.12 source line and are patched when available.
            async_module = None

        if _ACTIVE_INSTANCES == 0:
            _INCLUDE_CONTENT = self._include_content
        elif _INCLUDE_CONTENT != self._include_content:
            logger.warning(
                "Dify instrumentation is already active with "
                "include_content=%s; keeping the first active configuration",
                _INCLUDE_CONTENT,
            )

        if not _IS_PATCHED:
            _patch_method(dify_client_cls, "_send_request", _wrap_send_request)
            _patch_method(
                dify_client_cls,
                "_send_request_with_files",
                _wrap_send_request_with_files,
            )
            for method_name in (
                "message_feedback",
                "get_application_parameters",
                "file_upload",
            ):
                _patch_method(dify_client_cls, method_name, _wrap_high_level_method)
            for method_name in (
                "create_chat_message",
                "get_conversation_messages",
                "get_conversations",
                "rename_conversation",
            ):
                _patch_method(chat_client_cls, method_name, _wrap_high_level_method)
            _patch_method(
                completion_client_cls,
                "create_completion_message",
                _wrap_high_level_method,
            )

            for class_name, method_names in (
                ("WorkflowClient", ("run", "run_specific_workflow")),
                (
                    "KnowledgeBaseClient",
                    ("run_datasource_node", "run_rag_pipeline"),
                ),
            ):
                cls = getattr(module, class_name, None)
                if cls is None:
                    continue
                for method_name in method_names:
                    _patch_method(cls, method_name, _wrap_high_level_method)

            if async_module is not None:
                async_dify_client_cls = getattr(async_module, "AsyncDifyClient", None)
                if async_dify_client_cls is not None:
                    _patch_method(
                        async_dify_client_cls,
                        "_send_request",
                        _wrap_async_send_request,
                    )
                    _patch_method(
                        async_dify_client_cls,
                        "_send_request_with_files",
                        _wrap_async_send_request_with_files,
                    )
                    for method_name in (
                        "message_feedback",
                        "get_application_parameters",
                        "file_upload",
                    ):
                        _patch_method(
                            async_dify_client_cls,
                            method_name,
                            _wrap_async_high_level_method,
                        )

                for class_name, method_names in (
                    (
                        "AsyncChatClient",
                        (
                            "create_chat_message",
                            "get_conversation_messages",
                            "get_conversations",
                            "rename_conversation",
                        ),
                    ),
                    (
                        "AsyncCompletionClient",
                        ("create_completion_message",),
                    ),
                    (
                        "AsyncWorkflowClient",
                        ("run", "run_specific_workflow"),
                    ),
                    (
                        "AsyncKnowledgeBaseClient",
                        ("run_datasource_node", "run_rag_pipeline"),
                    ),
                ):
                    cls = getattr(async_module, class_name, None)
                    if cls is None:
                        continue
                    for method_name in method_names:
                        _patch_method(
                            cls,
                            method_name,
                            _wrap_async_high_level_method,
                        )
            _IS_PATCHED = True

        _ACTIVE_INSTANCES += 1
        self._is_instrumented = True
        logger.info("Dify instrumentation activated")

    def deactivate(self) -> None:
        """Restore patched Dify client methods."""
        global _ACTIVE_INSTANCES, _IS_PATCHED
        if not self._is_instrumented:
            return
        _ACTIVE_INSTANCES = max(0, _ACTIVE_INSTANCES - 1)
        self._is_instrumented = False
        if _ACTIVE_INSTANCES > 0:
            return
        for (cls, method_name), (original, replacement) in list(
            _PATCHED_METHODS.items()
        ):
            if getattr(cls, method_name, None) is replacement:
                setattr(cls, method_name, original)
            else:
                logger.warning(
                    "Dify method %s.%s changed after activation; "
                    "leaving the later patch installed",
                    cls.__name__,
                    method_name,
                )
        _PATCHED_METHODS.clear()
        _IS_PATCHED = False
        logger.info("Dify instrumentation deactivated")


DifyAIInstrumentor = DifyInstrumentor
