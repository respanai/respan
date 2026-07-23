"""Dify SDK instrumentation plugin for Respan."""

from __future__ import annotations

import importlib
import json
import logging
from typing import Any

from respan_instrumentation_dify._constants import (
    DIFY_CLIENT_MODULE,
    DIFY_INSTRUMENTATION_NAME,
)
from respan_instrumentation_dify._context import use_respan_params
from respan_instrumentation_dify._otel_emitter import capture_call_context
from respan_instrumentation_dify._otel_emitter import emit_dify_span
from respan_tracing.core.tracer import RespanTracer

logger = logging.getLogger(__name__)

_PATCHED_METHODS: dict[tuple[type[Any], str], Any] = {}
_IS_PATCHED = False
_INCLUDE_CONTENT = True


def _arg(args: tuple[Any, ...], index: int, kwargs: dict[str, Any], name: str, default: Any = None) -> Any:
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
    except Exception:
        return text


class _InstrumentedStreamingResponse:
    def __init__(self, *, response: Any, call_context: Any) -> None:
        self._response = response
        self._call_context = call_context
        self._events: list[Any] = []
        self._emitted = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    def __enter__(self) -> "_InstrumentedStreamingResponse":
        enter = getattr(self._response, "__enter__", None)
        if callable(enter):
            enter()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> Any:
        if exc is not None:
            self._emit_once(error=exc if isinstance(exc, Exception) else Exception(str(exc)))
        else:
            self._emit_once(error=None)
        exit_method = getattr(self._response, "__exit__", None)
        if callable(exit_method):
            return exit_method(exc_type, exc, tb)
        return None

    def close(self) -> Any:
        self._emit_once(error=None)
        return self._response.close()

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
        else:
            self._emit_once(error=None)

    def iter_content(self, *args: Any, **kwargs: Any) -> Any:
        chunks: list[Any] = []
        try:
            for chunk in self._response.iter_content(*args, **kwargs):
                chunks.append(chunk)
                yield chunk
        except Exception as exc:
            self._emit_once(error=exc)
            raise
        else:
            if not self._events:
                self._events.extend(_parse_sse_event(chunk) for chunk in chunks)
                self._events = [event for event in self._events if event is not None]
            self._emit_once(error=None)

    def _emit_once(self, *, error: Exception | None) -> None:
        if self._emitted:
            return
        self._emitted = True
        emit_dify_span(
            call_context=self._call_context,
            response=self._response,
            stream_events=self._events,
            error=error,
            include_content=_INCLUDE_CONTENT,
        )


def _wrap_send_request(original: Any) -> Any:
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        respan_params = _pop_respan_params(kwargs=kwargs)
        with use_respan_params(respan_params):
            method = _arg(args, 0, kwargs, "method", "")
            endpoint = _arg(args, 1, kwargs, "endpoint", "")
            request_json = _arg(args, 2, kwargs, "json")
            request_params = _arg(args, 3, kwargs, "params")
            stream = bool(_arg(args, 4, kwargs, "stream", False))
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
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        respan_params = _pop_respan_params(kwargs=kwargs)
        with use_respan_params(respan_params):
            return original(self, *args, **kwargs)

    return wrapper


def _patch_method(cls: type[Any], method_name: str, wrapper_factory: Any) -> None:
    key = (cls, method_name)
    if key in _PATCHED_METHODS:
        return
    original = getattr(cls, method_name, None)
    if original is None:
        return
    _PATCHED_METHODS[key] = original
    setattr(cls, method_name, wrapper_factory(original))


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
        global _IS_PATCHED, _INCLUDE_CONTENT
        if self._is_instrumented:
            return
        if not self._is_respan_tracing_enabled():
            logger.info("Dify instrumentation skipped because Respan tracing is disabled")
            return

        try:
            module = importlib.import_module(DIFY_CLIENT_MODULE)
            dify_client_cls = getattr(module, "DifyClient")
            chat_client_cls = getattr(module, "ChatClient")
            completion_client_cls = getattr(module, "CompletionClient")
        except (AttributeError, ImportError) as exc:
            logger.warning(
                "Failed to activate Dify instrumentation - missing dependency: %s",
                exc,
            )
            return

        _INCLUDE_CONTENT = self._include_content

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
            _IS_PATCHED = True

        self._is_instrumented = True
        logger.info("Dify instrumentation activated")

    def deactivate(self) -> None:
        """Restore patched Dify client methods."""
        global _IS_PATCHED
        for (cls, method_name), original in list(_PATCHED_METHODS.items()):
            setattr(cls, method_name, original)
        _PATCHED_METHODS.clear()
        _IS_PATCHED = False
        self._is_instrumented = False
        logger.info("Dify instrumentation deactivated")


DifyAIInstrumentor = DifyInstrumentor
