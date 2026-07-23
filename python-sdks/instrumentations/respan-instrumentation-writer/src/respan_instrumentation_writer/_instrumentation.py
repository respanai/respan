"""Writer SDK instrumentation plugin for Respan."""

from __future__ import annotations

import importlib
import logging
import time
from typing import Any, Callable

from respan_instrumentation_writer._constants import (
    ANALYZE_METHOD_NAME,
    ASYNC_APPLICATIONS_CLASS_NAME,
    ASYNC_CHAT_CLASS_NAME,
    ASYNC_COMPLETIONS_CLASS_NAME,
    ASYNC_GRAPHS_CLASS_NAME,
    ASYNC_TOOLS_CLASS_NAME,
    ASYNC_TRANSLATION_CLASS_NAME,
    ASYNC_VISION_CLASS_NAME,
    CHAT_METHOD_NAME,
    CREATE_METHOD_NAME,
    FILE_ID_KEY,
    GENERATE_CONTENT_METHOD_NAME,
    APPLICATION_ID_KEY,
    PARSE_PDF_METHOD_NAME,
    QUESTION_METHOD_NAME,
    STREAM_KEY,
    SYNC_APPLICATIONS_CLASS_NAME,
    SYNC_CHAT_CLASS_NAME,
    SYNC_COMPLETIONS_CLASS_NAME,
    SYNC_GRAPHS_CLASS_NAME,
    SYNC_TOOLS_CLASS_NAME,
    SYNC_TRANSLATION_CLASS_NAME,
    SYNC_VISION_CLASS_NAME,
    TRANSLATE_METHOD_NAME,
    WEB_SEARCH_METHOD_NAME,
    WRITER_APPLICATIONS_MODULE,
    WRITER_APPLICATION_GENERATE_SPAN_NAME,
    WRITER_CHAT_SPAN_NAME,
    WRITER_CHAT_MODULE,
    WRITER_COMPLETION_SPAN_NAME,
    WRITER_COMPLETIONS_MODULE,
    WRITER_GRAPH_QUESTION_SPAN_NAME,
    WRITER_GRAPHS_MODULE,
    WRITER_INSTRUMENTATION_NAME,
    WRITER_PARSE_PDF_TOOL_NAME,
    WRITER_TOOLS_MODULE,
    WRITER_TRANSLATION_MODULE,
    WRITER_TRANSLATION_SPAN_NAME,
    WRITER_VISION_MODULE,
    WRITER_VISION_SPAN_NAME,
    WRITER_WEB_SEARCH_TOOL_NAME,
)
from respan_instrumentation_writer._otel_emitter import emit_writer_span
from respan_instrumentation_writer._translator import (
    build_application_generate_attrs,
    build_chat_attrs,
    build_completion_attrs,
    build_graph_question_attrs,
    build_tool_attrs,
    build_translation_attrs,
    build_vision_attrs,
    request_kwargs_with_positionals,
)
from respan_tracing.core.tracer import RespanTracer

logger = logging.getLogger(__name__)

_original_sync_chat = None
_original_async_chat = None
_original_sync_completion_create = None
_original_async_completion_create = None
_original_sync_graph_question = None
_original_async_graph_question = None
_original_sync_application_generate = None
_original_async_application_generate = None
_original_sync_vision_analyze = None
_original_async_vision_analyze = None
_original_sync_translation_translate = None
_original_async_translation_translate = None
_original_sync_tool_web_search = None
_original_async_tool_web_search = None
_original_sync_tool_parse_pdf = None
_original_async_tool_parse_pdf = None

_ITERABLE_REQUEST_FIELDS = {
    "graph_ids",
    "inputs",
    "messages",
    "tools",
    "variables",
}


def _get_module_attr(module_path: str, attr_name: str) -> Any:
    module = importlib.import_module(module_path)
    attr_value = getattr(module, attr_name, None)
    if attr_value is None:
        raise AttributeError(f"{module_path}.{attr_name}")
    return attr_value


def _load_resource_classes() -> dict[str, type[Any]]:
    return {
        "sync_chat": _get_module_attr(WRITER_CHAT_MODULE, SYNC_CHAT_CLASS_NAME),
        "async_chat": _get_module_attr(WRITER_CHAT_MODULE, ASYNC_CHAT_CLASS_NAME),
        "sync_completions": _get_module_attr(
            WRITER_COMPLETIONS_MODULE,
            SYNC_COMPLETIONS_CLASS_NAME,
        ),
        "async_completions": _get_module_attr(
            WRITER_COMPLETIONS_MODULE,
            ASYNC_COMPLETIONS_CLASS_NAME,
        ),
        "sync_graphs": _get_module_attr(WRITER_GRAPHS_MODULE, SYNC_GRAPHS_CLASS_NAME),
        "async_graphs": _get_module_attr(WRITER_GRAPHS_MODULE, ASYNC_GRAPHS_CLASS_NAME),
        "sync_applications": _get_module_attr(
            WRITER_APPLICATIONS_MODULE,
            SYNC_APPLICATIONS_CLASS_NAME,
        ),
        "async_applications": _get_module_attr(
            WRITER_APPLICATIONS_MODULE,
            ASYNC_APPLICATIONS_CLASS_NAME,
        ),
        "sync_vision": _get_module_attr(WRITER_VISION_MODULE, SYNC_VISION_CLASS_NAME),
        "async_vision": _get_module_attr(WRITER_VISION_MODULE, ASYNC_VISION_CLASS_NAME),
        "sync_translation": _get_module_attr(
            WRITER_TRANSLATION_MODULE,
            SYNC_TRANSLATION_CLASS_NAME,
        ),
        "async_translation": _get_module_attr(
            WRITER_TRANSLATION_MODULE,
            ASYNC_TRANSLATION_CLASS_NAME,
        ),
        "sync_tools": _get_module_attr(WRITER_TOOLS_MODULE, SYNC_TOOLS_CLASS_NAME),
        "async_tools": _get_module_attr(WRITER_TOOLS_MODULE, ASYNC_TOOLS_CLASS_NAME),
    }


def _is_omitted(value: Any) -> bool:
    return type(value).__name__ in {"Omit", "NotGiven"}


def _materialize_iterable(value: Any) -> Any:
    if value is None or _is_omitted(value):
        return value
    if isinstance(value, (str, bytes, dict, list, tuple)):
        return value
    try:
        return list(value)
    except TypeError:
        return value


def _snapshot_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    snapshot = dict(kwargs)
    for field_name in _ITERABLE_REQUEST_FIELDS:
        if field_name in snapshot:
            snapshot[field_name] = _materialize_iterable(snapshot[field_name])
    return snapshot


def _is_stream_request(request_kwargs: dict[str, Any]) -> bool:
    return request_kwargs.get(STREAM_KEY) is True


def _emit_safely(
    *,
    span_name: str,
    attrs_builder: Callable[..., dict[str, Any]],
    request_kwargs: dict[str, Any],
    start_ns: int,
    response_or_chunks: Any = None,
    error_message: str | None = None,
    status_code: int = 200,
) -> None:
    try:
        attrs = attrs_builder(
            request_kwargs=request_kwargs,
            response_or_chunks=response_or_chunks,
        )
        emit_writer_span(
            name=span_name,
            attrs=attrs,
            start_ns=start_ns,
            error_message=error_message,
            status_code=status_code,
        )
    except Exception:
        logger.debug("Failed to build Writer span attrs", exc_info=True)


def _emit_unary_safely(
    *,
    span_name: str,
    attrs_builder: Callable[..., dict[str, Any]],
    request_kwargs: dict[str, Any],
    start_ns: int,
    response: Any = None,
    error_message: str | None = None,
    status_code: int = 200,
) -> None:
    try:
        attrs = attrs_builder(request_kwargs=request_kwargs, response=response)
        emit_writer_span(
            name=span_name,
            attrs=attrs,
            start_ns=start_ns,
            error_message=error_message,
            status_code=status_code,
        )
    except Exception:
        logger.debug("Failed to build Writer span attrs", exc_info=True)


class _SyncStreamCapture:
    def __init__(
        self,
        *,
        stream: Any,
        span_name: str,
        attrs_builder: Callable[..., dict[str, Any]],
        request_kwargs: dict[str, Any],
        start_ns: int,
    ) -> None:
        self._stream = stream
        self._iterator = iter(stream)
        self._span_name = span_name
        self._attrs_builder = attrs_builder
        self._request_kwargs = request_kwargs
        self._start_ns = start_ns
        self._chunks: list[Any] = []
        self._emitted = False
        self.response = getattr(stream, "response", None)

    def __iter__(self) -> "_SyncStreamCapture":
        return self

    def __next__(self) -> Any:
        try:
            chunk = next(self._iterator)
        except StopIteration:
            self._emit_success()
            raise
        except Exception as exc:
            self._emit_error(exc)
            raise
        self._chunks.append(chunk)
        return chunk

    def __enter__(self) -> "_SyncStreamCapture":
        enter = getattr(self._stream, "__enter__", None)
        if callable(enter):
            entered = enter()
            if entered is not self._stream:
                self._stream = entered
                self._iterator = iter(entered)
                self.response = getattr(entered, "response", self.response)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> Any:
        exit_method = getattr(self._stream, "__exit__", None)
        result = None
        if callable(exit_method):
            result = exit_method(exc_type, exc_val, exc_tb)
        if exc_val is not None:
            self._emit_error(exc_val)
        elif not self._emitted:
            self._emit_success()
        return result

    def close(self) -> None:
        close = getattr(self._stream, "close", None)
        if callable(close):
            close()
        if not self._emitted:
            self._emit_success()

    def _emit_success(self) -> None:
        if self._emitted:
            return
        self._emitted = True
        _emit_safely(
            span_name=self._span_name,
            attrs_builder=self._attrs_builder,
            request_kwargs=self._request_kwargs,
            start_ns=self._start_ns,
            response_or_chunks=self._chunks,
        )

    def _emit_error(self, exc: BaseException) -> None:
        if self._emitted:
            return
        self._emitted = True
        _emit_safely(
            span_name=self._span_name,
            attrs_builder=self._attrs_builder,
            request_kwargs=self._request_kwargs,
            start_ns=self._start_ns,
            response_or_chunks=self._chunks,
            error_message=str(exc),
            status_code=500,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


class _AsyncStreamCapture:
    def __init__(
        self,
        *,
        stream: Any,
        span_name: str,
        attrs_builder: Callable[..., dict[str, Any]],
        request_kwargs: dict[str, Any],
        start_ns: int,
    ) -> None:
        self._stream = stream
        self._iterator = None
        self._span_name = span_name
        self._attrs_builder = attrs_builder
        self._request_kwargs = request_kwargs
        self._start_ns = start_ns
        self._chunks: list[Any] = []
        self._emitted = False
        self.response = getattr(stream, "response", None)

    def __aiter__(self) -> "_AsyncStreamCapture":
        self._iterator = self._stream.__aiter__()
        return self

    async def __anext__(self) -> Any:
        if self._iterator is None:
            self._iterator = self._stream.__aiter__()
        try:
            chunk = await self._iterator.__anext__()
        except StopAsyncIteration:
            self._emit_success()
            raise
        except Exception as exc:
            self._emit_error(exc)
            raise
        self._chunks.append(chunk)
        return chunk

    async def __aenter__(self) -> "_AsyncStreamCapture":
        enter = getattr(self._stream, "__aenter__", None)
        if callable(enter):
            entered = await enter()
            if entered is not self._stream:
                self._stream = entered
                self._iterator = entered.__aiter__()
                self.response = getattr(entered, "response", self.response)
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> Any:
        exit_method = getattr(self._stream, "__aexit__", None)
        result = None
        if callable(exit_method):
            result = await exit_method(exc_type, exc_val, exc_tb)
        if exc_val is not None:
            self._emit_error(exc_val)
        elif not self._emitted:
            self._emit_success()
        return result

    async def close(self) -> None:
        close = getattr(self._stream, "close", None)
        if callable(close):
            result = close()
            if hasattr(result, "__await__"):
                await result
        if not self._emitted:
            self._emit_success()

    def _emit_success(self) -> None:
        if self._emitted:
            return
        self._emitted = True
        _emit_safely(
            span_name=self._span_name,
            attrs_builder=self._attrs_builder,
            request_kwargs=self._request_kwargs,
            start_ns=self._start_ns,
            response_or_chunks=self._chunks,
        )

    def _emit_error(self, exc: BaseException) -> None:
        if self._emitted:
            return
        self._emitted = True
        _emit_safely(
            span_name=self._span_name,
            attrs_builder=self._attrs_builder,
            request_kwargs=self._request_kwargs,
            start_ns=self._start_ns,
            response_or_chunks=self._chunks,
            error_message=str(exc),
            status_code=500,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


def _wrap_sync_streamable(
    *,
    original: Any,
    span_name: str,
    attrs_builder: Callable[..., dict[str, Any]],
    positional_names: tuple[str, ...] = (),
) -> Any:
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        kwargs = _snapshot_kwargs(kwargs)
        request_kwargs = request_kwargs_with_positionals(
            kwargs=kwargs,
            positional_values=args,
            positional_names=positional_names,
        )
        start_ns = time.time_ns()
        try:
            response = original(self, *args, **kwargs)
        except Exception as exc:
            _emit_safely(
                span_name=span_name,
                attrs_builder=attrs_builder,
                request_kwargs=request_kwargs,
                start_ns=start_ns,
                error_message=str(exc),
                status_code=500,
            )
            raise

        if _is_stream_request(request_kwargs):
            return _SyncStreamCapture(
                stream=response,
                span_name=span_name,
                attrs_builder=attrs_builder,
                request_kwargs=request_kwargs,
                start_ns=start_ns,
            )

        _emit_safely(
            span_name=span_name,
            attrs_builder=attrs_builder,
            request_kwargs=request_kwargs,
            start_ns=start_ns,
            response_or_chunks=response,
        )
        return response

    return wrapper


def _wrap_async_streamable(
    *,
    original: Any,
    span_name: str,
    attrs_builder: Callable[..., dict[str, Any]],
    positional_names: tuple[str, ...] = (),
) -> Any:
    async def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        kwargs = _snapshot_kwargs(kwargs)
        request_kwargs = request_kwargs_with_positionals(
            kwargs=kwargs,
            positional_values=args,
            positional_names=positional_names,
        )
        start_ns = time.time_ns()
        try:
            response = await original(self, *args, **kwargs)
        except Exception as exc:
            _emit_safely(
                span_name=span_name,
                attrs_builder=attrs_builder,
                request_kwargs=request_kwargs,
                start_ns=start_ns,
                error_message=str(exc),
                status_code=500,
            )
            raise

        if _is_stream_request(request_kwargs):
            return _AsyncStreamCapture(
                stream=response,
                span_name=span_name,
                attrs_builder=attrs_builder,
                request_kwargs=request_kwargs,
                start_ns=start_ns,
            )

        _emit_safely(
            span_name=span_name,
            attrs_builder=attrs_builder,
            request_kwargs=request_kwargs,
            start_ns=start_ns,
            response_or_chunks=response,
        )
        return response

    return wrapper


def _wrap_sync_unary(
    *,
    original: Any,
    span_name: str,
    attrs_builder: Callable[..., dict[str, Any]],
    positional_names: tuple[str, ...] = (),
) -> Any:
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        kwargs = _snapshot_kwargs(kwargs)
        request_kwargs = request_kwargs_with_positionals(
            kwargs=kwargs,
            positional_values=args,
            positional_names=positional_names,
        )
        start_ns = time.time_ns()
        try:
            response = original(self, *args, **kwargs)
        except Exception as exc:
            _emit_unary_safely(
                span_name=span_name,
                attrs_builder=attrs_builder,
                request_kwargs=request_kwargs,
                start_ns=start_ns,
                error_message=str(exc),
                status_code=500,
            )
            raise

        _emit_unary_safely(
            span_name=span_name,
            attrs_builder=attrs_builder,
            request_kwargs=request_kwargs,
            start_ns=start_ns,
            response=response,
        )
        return response

    return wrapper


def _wrap_async_unary(
    *,
    original: Any,
    span_name: str,
    attrs_builder: Callable[..., dict[str, Any]],
    positional_names: tuple[str, ...] = (),
) -> Any:
    async def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        kwargs = _snapshot_kwargs(kwargs)
        request_kwargs = request_kwargs_with_positionals(
            kwargs=kwargs,
            positional_values=args,
            positional_names=positional_names,
        )
        start_ns = time.time_ns()
        try:
            response = await original(self, *args, **kwargs)
        except Exception as exc:
            _emit_unary_safely(
                span_name=span_name,
                attrs_builder=attrs_builder,
                request_kwargs=request_kwargs,
                start_ns=start_ns,
                error_message=str(exc),
                status_code=500,
            )
            raise

        _emit_unary_safely(
            span_name=span_name,
            attrs_builder=attrs_builder,
            request_kwargs=request_kwargs,
            start_ns=start_ns,
            response=response,
        )
        return response

    return wrapper


def _make_tool_attrs_builder(tool_name: str) -> Callable[..., dict[str, Any]]:
    def builder(*, request_kwargs: dict[str, Any], response: Any = None) -> dict[str, Any]:
        return build_tool_attrs(
            tool_name=tool_name,
            request_kwargs=request_kwargs,
            response=response,
        )

    return builder


class WriterInstrumentor:
    """Respan instrumentor for the Writer Python SDK."""

    name = WRITER_INSTRUMENTATION_NAME

    def __init__(self) -> None:
        self._is_instrumented = False

    @staticmethod
    def _is_respan_tracing_enabled() -> bool:
        tracer = getattr(RespanTracer, "_instance", None)
        if tracer is None:
            return True
        return bool(getattr(tracer, "is_enabled", True))

    def activate(self) -> None:
        """Monkey-patch the Writer SDK."""
        global _original_sync_chat, _original_async_chat
        global _original_sync_completion_create, _original_async_completion_create
        global _original_sync_graph_question, _original_async_graph_question
        global _original_sync_application_generate, _original_async_application_generate
        global _original_sync_vision_analyze, _original_async_vision_analyze
        global _original_sync_translation_translate, _original_async_translation_translate
        global _original_sync_tool_web_search, _original_async_tool_web_search
        global _original_sync_tool_parse_pdf, _original_async_tool_parse_pdf

        if self._is_instrumented:
            return

        if not self._is_respan_tracing_enabled():
            logger.info(
                "Writer instrumentation skipped because Respan tracing is disabled"
            )
            return

        try:
            classes = _load_resource_classes()
        except (AttributeError, ImportError) as exc:
            logger.warning(
                "Failed to activate Writer instrumentation - missing dependency: %s",
                exc,
            )
            return

        try:
            sync_chat = classes["sync_chat"]
            async_chat = classes["async_chat"]
            sync_completions = classes["sync_completions"]
            async_completions = classes["async_completions"]
            sync_graphs = classes["sync_graphs"]
            async_graphs = classes["async_graphs"]
            sync_applications = classes["sync_applications"]
            async_applications = classes["async_applications"]
            sync_vision = classes["sync_vision"]
            async_vision = classes["async_vision"]
            sync_translation = classes["sync_translation"]
            async_translation = classes["async_translation"]
            sync_tools = classes["sync_tools"]
            async_tools = classes["async_tools"]

            if _original_sync_chat is None:
                _original_sync_chat = getattr(sync_chat, CHAT_METHOD_NAME)
            setattr(
                sync_chat,
                CHAT_METHOD_NAME,
                _wrap_sync_streamable(
                    original=_original_sync_chat,
                    span_name=WRITER_CHAT_SPAN_NAME,
                    attrs_builder=build_chat_attrs,
                ),
            )

            if _original_async_chat is None:
                _original_async_chat = getattr(async_chat, CHAT_METHOD_NAME)
            setattr(
                async_chat,
                CHAT_METHOD_NAME,
                _wrap_async_streamable(
                    original=_original_async_chat,
                    span_name=WRITER_CHAT_SPAN_NAME,
                    attrs_builder=build_chat_attrs,
                ),
            )

            if _original_sync_completion_create is None:
                _original_sync_completion_create = getattr(
                    sync_completions,
                    CREATE_METHOD_NAME,
                )
            setattr(
                sync_completions,
                CREATE_METHOD_NAME,
                _wrap_sync_streamable(
                    original=_original_sync_completion_create,
                    span_name=WRITER_COMPLETION_SPAN_NAME,
                    attrs_builder=build_completion_attrs,
                ),
            )

            if _original_async_completion_create is None:
                _original_async_completion_create = getattr(
                    async_completions,
                    CREATE_METHOD_NAME,
                )
            setattr(
                async_completions,
                CREATE_METHOD_NAME,
                _wrap_async_streamable(
                    original=_original_async_completion_create,
                    span_name=WRITER_COMPLETION_SPAN_NAME,
                    attrs_builder=build_completion_attrs,
                ),
            )

            if _original_sync_graph_question is None:
                _original_sync_graph_question = getattr(sync_graphs, QUESTION_METHOD_NAME)
            setattr(
                sync_graphs,
                QUESTION_METHOD_NAME,
                _wrap_sync_streamable(
                    original=_original_sync_graph_question,
                    span_name=WRITER_GRAPH_QUESTION_SPAN_NAME,
                    attrs_builder=build_graph_question_attrs,
                ),
            )

            if _original_async_graph_question is None:
                _original_async_graph_question = getattr(
                    async_graphs,
                    QUESTION_METHOD_NAME,
                )
            setattr(
                async_graphs,
                QUESTION_METHOD_NAME,
                _wrap_async_streamable(
                    original=_original_async_graph_question,
                    span_name=WRITER_GRAPH_QUESTION_SPAN_NAME,
                    attrs_builder=build_graph_question_attrs,
                ),
            )

            if _original_sync_application_generate is None:
                _original_sync_application_generate = getattr(
                    sync_applications,
                    GENERATE_CONTENT_METHOD_NAME,
                )
            setattr(
                sync_applications,
                GENERATE_CONTENT_METHOD_NAME,
                _wrap_sync_streamable(
                    original=_original_sync_application_generate,
                    span_name=WRITER_APPLICATION_GENERATE_SPAN_NAME,
                    attrs_builder=build_application_generate_attrs,
                    positional_names=(APPLICATION_ID_KEY,),
                ),
            )

            if _original_async_application_generate is None:
                _original_async_application_generate = getattr(
                    async_applications,
                    GENERATE_CONTENT_METHOD_NAME,
                )
            setattr(
                async_applications,
                GENERATE_CONTENT_METHOD_NAME,
                _wrap_async_streamable(
                    original=_original_async_application_generate,
                    span_name=WRITER_APPLICATION_GENERATE_SPAN_NAME,
                    attrs_builder=build_application_generate_attrs,
                    positional_names=(APPLICATION_ID_KEY,),
                ),
            )

            if _original_sync_vision_analyze is None:
                _original_sync_vision_analyze = getattr(sync_vision, ANALYZE_METHOD_NAME)
            setattr(
                sync_vision,
                ANALYZE_METHOD_NAME,
                _wrap_sync_unary(
                    original=_original_sync_vision_analyze,
                    span_name=WRITER_VISION_SPAN_NAME,
                    attrs_builder=build_vision_attrs,
                ),
            )

            if _original_async_vision_analyze is None:
                _original_async_vision_analyze = getattr(
                    async_vision,
                    ANALYZE_METHOD_NAME,
                )
            setattr(
                async_vision,
                ANALYZE_METHOD_NAME,
                _wrap_async_unary(
                    original=_original_async_vision_analyze,
                    span_name=WRITER_VISION_SPAN_NAME,
                    attrs_builder=build_vision_attrs,
                ),
            )

            if _original_sync_translation_translate is None:
                _original_sync_translation_translate = getattr(
                    sync_translation,
                    TRANSLATE_METHOD_NAME,
                )
            setattr(
                sync_translation,
                TRANSLATE_METHOD_NAME,
                _wrap_sync_unary(
                    original=_original_sync_translation_translate,
                    span_name=WRITER_TRANSLATION_SPAN_NAME,
                    attrs_builder=build_translation_attrs,
                ),
            )

            if _original_async_translation_translate is None:
                _original_async_translation_translate = getattr(
                    async_translation,
                    TRANSLATE_METHOD_NAME,
                )
            setattr(
                async_translation,
                TRANSLATE_METHOD_NAME,
                _wrap_async_unary(
                    original=_original_async_translation_translate,
                    span_name=WRITER_TRANSLATION_SPAN_NAME,
                    attrs_builder=build_translation_attrs,
                ),
            )

            if _original_sync_tool_web_search is None:
                _original_sync_tool_web_search = getattr(sync_tools, WEB_SEARCH_METHOD_NAME)
            setattr(
                sync_tools,
                WEB_SEARCH_METHOD_NAME,
                _wrap_sync_unary(
                    original=_original_sync_tool_web_search,
                    span_name=WRITER_WEB_SEARCH_TOOL_NAME,
                    attrs_builder=_make_tool_attrs_builder(WRITER_WEB_SEARCH_TOOL_NAME),
                ),
            )

            if _original_async_tool_web_search is None:
                _original_async_tool_web_search = getattr(
                    async_tools,
                    WEB_SEARCH_METHOD_NAME,
                )
            setattr(
                async_tools,
                WEB_SEARCH_METHOD_NAME,
                _wrap_async_unary(
                    original=_original_async_tool_web_search,
                    span_name=WRITER_WEB_SEARCH_TOOL_NAME,
                    attrs_builder=_make_tool_attrs_builder(WRITER_WEB_SEARCH_TOOL_NAME),
                ),
            )

            if _original_sync_tool_parse_pdf is None:
                _original_sync_tool_parse_pdf = getattr(sync_tools, PARSE_PDF_METHOD_NAME)
            setattr(
                sync_tools,
                PARSE_PDF_METHOD_NAME,
                _wrap_sync_unary(
                    original=_original_sync_tool_parse_pdf,
                    span_name=WRITER_PARSE_PDF_TOOL_NAME,
                    attrs_builder=_make_tool_attrs_builder(WRITER_PARSE_PDF_TOOL_NAME),
                    positional_names=(FILE_ID_KEY,),
                ),
            )

            if _original_async_tool_parse_pdf is None:
                _original_async_tool_parse_pdf = getattr(
                    async_tools,
                    PARSE_PDF_METHOD_NAME,
                )
            setattr(
                async_tools,
                PARSE_PDF_METHOD_NAME,
                _wrap_async_unary(
                    original=_original_async_tool_parse_pdf,
                    span_name=WRITER_PARSE_PDF_TOOL_NAME,
                    attrs_builder=_make_tool_attrs_builder(WRITER_PARSE_PDF_TOOL_NAME),
                    positional_names=(FILE_ID_KEY,),
                ),
            )
        except Exception as exc:
            logger.warning("Failed to activate Writer instrumentation: %s", exc)
            self.deactivate()
            return

        self._is_instrumented = True
        logger.info("Writer instrumentation activated")

    def deactivate(self) -> None:
        """Deactivate the instrumentation."""
        global _original_sync_chat, _original_async_chat
        global _original_sync_completion_create, _original_async_completion_create
        global _original_sync_graph_question, _original_async_graph_question
        global _original_sync_application_generate, _original_async_application_generate
        global _original_sync_vision_analyze, _original_async_vision_analyze
        global _original_sync_translation_translate, _original_async_translation_translate
        global _original_sync_tool_web_search, _original_async_tool_web_search
        global _original_sync_tool_parse_pdf, _original_async_tool_parse_pdf

        try:
            classes = _load_resource_classes()
        except Exception:
            classes = {}

        restore_targets = (
            ("sync_chat", CHAT_METHOD_NAME, "_original_sync_chat"),
            ("async_chat", CHAT_METHOD_NAME, "_original_async_chat"),
            (
                "sync_completions",
                CREATE_METHOD_NAME,
                "_original_sync_completion_create",
            ),
            (
                "async_completions",
                CREATE_METHOD_NAME,
                "_original_async_completion_create",
            ),
            ("sync_graphs", QUESTION_METHOD_NAME, "_original_sync_graph_question"),
            ("async_graphs", QUESTION_METHOD_NAME, "_original_async_graph_question"),
            (
                "sync_applications",
                GENERATE_CONTENT_METHOD_NAME,
                "_original_sync_application_generate",
            ),
            (
                "async_applications",
                GENERATE_CONTENT_METHOD_NAME,
                "_original_async_application_generate",
            ),
            ("sync_vision", ANALYZE_METHOD_NAME, "_original_sync_vision_analyze"),
            ("async_vision", ANALYZE_METHOD_NAME, "_original_async_vision_analyze"),
            (
                "sync_translation",
                TRANSLATE_METHOD_NAME,
                "_original_sync_translation_translate",
            ),
            (
                "async_translation",
                TRANSLATE_METHOD_NAME,
                "_original_async_translation_translate",
            ),
            ("sync_tools", WEB_SEARCH_METHOD_NAME, "_original_sync_tool_web_search"),
            ("async_tools", WEB_SEARCH_METHOD_NAME, "_original_async_tool_web_search"),
            ("sync_tools", PARSE_PDF_METHOD_NAME, "_original_sync_tool_parse_pdf"),
            ("async_tools", PARSE_PDF_METHOD_NAME, "_original_async_tool_parse_pdf"),
        )

        for class_key, method_name, original_name in restore_targets:
            original = globals().get(original_name)
            resource_class = classes.get(class_key)
            if original is not None and resource_class is not None:
                setattr(resource_class, method_name, original)
            globals()[original_name] = None

        self._is_instrumented = False
        logger.info("Writer instrumentation deactivated")
