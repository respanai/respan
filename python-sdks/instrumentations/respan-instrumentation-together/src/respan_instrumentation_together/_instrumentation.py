"""Together AI SDK instrumentation plugin for Respan."""

from __future__ import annotations

import importlib
import logging
import time
from typing import Any

from respan_instrumentation_together._constants import (
    ASYNC_COMPLETIONS_RESOURCE_CLASS_NAME,
    ASYNC_EMBEDDINGS_RESOURCE_CLASS_NAME,
    ASYNC_IMAGES_RESOURCE_CLASS_NAME,
    ASYNC_RERANK_RESOURCE_CLASS_NAME,
    COMPLETIONS_RESOURCE_CLASS_NAME,
    CREATE_METHOD_NAME,
    EMBEDDINGS_RESOURCE_CLASS_NAME,
    GENERATE_METHOD_NAME,
    IMAGES_RESOURCE_CLASS_NAME,
    RERANK_RESOURCE_CLASS_NAME,
    STREAM_KEY,
    TOGETHER_CHAT_COMPLETIONS_MODULE,
    TOGETHER_EMBEDDINGS_MODULE,
    TOGETHER_IMAGES_MODULE,
    TOGETHER_INSTRUMENTATION_NAME,
    TOGETHER_RERANK_MODULE,
    TOGETHER_TEXT_COMPLETIONS_MODULE,
)
from respan_instrumentation_together._otel_emitter import emit_together_span
from respan_tracing.core.tracer import RespanTracer

logger = logging.getLogger(__name__)

_original_sync_chat_create = None
_original_async_chat_create = None
_original_sync_completion_create = None
_original_async_completion_create = None
_original_sync_embedding_create = None
_original_async_embedding_create = None
_original_sync_image_generate = None
_original_async_image_generate = None
_original_sync_rerank_create = None
_original_async_rerank_create = None


def _get_module_attr(module_path: str, attr_name: str) -> Any:
    module = importlib.import_module(module_path)
    attr_value = getattr(module, attr_name, None)
    if attr_value is None:
        raise AttributeError(f"{module_path}.{attr_name}")
    return attr_value


def _load_class(module_path: str, class_name: str) -> type[Any]:
    return _get_module_attr(module_path=module_path, attr_name=class_name)


def _is_stream_requested(kwargs: dict[str, Any]) -> bool:
    return kwargs.get(STREAM_KEY) is True


def _emit_span_safely(
    *,
    operation: str,
    kwargs: dict[str, Any],
    start_ns: int,
    response_or_chunks: Any = None,
    error_message: str | None = None,
    status_code: int = 200,
) -> None:
    emit_together_span(
        operation=operation,
        request_kwargs=dict(kwargs),
        start_ns=start_ns,
        response_or_chunks=response_or_chunks,
        error_message=error_message,
        status_code=status_code,
    )


class _InstrumentedStream:
    def __init__(self, *, stream: Any, operation: str, kwargs: dict[str, Any], start_ns: int) -> None:
        self._stream = stream
        self._operation = operation
        self._kwargs = dict(kwargs)
        self._start_ns = start_ns
        self._chunks: list[Any] = []
        self._emitted = False

    def __iter__(self) -> "_InstrumentedStream":
        return self

    def __next__(self) -> Any:
        try:
            chunk = next(self._stream)
        except StopIteration:
            self._emit_once()
            raise
        except Exception as exc:
            self._emit_once(error_message=str(exc), status_code=500)
            raise
        self._chunks.append(chunk)
        return chunk

    def __enter__(self) -> "_InstrumentedStream":
        enter = getattr(self._stream, "__enter__", None)
        if callable(enter):
            enter()
        return self

    def __exit__(self, exc_type: Any, exc: Any, exc_tb: Any) -> Any:
        if exc is not None:
            self._emit_once(error_message=str(exc), status_code=500)
        else:
            self._emit_once()
        exit_method = getattr(self._stream, "__exit__", None)
        if callable(exit_method):
            return exit_method(exc_type, exc, exc_tb)
        return None

    def close(self) -> None:
        self._emit_once()
        close = getattr(self._stream, "close", None)
        if callable(close):
            close()

    def _emit_once(
        self,
        *,
        error_message: str | None = None,
        status_code: int = 200,
    ) -> None:
        if self._emitted:
            return
        self._emitted = True
        _emit_span_safely(
            operation=self._operation,
            kwargs=self._kwargs,
            start_ns=self._start_ns,
            response_or_chunks=self._chunks,
            error_message=error_message,
            status_code=status_code,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


class _InstrumentedAsyncStream:
    def __init__(self, *, stream: Any, operation: str, kwargs: dict[str, Any], start_ns: int) -> None:
        self._stream = stream
        self._operation = operation
        self._kwargs = dict(kwargs)
        self._start_ns = start_ns
        self._chunks: list[Any] = []
        self._emitted = False

    def __aiter__(self) -> "_InstrumentedAsyncStream":
        return self

    async def __anext__(self) -> Any:
        try:
            chunk = await self._stream.__anext__()
        except StopAsyncIteration:
            self._emit_once()
            raise
        except Exception as exc:
            self._emit_once(error_message=str(exc), status_code=500)
            raise
        self._chunks.append(chunk)
        return chunk

    async def __aenter__(self) -> "_InstrumentedAsyncStream":
        enter = getattr(self._stream, "__aenter__", None)
        if callable(enter):
            await enter()
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, exc_tb: Any) -> Any:
        if exc is not None:
            self._emit_once(error_message=str(exc), status_code=500)
        else:
            self._emit_once()
        exit_method = getattr(self._stream, "__aexit__", None)
        if callable(exit_method):
            return await exit_method(exc_type, exc, exc_tb)
        return None

    async def close(self) -> None:
        self._emit_once()
        close = getattr(self._stream, "close", None)
        if callable(close):
            result = close()
            if hasattr(result, "__await__"):
                await result

    def _emit_once(
        self,
        *,
        error_message: str | None = None,
        status_code: int = 200,
    ) -> None:
        if self._emitted:
            return
        self._emitted = True
        _emit_span_safely(
            operation=self._operation,
            kwargs=self._kwargs,
            start_ns=self._start_ns,
            response_or_chunks=self._chunks,
            error_message=error_message,
            status_code=status_code,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


def _wrap_sync_create(original: Any, *, operation: str) -> Any:
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        start_ns = time.time_ns()
        try:
            response = original(self, *args, **kwargs)
        except Exception as exc:
            _emit_span_safely(
                operation=operation,
                kwargs=kwargs,
                start_ns=start_ns,
                error_message=str(exc),
                status_code=500,
            )
            raise

        if _is_stream_requested(kwargs):
            return _InstrumentedStream(
                stream=response,
                operation=operation,
                kwargs=kwargs,
                start_ns=start_ns,
            )

        _emit_span_safely(
            operation=operation,
            kwargs=kwargs,
            start_ns=start_ns,
            response_or_chunks=response,
        )
        return response

    return wrapper


def _wrap_async_create(original: Any, *, operation: str) -> Any:
    async def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        start_ns = time.time_ns()
        try:
            response = await original(self, *args, **kwargs)
        except Exception as exc:
            _emit_span_safely(
                operation=operation,
                kwargs=kwargs,
                start_ns=start_ns,
                error_message=str(exc),
                status_code=500,
            )
            raise

        if _is_stream_requested(kwargs):
            return _InstrumentedAsyncStream(
                stream=response,
                operation=operation,
                kwargs=kwargs,
                start_ns=start_ns,
            )

        _emit_span_safely(
            operation=operation,
            kwargs=kwargs,
            start_ns=start_ns,
            response_or_chunks=response,
        )
        return response

    return wrapper


class TogetherInstrumentor:
    """Respan instrumentor for the Together AI Python SDK."""

    name = TOGETHER_INSTRUMENTATION_NAME

    def __init__(self) -> None:
        self._is_instrumented = False

    @staticmethod
    def _is_respan_tracing_enabled() -> bool:
        tracer = getattr(RespanTracer, "_instance", None)
        if tracer is None:
            return True
        return bool(getattr(tracer, "is_enabled", True))

    def activate(self) -> None:
        """Monkey-patch Together SDK resources."""
        global _original_sync_chat_create, _original_async_chat_create
        global _original_sync_completion_create, _original_async_completion_create
        global _original_sync_embedding_create, _original_async_embedding_create
        global _original_sync_image_generate, _original_async_image_generate
        global _original_sync_rerank_create, _original_async_rerank_create

        if self._is_instrumented:
            return

        if not self._is_respan_tracing_enabled():
            logger.info("Together instrumentation skipped because Respan tracing is disabled")
            return

        try:
            SyncChatCompletions = _load_class(
                TOGETHER_CHAT_COMPLETIONS_MODULE,
                COMPLETIONS_RESOURCE_CLASS_NAME,
            )
            AsyncChatCompletions = _load_class(
                TOGETHER_CHAT_COMPLETIONS_MODULE,
                ASYNC_COMPLETIONS_RESOURCE_CLASS_NAME,
            )
            SyncCompletions = _load_class(
                TOGETHER_TEXT_COMPLETIONS_MODULE,
                COMPLETIONS_RESOURCE_CLASS_NAME,
            )
            AsyncCompletions = _load_class(
                TOGETHER_TEXT_COMPLETIONS_MODULE,
                ASYNC_COMPLETIONS_RESOURCE_CLASS_NAME,
            )
            SyncEmbeddings = _load_class(
                TOGETHER_EMBEDDINGS_MODULE,
                EMBEDDINGS_RESOURCE_CLASS_NAME,
            )
            AsyncEmbeddings = _load_class(
                TOGETHER_EMBEDDINGS_MODULE,
                ASYNC_EMBEDDINGS_RESOURCE_CLASS_NAME,
            )
            SyncImages = _load_class(
                TOGETHER_IMAGES_MODULE,
                IMAGES_RESOURCE_CLASS_NAME,
            )
            AsyncImages = _load_class(
                TOGETHER_IMAGES_MODULE,
                ASYNC_IMAGES_RESOURCE_CLASS_NAME,
            )
            SyncRerank = _load_class(
                TOGETHER_RERANK_MODULE,
                RERANK_RESOURCE_CLASS_NAME,
            )
            AsyncRerank = _load_class(
                TOGETHER_RERANK_MODULE,
                ASYNC_RERANK_RESOURCE_CLASS_NAME,
            )
        except ImportError as exc:
            # SDK genuinely absent - expected when the app doesn't use Together
            # (the together SDK is an optional extra).
            logger.debug(
                "Together instrumentation inactive - missing dependency: %s",
                exc,
            )
            return
        except AttributeError as exc:
            # SDK installed but incompatible (a class moved/renamed) - surface it
            # so a broken install isn't silently left untraced.
            logger.warning(
                "together is installed but incompatible - instrumentation inactive: %s",
                exc,
            )
            return
        except Exception as exc:
            logger.warning("Failed to activate Together instrumentation: %s", exc)
            return

        try:
            if _original_sync_chat_create is None:
                _original_sync_chat_create = getattr(SyncChatCompletions, CREATE_METHOD_NAME)
            setattr(
                SyncChatCompletions,
                CREATE_METHOD_NAME,
                _wrap_sync_create(_original_sync_chat_create, operation="chat"),
            )

            if _original_async_chat_create is None:
                _original_async_chat_create = getattr(AsyncChatCompletions, CREATE_METHOD_NAME)
            setattr(
                AsyncChatCompletions,
                CREATE_METHOD_NAME,
                _wrap_async_create(_original_async_chat_create, operation="chat"),
            )

            if _original_sync_completion_create is None:
                _original_sync_completion_create = getattr(SyncCompletions, CREATE_METHOD_NAME)
            setattr(
                SyncCompletions,
                CREATE_METHOD_NAME,
                _wrap_sync_create(_original_sync_completion_create, operation="completion"),
            )

            if _original_async_completion_create is None:
                _original_async_completion_create = getattr(AsyncCompletions, CREATE_METHOD_NAME)
            setattr(
                AsyncCompletions,
                CREATE_METHOD_NAME,
                _wrap_async_create(_original_async_completion_create, operation="completion"),
            )

            if _original_sync_embedding_create is None:
                _original_sync_embedding_create = getattr(SyncEmbeddings, CREATE_METHOD_NAME)
            setattr(
                SyncEmbeddings,
                CREATE_METHOD_NAME,
                _wrap_sync_create(_original_sync_embedding_create, operation="embedding"),
            )

            if _original_async_embedding_create is None:
                _original_async_embedding_create = getattr(AsyncEmbeddings, CREATE_METHOD_NAME)
            setattr(
                AsyncEmbeddings,
                CREATE_METHOD_NAME,
                _wrap_async_create(_original_async_embedding_create, operation="embedding"),
            )

            if _original_sync_image_generate is None:
                _original_sync_image_generate = getattr(SyncImages, GENERATE_METHOD_NAME)
            setattr(
                SyncImages,
                GENERATE_METHOD_NAME,
                _wrap_sync_create(_original_sync_image_generate, operation="image"),
            )

            if _original_async_image_generate is None:
                _original_async_image_generate = getattr(AsyncImages, GENERATE_METHOD_NAME)
            setattr(
                AsyncImages,
                GENERATE_METHOD_NAME,
                _wrap_async_create(_original_async_image_generate, operation="image"),
            )

            if _original_sync_rerank_create is None:
                _original_sync_rerank_create = getattr(SyncRerank, CREATE_METHOD_NAME)
            setattr(
                SyncRerank,
                CREATE_METHOD_NAME,
                _wrap_sync_create(_original_sync_rerank_create, operation="rerank"),
            )

            if _original_async_rerank_create is None:
                _original_async_rerank_create = getattr(AsyncRerank, CREATE_METHOD_NAME)
            setattr(
                AsyncRerank,
                CREATE_METHOD_NAME,
                _wrap_async_create(_original_async_rerank_create, operation="rerank"),
            )
        except Exception as exc:
            logger.warning("Failed to activate Together instrumentation: %s", exc)
            self.deactivate()
            return

        self._is_instrumented = True
        logger.info("Together instrumentation activated")

    def deactivate(self) -> None:
        """Restore original Together SDK methods."""
        global _original_sync_chat_create, _original_async_chat_create
        global _original_sync_completion_create, _original_async_completion_create
        global _original_sync_embedding_create, _original_async_embedding_create
        global _original_sync_image_generate, _original_async_image_generate
        global _original_sync_rerank_create, _original_async_rerank_create

        if not self._is_instrumented:
            return

        try:
            SyncChatCompletions = _load_class(
                TOGETHER_CHAT_COMPLETIONS_MODULE,
                COMPLETIONS_RESOURCE_CLASS_NAME,
            )
            AsyncChatCompletions = _load_class(
                TOGETHER_CHAT_COMPLETIONS_MODULE,
                ASYNC_COMPLETIONS_RESOURCE_CLASS_NAME,
            )
            SyncCompletions = _load_class(
                TOGETHER_TEXT_COMPLETIONS_MODULE,
                COMPLETIONS_RESOURCE_CLASS_NAME,
            )
            AsyncCompletions = _load_class(
                TOGETHER_TEXT_COMPLETIONS_MODULE,
                ASYNC_COMPLETIONS_RESOURCE_CLASS_NAME,
            )
            SyncEmbeddings = _load_class(
                TOGETHER_EMBEDDINGS_MODULE,
                EMBEDDINGS_RESOURCE_CLASS_NAME,
            )
            AsyncEmbeddings = _load_class(
                TOGETHER_EMBEDDINGS_MODULE,
                ASYNC_EMBEDDINGS_RESOURCE_CLASS_NAME,
            )
            SyncImages = _load_class(
                TOGETHER_IMAGES_MODULE,
                IMAGES_RESOURCE_CLASS_NAME,
            )
            AsyncImages = _load_class(
                TOGETHER_IMAGES_MODULE,
                ASYNC_IMAGES_RESOURCE_CLASS_NAME,
            )
            SyncRerank = _load_class(
                TOGETHER_RERANK_MODULE,
                RERANK_RESOURCE_CLASS_NAME,
            )
            AsyncRerank = _load_class(
                TOGETHER_RERANK_MODULE,
                ASYNC_RERANK_RESOURCE_CLASS_NAME,
            )

            if _original_sync_chat_create is not None:
                setattr(SyncChatCompletions, CREATE_METHOD_NAME, _original_sync_chat_create)
            if _original_async_chat_create is not None:
                setattr(AsyncChatCompletions, CREATE_METHOD_NAME, _original_async_chat_create)
            if _original_sync_completion_create is not None:
                setattr(SyncCompletions, CREATE_METHOD_NAME, _original_sync_completion_create)
            if _original_async_completion_create is not None:
                setattr(AsyncCompletions, CREATE_METHOD_NAME, _original_async_completion_create)
            if _original_sync_embedding_create is not None:
                setattr(SyncEmbeddings, CREATE_METHOD_NAME, _original_sync_embedding_create)
            if _original_async_embedding_create is not None:
                setattr(AsyncEmbeddings, CREATE_METHOD_NAME, _original_async_embedding_create)
            if _original_sync_image_generate is not None:
                setattr(SyncImages, GENERATE_METHOD_NAME, _original_sync_image_generate)
            if _original_async_image_generate is not None:
                setattr(AsyncImages, GENERATE_METHOD_NAME, _original_async_image_generate)
            if _original_sync_rerank_create is not None:
                setattr(SyncRerank, CREATE_METHOD_NAME, _original_sync_rerank_create)
            if _original_async_rerank_create is not None:
                setattr(AsyncRerank, CREATE_METHOD_NAME, _original_async_rerank_create)
        except Exception as exc:
            logger.warning("Failed to deactivate Together instrumentation cleanly: %s", exc)
        finally:
            self._is_instrumented = False
            logger.info("Together instrumentation deactivated")
