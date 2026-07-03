"""Google Gen AI SDK instrumentation plugin for Respan."""

from __future__ import annotations

import importlib
import logging
import time
from collections.abc import AsyncIterator, Iterator
from typing import Any

from respan_instrumentation_google_genai._constants import (
    ASYNC_MODELS_CLASS_NAME,
    GENERATE_CONTENT_METHOD_NAME,
    GENERATE_CONTENT_STREAM_METHOD_NAME,
    GOOGLE_GENAI_INSTRUMENTATION_NAME,
    GOOGLE_GENAI_MODELS_MODULE,
    MODELS_CLASS_NAME,
)
from respan_instrumentation_google_genai._otel_emitter import (
    emit_generate_content_span,
)
from respan_instrumentation_google_genai._translator import request_kwargs_from_call
from respan_tracing.core.tracer import RespanTracer

logger = logging.getLogger(__name__)

_original_sync_generate_content = None
_original_sync_generate_content_stream = None
_original_async_generate_content = None
_original_async_generate_content_stream = None


def _get_module_attr(module_path: str, attr_name: str) -> Any:
    module = importlib.import_module(module_path)
    attr_value = getattr(module, attr_name, None)
    if attr_value is None:
        raise AttributeError(f"{module_path}.{attr_name}")
    return attr_value


def _load_models_classes() -> tuple[type[Any], type[Any]]:
    return (
        _get_module_attr(GOOGLE_GENAI_MODELS_MODULE, MODELS_CLASS_NAME),
        _get_module_attr(GOOGLE_GENAI_MODELS_MODULE, ASYNC_MODELS_CLASS_NAME),
    )


def _emit_span_safely(
    *,
    kwargs: dict[str, Any],
    start_ns: int,
    response_or_chunks: Any = None,
    error_message: str | None = None,
    status_code: int = 200,
) -> None:
    emit_generate_content_span(
        request_kwargs=request_kwargs_from_call(kwargs=kwargs),
        start_ns=start_ns,
        response_or_chunks=response_or_chunks,
        error_message=error_message,
        status_code=status_code,
    )


def _wrap_sync_generate_content(original: Any) -> Any:
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        start_ns = time.time_ns()
        try:
            response = original(self, *args, **kwargs)
        except Exception as exc:
            _emit_span_safely(
                kwargs=kwargs,
                start_ns=start_ns,
                error_message=str(exc),
                status_code=500,
            )
            raise

        _emit_span_safely(
            kwargs=kwargs,
            start_ns=start_ns,
            response_or_chunks=response,
        )
        return response

    return wrapper


def _instrument_sync_stream(
    *,
    iterator: Iterator[Any],
    kwargs: dict[str, Any],
    start_ns: int,
) -> Iterator[Any]:
    chunks: list[Any] = []
    try:
        for chunk in iterator:
            chunks.append(chunk)
            yield chunk
    except Exception as exc:
        _emit_span_safely(
            kwargs=kwargs,
            start_ns=start_ns,
            response_or_chunks=chunks,
            error_message=str(exc),
            status_code=500,
        )
        raise
    else:
        _emit_span_safely(
            kwargs=kwargs,
            start_ns=start_ns,
            response_or_chunks=chunks,
        )


def _wrap_sync_generate_content_stream(original: Any) -> Any:
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Iterator[Any]:
        start_ns = time.time_ns()
        try:
            iterator = original(self, *args, **kwargs)
        except Exception as exc:
            _emit_span_safely(
                kwargs=kwargs,
                start_ns=start_ns,
                error_message=str(exc),
                status_code=500,
            )
            raise
        return _instrument_sync_stream(
            iterator=iterator,
            kwargs=kwargs,
            start_ns=start_ns,
        )

    return wrapper


def _wrap_async_generate_content(original: Any) -> Any:
    async def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        start_ns = time.time_ns()
        try:
            response = await original(self, *args, **kwargs)
        except Exception as exc:
            _emit_span_safely(
                kwargs=kwargs,
                start_ns=start_ns,
                error_message=str(exc),
                status_code=500,
            )
            raise

        _emit_span_safely(
            kwargs=kwargs,
            start_ns=start_ns,
            response_or_chunks=response,
        )
        return response

    return wrapper


async def _instrument_async_stream(
    *,
    async_iterator: AsyncIterator[Any],
    kwargs: dict[str, Any],
    start_ns: int,
) -> AsyncIterator[Any]:
    chunks: list[Any] = []
    try:
        async for chunk in async_iterator:
            chunks.append(chunk)
            yield chunk
    except Exception as exc:
        _emit_span_safely(
            kwargs=kwargs,
            start_ns=start_ns,
            response_or_chunks=chunks,
            error_message=str(exc),
            status_code=500,
        )
        raise
    else:
        _emit_span_safely(
            kwargs=kwargs,
            start_ns=start_ns,
            response_or_chunks=chunks,
        )


def _wrap_async_generate_content_stream(original: Any) -> Any:
    async def wrapper(self: Any, *args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        start_ns = time.time_ns()
        try:
            async_iterator = await original(self, *args, **kwargs)
        except Exception as exc:
            _emit_span_safely(
                kwargs=kwargs,
                start_ns=start_ns,
                error_message=str(exc),
                status_code=500,
            )
            raise
        return _instrument_async_stream(
            async_iterator=async_iterator,
            kwargs=kwargs,
            start_ns=start_ns,
        )

    return wrapper


class GoogleGenAIInstrumentor:
    """Respan instrumentor for the Google Gen AI Python SDK."""

    name = GOOGLE_GENAI_INSTRUMENTATION_NAME

    def __init__(self) -> None:
        self._is_instrumented = False

    @staticmethod
    def _is_respan_tracing_enabled() -> bool:
        tracer = getattr(RespanTracer, "_instance", None)
        if tracer is None:
            return True
        return bool(getattr(tracer, "is_enabled", True))

    def activate(self) -> None:
        """Monkey-patch Google Gen AI model generation methods."""
        global _original_sync_generate_content
        global _original_sync_generate_content_stream
        global _original_async_generate_content
        global _original_async_generate_content_stream

        if self._is_instrumented:
            return

        if not self._is_respan_tracing_enabled():
            logger.info(
                "Google Gen AI instrumentation skipped because Respan tracing is disabled"
            )
            return

        try:
            Models, AsyncModels = _load_models_classes()
        except ImportError as exc:
            # SDK genuinely absent - expected when the app doesn't use Google Gen AI
            # (the google-genai SDK is an optional extra).
            logger.debug(
                "Google Gen AI instrumentation inactive - missing dependency: %s",
                exc,
            )
            return
        except AttributeError as exc:
            # SDK installed but incompatible (a class moved/renamed) - surface it
            # so a broken install isn't silently left untraced.
            logger.warning(
                "google-genai is installed but incompatible - instrumentation inactive: %s",
                exc,
            )
            return
        except Exception as exc:
            logger.warning("Failed to activate Google Gen AI instrumentation: %s", exc)
            return

        try:
            if _original_sync_generate_content is None:
                _original_sync_generate_content = getattr(
                    Models,
                    GENERATE_CONTENT_METHOD_NAME,
                )
            setattr(
                Models,
                GENERATE_CONTENT_METHOD_NAME,
                _wrap_sync_generate_content(_original_sync_generate_content),
            )

            if _original_sync_generate_content_stream is None:
                _original_sync_generate_content_stream = getattr(
                    Models,
                    GENERATE_CONTENT_STREAM_METHOD_NAME,
                )
            setattr(
                Models,
                GENERATE_CONTENT_STREAM_METHOD_NAME,
                _wrap_sync_generate_content_stream(
                    _original_sync_generate_content_stream
                ),
            )

            if _original_async_generate_content is None:
                _original_async_generate_content = getattr(
                    AsyncModels,
                    GENERATE_CONTENT_METHOD_NAME,
                )
            setattr(
                AsyncModels,
                GENERATE_CONTENT_METHOD_NAME,
                _wrap_async_generate_content(_original_async_generate_content),
            )

            if _original_async_generate_content_stream is None:
                _original_async_generate_content_stream = getattr(
                    AsyncModels,
                    GENERATE_CONTENT_STREAM_METHOD_NAME,
                )
            setattr(
                AsyncModels,
                GENERATE_CONTENT_STREAM_METHOD_NAME,
                _wrap_async_generate_content_stream(
                    _original_async_generate_content_stream
                ),
            )
        except Exception as exc:
            logger.warning("Failed to activate Google Gen AI instrumentation: %s", exc)
            self.deactivate()
            return

        self._is_instrumented = True
        logger.info("Google Gen AI instrumentation activated")

    def deactivate(self) -> None:
        """Restore original Google Gen AI SDK methods."""
        global _original_sync_generate_content
        global _original_sync_generate_content_stream
        global _original_async_generate_content
        global _original_async_generate_content_stream

        if not self._is_instrumented:
            return

        try:
            Models, AsyncModels = _load_models_classes()
            if _original_sync_generate_content is not None:
                setattr(
                    Models,
                    GENERATE_CONTENT_METHOD_NAME,
                    _original_sync_generate_content,
                )
                _original_sync_generate_content = None
            if _original_sync_generate_content_stream is not None:
                setattr(
                    Models,
                    GENERATE_CONTENT_STREAM_METHOD_NAME,
                    _original_sync_generate_content_stream,
                )
                _original_sync_generate_content_stream = None
            if _original_async_generate_content is not None:
                setattr(
                    AsyncModels,
                    GENERATE_CONTENT_METHOD_NAME,
                    _original_async_generate_content,
                )
                _original_async_generate_content = None
            if _original_async_generate_content_stream is not None:
                setattr(
                    AsyncModels,
                    GENERATE_CONTENT_STREAM_METHOD_NAME,
                    _original_async_generate_content_stream,
                )
                _original_async_generate_content_stream = None
        except Exception:
            logger.debug("Failed to restore Google Gen AI methods", exc_info=True)

        self._is_instrumented = False
        logger.info("Google Gen AI instrumentation deactivated")
