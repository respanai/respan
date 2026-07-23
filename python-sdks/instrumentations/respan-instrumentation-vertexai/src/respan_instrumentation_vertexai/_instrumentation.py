"""Vertex AI SDK instrumentation plugin for Respan."""

from __future__ import annotations

import importlib
import logging
import time
from collections.abc import AsyncIterator, Iterator
from typing import Any

from respan_instrumentation_vertexai._constants import (
    CHAT_SESSION_CLASS_NAME,
    GENERATE_CONTENT_ASYNC_METHOD_NAME,
    GENERATE_CONTENT_METHOD_NAME,
    GENERATIVE_MODEL_CLASS_NAME,
    SEND_MESSAGE_ASYNC_METHOD_NAME,
    SEND_MESSAGE_METHOD_NAME,
    VERTEXAI_CHAT_SPAN_NAME,
    VERTEXAI_GENERATE_CONTENT_SPAN_NAME,
    VERTEXAI_GENERATIVE_MODELS_MODULE,
    VERTEXAI_INSTRUMENTATION_NAME,
)
from respan_instrumentation_vertexai._otel_emitter import emit_generate_content_span
from respan_instrumentation_vertexai._translator import request_payload_from_call
from respan_tracing.core.tracer import RespanTracer

logger = logging.getLogger(__name__)

_original_generate_content = None
_original_generate_content_async = None
_original_send_message = None
_original_send_message_async = None


def _get_module_attr(module_path: str, attr_name: str) -> Any:
    module = importlib.import_module(module_path)
    attr_value = getattr(module, attr_name, None)
    if attr_value is None:
        raise AttributeError(f"{module_path}.{attr_name}")
    return attr_value


def _load_vertexai_classes() -> tuple[type[Any], type[Any]]:
    return (
        _get_module_attr(
            VERTEXAI_GENERATIVE_MODELS_MODULE, GENERATIVE_MODEL_CLASS_NAME
        ),
        _get_module_attr(VERTEXAI_GENERATIVE_MODELS_MODULE, CHAT_SESSION_CLASS_NAME),
    )


def _emit_span_safely(
    *,
    instance: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    start_ns: int,
    span_name: str,
    response_or_chunks: Any = None,
    error_message: str | None = None,
    status_code: int = 200,
) -> None:
    emit_generate_content_span(
        request_payload=request_payload_from_call(
            instance=instance,
            args=args,
            kwargs=kwargs,
        ),
        start_ns=start_ns,
        response_or_chunks=response_or_chunks,
        span_name=span_name,
        error_message=error_message,
        status_code=status_code,
    )


def _wrap_sync_iterator(
    *,
    iterator: Iterator[Any],
    instance: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    start_ns: int,
    span_name: str,
) -> Iterator[Any]:
    chunks: list[Any] = []
    try:
        for chunk in iterator:
            chunks.append(chunk)
            yield chunk
    except Exception as exc:
        _emit_span_safely(
            instance=instance,
            args=args,
            kwargs=kwargs,
            start_ns=start_ns,
            span_name=span_name,
            response_or_chunks=chunks,
            error_message=str(exc),
            status_code=500,
        )
        raise
    else:
        _emit_span_safely(
            instance=instance,
            args=args,
            kwargs=kwargs,
            start_ns=start_ns,
            span_name=span_name,
            response_or_chunks=chunks,
        )


async def _wrap_async_iterator(
    *,
    async_iterator: AsyncIterator[Any],
    instance: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    start_ns: int,
    span_name: str,
) -> AsyncIterator[Any]:
    chunks: list[Any] = []
    try:
        async for chunk in async_iterator:
            chunks.append(chunk)
            yield chunk
    except Exception as exc:
        _emit_span_safely(
            instance=instance,
            args=args,
            kwargs=kwargs,
            start_ns=start_ns,
            span_name=span_name,
            response_or_chunks=chunks,
            error_message=str(exc),
            status_code=500,
        )
        raise
    else:
        _emit_span_safely(
            instance=instance,
            args=args,
            kwargs=kwargs,
            start_ns=start_ns,
            span_name=span_name,
            response_or_chunks=chunks,
        )


def _is_sync_stream(value: Any) -> bool:
    if isinstance(value, (str, bytes, bytearray)):
        return False
    return hasattr(value, "__iter__") and not hasattr(value, "candidates")


def _is_async_stream(value: Any) -> bool:
    return hasattr(value, "__aiter__")


def _wrap_sync_method(original: Any, span_name: str) -> Any:
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        start_ns = time.time_ns()
        try:
            response = original(self, *args, **kwargs)
        except Exception as exc:
            _emit_span_safely(
                instance=self,
                args=args,
                kwargs=kwargs,
                start_ns=start_ns,
                span_name=span_name,
                error_message=str(exc),
                status_code=500,
            )
            raise

        if _is_sync_stream(response):
            return _wrap_sync_iterator(
                iterator=response,
                instance=self,
                args=args,
                kwargs=kwargs,
                start_ns=start_ns,
                span_name=span_name,
            )

        _emit_span_safely(
            instance=self,
            args=args,
            kwargs=kwargs,
            start_ns=start_ns,
            span_name=span_name,
            response_or_chunks=response,
        )
        return response

    return wrapper


def _wrap_async_method(original: Any, span_name: str) -> Any:
    async def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        start_ns = time.time_ns()
        try:
            response = await original(self, *args, **kwargs)
        except Exception as exc:
            _emit_span_safely(
                instance=self,
                args=args,
                kwargs=kwargs,
                start_ns=start_ns,
                span_name=span_name,
                error_message=str(exc),
                status_code=500,
            )
            raise

        if _is_async_stream(response):
            return _wrap_async_iterator(
                async_iterator=response,
                instance=self,
                args=args,
                kwargs=kwargs,
                start_ns=start_ns,
                span_name=span_name,
            )

        _emit_span_safely(
            instance=self,
            args=args,
            kwargs=kwargs,
            start_ns=start_ns,
            span_name=span_name,
            response_or_chunks=response,
        )
        return response

    return wrapper


class VertexAIInstrumentor:
    """Respan instrumentor for the Google Vertex AI Python SDK."""

    name = VERTEXAI_INSTRUMENTATION_NAME

    def __init__(self) -> None:
        self._is_instrumented = False

    @staticmethod
    def _is_respan_tracing_enabled() -> bool:
        tracer = getattr(RespanTracer, "_instance", None)
        if tracer is None:
            return True
        return bool(getattr(tracer, "is_enabled", True))

    def activate(self) -> None:
        """Monkey-patch Vertex AI generation methods."""
        global _original_generate_content
        global _original_generate_content_async
        global _original_send_message
        global _original_send_message_async

        if self._is_instrumented:
            return

        if not self._is_respan_tracing_enabled():
            logger.info(
                "Vertex AI instrumentation skipped because Respan tracing is disabled"
            )
            return

        try:
            GenerativeModel, ChatSession = _load_vertexai_classes()
        except ImportError as exc:
            # SDK genuinely absent - expected when the app doesn't use Vertex AI
            # (the google-cloud-aiplatform SDK is an optional extra).
            logger.debug(
                "Vertex AI instrumentation inactive - missing dependency: %s",
                exc,
            )
            return
        except AttributeError as exc:
            # SDK installed but incompatible (a class moved/renamed) - surface it
            # so a broken install isn't silently left untraced.
            logger.warning(
                "google-cloud-aiplatform is installed but incompatible - Vertex AI instrumentation inactive: %s",
                exc,
            )
            return
        except Exception as exc:
            logger.warning("Failed to activate Vertex AI instrumentation: %s", exc)
            return

        try:
            if _original_generate_content is None:
                _original_generate_content = getattr(
                    GenerativeModel,
                    GENERATE_CONTENT_METHOD_NAME,
                )
            setattr(
                GenerativeModel,
                GENERATE_CONTENT_METHOD_NAME,
                _wrap_sync_method(
                    _original_generate_content,
                    VERTEXAI_GENERATE_CONTENT_SPAN_NAME,
                ),
            )

            if _original_generate_content_async is None:
                _original_generate_content_async = getattr(
                    GenerativeModel,
                    GENERATE_CONTENT_ASYNC_METHOD_NAME,
                )
            setattr(
                GenerativeModel,
                GENERATE_CONTENT_ASYNC_METHOD_NAME,
                _wrap_async_method(
                    _original_generate_content_async,
                    VERTEXAI_GENERATE_CONTENT_SPAN_NAME,
                ),
            )

            if _original_send_message is None:
                _original_send_message = getattr(ChatSession, SEND_MESSAGE_METHOD_NAME)
            setattr(
                ChatSession,
                SEND_MESSAGE_METHOD_NAME,
                _wrap_sync_method(_original_send_message, VERTEXAI_CHAT_SPAN_NAME),
            )

            if _original_send_message_async is None:
                _original_send_message_async = getattr(
                    ChatSession,
                    SEND_MESSAGE_ASYNC_METHOD_NAME,
                )
            setattr(
                ChatSession,
                SEND_MESSAGE_ASYNC_METHOD_NAME,
                _wrap_async_method(
                    _original_send_message_async,
                    VERTEXAI_CHAT_SPAN_NAME,
                ),
            )
        except Exception as exc:
            logger.warning("Failed to activate Vertex AI instrumentation: %s", exc)
            self.deactivate()
            return

        self._is_instrumented = True
        logger.info("Vertex AI instrumentation activated")

    def deactivate(self) -> None:
        """Restore original Vertex AI SDK methods."""
        global _original_generate_content
        global _original_generate_content_async
        global _original_send_message
        global _original_send_message_async

        if not self._is_instrumented:
            return

        try:
            GenerativeModel, ChatSession = _load_vertexai_classes()
            if _original_generate_content is not None:
                setattr(
                    GenerativeModel,
                    GENERATE_CONTENT_METHOD_NAME,
                    _original_generate_content,
                )
                _original_generate_content = None
            if _original_generate_content_async is not None:
                setattr(
                    GenerativeModel,
                    GENERATE_CONTENT_ASYNC_METHOD_NAME,
                    _original_generate_content_async,
                )
                _original_generate_content_async = None
            if _original_send_message is not None:
                setattr(ChatSession, SEND_MESSAGE_METHOD_NAME, _original_send_message)
                _original_send_message = None
            if _original_send_message_async is not None:
                setattr(
                    ChatSession,
                    SEND_MESSAGE_ASYNC_METHOD_NAME,
                    _original_send_message_async,
                )
                _original_send_message_async = None
        except Exception:
            logger.debug("Failed to restore Vertex AI methods", exc_info=True)

        self._is_instrumented = False
        logger.info("Vertex AI instrumentation deactivated")
