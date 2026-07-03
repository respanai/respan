"""Ollama SDK instrumentation plugin for Respan."""

from __future__ import annotations

import importlib
import inspect
import logging
import time
from collections.abc import AsyncIterator, Iterator
from typing import Any, Callable

from respan_instrumentation_ollama._constants import (
    ASYNC_CLIENT_CLASS_NAME,
    CHAT_METHOD_NAME,
    CLIENT_CLASS_NAME,
    EMBED_METHOD_NAME,
    EMBEDDINGS_METHOD_NAME,
    GENERATE_METHOD_NAME,
    OLLAMA_CLIENT_MODULE,
    OLLAMA_INSTRUMENTATION_NAME,
    STREAM_KEY,
)
from respan_instrumentation_ollama._otel_emitter import (
    emit_chat_span,
    emit_embedding_span,
    emit_generate_span,
)
from respan_tracing.core.tracer import RespanTracer

logger = logging.getLogger(__name__)

_original_sync_chat = None
_original_async_chat = None
_original_sync_generate = None
_original_async_generate = None
_original_sync_embed = None
_original_async_embed = None
_original_sync_embeddings = None
_original_async_embeddings = None


def _get_module_attr(module_path: str, attr_name: str) -> Any:
    module = importlib.import_module(module_path)
    attr_value = getattr(module, attr_name, None)
    if attr_value is None:
        raise AttributeError(f"{module_path}.{attr_name}")
    return attr_value


def _load_client_classes() -> tuple[type[Any], type[Any]]:
    return (
        _get_module_attr(OLLAMA_CLIENT_MODULE, CLIENT_CLASS_NAME),
        _get_module_attr(OLLAMA_CLIENT_MODULE, ASYNC_CLIENT_CLASS_NAME),
    )


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
    bound.apply_defaults()
    return {
        key: value
        for key, value in bound.arguments.items()
        if key not in {"self", "cls"}
    }


def _instrument_sync_stream(
    *,
    iterator: Iterator[Any],
    emit_span: Callable[..., None],
    request_kwargs: dict[str, Any],
    start_ns: int,
) -> Iterator[Any]:
    chunks: list[Any] = []
    try:
        for chunk in iterator:
            chunks.append(chunk)
            yield chunk
    except Exception as exc:
        emit_span(
            request_kwargs=request_kwargs,
            response_or_chunks=chunks,
            start_ns=start_ns,
            error_message=str(exc),
            status_code=500,
        )
        raise
    else:
        emit_span(
            request_kwargs=request_kwargs,
            response_or_chunks=chunks,
            start_ns=start_ns,
        )


async def _instrument_async_stream(
    *,
    async_iterator: AsyncIterator[Any],
    emit_span: Callable[..., None],
    request_kwargs: dict[str, Any],
    start_ns: int,
) -> AsyncIterator[Any]:
    chunks: list[Any] = []
    try:
        async for chunk in async_iterator:
            chunks.append(chunk)
            yield chunk
    except Exception as exc:
        emit_span(
            request_kwargs=request_kwargs,
            response_or_chunks=chunks,
            start_ns=start_ns,
            error_message=str(exc),
            status_code=500,
        )
        raise
    else:
        emit_span(
            request_kwargs=request_kwargs,
            response_or_chunks=chunks,
            start_ns=start_ns,
        )


def _wrap_sync_llm_call(
    original: Callable[..., Any],
    *,
    emit_span: Callable[..., None],
) -> Callable[..., Any]:
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        request_kwargs = _request_kwargs_from_call(original, self, args, kwargs)
        start_ns = time.time_ns()
        try:
            response = original(self, *args, **kwargs)
        except Exception as exc:
            emit_span(
                request_kwargs=request_kwargs,
                start_ns=start_ns,
                error_message=str(exc),
                status_code=500,
            )
            raise

        if request_kwargs.get(STREAM_KEY):
            return _instrument_sync_stream(
                iterator=response,
                emit_span=emit_span,
                request_kwargs=request_kwargs,
                start_ns=start_ns,
            )

        emit_span(
            request_kwargs=request_kwargs,
            response_or_chunks=response,
            start_ns=start_ns,
        )
        return response

    return wrapper


def _wrap_async_llm_call(
    original: Callable[..., Any],
    *,
    emit_span: Callable[..., None],
) -> Callable[..., Any]:
    async def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        request_kwargs = _request_kwargs_from_call(original, self, args, kwargs)
        start_ns = time.time_ns()
        try:
            response = await original(self, *args, **kwargs)
        except Exception as exc:
            emit_span(
                request_kwargs=request_kwargs,
                start_ns=start_ns,
                error_message=str(exc),
                status_code=500,
            )
            raise

        if request_kwargs.get(STREAM_KEY):
            return _instrument_async_stream(
                async_iterator=response,
                emit_span=emit_span,
                request_kwargs=request_kwargs,
                start_ns=start_ns,
            )

        emit_span(
            request_kwargs=request_kwargs,
            response_or_chunks=response,
            start_ns=start_ns,
        )
        return response

    return wrapper


def _wrap_sync_embedding_call(
    original: Callable[..., Any],
    *,
    method_name: str,
) -> Callable[..., Any]:
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        request_kwargs = _request_kwargs_from_call(original, self, args, kwargs)
        start_ns = time.time_ns()
        try:
            response = original(self, *args, **kwargs)
        except Exception as exc:
            emit_embedding_span(
                request_kwargs=request_kwargs,
                start_ns=start_ns,
                error_message=str(exc),
                status_code=500,
                method_name=method_name,
            )
            raise

        emit_embedding_span(
            request_kwargs=request_kwargs,
            response=response,
            start_ns=start_ns,
            method_name=method_name,
        )
        return response

    return wrapper


def _wrap_async_embedding_call(
    original: Callable[..., Any],
    *,
    method_name: str,
) -> Callable[..., Any]:
    async def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        request_kwargs = _request_kwargs_from_call(original, self, args, kwargs)
        start_ns = time.time_ns()
        try:
            response = await original(self, *args, **kwargs)
        except Exception as exc:
            emit_embedding_span(
                request_kwargs=request_kwargs,
                start_ns=start_ns,
                error_message=str(exc),
                status_code=500,
                method_name=method_name,
            )
            raise

        emit_embedding_span(
            request_kwargs=request_kwargs,
            response=response,
            start_ns=start_ns,
            method_name=method_name,
        )
        return response

    return wrapper


class OllamaInstrumentor:
    """Respan instrumentor for the official Ollama Python SDK."""

    name = OLLAMA_INSTRUMENTATION_NAME

    def __init__(self) -> None:
        self._is_instrumented = False

    @staticmethod
    def _is_respan_tracing_enabled() -> bool:
        tracer = getattr(RespanTracer, "_instance", None)
        if tracer is None:
            return True
        return bool(getattr(tracer, "is_enabled", True))

    def activate(self) -> None:
        """Monkey-patch Ollama client generation and embedding methods."""
        global _original_sync_chat, _original_async_chat
        global _original_sync_generate, _original_async_generate
        global _original_sync_embed, _original_async_embed
        global _original_sync_embeddings, _original_async_embeddings

        if self._is_instrumented:
            return

        if not self._is_respan_tracing_enabled():
            logger.info(
                "Ollama instrumentation skipped because Respan tracing is disabled"
            )
            return

        try:
            Client, AsyncClient = _load_client_classes()
        except ImportError as exc:
            # SDK genuinely absent - expected when the app doesn't use Ollama
            # (the ollama SDK is an optional extra).
            logger.debug(
                "Ollama instrumentation inactive - missing dependency: %s",
                exc,
            )
            return
        except AttributeError as exc:
            # SDK installed but incompatible (a class moved/renamed) - surface it
            # so a broken install isn't silently left untraced.
            logger.warning(
                "ollama is installed but incompatible - instrumentation inactive: %s",
                exc,
            )
            return
        except Exception as exc:
            logger.warning("Failed to activate Ollama instrumentation: %s", exc)
            return

        try:
            if _original_sync_chat is None:
                _original_sync_chat = getattr(Client, CHAT_METHOD_NAME)
                setattr(
                    Client,
                    CHAT_METHOD_NAME,
                    _wrap_sync_llm_call(
                        _original_sync_chat,
                        emit_span=emit_chat_span,
                    ),
                )

            if _original_async_chat is None:
                _original_async_chat = getattr(AsyncClient, CHAT_METHOD_NAME)
                setattr(
                    AsyncClient,
                    CHAT_METHOD_NAME,
                    _wrap_async_llm_call(
                        _original_async_chat,
                        emit_span=emit_chat_span,
                    ),
                )

            if _original_sync_generate is None:
                _original_sync_generate = getattr(Client, GENERATE_METHOD_NAME)
                setattr(
                    Client,
                    GENERATE_METHOD_NAME,
                    _wrap_sync_llm_call(
                        _original_sync_generate,
                        emit_span=emit_generate_span,
                    ),
                )

            if _original_async_generate is None:
                _original_async_generate = getattr(AsyncClient, GENERATE_METHOD_NAME)
                setattr(
                    AsyncClient,
                    GENERATE_METHOD_NAME,
                    _wrap_async_llm_call(
                        _original_async_generate,
                        emit_span=emit_generate_span,
                    ),
                )

            if _original_sync_embed is None and hasattr(Client, EMBED_METHOD_NAME):
                _original_sync_embed = getattr(Client, EMBED_METHOD_NAME)
                setattr(
                    Client,
                    EMBED_METHOD_NAME,
                    _wrap_sync_embedding_call(
                        _original_sync_embed,
                        method_name=EMBED_METHOD_NAME,
                    ),
                )

            if _original_async_embed is None and hasattr(
                AsyncClient, EMBED_METHOD_NAME
            ):
                _original_async_embed = getattr(AsyncClient, EMBED_METHOD_NAME)
                setattr(
                    AsyncClient,
                    EMBED_METHOD_NAME,
                    _wrap_async_embedding_call(
                        _original_async_embed,
                        method_name=EMBED_METHOD_NAME,
                    ),
                )

            if _original_sync_embeddings is None and hasattr(
                Client,
                EMBEDDINGS_METHOD_NAME,
            ):
                _original_sync_embeddings = getattr(Client, EMBEDDINGS_METHOD_NAME)
                setattr(
                    Client,
                    EMBEDDINGS_METHOD_NAME,
                    _wrap_sync_embedding_call(
                        _original_sync_embeddings,
                        method_name=EMBEDDINGS_METHOD_NAME,
                    ),
                )

            if _original_async_embeddings is None and hasattr(
                AsyncClient,
                EMBEDDINGS_METHOD_NAME,
            ):
                _original_async_embeddings = getattr(
                    AsyncClient, EMBEDDINGS_METHOD_NAME
                )
                setattr(
                    AsyncClient,
                    EMBEDDINGS_METHOD_NAME,
                    _wrap_async_embedding_call(
                        _original_async_embeddings,
                        method_name=EMBEDDINGS_METHOD_NAME,
                    ),
                )
        except Exception:
            logger.exception("Failed to patch Ollama client methods")
            self.deactivate()
            return

        self._is_instrumented = True
        logger.info("Ollama instrumentation activated")

    def deactivate(self) -> None:
        """Restore Ollama client methods."""
        global _original_sync_chat, _original_async_chat
        global _original_sync_generate, _original_async_generate
        global _original_sync_embed, _original_async_embed
        global _original_sync_embeddings, _original_async_embeddings

        try:
            Client, AsyncClient = _load_client_classes()
        except Exception:
            Client = AsyncClient = None

        if Client is not None:
            if _original_sync_chat is not None:
                setattr(Client, CHAT_METHOD_NAME, _original_sync_chat)
            if _original_sync_generate is not None:
                setattr(Client, GENERATE_METHOD_NAME, _original_sync_generate)
            if _original_sync_embed is not None:
                setattr(Client, EMBED_METHOD_NAME, _original_sync_embed)
            if _original_sync_embeddings is not None:
                setattr(Client, EMBEDDINGS_METHOD_NAME, _original_sync_embeddings)

        if AsyncClient is not None:
            if _original_async_chat is not None:
                setattr(AsyncClient, CHAT_METHOD_NAME, _original_async_chat)
            if _original_async_generate is not None:
                setattr(AsyncClient, GENERATE_METHOD_NAME, _original_async_generate)
            if _original_async_embed is not None:
                setattr(AsyncClient, EMBED_METHOD_NAME, _original_async_embed)
            if _original_async_embeddings is not None:
                setattr(
                    AsyncClient,
                    EMBEDDINGS_METHOD_NAME,
                    _original_async_embeddings,
                )

        _original_sync_chat = None
        _original_async_chat = None
        _original_sync_generate = None
        _original_async_generate = None
        _original_sync_embed = None
        _original_async_embed = None
        _original_sync_embeddings = None
        _original_async_embeddings = None
        self._is_instrumented = False
        logger.info("Ollama instrumentation deactivated")
