"""Aleph Alpha SDK instrumentation plugin for Respan."""

from __future__ import annotations

import importlib
import logging
import time
from collections.abc import AsyncIterator
from typing import Any

from respan_instrumentation_aleph_alpha._constants import (
    ALEPH_ALPHA_CLIENT_MODULE,
    ALEPH_ALPHA_INSTRUMENTATION_NAME,
    ASYNC_CLIENT_CLASS_NAME,
    CLIENT_CLASS_NAME,
    METHOD_BATCH_SEMANTIC_EMBED,
    METHOD_CHAT,
    METHOD_CHAT_WITH_STREAMING,
    METHOD_COMPLETE,
    METHOD_COMPLETE_WITH_STREAMING,
    METHOD_EMBED,
    METHOD_EMBEDDINGS,
    METHOD_EVALUATE,
    METHOD_EXPLAIN,
    METHOD_INSTRUCTABLE_EMBED,
    METHOD_SEMANTIC_EMBED,
    MODEL_KEY,
    OPERATION_BATCH_SEMANTIC_EMBED,
    OPERATION_CHAT,
    OPERATION_CHAT_STREAM,
    OPERATION_COMPLETE,
    OPERATION_COMPLETE_STREAM,
    OPERATION_EMBED,
    OPERATION_EMBEDDINGS,
    OPERATION_EVALUATE,
    OPERATION_EXPLAIN,
    OPERATION_INSTRUCTABLE_EMBED,
    OPERATION_SEMANTIC_EMBED,
    REQUEST_KEY,
)
from respan_instrumentation_aleph_alpha._otel_emitter import emit_aleph_alpha_span
from respan_tracing.core.tracer import RespanTracer

logger = logging.getLogger(__name__)

_original_methods: dict[tuple[str, str], Any] = {}


def _get_module_attr(module_path: str, attr_name: str) -> Any:
    module = importlib.import_module(module_path)
    attr_value = getattr(module, attr_name, None)
    if attr_value is None:
        raise AttributeError(f"{module_path}.{attr_name}")
    return attr_value


def _load_client_classes() -> tuple[type[Any], type[Any]]:
    return (
        _get_module_attr(ALEPH_ALPHA_CLIENT_MODULE, CLIENT_CLASS_NAME),
        _get_module_attr(ALEPH_ALPHA_CLIENT_MODULE, ASYNC_CLIENT_CLASS_NAME),
    )


def _request_model_from_call(args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[Any, str | None]:
    request = kwargs.get(REQUEST_KEY)
    if request is None and args:
        request = args[0]
    model = kwargs.get(MODEL_KEY)
    if model is None and len(args) > 1:
        model = args[1]
    return request, str(model) if model is not None else None


def _emit_span_safely(
    *,
    operation: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    start_ns: int,
    response_or_items: Any = None,
    error_message: str | None = None,
    status_code: int = 200,
) -> None:
    request, model = _request_model_from_call(args=args, kwargs=kwargs)
    emit_aleph_alpha_span(
        operation=operation,
        request=request,
        model=model,
        start_ns=start_ns,
        response_or_items=response_or_items,
        error_message=error_message,
        status_code=status_code,
    )


def _wrap_sync_method(original: Any, operation: str) -> Any:
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        start_ns = time.time_ns()
        try:
            response = original(self, *args, **kwargs)
        except Exception as exc:
            _emit_span_safely(
                operation=operation,
                args=args,
                kwargs=kwargs,
                start_ns=start_ns,
                error_message=str(exc),
                status_code=500,
            )
            raise

        _emit_span_safely(
            operation=operation,
            args=args,
            kwargs=kwargs,
            start_ns=start_ns,
            response_or_items=response,
        )
        return response

    return wrapper


def _wrap_async_method(original: Any, operation: str) -> Any:
    async def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        start_ns = time.time_ns()
        try:
            response = await original(self, *args, **kwargs)
        except Exception as exc:
            _emit_span_safely(
                operation=operation,
                args=args,
                kwargs=kwargs,
                start_ns=start_ns,
                error_message=str(exc),
                status_code=500,
            )
            raise

        _emit_span_safely(
            operation=operation,
            args=args,
            kwargs=kwargs,
            start_ns=start_ns,
            response_or_items=response,
        )
        return response

    return wrapper


async def _instrument_async_stream(
    *,
    async_iterator: AsyncIterator[Any],
    operation: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    start_ns: int,
) -> AsyncIterator[Any]:
    items: list[Any] = []
    try:
        async for item in async_iterator:
            items.append(item)
            yield item
    except Exception as exc:
        _emit_span_safely(
            operation=operation,
            args=args,
            kwargs=kwargs,
            start_ns=start_ns,
            response_or_items=items,
            error_message=str(exc),
            status_code=500,
        )
        raise
    else:
        _emit_span_safely(
            operation=operation,
            args=args,
            kwargs=kwargs,
            start_ns=start_ns,
            response_or_items=items,
        )


def _wrap_async_stream_method(original: Any, operation: str) -> Any:
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        start_ns = time.time_ns()
        try:
            async_iterator = original(self, *args, **kwargs)
        except Exception as exc:
            _emit_span_safely(
                operation=operation,
                args=args,
                kwargs=kwargs,
                start_ns=start_ns,
                error_message=str(exc),
                status_code=500,
            )
            raise
        return _instrument_async_stream(
            async_iterator=async_iterator,
            operation=operation,
            args=args,
            kwargs=kwargs,
            start_ns=start_ns,
        )

    return wrapper


def _patch_method(
    *,
    class_name: str,
    cls: type[Any],
    method_name: str,
    operation: str,
    wrapper_factory: Any,
) -> None:
    key = (class_name, method_name)
    if _original_methods.get(key) is None:
        _original_methods[key] = getattr(cls, method_name)
    setattr(cls, method_name, wrapper_factory(_original_methods[key], operation))


class AlephAlphaInstrumentor:
    """Respan instrumentor for the Aleph Alpha Python SDK."""

    name = ALEPH_ALPHA_INSTRUMENTATION_NAME

    def __init__(self) -> None:
        self._is_instrumented = False

    @staticmethod
    def _is_respan_tracing_enabled() -> bool:
        tracer = getattr(RespanTracer, "_instance", None)
        if tracer is None:
            return True
        return bool(getattr(tracer, "is_enabled", True))

    def activate(self) -> None:
        """Monkey-patch Aleph Alpha client model methods."""
        if self._is_instrumented:
            return

        if not self._is_respan_tracing_enabled():
            logger.info(
                "Aleph Alpha instrumentation skipped because Respan tracing is disabled"
            )
            return

        try:
            Client, AsyncClient = _load_client_classes()
        except (AttributeError, ImportError) as exc:
            logger.warning(
                "Failed to activate Aleph Alpha instrumentation - missing dependency: %s",
                exc,
            )
            return
        except Exception as exc:
            logger.warning("Failed to activate Aleph Alpha instrumentation: %s", exc)
            return

        try:
            for method_name, operation in (
                (METHOD_COMPLETE, OPERATION_COMPLETE),
                (METHOD_CHAT, OPERATION_CHAT),
                (METHOD_EMBED, OPERATION_EMBED),
                (METHOD_EMBEDDINGS, OPERATION_EMBEDDINGS),
                (METHOD_SEMANTIC_EMBED, OPERATION_SEMANTIC_EMBED),
                (METHOD_BATCH_SEMANTIC_EMBED, OPERATION_BATCH_SEMANTIC_EMBED),
                (METHOD_INSTRUCTABLE_EMBED, OPERATION_INSTRUCTABLE_EMBED),
                (METHOD_EVALUATE, OPERATION_EVALUATE),
                (METHOD_EXPLAIN, OPERATION_EXPLAIN),
            ):
                if hasattr(Client, method_name):
                    _patch_method(
                        class_name=CLIENT_CLASS_NAME,
                        cls=Client,
                        method_name=method_name,
                        operation=operation,
                        wrapper_factory=_wrap_sync_method,
                    )

            for method_name, operation in (
                (METHOD_COMPLETE, OPERATION_COMPLETE),
                (METHOD_CHAT, OPERATION_CHAT),
                (METHOD_EMBED, OPERATION_EMBED),
                (METHOD_SEMANTIC_EMBED, OPERATION_SEMANTIC_EMBED),
                (METHOD_BATCH_SEMANTIC_EMBED, OPERATION_BATCH_SEMANTIC_EMBED),
                (METHOD_INSTRUCTABLE_EMBED, OPERATION_INSTRUCTABLE_EMBED),
                (METHOD_EVALUATE, OPERATION_EVALUATE),
                (METHOD_EXPLAIN, OPERATION_EXPLAIN),
            ):
                if hasattr(AsyncClient, method_name):
                    _patch_method(
                        class_name=ASYNC_CLIENT_CLASS_NAME,
                        cls=AsyncClient,
                        method_name=method_name,
                        operation=operation,
                        wrapper_factory=_wrap_async_method,
                    )

            for method_name, operation in (
                (METHOD_COMPLETE_WITH_STREAMING, OPERATION_COMPLETE_STREAM),
                (METHOD_CHAT_WITH_STREAMING, OPERATION_CHAT_STREAM),
            ):
                if hasattr(AsyncClient, method_name):
                    _patch_method(
                        class_name=ASYNC_CLIENT_CLASS_NAME,
                        cls=AsyncClient,
                        method_name=method_name,
                        operation=operation,
                        wrapper_factory=_wrap_async_stream_method,
                    )
        except Exception as exc:
            logger.warning("Failed to activate Aleph Alpha instrumentation: %s", exc)
            self.deactivate()
            return

        self._is_instrumented = True
        logger.info("Aleph Alpha SDK instrumentation activated")

    def deactivate(self) -> None:
        """Restore original Aleph Alpha SDK methods."""
        if not self._is_instrumented and not _original_methods:
            return

        try:
            Client, AsyncClient = _load_client_classes()
            class_map = {
                CLIENT_CLASS_NAME: Client,
                ASYNC_CLIENT_CLASS_NAME: AsyncClient,
            }
            for (class_name, method_name), original in list(_original_methods.items()):
                setattr(class_map[class_name], method_name, original)
                _original_methods.pop((class_name, method_name), None)
        except Exception:
            logger.debug("Failed to restore Aleph Alpha methods", exc_info=True)

        self._is_instrumented = False
        logger.info("Aleph Alpha SDK instrumentation deactivated")
