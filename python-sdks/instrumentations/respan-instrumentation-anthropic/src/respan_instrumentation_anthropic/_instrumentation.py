"""Anthropic SDK instrumentation plugin for Respan."""

from __future__ import annotations

import importlib
import logging
import time
from typing import Any

from respan_instrumentation_anthropic._constants import (
    ANTHROPIC_BETA_SESSIONS_MODULE,
    ANTHROPIC_CHAT_SPAN_NAME,
    ANTHROPIC_INSTRUMENTATION_NAME,
    ANTHROPIC_RESOURCES_MODULE,
    ASYNC_EVENTS_CLASS_NAME,
    ASYNC_MESSAGES_CLASS_NAME,
    CREATE_METHOD_NAME,
    EVENTS_CLASS_NAME,
    GET_FINAL_MESSAGE_METHOD_NAME,
    MESSAGES_CLASS_NAME,
    SESSION_ERROR_EVENT,
    STREAM_METHOD_NAME,
)
from respan_instrumentation_anthropic._managed_agents import (
    _wrap_async_events_stream,
    _wrap_sync_events_stream,
)
from respan_instrumentation_anthropic._messages import (
    _build_error_attrs,
    _emit_message_spans,
    _emit_span,
)

logger = logging.getLogger(__name__)

_original_sync_create = None
_original_async_create = None
_original_sync_stream = None
_original_async_stream = None
_original_sync_events_stream = None
_original_async_events_stream = None


def _get_module_attr(module_path: str, attr_name: str) -> Any:
    module = importlib.import_module(module_path)
    attr_value = getattr(module, attr_name, None)
    if attr_value is None:
        raise AttributeError(f"{module_path}.{attr_name}")
    return attr_value


def _load_messages_classes() -> tuple[type[Any], type[Any]]:
    return (
        _get_module_attr(
            module_path=ANTHROPIC_RESOURCES_MODULE,
            attr_name=MESSAGES_CLASS_NAME,
        ),
        _get_module_attr(
            module_path=ANTHROPIC_RESOURCES_MODULE,
            attr_name=ASYNC_MESSAGES_CLASS_NAME,
        ),
    )


def _load_events_classes() -> tuple[type[Any], type[Any]]:
    return (
        _get_module_attr(
            module_path=ANTHROPIC_BETA_SESSIONS_MODULE,
            attr_name=EVENTS_CLASS_NAME,
        ),
        _get_module_attr(
            module_path=ANTHROPIC_BETA_SESSIONS_MODULE,
            attr_name=ASYNC_EVENTS_CLASS_NAME,
        ),
    )


def _emit_message_span_safely(
    *, kwargs: dict[str, Any], message: Any, start_ns: int
) -> None:
    try:
        _emit_message_spans(kwargs=kwargs, message=message, start_ns=start_ns)
    except Exception:
        logger.debug("Failed to build Anthropic span attrs", exc_info=True)


def _emit_error_span(*, kwargs: dict[str, Any], start_ns: int, exc: Exception) -> None:
    _emit_span(
        attrs=_build_error_attrs(kwargs=kwargs),
        start_ns=start_ns,
        error_message=str(exc),
    )


def _wrap_sync_create(original: Any) -> Any:
    """Wrap ``Messages.create()`` for the sync Anthropic client."""

    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        start_ns = time.time_ns()
        try:
            message = original(self, *args, **kwargs)
        except Exception as exc:
            _emit_error_span(kwargs=kwargs, start_ns=start_ns, exc=exc)
            raise

        _emit_message_span_safely(kwargs=kwargs, message=message, start_ns=start_ns)
        return message

    return wrapper


def _wrap_async_create(original: Any) -> Any:
    """Wrap ``AsyncMessages.create()`` for the async Anthropic client."""

    async def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        start_ns = time.time_ns()
        try:
            message = await original(self, *args, **kwargs)
        except Exception as exc:
            _emit_error_span(kwargs=kwargs, start_ns=start_ns, exc=exc)
            raise

        _emit_message_span_safely(kwargs=kwargs, message=message, start_ns=start_ns)
        return message

    return wrapper


def _wrap_sync_stream(original: Any) -> Any:
    """Wrap ``Messages.stream()`` for the sync Anthropic client."""

    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        start_ns = time.time_ns()
        stream_cm = original(self, *args, **kwargs)

        class _InstrumentedStream:
            """Proxy that delegates to the real MessageStream context manager."""

            def __init__(self, cm: Any) -> None:
                self._cm = cm
                self._stream = None

            def __enter__(self) -> Any:
                self._stream = self._cm.__enter__()
                return self._stream

            def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> Any:
                result = self._cm.__exit__(exc_type, exc_val, exc_tb)
                try:
                    final_message_getter = getattr(
                        self._stream, GET_FINAL_MESSAGE_METHOD_NAME, None
                    )
                    if callable(final_message_getter):
                        _emit_message_span_safely(
                            kwargs=kwargs,
                            message=final_message_getter(),
                            start_ns=start_ns,
                        )
                    elif exc_val is not None:
                        _emit_error_span(
                            kwargs=kwargs,
                            start_ns=start_ns,
                            exc=exc_val,
                        )
                except Exception:
                    logger.debug("Failed to emit stream span", exc_info=True)
                return result

            def __iter__(self) -> Any:
                return iter(self._cm)

        return _InstrumentedStream(cm=stream_cm)

    return wrapper


def _wrap_async_stream(original: Any) -> Any:
    """Wrap ``AsyncMessages.stream()`` for the async Anthropic client."""

    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        start_ns = time.time_ns()
        stream_cm = original(self, *args, **kwargs)

        class _InstrumentedAsyncStream:
            """Proxy that delegates to the real AsyncMessageStream."""

            def __init__(self, cm: Any) -> None:
                self._cm = cm
                self._stream = None

            async def __aenter__(self) -> Any:
                self._stream = await self._cm.__aenter__()
                return self._stream

            async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> Any:
                result = await self._cm.__aexit__(exc_type, exc_val, exc_tb)
                try:
                    final_message_getter = getattr(
                        self._stream, GET_FINAL_MESSAGE_METHOD_NAME, None
                    )
                    if callable(final_message_getter):
                        _emit_message_span_safely(
                            kwargs=kwargs,
                            message=final_message_getter(),
                            start_ns=start_ns,
                        )
                    elif exc_val is not None:
                        _emit_error_span(
                            kwargs=kwargs,
                            start_ns=start_ns,
                            exc=exc_val,
                        )
                except Exception:
                    logger.debug("Failed to emit async stream span", exc_info=True)
                return result

            def __aiter__(self) -> Any:
                return self._cm.__aiter__()

        return _InstrumentedAsyncStream(cm=stream_cm)

    return wrapper


class AnthropicInstrumentor:
    """Respan instrumentor for the Anthropic SDK."""

    name = ANTHROPIC_INSTRUMENTATION_NAME

    def __init__(self) -> None:
        self._is_instrumented = False

    def activate(self) -> None:
        """Monkey-patch the Anthropic SDK."""
        global _original_sync_create, _original_async_create
        global _original_sync_stream, _original_async_stream
        global _original_sync_events_stream, _original_async_events_stream

        try:
            Messages, AsyncMessages = _load_messages_classes()
        except ImportError as exc:
            # SDK genuinely absent — expected when the app doesn't use Anthropic
            # (the anthropic SDK is an optional extra).
            logger.debug(
                "Anthropic instrumentation inactive — missing dependency: %s",
                exc,
            )
            return
        except AttributeError as exc:
            # SDK installed but incompatible (a class moved/renamed) — surface it
            # so a broken install isn't silently left untraced.
            logger.warning(
                "anthropic is installed but incompatible — instrumentation inactive: %s",
                exc,
            )
            return
        except Exception as exc:
            logger.warning("Failed to activate Anthropic instrumentation: %s", exc)
            return

        self._is_instrumented = True

        try:
            if _original_sync_create is None:
                _original_sync_create = getattr(Messages, CREATE_METHOD_NAME)
            setattr(
                Messages,
                CREATE_METHOD_NAME,
                _wrap_sync_create(original=_original_sync_create),
            )

            if _original_async_create is None:
                _original_async_create = getattr(AsyncMessages, CREATE_METHOD_NAME)
            setattr(
                AsyncMessages,
                CREATE_METHOD_NAME,
                _wrap_async_create(original=_original_async_create),
            )

            if hasattr(Messages, STREAM_METHOD_NAME):
                if _original_sync_stream is None:
                    _original_sync_stream = getattr(Messages, STREAM_METHOD_NAME)
                setattr(
                    Messages,
                    STREAM_METHOD_NAME,
                    _wrap_sync_stream(original=_original_sync_stream),
                )

            if hasattr(AsyncMessages, STREAM_METHOD_NAME):
                if _original_async_stream is None:
                    _original_async_stream = getattr(AsyncMessages, STREAM_METHOD_NAME)
                setattr(
                    AsyncMessages,
                    STREAM_METHOD_NAME,
                    _wrap_async_stream(original=_original_async_stream),
                )
        except Exception as exc:
            logger.warning("Failed to activate Anthropic instrumentation: %s", exc)
            self.deactivate()
            return

        try:
            Events, AsyncEvents = _load_events_classes()

            if _original_sync_events_stream is None:
                _original_sync_events_stream = getattr(Events, STREAM_METHOD_NAME)
            setattr(
                Events,
                STREAM_METHOD_NAME,
                _wrap_sync_events_stream(original=_original_sync_events_stream),
            )

            if _original_async_events_stream is None:
                _original_async_events_stream = getattr(AsyncEvents, STREAM_METHOD_NAME)
            setattr(
                AsyncEvents,
                STREAM_METHOD_NAME,
                _wrap_async_events_stream(original=_original_async_events_stream),
            )

            logger.info("Anthropic Managed Agents instrumentation activated")
        except (AttributeError, ImportError):
            logger.debug(
                "Managed Agents beta not available in installed anthropic SDK; skipping"
            )
        except Exception as exc:
            logger.warning(
                "Failed to activate Managed Agents instrumentation: %s", exc
            )

        logger.info("Anthropic SDK instrumentation activated")

    def deactivate(self) -> None:
        """Restore original Anthropic SDK methods."""
        global _original_sync_create, _original_async_create
        global _original_sync_stream, _original_async_stream
        global _original_sync_events_stream, _original_async_events_stream

        if not self._is_instrumented:
            return

        try:
            Messages, AsyncMessages = _load_messages_classes()

            if _original_sync_create is not None:
                setattr(Messages, CREATE_METHOD_NAME, _original_sync_create)
                _original_sync_create = None

            if _original_async_create is not None:
                setattr(AsyncMessages, CREATE_METHOD_NAME, _original_async_create)
                _original_async_create = None

            if _original_sync_stream is not None:
                setattr(Messages, STREAM_METHOD_NAME, _original_sync_stream)
                _original_sync_stream = None

            if _original_async_stream is not None:
                setattr(AsyncMessages, STREAM_METHOD_NAME, _original_async_stream)
                _original_async_stream = None
        except Exception:
            logger.debug("Failed to restore Anthropic message methods", exc_info=True)

        try:
            Events, AsyncEvents = _load_events_classes()

            if _original_sync_events_stream is not None:
                setattr(Events, STREAM_METHOD_NAME, _original_sync_events_stream)
                _original_sync_events_stream = None

            if _original_async_events_stream is not None:
                setattr(
                    AsyncEvents,
                    STREAM_METHOD_NAME,
                    _original_async_events_stream,
                )
                _original_async_events_stream = None
        except Exception:
            logger.debug("Failed to restore Anthropic managed-agent methods", exc_info=True)

        self._is_instrumented = False
        logger.info("Anthropic SDK instrumentation deactivated")
