"""LiteLLM instrumentation plugin for Respan."""

from __future__ import annotations

from datetime import datetime, timezone
import importlib
import logging
from types import SimpleNamespace
from typing import Any

from respan_instrumentation_litellm._callback import RespanLiteLLMCallback
from respan_instrumentation_litellm._callback import _current_otel_parent
from respan_instrumentation_litellm._constants import (
    CONTENT_KEY,
    DELTA_KEY,
    LITELLM_INSTRUMENTATION_NAME,
    MESSAGE_KEY,
    METADATA_KEY,
    MODEL_KEY,
    RESPAN_SKIP_CALLBACK_KEY,
    STREAM_KEY,
    TOOL_CALLS_KEY,
    USAGE_KEY,
)
from respan_tracing.core.tracer import RespanTracer

logger = logging.getLogger(__name__)


class _StreamAccumulator:
    def __init__(self, *, model: Any = None) -> None:
        self.model = model
        self.content_parts: list[str] = []
        self.usage: Any = None
        self.tool_calls: Any = None

    @staticmethod
    def _get(value: Any, key: str, default: Any = None) -> Any:
        if isinstance(value, dict):
            return value.get(key, default)
        return getattr(value, key, default)

    def add(self, chunk: Any) -> None:
        if self.model is None:
            self.model = self._get(chunk, MODEL_KEY)

        usage = self._get(chunk, USAGE_KEY)
        if usage is not None:
            self.usage = usage

        choices = self._get(chunk, "choices", []) or []
        if not choices:
            return

        choice = choices[0]
        message = self._get(choice, DELTA_KEY) or self._get(choice, MESSAGE_KEY)
        content = self._get(message, CONTENT_KEY)
        if content:
            self.content_parts.append(str(content))

        tool_calls = self._get(message, TOOL_CALLS_KEY)
        if tool_calls:
            self.tool_calls = tool_calls

    def response(self) -> Any:
        usage = self.usage
        if isinstance(usage, dict):
            usage = SimpleNamespace(**usage)
        content = "".join(self.content_parts)
        return SimpleNamespace(
            model=self.model,
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=content,
                        tool_calls=self.tool_calls,
                    )
                )
            ],
            usage=usage,
        )


def _callback_list(callbacks: Any) -> list[Any]:
    if callbacks is None:
        return []
    if isinstance(callbacks, list):
        return callbacks
    if isinstance(callbacks, tuple):
        return list(callbacks)
    return [callbacks]


def _with_skip_marker(kwargs: dict[str, Any]) -> dict[str, Any]:
    marked_kwargs = dict(kwargs)
    metadata = marked_kwargs.get(METADATA_KEY)
    if isinstance(metadata, dict):
        metadata = dict(metadata)
    elif metadata is None:
        metadata = {}
    else:
        metadata = {"value": metadata}
    metadata[RESPAN_SKIP_CALLBACK_KEY] = True
    marked_kwargs[METADATA_KEY] = metadata
    return marked_kwargs


class _InstrumentedStream:
    def __init__(
        self,
        *,
        stream: Any,
        callback: RespanLiteLLMCallback,
        kwargs: dict[str, Any],
        start_time: datetime,
        parent_context: tuple[str | None, str | None],
    ) -> None:
        self._stream = stream
        self._iterator = iter(stream)
        self._callback = callback
        self._kwargs = kwargs
        self._start_time = start_time
        self._accumulator = _StreamAccumulator(model=kwargs.get(MODEL_KEY))
        self._parent_context = parent_context
        self._emitted = False

    def __iter__(self):
        return self

    def __next__(self):
        try:
            chunk = next(self._iterator)
        except StopIteration:
            self._emit_once(error=None)
            raise
        except Exception as exc:
            self._emit_once(error=exc)
            raise
        self._accumulator.add(chunk=chunk)
        return chunk

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)

    def _emit_once(self, *, error: Exception | None) -> None:
        if self._emitted:
            return
        self._emitted = True
        self._callback._emit_event(
            kwargs=self._kwargs,
            response_obj=self._accumulator.response(),
            start_time=self._start_time,
            end_time=datetime.now(timezone.utc),
            error=error,
            parent_context=self._parent_context,
        )


class _AsyncInstrumentedStream:
    def __init__(
        self,
        *,
        stream: Any,
        callback: RespanLiteLLMCallback,
        kwargs: dict[str, Any],
        start_time: datetime,
        parent_context: tuple[str | None, str | None],
    ) -> None:
        self._stream = stream
        self._aiterator = stream.__aiter__()
        self._callback = callback
        self._kwargs = kwargs
        self._start_time = start_time
        self._accumulator = _StreamAccumulator(model=kwargs.get(MODEL_KEY))
        self._parent_context = parent_context
        self._emitted = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            chunk = await self._aiterator.__anext__()
        except StopAsyncIteration:
            self._emit_once(error=None)
            raise
        except Exception as exc:
            self._emit_once(error=exc)
            raise
        self._accumulator.add(chunk=chunk)
        return chunk

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)

    def _emit_once(self, *, error: Exception | None) -> None:
        if self._emitted:
            return
        self._emitted = True
        self._callback._emit_event(
            kwargs=self._kwargs,
            response_obj=self._accumulator.response(),
            start_time=self._start_time,
            end_time=datetime.now(timezone.utc),
            error=error,
            parent_context=self._parent_context,
        )


class LiteLLMInstrumentor:
    """Respan instrumentor for LiteLLM.

    Registers a LiteLLM ``CustomLogger`` callback. LiteLLM invokes the callback
    for sync and async completion events, and the callback injects canonical
    chat spans into the active Respan tracing pipeline. Streaming completions
    are wrapped so a span is emitted after the stream has been consumed.
    """

    name = LITELLM_INSTRUMENTATION_NAME

    def __init__(self, *, include_content: bool = True) -> None:
        self._include_content = include_content
        self._callback: RespanLiteLLMCallback | None = None
        self._litellm_module: Any = None
        self._original_completion: Any = None
        self._original_acompletion: Any = None
        self._is_instrumented = False

    @staticmethod
    def _is_respan_tracing_enabled() -> bool:
        tracer = getattr(RespanTracer, "_instance", None)
        if tracer is None:
            return True
        return bool(getattr(tracer, "is_enabled", True))

    def activate(self) -> None:
        """Register the Respan LiteLLM callback and streaming wrappers."""
        if self._is_instrumented:
            return

        if not self._is_respan_tracing_enabled():
            logger.info(
                "LiteLLM instrumentation skipped because Respan tracing is disabled"
            )
            return

        try:
            litellm_module = importlib.import_module("litellm")
        except ImportError as exc:
            logger.warning(
                "Failed to activate LiteLLM instrumentation - missing dependency: %s",
                exc,
            )
            return

        callback = RespanLiteLLMCallback(include_content=self._include_content)
        callbacks = _callback_list(getattr(litellm_module, "callbacks", []))
        if callback not in callbacks:
            callbacks.append(callback)
        setattr(litellm_module, "callbacks", callbacks)

        self._callback = callback
        self._litellm_module = litellm_module
        self._patch_streaming_methods(litellm_module=litellm_module, callback=callback)
        self._is_instrumented = True
        logger.info("LiteLLM instrumentation activated")

    def _patch_streaming_methods(
        self,
        *,
        litellm_module: Any,
        callback: RespanLiteLLMCallback,
    ) -> None:
        self._original_completion = getattr(litellm_module, "completion", None)
        self._original_acompletion = getattr(litellm_module, "acompletion", None)

        if self._original_completion is not None:
            original_completion = self._original_completion

            def completion_wrapper(*args: Any, **kwargs: Any) -> Any:
                if not kwargs.get(STREAM_KEY):
                    return original_completion(*args, **kwargs)

                event_kwargs = dict(kwargs)
                start_time = datetime.now(timezone.utc)
                parent_context = _current_otel_parent()
                try:
                    stream = original_completion(*args, **_with_skip_marker(kwargs))
                except Exception as exc:
                    callback._emit_event(
                        kwargs=event_kwargs,
                        response_obj=None,
                        start_time=start_time,
                        end_time=datetime.now(timezone.utc),
                        error=exc,
                        parent_context=parent_context,
                    )
                    raise
                return _InstrumentedStream(
                    stream=stream,
                    callback=callback,
                    kwargs=event_kwargs,
                    start_time=start_time,
                    parent_context=parent_context,
                )

            setattr(litellm_module, "completion", completion_wrapper)

        if self._original_acompletion is not None:
            original_acompletion = self._original_acompletion

            async def acompletion_wrapper(*args: Any, **kwargs: Any) -> Any:
                if not kwargs.get(STREAM_KEY):
                    return await original_acompletion(*args, **kwargs)

                event_kwargs = dict(kwargs)
                start_time = datetime.now(timezone.utc)
                parent_context = _current_otel_parent()
                try:
                    stream = await original_acompletion(
                        *args, **_with_skip_marker(kwargs)
                    )
                except Exception as exc:
                    callback._emit_event(
                        kwargs=event_kwargs,
                        response_obj=None,
                        start_time=start_time,
                        end_time=datetime.now(timezone.utc),
                        error=exc,
                        parent_context=parent_context,
                    )
                    raise
                return _AsyncInstrumentedStream(
                    stream=stream,
                    callback=callback,
                    kwargs=event_kwargs,
                    start_time=start_time,
                    parent_context=parent_context,
                )

            setattr(litellm_module, "acompletion", acompletion_wrapper)

    def deactivate(self) -> None:
        """Remove the Respan LiteLLM callback and restore wrapped methods."""
        if self._litellm_module is not None:
            if self._original_completion is not None:
                setattr(self._litellm_module, "completion", self._original_completion)
            if self._original_acompletion is not None:
                setattr(self._litellm_module, "acompletion", self._original_acompletion)

        if self._litellm_module is not None and self._callback is not None:
            callbacks = _callback_list(getattr(self._litellm_module, "callbacks", []))
            callbacks = [
                existing_callback
                for existing_callback in callbacks
                if existing_callback is not self._callback
            ]
            setattr(self._litellm_module, "callbacks", callbacks)

        self._callback = None
        self._litellm_module = None
        self._original_completion = None
        self._original_acompletion = None
        self._is_instrumented = False
        logger.info("LiteLLM instrumentation deactivated")


LitellmInstrumentor = LiteLLMInstrumentor
