"""OpenAI Agents SDK instrumentation plugin for Respan."""

from __future__ import annotations

import contextvars
import functools
import logging
import threading
import time
from collections import OrderedDict
from collections.abc import AsyncIterator, Mapping
from typing import Any

from agents.tracing.processor_interface import TracingProcessor
from agents.tracing.span_data import AgentSpanData
from agents.tracing.spans import Span
from agents.tracing.traces import Trace
from opentelemetry import trace as otel_trace

from respan_instrumentation_openai_agents._otel_emitter import (
    clear_trace,
    clear_traces,
    emit_sdk_item,
    register_trace,
)

logger = logging.getLogger(__name__)

_MAX_SEEN_ITEMS = 8_192
_STREAMING: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "respan_openai_agents_streaming", default=False
)
_STREAM_PATCH_LOCK = threading.RLock()
_STREAM_ORIGINALS: dict[tuple[type[Any], str], Any] = {}


def _stream_patch_targets() -> tuple[type[Any], ...]:
    targets: list[type[Any]] = []
    try:
        from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel

        targets.append(OpenAIChatCompletionsModel)
    except ImportError:
        pass
    try:
        from agents.models.openai_responses import OpenAIResponsesModel

        targets.append(OpenAIResponsesModel)
    except ImportError:
        pass
    return tuple(targets)


def _wrap_stream_method(original: Any) -> Any:
    @functools.wraps(original)
    def wrapped(*args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        source = original(*args, **kwargs)

        async def iterate() -> AsyncIterator[Any]:
            token = _STREAMING.set(True)
            active_error: BaseException | None = None
            try:
                async for item in source:
                    yield item
            except BaseException as exc:
                if not isinstance(exc, GeneratorExit):
                    active_error = exc
                raise
            finally:
                try:
                    close = getattr(source, "aclose", None)
                    if callable(close):
                        await close()
                except BaseException:
                    if active_error is None:
                        raise
                    logger.debug(
                        "Failed to close OpenAI Agents streaming source",
                        exc_info=True,
                    )
                finally:
                    _STREAMING.reset(token)

        return iterate()

    wrapped.__respan_openai_agents_stream_patch__ = True
    return wrapped


def _install_stream_patches() -> None:
    """Mark native OpenAI model streaming spans without changing SDK output."""
    with _STREAM_PATCH_LOCK:
        for target in _stream_patch_targets():
            key = (target, "stream_response")
            if key in _STREAM_ORIGINALS:
                continue
            original = getattr(target, "stream_response", None)
            if not callable(original):
                continue
            wrapped = _wrap_stream_method(original)
            target.stream_response = wrapped
            _STREAM_ORIGINALS[key] = (original, wrapped)


def _remove_stream_patches() -> None:
    with _STREAM_PATCH_LOCK:
        for (target, name), (original, wrapped) in tuple(_STREAM_ORIGINALS.items()):
            if getattr(target, name, None) is wrapped:
                setattr(target, name, original)
            _STREAM_ORIGINALS.pop((target, name), None)


class _RespanTracingProcessor(TracingProcessor):
    """Convert completed OpenAI Agents items into the active OTEL pipeline."""

    def __init__(self, *, metadata: Mapping[str, Any] | None = None) -> None:
        self._metadata = dict(metadata or {})
        self._lock = threading.RLock()
        self._seen: OrderedDict[tuple[str, str], None] = OrderedDict()
        self._open_spans: dict[tuple[str, str], Span[Any]] = {}
        self._trace_starts: dict[str, int] = {}

    def _mark_once(self, kind: str, identifier: str) -> bool:
        key = (kind, identifier)
        with self._lock:
            if key in self._seen:
                return False
            self._seen[key] = None
            self._seen.move_to_end(key)
            while len(self._seen) > _MAX_SEEN_ITEMS:
                self._seen.popitem(last=False)
        return True

    def _agent_context(self, span: Span[Any]) -> dict[str, Any]:
        trace_id = span.trace_id
        parent_id = span.parent_id
        with self._lock:
            for _ in range(32):
                if not parent_id:
                    break
                parent = self._open_spans.get((trace_id, parent_id))
                if parent is None:
                    break
                data = parent.span_data
                if isinstance(data, AgentSpanData):
                    return {
                        "handoffs": tuple(data.handoffs or ()),
                        "output_type": data.output_type,
                        "tools": tuple(data.tools or ()),
                    }
                parent_id = parent.parent_id
        return {}

    def on_trace_start(self, trace: Trace) -> None:
        with self._lock:
            self._trace_starts.setdefault(trace.trace_id, time.time_ns())
        register_trace(trace)

    def on_trace_end(self, trace: Trace) -> None:
        end_ns = time.time_ns()
        with self._lock:
            start_ns = self._trace_starts.pop(trace.trace_id, end_ns)
        if not self._mark_once("trace", trace.trace_id):
            return
        emit_sdk_item(
            trace,
            extra_metadata=self._metadata,
            trace_start_ns=start_ns,
            trace_end_ns=end_ns,
        )
        clear_trace(trace.trace_id)
        with self._lock:
            stale = [key for key in self._open_spans if key[0] == trace.trace_id]
            for key in stale:
                self._open_spans.pop(key, None)

    def on_span_start(self, span: Span[Any]) -> None:
        with self._lock:
            self._open_spans[(span.trace_id, span.span_id)] = span

    def on_span_end(self, span: Span[Any]) -> None:
        if not self._mark_once("span", f"{span.trace_id}:{span.span_id}"):
            return
        agent_context = self._agent_context(span)
        emit_sdk_item(
            span,
            extra_metadata=self._metadata,
            agent_context=agent_context,
            is_streaming=_STREAMING.get(),
        )
        with self._lock:
            self._open_spans.pop((span.trace_id, span.span_id), None)

    def shutdown(self) -> None:
        self.force_flush()
        with self._lock:
            self._open_spans.clear()
            self._trace_starts.clear()
            self._seen.clear()
        clear_traces()

    def force_flush(self) -> None:
        force_flush = getattr(otel_trace.get_tracer_provider(), "force_flush", None)
        if callable(force_flush):
            try:
                force_flush()
            except Exception:
                logger.exception("Failed to flush OpenAI Agents OTEL spans")


_LIFECYCLE_LOCK = threading.RLock()
_SHARED_PROCESSOR: _RespanTracingProcessor | None = None
_ACTIVATION_COUNT = 0
_PREVIOUS_PROCESSORS: tuple[TracingProcessor, ...] = ()


def _current_processors() -> tuple[TracingProcessor, ...]:
    from agents.tracing import get_trace_provider

    provider = get_trace_provider()
    multi = getattr(provider, "_multi_processor", None)
    processors = getattr(multi, "_processors", ())
    return tuple(processors) if processors is not None else ()


class OpenAIAgentsInstrumentor:
    """Install one ref-counted OpenAI Agents tracing processor per process."""

    name = "openai-agents"

    def __init__(self) -> None:
        self._processor: _RespanTracingProcessor | None = None
        self._is_active = False

    def activate(self) -> None:
        """Replace the OpenAI tracing backend while preserving it for deactivation."""
        global _ACTIVATION_COUNT, _PREVIOUS_PROCESSORS, _SHARED_PROCESSOR

        with _LIFECYCLE_LOCK:
            if self._is_active:
                return
            if _ACTIVATION_COUNT:
                _ACTIVATION_COUNT += 1
                self._processor = _SHARED_PROCESSOR
                self._is_active = True
                return

            from agents.tracing import set_trace_processors

            previous = _current_processors()
            processor = _RespanTracingProcessor()
            try:
                _install_stream_patches()
                set_trace_processors([processor])
            except Exception:
                _remove_stream_patches()
                try:
                    set_trace_processors(list(previous))
                except Exception:
                    logger.exception("Failed to roll back OpenAI Agents processors")
                logger.exception("Failed to activate OpenAI Agents instrumentation")
                return

            _PREVIOUS_PROCESSORS = previous
            _SHARED_PROCESSOR = processor
            _ACTIVATION_COUNT = 1
            self._processor = processor
            self._is_active = True
            logger.info("OpenAI Agents SDK instrumentation activated")

    def deactivate(self) -> None:
        """Release this activation and restore the prior SDK processors last."""
        global _ACTIVATION_COUNT, _PREVIOUS_PROCESSORS, _SHARED_PROCESSOR

        with _LIFECYCLE_LOCK:
            if not self._is_active:
                return
            self._is_active = False
            self._processor = None
            _ACTIVATION_COUNT -= 1
            if _ACTIVATION_COUNT:
                return

            from agents.tracing import set_trace_processors

            processor = _SHARED_PROCESSOR
            current = _current_processors()
            replacement_items: list[TracingProcessor] = []
            for item in (*_PREVIOUS_PROCESSORS, *current):
                if item is processor:
                    continue
                if any(existing is item for existing in replacement_items):
                    continue
                replacement_items.append(item)
            replacement = tuple(replacement_items)
            try:
                set_trace_processors(list(replacement))
            finally:
                _remove_stream_patches()
                if processor is not None:
                    processor.shutdown()
                _SHARED_PROCESSOR = None
                _PREVIOUS_PROCESSORS = ()
                logger.info("OpenAI Agents SDK instrumentation deactivated")
