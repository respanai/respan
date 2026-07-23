"""Thread-safe assembly of asynchronous CrewAI lifecycle events into spans."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
import logging
import threading
import time
from typing import Any

from opentelemetry import context as context_api
from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.trace import Status, StatusCode

from respan_sdk.constants import ERROR_MESSAGE_ATTR

from respan_instrumentation_crewai._constants import (
    MAX_BUFFERED_ENTRIES,
    MAX_OPEN_SPAN_AGE_SECONDS,
)
from respan_instrumentation_crewai._serialization import json_attribute

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SpanStartSpec:
    """Everything needed to start one canonical Respan span."""

    name: str
    attributes: dict[str, Any] = field(default_factory=dict)
    parent_hint_keys: tuple[str, ...] = ()
    remember_hint_keys: tuple[str, ...] = ()
    correlation_keys: tuple[str, ...] = ()


@dataclass(frozen=True)
class SpanEndSpec:
    """Everything learned when a CrewAI operation finishes."""

    output: Any = None
    error: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _ParentInfo:
    context: Context
    workflow_name: str | None = None


@dataclass
class _TransparentScope:
    parent_info: _ParentInfo
    opened_at_monotonic: float


@dataclass
class _OpenSpan:
    span: trace.Span
    parent_info: _ParentInfo
    opened_at_monotonic: float
    correlation_keys: tuple[str, ...]


@dataclass(frozen=True)
class _PendingEnd:
    spec: SpanEndSpec
    end_time_ns: int | None


@dataclass(frozen=True)
class _PendingStart:
    event: Any
    spec: SpanStartSpec | None
    ambient_parent: _ParentInfo


class CrewAIEventAssembler:
    """Correlate CrewAI start/end events without relying on thread-local spans."""

    def __init__(self, tracer: trace.Tracer) -> None:
        self._tracer = tracer
        self._open_spans: OrderedDict[str, _OpenSpan] = OrderedDict()
        self._transparent_scopes: OrderedDict[str, _TransparentScope] = OrderedDict()
        self._finished_contexts: OrderedDict[str, _ParentInfo] = OrderedDict()
        self._context_hints: OrderedDict[str, _ParentInfo] = OrderedDict()
        self._correlations: OrderedDict[str, str] = OrderedDict()
        self._pending_starts: OrderedDict[str, list[_PendingStart]] = OrderedDict()
        self._pending_start_count = 0
        self._pending_ends: OrderedDict[str, _PendingEnd] = OrderedDict()
        self._pending_correlation_ends: OrderedDict[str, _PendingEnd] = OrderedDict()
        self._pending_scope_ends: OrderedDict[str, None] = OrderedDict()
        self._lock = threading.RLock()

    @staticmethod
    def _time_ns(value: Any) -> int | None:
        if not isinstance(value, datetime):
            return None
        if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
            return None
        return int(value.astimezone(timezone.utc).timestamp() * 1_000_000_000)

    @classmethod
    def _start_time_ns(cls, event: Any) -> int | None:
        return cls._time_ns(getattr(event, "timestamp", None))

    @classmethod
    def _end_time_ns(cls, event: Any) -> int | None:
        return cls._time_ns(getattr(event, "finished_at", None)) or cls._time_ns(
            getattr(event, "timestamp", None)
        )

    @staticmethod
    def _event_id(event: Any) -> str | None:
        value = getattr(event, "event_id", None)
        return str(value) if value else None

    @staticmethod
    def _parent_event_id(event: Any) -> str | None:
        value = getattr(event, "parent_event_id", None)
        return str(value) if value else None

    @staticmethod
    def _ambient_parent() -> _ParentInfo:
        current_context = context_api.get_current()
        workflow_name = context_api.get_value(
            SpanAttributes.TRACELOOP_WORKFLOW_NAME,
            context=current_context,
        )
        if not workflow_name:
            workflow_name = context_api.get_value(
                SpanAttributes.TRACELOOP_ENTITY_NAME,
                context=current_context,
            )
        return _ParentInfo(
            context=current_context,
            workflow_name=str(workflow_name) if workflow_name else None,
        )

    def _trim_cache_locked(self, cache: OrderedDict[Any, Any], label: str) -> None:
        while len(cache) > MAX_BUFFERED_ENTRIES:
            evicted_key, _ = cache.popitem(last=False)
            logger.warning(
                "Evicting oldest CrewAI %s entry for %s after reaching %d entries",
                label,
                evicted_key,
                MAX_BUFFERED_ENTRIES,
            )

    def _remember_finished_locked(self, event_id: str, info: _ParentInfo) -> None:
        self._finished_contexts.pop(event_id, None)
        self._finished_contexts[event_id] = info
        self._trim_cache_locked(self._finished_contexts, "finished context")

    def _remember_hints_locked(
        self,
        hint_keys: tuple[str, ...],
        info: _ParentInfo,
    ) -> None:
        for hint_key in hint_keys:
            if not hint_key:
                continue
            self._context_hints.pop(hint_key, None)
            self._context_hints[hint_key] = info
        self._trim_cache_locked(self._context_hints, "context hint")

    def _remember_correlations_locked(
        self,
        event_id: str,
        correlation_keys: tuple[str, ...],
    ) -> None:
        for correlation_key in correlation_keys:
            if not correlation_key:
                continue
            self._correlations.pop(correlation_key, None)
            self._correlations[correlation_key] = event_id
        self._trim_cache_locked(self._correlations, "correlation")

    def _forget_correlations_locked(
        self,
        event_id: str,
        entry: _OpenSpan,
    ) -> None:
        for correlation_key in entry.correlation_keys:
            if self._correlations.get(correlation_key) == event_id:
                self._correlations.pop(correlation_key, None)

    def _context_for_event_locked(self, event_id: str | None) -> _ParentInfo | None:
        if not event_id:
            return None
        entry = self._open_spans.get(event_id)
        if entry is not None:
            return entry.parent_info
        scope = self._transparent_scopes.get(event_id)
        if scope is not None:
            return scope.parent_info
        finished = self._finished_contexts.pop(event_id, None)
        if finished is not None:
            self._finished_contexts[event_id] = finished
        return finished

    def _hint_context_locked(self, hint_keys: tuple[str, ...]) -> _ParentInfo | None:
        for hint_key in hint_keys:
            info = self._context_hints.pop(hint_key, None)
            if info is not None:
                self._context_hints[hint_key] = info
                return info
        return None

    def _mark_abandoned(self, entry: _OpenSpan, message: str) -> None:
        try:
            entry.span.set_attribute(ERROR_MESSAGE_ATTR, message)
            entry.span.set_status(Status(StatusCode.ERROR, message))
            entry.span.end()
        except Exception:
            logger.debug("Failed to close abandoned CrewAI span", exc_info=True)

    def _evict_stale_locked(self) -> None:
        cutoff = time.monotonic() - MAX_OPEN_SPAN_AGE_SECONDS
        while self._open_spans:
            event_id, entry = next(iter(self._open_spans.items()))
            if entry.opened_at_monotonic > cutoff:
                break
            self._open_spans.pop(event_id, None)
            self._forget_correlations_locked(event_id, entry)
            self._remember_finished_locked(event_id, entry.parent_info)
            self._mark_abandoned(
                entry,
                "CrewAI lifecycle span exceeded the maximum open age",
            )

        while len(self._open_spans) >= MAX_BUFFERED_ENTRIES:
            event_id, entry = self._open_spans.popitem(last=False)
            self._forget_correlations_locked(event_id, entry)
            self._remember_finished_locked(event_id, entry.parent_info)
            self._mark_abandoned(
                entry,
                "CrewAI lifecycle span evicted after reaching the buffer limit",
            )

        while self._transparent_scopes:
            scope_id, scope = next(iter(self._transparent_scopes.items()))
            if scope.opened_at_monotonic > cutoff:
                break
            self._transparent_scopes.pop(scope_id, None)
            self._remember_finished_locked(scope_id, scope.parent_info)

        while len(self._transparent_scopes) >= MAX_BUFFERED_ENTRIES:
            scope_id, scope = self._transparent_scopes.popitem(last=False)
            self._remember_finished_locked(scope_id, scope.parent_info)
            logger.warning(
                "Evicting oldest CrewAI transparent scope for %s",
                scope_id,
            )

    def _resolve_parent_locked(
        self,
        event: Any,
        hint_keys: tuple[str, ...],
    ) -> tuple[str | None, _ParentInfo | None]:
        parent_event_id = self._parent_event_id(event)
        parent_info = self._context_for_event_locked(parent_event_id)
        if parent_info is None:
            parent_info = self._hint_context_locked(hint_keys)
        return parent_event_id, parent_info

    def _queue_pending_start_locked(
        self,
        parent_event_id: str,
        pending: _PendingStart,
    ) -> list[_PendingStart]:
        queue = self._pending_starts.setdefault(parent_event_id, [])
        queue.append(pending)
        self._pending_start_count += 1
        self._pending_starts.move_to_end(parent_event_id)

        evicted: list[_PendingStart] = []
        while self._pending_start_count > MAX_BUFFERED_ENTRIES:
            evicted_parent, evicted_queue = next(iter(self._pending_starts.items()))
            evicted.append(evicted_queue.pop(0))
            self._pending_start_count -= 1
            if not evicted_queue:
                self._pending_starts.pop(evicted_parent, None)
            logger.warning(
                "Starting orphaned CrewAI event after evicting unknown parent %s",
                evicted_parent,
            )
        return evicted

    def start_span(self, event: Any, spec: SpanStartSpec) -> None:
        """Start a span, buffering briefly reordered children by parent event ID."""
        self._start(event, spec, self._ambient_parent(), allow_queue=True)

    def open_scope(
        self,
        event: Any,
        *,
        parent_hint_keys: tuple[str, ...] = (),
        remember_hint_keys: tuple[str, ...] = (),
    ) -> None:
        """Track a non-emitted CrewAI scope so instrumented descendants stay linked."""
        spec = SpanStartSpec(
            name="",
            parent_hint_keys=parent_hint_keys,
            remember_hint_keys=remember_hint_keys,
        )
        self._start(
            event, spec, self._ambient_parent(), allow_queue=True, is_scope=True
        )

    def _start(
        self,
        event: Any,
        spec: SpanStartSpec,
        ambient_parent: _ParentInfo,
        *,
        allow_queue: bool,
        is_scope: bool = False,
    ) -> None:
        event_id = self._event_id(event)
        if not event_id:
            return

        evicted_pending: list[_PendingStart] = []
        pending_children: list[_PendingStart] = []
        pending_end: _PendingEnd | None = None
        open_entry: _OpenSpan | None = None

        with self._lock:
            self._evict_stale_locked()
            if (
                event_id in self._open_spans
                or event_id in self._transparent_scopes
                or event_id in self._finished_contexts
            ):
                return

            parent_event_id, parent_info = self._resolve_parent_locked(
                event,
                spec.parent_hint_keys,
            )
            if parent_event_id and parent_info is None and allow_queue:
                evicted_pending = self._queue_pending_start_locked(
                    parent_event_id,
                    _PendingStart(
                        event=event,
                        spec=None if is_scope else spec,
                        ambient_parent=ambient_parent,
                    ),
                )
                queued = True
            else:
                queued = False
                parent_info = parent_info or ambient_parent

                if is_scope:
                    if event_id in self._pending_scope_ends:
                        self._pending_scope_ends.pop(event_id, None)
                        self._remember_finished_locked(event_id, parent_info)
                    else:
                        self._transparent_scopes[event_id] = _TransparentScope(
                            parent_info=parent_info,
                            opened_at_monotonic=time.monotonic(),
                        )
                    self._remember_hints_locked(
                        spec.remember_hint_keys,
                        parent_info,
                    )
                else:
                    attributes = dict(spec.attributes)
                    if parent_info.workflow_name:
                        attributes.setdefault(
                            SpanAttributes.TRACELOOP_WORKFLOW_NAME,
                            parent_info.workflow_name,
                        )

                    span = self._tracer.start_span(
                        spec.name,
                        context=parent_info.context,
                        attributes=attributes,
                        start_time=self._start_time_ns(event),
                        record_exception=False,
                        set_status_on_exception=False,
                    )
                    span_context = trace.set_span_in_context(span, parent_info.context)
                    workflow_name = attributes.get(
                        SpanAttributes.TRACELOOP_WORKFLOW_NAME
                    )
                    span_info = _ParentInfo(
                        context=span_context,
                        workflow_name=(
                            str(workflow_name)
                            if workflow_name
                            else parent_info.workflow_name
                        ),
                    )
                    open_entry = _OpenSpan(
                        span=span,
                        parent_info=span_info,
                        opened_at_monotonic=time.monotonic(),
                        correlation_keys=spec.correlation_keys,
                    )
                    self._open_spans[event_id] = open_entry
                    self._remember_hints_locked(
                        spec.remember_hint_keys,
                        span_info,
                    )
                    self._remember_correlations_locked(
                        event_id,
                        spec.correlation_keys,
                    )
                    pending_end = self._pending_ends.pop(event_id, None)
                    if pending_end is None:
                        for correlation_key in spec.correlation_keys:
                            pending_end = self._pending_correlation_ends.pop(
                                correlation_key,
                                None,
                            )
                            if pending_end is not None:
                                break
                    if pending_end is not None:
                        self._open_spans.pop(event_id, None)
                        self._forget_correlations_locked(event_id, open_entry)
                        self._remember_finished_locked(event_id, span_info)

                pending_children = self._pending_starts.pop(event_id, [])
                self._pending_start_count -= len(pending_children)

        for pending in evicted_pending:
            self._resume_pending(pending, allow_queue=False)
        if queued:
            return

        if pending_end is not None and open_entry is not None:
            self._finalize(open_entry, pending_end.spec, pending_end.end_time_ns)
        for pending in pending_children:
            self._resume_pending(pending, allow_queue=True)

    def _resume_pending(self, pending: _PendingStart, *, allow_queue: bool) -> None:
        if pending.spec is None:
            scope_spec = SpanStartSpec(name="")
            self._start(
                pending.event,
                scope_spec,
                pending.ambient_parent,
                allow_queue=allow_queue,
                is_scope=True,
            )
            return
        self._start(
            pending.event,
            pending.spec,
            pending.ambient_parent,
            allow_queue=allow_queue,
        )

    def end_span(
        self,
        event: Any,
        spec: SpanEndSpec,
        *,
        correlation_keys: tuple[str, ...] = (),
    ) -> None:
        """Finish a span or buffer an end event that raced ahead of its start."""
        started_event_id = getattr(event, "started_event_id", None)
        event_id = str(started_event_id) if started_event_id else None
        end_time_ns = self._end_time_ns(event)
        pending = _PendingEnd(spec=spec, end_time_ns=end_time_ns)

        with self._lock:
            self._evict_stale_locked()
            if event_id is None:
                for correlation_key in correlation_keys:
                    correlated_event_id = self._correlations.get(correlation_key)
                    if correlated_event_id:
                        event_id = correlated_event_id
                        break

            if event_id is None:
                if correlation_keys:
                    correlation_key = correlation_keys[0]
                    self._pending_correlation_ends.pop(correlation_key, None)
                    self._pending_correlation_ends[correlation_key] = pending
                    self._trim_cache_locked(
                        self._pending_correlation_ends,
                        "pending correlation end",
                    )
                return

            entry = self._open_spans.pop(event_id, None)
            if entry is None:
                if event_id in self._finished_contexts:
                    return
                self._pending_ends.pop(event_id, None)
                self._pending_ends[event_id] = pending
                self._trim_cache_locked(self._pending_ends, "pending end")
                return

            self._forget_correlations_locked(event_id, entry)
            self._remember_finished_locked(event_id, entry.parent_info)

        self._finalize(entry, spec, end_time_ns)

    def close_scope(self, event: Any) -> None:
        """Close a transparent scope while retaining its context for late children."""
        started_event_id = getattr(event, "started_event_id", None)
        if not started_event_id:
            return
        event_id = str(started_event_id)
        with self._lock:
            self._evict_stale_locked()
            scope = self._transparent_scopes.pop(event_id, None)
            if scope is not None:
                self._remember_finished_locked(event_id, scope.parent_info)
            elif event_id not in self._finished_contexts:
                self._pending_scope_ends.pop(event_id, None)
                self._pending_scope_ends[event_id] = None
                self._trim_cache_locked(
                    self._pending_scope_ends,
                    "pending scope end",
                )

    @staticmethod
    def _finalize(
        entry: _OpenSpan,
        spec: SpanEndSpec,
        end_time_ns: int | None,
    ) -> None:
        span = entry.span
        try:
            if spec.output is not None:
                span.set_attribute(
                    SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
                    json_attribute(spec.output),
                )
            if spec.attributes:
                span.set_attributes(spec.attributes)
            if spec.error:
                span.set_attribute(ERROR_MESSAGE_ATTR, spec.error)
                span.set_status(Status(StatusCode.ERROR, spec.error))
            else:
                span.set_status(StatusCode.OK)
        finally:
            span.end(end_time=end_time_ns)

    def shutdown(self) -> None:
        """Flush orphan starts and close every still-open lifecycle span."""
        with self._lock:
            pending_starts = [
                pending
                for pending_group in self._pending_starts.values()
                for pending in pending_group
            ]
            self._pending_starts.clear()
            self._pending_start_count = 0

        for pending in pending_starts:
            self._resume_pending(pending, allow_queue=False)

        with self._lock:
            open_entries = list(self._open_spans.values())
            self._open_spans.clear()
            self._transparent_scopes.clear()
            self._finished_contexts.clear()
            self._context_hints.clear()
            self._correlations.clear()
            self._pending_ends.clear()
            self._pending_correlation_ends.clear()
            self._pending_scope_ends.clear()

        for entry in open_entries:
            self._mark_abandoned(
                entry,
                "CrewAI instrumentation deactivated before lifecycle completion",
            )
