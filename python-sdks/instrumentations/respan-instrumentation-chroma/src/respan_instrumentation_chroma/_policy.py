"""Latch capture vetoes and retain bounded, bodyless local ancestry."""

from __future__ import annotations

import os
import weakref
from collections import OrderedDict
from typing import Any

from opentelemetry import context, trace
from opentelemetry.context import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.sdk.trace import SpanProcessor
from opentelemetry.sdk.util import BoundedList
from opentelemetry.semconv._incubating.attributes.error_attributes import ERROR_MESSAGE
from opentelemetry.semconv_ai import SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY
from opentelemetry.trace import Status, StatusCode
from respan_tracing.constants.context_constants import ENABLE_CONTENT_TRACING_KEY

_OFF = {"false", "0", "no", "off"}
_CAPTURE_KEYS = (
    ENABLE_CONTENT_TRACING_KEY,
    "trace_content",
    "override_enable_content_tracing",
)


def suppressed(carrier: Any = None) -> bool:
    return bool(
        context.get_value(_SUPPRESS_INSTRUMENTATION_KEY, carrier)
        or context.get_value(SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY, carrier)
    )


def context_capture(carrier: Any = None) -> bool:
    return not suppressed(carrier) and explicit_capture(carrier)


def explicit_capture(carrier: Any = None) -> bool:
    return all(
        context.get_value(key, carrier) is not False for key in _CAPTURE_KEYS
    ) and all(
        os.getenv(key, "true").strip().lower() not in _OFF
        for key in ("TRACELOOP_TRACE_CONTENT", "RESPAN_TRACE_CONTENT")
    )


class PrivacyObserver(SpanProcessor):
    def __init__(self) -> None:
        self.spans: weakref.WeakValueDictionary[tuple[int, int], Any] = (
            weakref.WeakValueDictionary()
        )
        self.parents: OrderedDict[tuple[int, int], tuple[int, int] | None] = (
            OrderedDict()
        )
        self.vetoes: set[tuple[int, int]] = set()
        self.states: weakref.WeakSet[Any] = weakref.WeakSet()

    def on_start(self, span: Any, parent_context: Any = None) -> None:
        try:
            child = span.get_span_context()
            key = (child.trace_id, child.span_id)
            parent = trace.get_current_span(parent_context).get_span_context()
            self.spans[key] = span
            self.parents[key] = (
                (parent.trace_id, parent.span_id)
                if parent.is_valid and not parent.is_remote
                else None
            )
            if not context_capture(parent_context) or not context_capture():
                self.vetoes.add(key)
            self._prune()
        except Exception:  # noqa: BLE001
            self.notice()

    def on_end(self, span: Any) -> None:
        owned = []
        try:
            sc = span.get_span_context()
            key = (sc.trace_id, sc.span_id)
            for state in list(self.states):
                if state.span is not None and state.span.get_span_context() == sc:
                    # End snapshots share immutable attributes and old event/status
                    # objects. Keep ownership before notice can discard state.
                    owned.append((state, set(state.keys)))
            if not context_capture() or any(
                (span.attributes or {}).get(flag) is False for flag in _CAPTURE_KEYS
            ):
                self.vetoes.add(key)
            self.notice()
            self._prune()
        except Exception:  # noqa: BLE001
            for state, _ in owned:
                state.policy.allowed = False
            self.notice()
        finally:
            for state, keys in owned:
                if not state.policy.allowed:
                    # Mutate the actual ReadableSpan before exporter processors,
                    # rather than only replacing storage on the ended SDK Span.
                    span._attributes = {
                        field: value
                        for field, value in (span.attributes or {}).items()
                        if field not in keys and field != ERROR_MESSAGE
                    }
                    span._events = BoundedList(maxlen=None)
                    if span.status.status_code == StatusCode.ERROR:
                        span._status = Status(StatusCode.ERROR)

    def notice(self) -> None:
        """Called before detach and after ancestor end, while veto is visible."""
        for state in list(self.states):
            if not state.done:
                state.safe("check")

    def _prune(self) -> None:
        # Active ancestors remain observed. At most 4096 finished identity/policy
        # records are retained; evicted local ancestry subsequently fails closed.
        ended = [
            key
            for key in self.parents
            if (local := self.spans.get(key)) is None or not local.is_recording()
        ]
        for key in ended[:-4096]:
            self.parents.pop(key, None)
            self.vetoes.discard(key)

    def allows(self, parent: Any) -> bool:
        sc = parent.get_span_context()
        if not sc.is_valid or sc.is_remote:
            return True
        key = (sc.trace_id, sc.span_id)
        while key is not None:
            if key not in self.parents or key in self.vetoes:
                return False
            local = self.spans.get(key)
            if local is not None and any(
                (local.attributes or {}).get(flag) is False for flag in _CAPTURE_KEYS
            ):
                return False
            key = self.parents.get(key)
        return True


class CapturePolicy:
    def __init__(
        self, observer: PrivacyObserver, enabled: bool, supplied: Any = None
    ) -> None:
        self.observer = observer
        self.context = context.get_current()
        self.supplied = supplied
        self.parent = trace.get_current_span(self.context)
        self.supplied_parent = (
            trace.get_current_span(supplied) if supplied is not None else None
        )
        self.allowed = enabled
        self.check()

    def check(self) -> bool:
        self.allowed = (
            self.allowed
            and context_capture(self.context)
            and context_capture()
            and self.observer.allows(self.parent)
        )
        if self.supplied is not None:
            self.allowed = (
                self.allowed
                and context_capture(self.supplied)
                and self.observer.allows(self.supplied_parent)
            )
        self.allowed = self.allowed and self.observer.allows(trace.get_current_span())
        return self.allowed
