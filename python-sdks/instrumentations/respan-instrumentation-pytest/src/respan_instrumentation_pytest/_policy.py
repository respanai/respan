"""Bodyless bounds for pytest-owned spans and their observed ancestors."""

import os
import threading
from collections import OrderedDict

from opentelemetry import context, trace
from opentelemetry.sdk.trace import SpanProcessor
from opentelemetry.sdk.util import BoundedList
from opentelemetry.semconv._incubating.attributes.error_attributes import ERROR_MESSAGE
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.trace import Status
from respan_tracing.constants.context_constants import ENABLE_CONTENT_TRACING_KEY


def permitted(ctx=None):
    return (
        not suppressed()
        and not suppressed(ctx)
        and context.get_value(ENABLE_CONTENT_TRACING_KEY) is not False
        and context.get_value(ENABLE_CONTENT_TRACING_KEY, ctx) is not False
        and all(
            os.getenv(k, "true").strip().lower() not in {"false", "0", "off", "no"}
            for k in [
                "TRACELOOP_TRACE_CONTENT",
                "RESPAN_TRACE_CONTENT",
                "RESPAN_PYTEST_CAPTURE_CONTENT",
            ]
        )
    )


def suppressed(ctx=None):
    from opentelemetry.context import _SUPPRESS_INSTRUMENTATION_KEY
    from opentelemetry.semconv_ai import SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY

    return bool(
        context.get_value(_SUPPRESS_INSTRUMENTATION_KEY, ctx)
        or context.get_value(SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY, ctx)
    )


def key(span):
    sc = span.get_span_context() if hasattr(span, "get_span_context") else span.context
    return (sc.trace_id, sc.span_id) if sc.is_valid else None


class Policy(SpanProcessor):
    def __init__(self, processor):
        self.active = {}
        self.closed = OrderedDict()
        self.enabled = True
        self.faulted = False
        self.lock = threading.RLock()
        self.processor = processor
        self.original_detach = context.detach

        def detach(token):
            try:
                if self.enabled and not permitted():
                    self.deny(key(trace.get_current_span()))
            except Exception:  # noqa: BLE001 - telemetry faults must preserve native pytest behavior.
                self.faulted = True
            return self.original_detach(token)

        self.guard = detach
        context.detach = detach
        try:
            processor.add_span_processor(self)
        except BaseException:
            self.close()
            raise

    def bound(self, k):
        with self.lock:
            if self.faulted:
                return False
            seen = set()
            while k and k not in seen:
                seen.add(k)
                s = self.active.get(k, self.closed.get(k))
                if s is None:
                    return True
                if not s[0]:
                    return False
                k = s[1]
            return True

    def enroll(self, span):
        k = key(span)
        with self.lock:
            if (
                k
                and k not in self.active
                and k not in self.closed
                and not span.get_span_context().is_remote
            ):
                self.active[k] = (
                    False,
                    (span.parent.trace_id, span.parent.span_id)
                    if getattr(span, "parent", None)
                    else None,
                )
            return self.bound(k)

    def deny(self, k):
        with self.lock:
            seen = set()
            while k and k not in seen:
                seen.add(k)
                s = self.active.get(k, self.closed.get(k, (False, None)))
                if k in self.active:
                    self.active[k] = (False, s[1])
                else:
                    self.closed[k] = (False, s[1])
                k = s[1]
            for child, s in list(self.active.items()):
                if not self.bound(child):
                    self.active[child] = (False, s[1])
            while len(self.closed) > 4096:
                self.closed.popitem(last=False)

    def on_start(self, span, parent_context=None):
        if not self.enabled:
            return
        try:
            parent = trace.get_current_span(parent_context)
            with self.lock:
                self.active[key(span)] = (
                    permitted(parent_context) and self.enroll(parent),
                    key(parent),
                )
        except Exception:  # noqa: BLE001 - telemetry faults must preserve native pytest behavior.
            self.faulted = True

    def on_end(self, span):
        if not self.enabled:
            return
        try:
            with self.lock:
                k = key(span)
                s = self.active.get(k, (False, None))
                if not s[0] or not permitted() or not self.bound(s[1]):
                    self.deny(k)
                self.closed[k] = self.active.pop(k, s)
                while len(self.closed) > 4096:
                    self.closed.popitem(last=False)
        except Exception:  # noqa: BLE001 - telemetry faults must preserve native pytest behavior.
            self.faulted = True

    def check(self, state):
        try:
            state.capture = bool(
                state.capture
                and self.enabled
                and permitted()
                and self.bound(key(state.span))
            )
        except Exception:  # noqa: BLE001 - telemetry faults must preserve native pytest behavior.
            state.capture = False
            self.faulted = True
        if not state.capture:
            self.clear(state.span)
        return state.capture and state.span.is_recording()

    @staticmethod
    def clear(span):
        attrs = getattr(span, "_attributes", None)
        if attrs is not None:
            for k in (
                SpanAttributes.TRACELOOP_ENTITY_INPUT,
                SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
                ERROR_MESSAGE,
            ):
                attrs.pop(k, None)
        if hasattr(span, "_events"):
            span._events = BoundedList(0)
        if hasattr(span, "status"):
            span._status = Status(span.status.status_code)

    def close(self):
        self.enabled = False
        if context.detach is self.guard:
            context.detach = self.original_detach
        lock = getattr(self.processor, "_lock", None)
        if lock is not None:
            with lock:
                self.processor._span_processors = tuple(
                    p for p in self.processor._span_processors if p is not self
                )
        self.active.clear()
        self.closed.clear()

    def shutdown(self):
        self.close()
