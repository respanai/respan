"""Bodyless content bounds for native calls and delayed hook generations."""

from __future__ import annotations

import os
import threading
import weakref
from collections import OrderedDict

from opentelemetry import context, trace
from opentelemetry.sdk.trace import SpanProcessor
from respan_tracing.constants.context_constants import ENABLE_CONTENT_TRACING_KEY

_OFF = {"false", "0", "off", "no"}


class Bound:
    def __init__(self, allowed, parent):
        self.allowed, self.parent = allowed, parent


def permitted(ctx=None):
    return (
        context.get_value(ENABLE_CONTENT_TRACING_KEY) is not False
        and context.get_value(ENABLE_CONTENT_TRACING_KEY, ctx) is not False
        and all(
            os.getenv(name, "true").strip().lower() not in _OFF
            for name in ("TRACELOOP_TRACE_CONTENT", "RESPAN_TRACE_CONTENT")
        )
    )


def suppressed():
    from opentelemetry.context import _SUPPRESS_INSTRUMENTATION_KEY
    from opentelemetry.semconv_ai import SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY

    return bool(
        context.get_value(_SUPPRESS_INSTRUMENTATION_KEY)
        or context.get_value(SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY)
    )


def span_key(span):
    sc = (
        span.get_span_context()
        if hasattr(span, "get_span_context")
        else getattr(span, "context", None)
    )
    return (sc.trace_id, sc.span_id) if sc is not None and sc.is_valid else None


class Policy(SpanProcessor):
    def __init__(self):
        self.active = {}
        self.denied = OrderedDict()
        self.lock = threading.RLock()
        self.bounds = weakref.WeakSet()
        self.detach = context.detach
        self.enabled = True
        self.faulted = False

        def guarded_detach(token):
            try:
                if self.enabled and not permitted():
                    self.deny(span_key(trace.get_current_span()))
            except Exception:  # noqa: BLE001 - Always preserve native context detachment.
                self.fail_closed()
            return self.detach(token)

        self.guard = guarded_detach
        context.detach = self.guard

    def deny(self, key):
        with self.lock:
            if key:
                self.denied[key] = True
                if key in self.active:
                    self.active[key] = (False, self.active[key][1])
                for child, state in list(self.active.items()):
                    if not self.ancestors(child):
                        self.active[child] = (False, state[1])
                for bound in self.bounds:
                    if not self.ancestors(bound.parent):
                        bound.allowed = False
                while len(self.denied) > 4096:
                    self.denied.popitem(last=False)

    def ancestors(self, key):
        with self.lock:
            if self.faulted:
                return False
            seen = set()
            while key and key not in seen:
                seen.add(key)
                if key in self.denied:
                    return False
                state = self.active.get(key)
                if state is None:
                    break
                if not state[0]:
                    return False
                key = state[1]
            return True

    def enroll(self, span):
        """An already-recording local parent has no trustworthy initial bound."""
        key = span_key(span)
        with self.lock:
            if key and span.is_recording() and key not in self.active:
                self.deny(key)
            return self.ancestors(key)

    def fail_closed(self):
        with self.lock:
            self.faulted = True
            for key, state in list(self.active.items()):
                self.active[key] = (False, state[1])
            for bound in list(self.bounds):
                bound.allowed = False

    def watch(self, parent):
        with self.lock:
            bound = Bound(self.ancestors(parent), parent)
            self.bounds.add(bound)
            return bound

    def on_start(self, span, parent_context=None):
        if not self.enabled:
            return
        try:
            with self.lock:
                parent_span = trace.get_current_span(parent_context)
                parent = span_key(parent_span)
                key = span_key(span)
                self.active[key] = (
                    permitted(parent_context) and self.enroll(parent_span),
                    parent,
                )
        except Exception:  # noqa: BLE001 - Optional policy observation cannot abort spans.
            self.fail_closed()

    def on_end(self, span):
        if not self.enabled:
            return
        try:
            with self.lock:
                key = span_key(span)
                state = self.active.pop(key, (False, None))
                if not state[0] or not permitted() or not self.ancestors(state[1]):
                    self.deny(key)
        except Exception:  # noqa: BLE001 - Optional policy observation cannot abort spans.
            self.fail_closed()

    def close(self):
        self.enabled = False
        if context.detach is self.guard:
            context.detach = self.detach
        self.active.clear()

    def shutdown(self):
        self.close()
