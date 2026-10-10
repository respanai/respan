"""Track start-bound privacy for recording spans and their ancestors."""

from __future__ import annotations

import os
import threading
from collections import OrderedDict

from opentelemetry import context, trace
from respan_tracing.constants.context_constants import ENABLE_CONTENT_TRACING_KEY


def permitted(ctx=None):
    return context.get_value(
        ENABLE_CONTENT_TRACING_KEY, ctx
    ) is not False and os.getenv(
        "TRACELOOP_TRACE_CONTENT", "true"
    ).strip().lower() not in {"false", "0", "no", "off"}


def key(span):
    sc = (
        span.get_span_context()
        if hasattr(span, "get_span_context")
        else getattr(span, "context", None)
    )
    return (sc.trace_id, sc.span_id) if sc is not None and sc.is_valid else None


class Policy:
    def __init__(self, capture):
        self.capture = capture
        self.active = {}
        self.denied = OrderedDict()
        self.lock = threading.RLock()

    def ancestors_allowed(self, parent):
        seen = set()
        while parent and parent not in seen:
            seen.add(parent)
            if parent in self.denied:
                return False
            state = self.active.get(parent)
            if state is None:
                break
            if not state[0]:
                return False
            parent = state[1]
        return True

    def start(self, span, parent_context=None):
        with self.lock:
            k = key(span)
            if k:
                parent = key(trace.get_current_span(parent_context))
                self.active[k] = (
                    self.capture
                    and permitted(parent_context)
                    and self.ancestors_allowed(parent),
                    parent,
                )

    def allowed(self, span):
        with self.lock:
            state = self.active.get(key(span))
            return bool(state and state[0] and self.ancestors_allowed(state[1]))

    def permitted_now(self, span):
        return permitted() and self.allowed(span)

    def deny(self, span):
        with self.lock:
            k = key(span)
            state = self.active.get(k)
            if state is not None:
                self.active[k] = (False, state[1])
            if k:
                self.denied[k] = True
                while len(self.denied) > 4096:
                    self.denied.popitem(last=False)

    def finish(self, span):
        with self.lock:
            k = key(span)
            initial, parent = self.active.pop(k, (False, None))
            allowed = initial and permitted() and self.ancestors_allowed(parent)
            if k and not allowed:
                self.denied[k] = True
                while len(self.denied) > 4096:
                    self.denied.popitem(last=False)
            return allowed
