"""Bodyless ancestor capture bounds, opt-out vetoes and OTel suppression."""

from __future__ import annotations

import os
import threading
from collections import OrderedDict

from opentelemetry import context, trace
from opentelemetry.sdk.trace import SpanProcessor
from opentelemetry.semconv_ai import SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY
from respan_tracing.constants.context_constants import ENABLE_CONTENT_TRACING_KEY


def content_allowed(setting=True, ctx=None):
    return bool(
        setting
        and context.get_value(ENABLE_CONTENT_TRACING_KEY, ctx) is not False
        and os.getenv("TRACELOOP_TRACE_CONTENT", "true").strip().lower()
        not in {"false", "0", "off", "no"}
    )


def suppressed():
    return bool(
        context.get_value(context._SUPPRESS_INSTRUMENTATION_KEY)
        or context.get_value(SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY)
    )


def span_key(span):
    value = (
        span.get_span_context() if hasattr(span, "get_span_context") else span.context
    )
    return (value.trace_id, value.span_id) if value.is_valid else None


class AncestorPolicy(SpanProcessor):
    def __init__(self, setting):
        self.setting = setting
        self.open = {}
        self.ended = OrderedDict()
        self.lock = threading.RLock()

    def _allowed(self, key):
        seen = set()
        while key is not None and key not in seen:
            seen.add(key)
            if key in self.ended:
                allowed, key = self.ended[key]
                if not allowed:
                    return False
                continue
            state = self.open.get(key)
            if state is None:
                return True
            allowed, key = state
            if not allowed:
                return False
        return True

    def on_start(self, span, parent_context=None):
        with self.lock:
            key = span_key(span)
            parent = span_key(trace.get_current_span(parent_context))
            if key is not None:
                self.open[key] = (
                    content_allowed(self.setting, parent_context)
                    and self._allowed(parent),
                    parent,
                )

    def observe(self, span):
        with self.lock:
            key = span_key(span)
            if content_allowed(self.setting):
                return self._allowed(key)
            seen = set()
            while key is not None and key not in seen:
                seen.add(key)
                state = self.open.get(key)
                if state is None:
                    parent = self.ended.get(key, (False, None))[1]
                    self.ended[key] = (False, parent)
                    break
                self.open[key] = (False, state[1])
                key = state[1]
            for child, state in list(self.open.items()):
                if not self._allowed(child):
                    self.open[child] = (False, state[1])
            return False

    def knows(self, span):
        with self.lock:
            key = span_key(span)
            return (
                key in self.open
                or key in self.ended
                or any(state[1] == key for state in self.open.values())
                or any(state[1] == key for state in self.ended.values())
            )

    def allowed(self, span):
        with self.lock:
            return self._allowed(span_key(span))

    def on_end(self, span):
        with self.lock:
            self.observe(span)
            key = span_key(span)
            initial, parent = self.open.pop(key, (False, None))
            if key is not None:
                self.ended[key] = (initial and self._allowed(parent), parent)
                while len(self.ended) > 4096:
                    self.ended.popitem(last=False)

    def clear(self):
        with self.lock:
            self.open.clear()
            self.ended.clear()

    def shutdown(self):
        self.clear()

    def force_flush(self, timeout_millis=30000):
        return True
