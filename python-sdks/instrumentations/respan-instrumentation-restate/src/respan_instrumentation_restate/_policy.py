"""Content vetoes survive ancestor completion and later context changes."""
# ruff: noqa: BLE001 -- privacy observation faults must fail closed.

from __future__ import annotations

import os
import threading
from collections import OrderedDict

from opentelemetry import context, trace
from opentelemetry.sdk.trace import SpanProcessor
from opentelemetry.semconv_ai import SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY
from respan_tracing.constants.context_constants import ENABLE_CONTENT_TRACING_KEY


def content_allowed(setting=True, parent_context=None):
    return bool(
        setting
        and context.get_value(ENABLE_CONTENT_TRACING_KEY) is not False
        and context.get_value(ENABLE_CONTENT_TRACING_KEY, parent_context) is not False
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
        self.enabled = True
        self.open = {}
        self.ended = OrderedDict()
        self.lock = threading.RLock()

    def _allowed(self, key):
        seen = set()
        while key is not None and key not in seen:
            seen.add(key)
            state = self.open.get(key, self.ended.get(key))
            if state is None or not state[0]:
                return False
            key = state[1]
        return key is None

    def on_start(self, span, parent_context=None):
        if not self.enabled:
            return
        try:
            with self.lock:
                key = span_key(span)
                parent_span = trace.get_current_span(parent_context)
                parent = span_key(parent_span)
                observed = parent in self.open or parent in self.ended
                if (
                    parent is not None
                    and not observed
                    and not parent_span.is_recording()
                ):
                    parent = None
                if key is not None:
                    self.open[key] = (
                        content_allowed(self.setting, parent_context)
                        and self._allowed(parent),
                        parent,
                    )
                    if not self.open[key][0]:
                        self._deny_chain(key)
        except Exception:
            self._deny_all()

    def _deny_all(self):
        with self.lock:
            for key, state in list(self.open.items()):
                self.open[key] = (False, state[1])

    def _deny_chain(self, key):
        seen = set()
        while key is not None and key not in seen:
            seen.add(key)
            state = self.open.get(key, self.ended.get(key, (False, None)))
            if key in self.open:
                self.open[key] = (False, state[1])
            else:
                self.ended[key] = (False, state[1])
            key = state[1]

    def observe(self, span):
        try:
            with self.lock:
                key = span_key(span)
                if content_allowed(self.setting):
                    return self.enabled and self._allowed(key)
                self._deny_chain(key)
                return False
        except Exception:
            self._deny_all()
            return False

    def on_end(self, span):
        if not self.enabled:
            return
        try:
            with self.lock:
                allowed = self.observe(span)
                key = span_key(span)
                state = self.open.pop(key, (False, None))
                if key is not None:
                    self.ended[key] = (allowed and state[0], state[1])
                    while len(self.ended) > 4096:
                        self.ended.popitem(last=False)
        except Exception:
            self._deny_all()

    def clear(self):
        with self.lock:
            self.open.clear()
            self.ended.clear()

    def shutdown(self):
        self.clear()

    def force_flush(self, timeout_millis=30000):
        return True
