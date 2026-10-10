"""Per-span privacy bounds and a vendor-owned pre-detach checkpoint."""

from __future__ import annotations

import os
import threading

from opentelemetry import context, trace
from opentelemetry.context import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.sdk.trace import SpanProcessor
from opentelemetry.semconv_ai import SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY
from respan_tracing.constants.context_constants import ENABLE_CONTENT_TRACING_KEY


def allowed(capture=True, parent_context=None):
    return (
        capture
        and os.getenv("TRACELOOP_TRACE_CONTENT", "true").strip().lower()
        not in {"false", "0", "off", "no"}
        and context.get_value(ENABLE_CONTENT_TRACING_KEY) is not False
        and context.get_value(ENABLE_CONTENT_TRACING_KEY, parent_context) is not False
    )


def suppressed(parent_context=None):
    return any(
        context.get_value(key) or context.get_value(key, parent_context)
        for key in (
            _SUPPRESS_INSTRUMENTATION_KEY,
            SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY,
        )
    )


def key(span):
    sc = span.get_span_context()
    return (sc.trace_id, sc.span_id) if sc.is_valid else None


class Policy(SpanProcessor):
    def __init__(self, capture):
        self.capture = capture
        self.live, self.parents, self.bounds, self.closed = {}, {}, {}, {}
        self.lock = threading.RLock()
        self.hook = None

    def on_start(self, span, parent_context=None):
        sid = key(span)
        parent_span = trace.get_current_span(parent_context)
        parent = key(parent_span)
        with self.lock:
            if (
                parent
                and parent not in self.bounds
                and parent not in self.closed
                and parent_span.is_recording()
            ):
                self.live[parent] = parent_span
                self.bounds[parent] = (
                    False  # Its initial privacy policy was unobserved.
                )
                ancestor = getattr(parent_span, "parent", None)
                self.parents[parent] = (
                    (ancestor.trace_id, ancestor.span_id)
                    if ancestor and ancestor.is_valid
                    else None
                )
            self.live[sid] = span
            self.parents[sid] = parent
            try:
                self.bounds[sid] = allowed(
                    self.capture, parent_context
                ) and self.ancestors(parent)
            except Exception:  # noqa: BLE001 - policy faults lower the bound.
                self.bounds[sid] = False

    def ancestors(self, sid):
        seen = set()
        with self.lock:
            while sid and sid not in seen:
                seen.add(sid)
                if not self.bounds.get(sid, self.closed.get(sid, True)):
                    return False
                sid = self.parents.get(sid)
        return True

    def veto(self, span):
        sid = key(span)
        with self.lock:
            seen = set()
            while sid and sid not in seen:
                seen.add(sid)
                if sid in self.bounds:
                    self.bounds[sid] = False
                if sid in self.closed:
                    self.closed[sid] = False
                sid = self.parents.get(sid)

    def observe(self):
        span = trace.get_current_span()
        sid = key(span)
        with self.lock:
            if sid not in self.live:
                return
            try:
                permitted = allowed(self.capture) and self.ancestors(sid)
            except Exception:  # noqa: BLE001 - policy faults cannot expose content.
                permitted = False
            if not permitted:
                self.veto(span)

    def on_end(self, span):
        sid = key(span)
        with self.lock:
            if sid not in self.live:
                return
            try:
                permitted = allowed(self.capture) and self.ancestors(sid)
            except Exception:  # noqa: BLE001 - retain the conservative bound.
                permitted = False
            self.closed[sid] = self.bounds.pop(sid, False) and permitted
            self.live.pop(sid, None)
            if len(self.closed) > 4096:
                # Keep finished ancestors referenced by active descendants.
                referenced = set()
                for live in self.live:
                    ancestor = self.parents.get(live)
                    while ancestor and ancestor not in referenced:
                        referenced.add(ancestor)
                        ancestor = self.parents.get(ancestor)
                for old in tuple(self.closed):
                    if old not in referenced:
                        self.closed.pop(old, None)
                        self.parents.pop(old, None)
                        break

    def install(self):
        original = context.detach

        def detached(token):
            try:
                self.observe()
            except Exception:  # noqa: BLE001, S110 - preserve native detach.
                pass
            return original(token)

        self.hook = (original, detached)
        context.detach = detached

    def shutdown(self):
        if self.hook and context.detach is self.hook[1]:
            context.detach = self.hook[0]
        self.hook = None
        with self.lock:
            self.live.clear()
            self.parents.clear()
            self.bounds.clear()
            self.closed.clear()

    def force_flush(self, timeout_millis=30000):
        return True
