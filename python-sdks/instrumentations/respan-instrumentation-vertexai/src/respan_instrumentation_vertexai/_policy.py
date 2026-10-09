"""Irreversible native context/ancestor content vetoes and suppression."""

# ruff: noqa: BLE001 -- telemetry privacy faults fail closed.
from __future__ import annotations

import os
import threading
import weakref
from collections import OrderedDict

from opentelemetry import context, trace
from opentelemetry.sdk.trace import SpanProcessor
from opentelemetry.semconv_ai import SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY
from respan_tracing.constants.context_constants import ENABLE_CONTENT_TRACING_KEY

_TRACED_CONTENT_OVERRIDE = "override_enable_content_tracing"
_TRACED_CONTENT_ATTRIBUTE = "traceloop.enable_content_tracing"


def suppressed(ctx=None):
    return any(
        bool(context.get_value(key, candidate))
        for candidate in (ctx, context.get_current())
        for key in (
            context._SUPPRESS_INSTRUMENTATION_KEY,
            SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY,
        )
    )


def content_allowed(setting=True, parent_context=None):
    return bool(
        setting
        and not suppressed(parent_context)
        and all(
            context.get_value(key, candidate) is not False
            for candidate in (parent_context, context.get_current())
            for key in (ENABLE_CONTENT_TRACING_KEY, _TRACED_CONTENT_OVERRIDE)
        )
        and all(
            os.getenv(name, "true").strip().lower() not in {"false", "0", "off", "no"}
            for name in ("TRACELOOP_TRACE_CONTENT", "RESPAN_TRACE_CONTENT")
        )
    )


def span_key(span):
    value = (
        span.get_span_context() if hasattr(span, "get_span_context") else span.context
    )
    return (value.trace_id, value.span_id) if value.is_valid else None


def _attributes_allowed(span):
    attrs = span.attributes or {}
    return all(
        attrs.get(key) is not False
        for key in (_TRACED_CONTENT_ATTRIBUTE, ENABLE_CONTENT_TRACING_KEY)
    )


class AncestorPolicy(SpanProcessor):
    def __init__(self, setting):
        self.setting = setting
        self.enabled = True
        self.open = {}
        self.ended = OrderedDict()
        self.lock = threading.RLock()

    def _deny_chain(self, key):
        seen = set()
        while key is not None and key not in seen:
            seen.add(key)
            state = self.open.get(key, self.ended.get(key, (False, None, None)))
            if key in self.open:
                self.open[key] = (False, state[1], state[2])
            else:
                self.ended[key] = (False, state[1], None)
            key = state[1]

    def _allowed(self, key):
        seen = set()
        while key is not None and key not in seen:
            seen.add(key)
            state = self.open.get(key, self.ended.get(key))
            if state is None or not state[0]:
                return False
            live = state[2]() if state[2] is not None else None
            if live is not None and not _attributes_allowed(live):
                self._deny_chain(key)
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
                if parent is not None and parent_span.get_span_context().is_remote:
                    parent = None
                if key is not None:
                    self.open[key] = (
                        content_allowed(self.setting, parent_context)
                        and self._allowed(parent),
                        parent,
                        weakref.ref(span),
                    )
                    if not self.open[key][0]:
                        self._deny_chain(key)
        except Exception:
            self._deny_all()

    def _deny_all(self):
        with self.lock:
            for key, state in list(self.open.items()):
                self.open[key] = (False, state[1], state[2])

    def observe(self, span):
        try:
            with self.lock:
                key = span_key(span)
                current = trace.get_current_span()
                current_key = span_key(current)
                current_allowed = (
                    current_key is None
                    or current.get_span_context().is_remote
                    or self._allowed(current_key)
                )
                if (
                    content_allowed(self.setting)
                    and current_allowed
                    and self._allowed(key)
                ):
                    return self.enabled
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
                allowed = _attributes_allowed(span) and self.observe(span)
                key = span_key(span)
                if not allowed:
                    self._deny_chain(key)
                state = self.open.pop(key, (False, None, None))
                if key is not None:
                    self.ended[key] = (allowed and state[0], state[1], None)
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
