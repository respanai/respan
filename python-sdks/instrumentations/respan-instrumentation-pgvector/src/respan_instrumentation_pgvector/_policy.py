"""Irreversible native context/ancestor content vetoes and suppression."""

# ruff: noqa: BLE001 -- telemetry privacy faults fail closed.
from __future__ import annotations

import os
import threading
import weakref
from collections import OrderedDict
from contextvars import ContextVar

from opentelemetry import context, trace
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor, _Span
from opentelemetry.semconv_ai import SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY
from respan_tracing.constants.context_constants import ENABLE_CONTENT_TRACING_KEY

CREATING_CALL = ContextVar("respan_pgvector_creating_call", default=None)

_TRACED_CONTENT_OVERRIDE = "override_enable_content_tracing"
_TRACED_CONTENT_ATTRIBUTE = "traceloop.enable_content_tracing"


def _flag(value):
    return (
        bool(value)
        if any(
            type(value) is item
            for item in (
                bool,
                int,
                float,
                str,
                list,
                tuple,
                dict,
            )
        )
        else value is not None
    )


def suppressed(ctx=None):
    return any(
        _flag(context.get_value(key, candidate))
        for candidate in (ctx, context.get_current())
        for key in (
            context._SUPPRESS_INSTRUMENTATION_KEY,
            SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY,
        )
    )


def content_allowed(setting=True, parent_context=None, *, honor_suppression=True):
    return bool(
        setting
        and (not honor_suppression or not suppressed(parent_context))
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
    if not any(
        type(span) is kind
        for kind in (Span, _Span, ReadableSpan, trace.NonRecordingSpan)
    ):
        return (-1, id(span))
    value = object.__getattribute__(span, "__dict__").get("_context")
    if type(value) is not trace.SpanContext:
        return (-1, id(span))
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
        self.calls = weakref.WeakValueDictionary()

    def _deny_chain(self, key):
        seen = set()
        while key is not None and key not in seen:
            seen.add(key)
            state = self.open.get(key, self.ended.get(key, (False, None, None)))
            if key in self.open:
                self.open[key] = (False, state[1], state[2])
            else:
                self.ended[key] = (False, state[1], None)
                while len(self.ended) > 4096:
                    self.ended.popitem(last=False)
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
            creator = CREATING_CALL.get()
            scope = getattr(span, "instrumentation_scope", None)
            if (
                creator is not None
                and creator.span is None
                and getattr(scope, "name", None) == "pgvector"
                and span.name == creator.creation_name
            ):
                creator.span = span
                self.calls[span_key(span)] = creator
            with self.lock:
                key = span_key(span)
                parent_span = trace.get_current_span(parent_context)
                parent = span_key(parent_span)
                if (
                    type(parent_span) is trace.NonRecordingSpan
                    and parent_span.get_span_context().is_remote
                ):
                    parent = None
                if key is not None:
                    self.open[key] = (
                        content_allowed(self.setting, parent_context)
                        and _attributes_allowed(span)
                        and self._allowed(parent),
                        parent,
                        weakref.ref(span, lambda _ref, key=key: self._forgotten(key)),
                    )
                    if not self.open[key][0]:
                        self._deny_chain(key)
        except Exception:
            self._deny_all()

    def _forgotten(self, key):
        with self.lock:
            state = self.open.pop(key, None)
            if state is not None:
                self.ended[key] = (False, state[1], None)
                self._deny_chain(key)
                while len(self.ended) > 4096:
                    self.ended.popitem(last=False)

    def _deny_all(self):
        with self.lock:
            for key, state in list(self.open.items()):
                self.open[key] = (False, state[1], state[2])

    def observe(self, span, *, honor_suppression=True):
        try:
            with self.lock:
                key = span_key(span)
                current = trace.get_current_span()
                current_key = span_key(current)
                current_allowed = (
                    current_key is None
                    or (
                        type(current) is trace.NonRecordingSpan
                        and current.get_span_context().is_remote
                    )
                    or self._allowed(current_key)
                )
                if (
                    content_allowed(self.setting, honor_suppression=honor_suppression)
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
                allowed = _attributes_allowed(span) and self.observe(
                    span, honor_suppression=False
                )
                key = span_key(span)
                if not allowed:
                    self._deny_chain(key)
                    call = self.calls.get(key)
                    if call is not None:
                        call.scrub(span)
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
            self.calls.clear()

    def shutdown(self):
        self.clear()

    def force_flush(self, timeout_millis=30000):
        return True
