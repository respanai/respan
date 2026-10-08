"""Bodyless native OTel ancestor bounds and runtime pre-detach privacy vetoes."""

from __future__ import annotations

import os
import threading
import weakref
from collections import OrderedDict

from opentelemetry import context, trace
from opentelemetry.sdk.trace import SpanProcessor
from opentelemetry.semconv_ai import SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY
from respan_tracing.constants.context_constants import ENABLE_CONTENT_TRACING_KEY

_CAPTURE_KEYS = (
    ENABLE_CONTENT_TRACING_KEY,
    "trace_content",
    "override_enable_content_tracing",
)


def suppressed(ctx=None):
    return bool(
        context.get_value(context._SUPPRESS_INSTRUMENTATION_KEY, ctx)
        or context.get_value(SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY, ctx)
    )


def capture_flags(ctx=None):
    return all(
        context.get_value(k) is not False and context.get_value(k, ctx) is not False
        for k in _CAPTURE_KEYS
    ) and all(
        os.getenv(k, "true").strip().lower() not in {"false", "0", "off", "no"}
        for k in ("TRACELOOP_TRACE_CONTENT", "RESPAN_TRACE_CONTENT")
    )


def permitted(ctx=None):
    return not suppressed() and not suppressed(ctx) and capture_flags(ctx)


def key(span):
    sc = span.get_span_context() if hasattr(span, "get_span_context") else span.context
    return (sc.trace_id, sc.span_id) if sc.is_valid else None


def span_permitted(span):
    attrs = getattr(span, "attributes", None) or {}
    return all(attrs.get(k) is not False for k in _CAPTURE_KEYS)


class Policy(SpanProcessor):
    def __init__(self, provider, scrub):
        self.provider = provider
        self.scrub = scrub
        self.active = {}
        self.refs = weakref.WeakValueDictionary()
        self.closed = OrderedDict()
        self.lock = threading.RLock()
        self.enabled = True
        self.faulted = False
        self.scrubbing = False
        self.runtime = context._RUNTIME_CONTEXT
        self.detach_present = "detach" in self.runtime.__dict__
        self.original_stored = self.runtime.__dict__.get("detach")
        self.original_detach = self.runtime.detach
        self.original_attach = self.runtime.attach

        def detach(token):
            try:
                if self.enabled and not capture_flags():
                    self.deny(key(trace.get_current_span()))
            except Exception:  # noqa: BLE001 - observer faults cannot alter native context exits.
                self.faulted = True
            return self.original_detach(token)

        self.guard = detach
        self.runtime.detach = detach
        try:
            active = getattr(provider, "_active_span_processor", None)
            if active is None:
                provider.add_span_processor(self)
            else:
                with active._lock:
                    active._span_processors = (self, *active._span_processors)
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
                state = self.active.get(k, self.closed.get(k))
                if state is None:
                    return False
                observed = self.refs.get(k)
                if not state[0] or (
                    observed is not None and not span_permitted(observed)
                ):
                    if k in self.active:
                        self.active[k] = (False, state[1])
                    return False
                k = state[1]
            return True

    def enroll(self, parent):
        k = key(parent)
        if not k or parent.get_span_context().is_remote:
            return True
        with self.lock:
            if k not in self.active and k not in self.closed:
                self.closed[k] = (False, None)
                while len(self.closed) > 4096:
                    self.closed.popitem(last=False)
            return self.bound(k)

    def deny(self, k, readable=None):
        with self.lock:
            if k:
                s = self.active.get(k, self.closed.get(k, (False, None)))
                if k in self.active:
                    self.active[k] = (False, s[1])
                else:
                    self.closed[k] = (False, s[1])
            if not self.scrubbing:
                self.scrubbing = True
                try:
                    self.scrub(readable)
                finally:
                    self.scrubbing = False
            while len(self.closed) > 4096:
                self.closed.popitem(last=False)

    def on_start(self, span, parent_context=None):
        if not self.enabled:
            return
        try:
            parent = trace.get_current_span(parent_context)
            k = key(span)
            with self.lock:
                self.active[k] = (
                    permitted(parent_context)
                    and span_permitted(span)
                    and self.enroll(parent),
                    None if parent.get_span_context().is_remote else key(parent),
                )
                self.refs[k] = span
        except Exception:  # noqa: BLE001 - observer faults must preserve native SDK behavior.
            self.faulted = True

    def on_end(self, span):
        if not self.enabled:
            return
        try:
            with self.lock:
                k = key(span)
                s = self.active.get(k, self.closed.get(k, (False, None)))
                if not permitted() or not span_permitted(span) or not self.bound(k):
                    self.deny(k, span)
                self.closed[k] = self.active.pop(k, self.closed.get(k, s))
                self.refs.pop(k, None)
                while len(self.closed) > 4096:
                    self.closed.popitem(last=False)
        except Exception:  # noqa: BLE001 - observer faults must preserve native SDK behavior.
            self.faulted = True

    def close(self):
        self.enabled = False
        if self.runtime.detach is self.guard:
            if self.detach_present:
                self.runtime.detach = self.original_stored
            else:
                delattr(self.runtime, "detach")
        processor = getattr(self.provider, "_active_span_processor", None)
        lock = getattr(processor, "_lock", None)
        if lock is not None:
            with lock:
                processor._span_processors = tuple(
                    p for p in processor._span_processors if p is not self
                )
        self.active.clear()
        self.refs.clear()
        self.closed.clear()

    def shutdown(self):
        self.close()
