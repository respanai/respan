"""Content bounds for Burr lifecycle spans and their observed ancestors."""

from __future__ import annotations

import logging
import os
import threading
from collections import OrderedDict
from typing import Any

from opentelemetry import context, trace
from opentelemetry.sdk.trace import (
    ConcurrentMultiSpanProcessor,
    SpanProcessor,
    SynchronousMultiSpanProcessor,
)
from opentelemetry.semconv_ai import SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY
from respan_tracing.constants.context_constants import ENABLE_CONTENT_TRACING_KEY

logger = logging.getLogger(__name__)
_OFF = {"false", "0", "off", "no"}


def suppressed(ctx: Any = None) -> bool:
    return bool(
        context.get_value(context._SUPPRESS_INSTRUMENTATION_KEY, ctx)
        or context.get_value(SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY, ctx)
    )


def permitted(setting: bool = True, ctx: Any = None) -> bool:
    return bool(
        setting
        and context.get_value(ENABLE_CONTENT_TRACING_KEY) is not False
        and context.get_value(ENABLE_CONTENT_TRACING_KEY, ctx) is not False
        and not suppressed()
        and not suppressed(ctx)
        and all(
            os.getenv(name, "true").strip().lower() not in _OFF
            for name in ("TRACELOOP_TRACE_CONTENT", "RESPAN_TRACE_CONTENT")
        )
    )


def key(span: Any) -> tuple[int, int] | None:
    sc = span.get_span_context()
    return (sc.trace_id, sc.span_id) if sc.is_valid else None


class CapturePolicy(SpanProcessor):
    """Observe bodyless parent bounds, including context exits before span ends."""

    def __init__(self, setting: bool) -> None:
        self.setting = setting
        self.active: dict[Any, list[Any]] = {}
        self.closed: OrderedDict[Any, list[Any]] = OrderedDict()
        self.providers: list[Any] = []
        self.lock = threading.RLock()
        self.enabled = True
        self.faulted = False
        self.scrub = lambda span: None
        self.detach = context.detach

        def guarded_detach(token: Any) -> Any:
            try:
                if self.enabled and not permitted(self.setting):
                    self.veto(key(trace.get_current_span()))
            except Exception:  # noqa: BLE001 - telemetry must preserve native detach.
                self.fail_closed()
            return self.detach(token)

        self.guard = guarded_detach
        context.detach = self.guard
        try:
            self.ensure_provider()
        except Exception:
            self.enabled = False
            if context.detach is self.guard:
                context.detach = self.detach
            raise

    def ensure_provider(self) -> None:
        provider = trace.get_tracer_provider()
        if provider not in self.providers and hasattr(provider, "add_span_processor"):
            provider.add_span_processor(self)
            self.providers.append(provider)

    def bound(self, span_key: Any) -> bool:
        with self.lock:
            if self.faulted:
                return False
            seen = set()
            while span_key and span_key not in seen:
                seen.add(span_key)
                state = self.active.get(span_key) or self.closed.get(span_key)
                if state is None:
                    break
                if not state[1]:
                    return False
                span_key = state[2]
            return True

    def enroll(self, span: Any, parent_context: Any = None) -> None:
        with self.lock:
            k = key(span)
            if not k or k in self.active or k in self.closed:
                return
            parent = trace.get_current_span(parent_context)
            p = key(parent)
            if p and p not in self.active and p not in self.closed:
                # A recording parent predating observation has no trustworthy
                # initial bound. Remote/nonrecording carriers remain usable.
                self.active[p] = [
                    parent,
                    parent.get_span_context().is_remote and not parent.is_recording(),
                    None,
                ]
            self.active[k] = [
                span,
                permitted(self.setting, parent_context) and self.bound(p),
                p,
            ]

    def on_start(self, span: Any, parent_context: Any = None) -> None:
        if self.enabled:
            try:
                self.enroll(span, parent_context)
            except Exception:  # noqa: BLE001 - policy observation cannot abort SDK spans.
                self.fail_closed()

    def veto(self, k: Any) -> None:
        with self.lock:
            seen = set()
            while k and k not in seen:
                seen.add(k)
                state = self.active.get(k) or self.closed.get(k)
                if state is None:
                    break
                state[1] = False
                k = state[2]
            for k, state in list(self.active.items()):
                if not self.bound(k):
                    state[1] = False
                    self.scrub(state[0])

    def fail_closed(self) -> None:
        with self.lock:
            self.faulted = True
            for state in self.active.values():
                state[1] = False
                try:
                    self.scrub(state[0])
                except Exception:  # noqa: BLE001 - a faulty span cannot stop cleanup.
                    logger.debug("Burr telemetry cleanup failed")

    def allowed(self, span: Any) -> bool:
        with self.lock:
            k = key(span)
            if not self.enabled or not permitted(self.setting) or not self.bound(k):
                self.veto(k)
                return False
            return span.is_recording() and self.bound(k)

    def on_end(self, span: Any) -> None:
        if self.enabled:
            try:
                with self.lock:
                    k = key(span)
                    if k not in self.active:
                        return
                    if not permitted(self.setting) or not self.bound(k):
                        self.veto(k)
                    state = self.active.pop(k, [None, False, None])
                    state[0] = None
                    self.closed[k] = state
                    while len(self.closed) > 4096:
                        self.closed.popitem(last=False)
            except Exception:  # noqa: BLE001 - observation cannot replace native results.
                self.fail_closed()

    def close(self) -> None:
        self.fail_closed()
        self.enabled = False
        if context.detach is self.guard:
            context.detach = self.detach
        for provider in self.providers:
            processor = getattr(provider, "_active_span_processor", None)
            if isinstance(
                processor, (SynchronousMultiSpanProcessor, ConcurrentMultiSpanProcessor)
            ):
                with processor._lock:
                    processor._span_processors = tuple(
                        item for item in processor._span_processors if item is not self
                    )
        self.providers.clear()
        self.active.clear()
        self.closed.clear()

    def shutdown(self) -> None:
        self.close()
