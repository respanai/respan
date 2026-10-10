"""Checkpoint native span scopes before their context is detached."""

from __future__ import annotations

import logging
from contextlib import contextmanager

from opentelemetry import context, trace
from opentelemetry.trace import INVALID_SPAN_CONTEXT, NonRecordingSpan

from ._policy import key, suppressed


class TracerGuard:
    def __init__(self, native, runtime):
        self.native = native
        self.runtime = runtime

    def __getattr__(self, name):
        return getattr(self.native, name)

    def start_span(self, *args, **kwargs):
        if self.runtime.active and (suppressed() or suppressed(kwargs.get("context"))):
            return NonRecordingSpan(INVALID_SPAN_CONTEXT)
        return self.native.start_span(*args, **kwargs)

    @contextmanager
    def start_as_current_span(self, *args, **kwargs):
        options = {
            k: kwargs.pop(k, default)
            for k, default in [
                ("end_on_exit", True),
                ("record_exception", True),
                ("set_status_on_exception", True),
            ]
        }
        span = self.start_span(*args, **kwargs)
        with trace.use_span(span, **options):
            try:
                yield span
            finally:
                try:
                    self.runtime.processor.policy.observe(span)
                except Exception:  # noqa: BLE001 - telemetry observes without replacing native behavior.
                    self.runtime.processor.policy.veto(key(span))


def install_detach(runtime):
    original = context.detach

    def detached(token):
        try:
            current = trace.get_current_span()
            if key(current) in runtime.processor.policy.open:
                runtime.processor.policy.observe(current)
                runtime.apply_call_bounds()
        except Exception:  # noqa: BLE001 - telemetry observes without replacing native behavior.
            logging.getLogger(__name__).debug("Skipped native detach observation")
        return original(token)

    runtime.patch(context, "detach", detached)
