"""Native OpenLIT tracer boundaries without replacing its SDK wrappers."""

import logging
from contextlib import contextmanager

from opentelemetry import context, trace
from opentelemetry.trace import INVALID_SPAN_CONTEXT, NonRecordingSpan

from ._policy import suppressed

logger = logging.getLogger(__name__)


class TracerGuard:
    def __init__(self, original, processor):
        self.original = original
        self.processor = processor

    def start_span(self, *args, **kwargs):
        if not self.processor.active or suppressed(kwargs.get("context")):
            return NonRecordingSpan(INVALID_SPAN_CONTEXT)
        return self.original.start_span(*args, **kwargs)

    @contextmanager
    def start_as_current_span(self, *args, **kwargs):
        if not self.processor.active:
            yield NonRecordingSpan(INVALID_SPAN_CONTEXT)
            return
        options = {
            k: kwargs.pop(k, True)
            for k in ("end_on_exit", "record_exception", "set_status_on_exception")
        }
        span = self.start_span(*args, **kwargs)
        with trace.use_span(span, **options):
            try:
                yield span
            finally:
                try:
                    self.processor.allowed(span)
                except Exception:  # noqa: BLE001 - telemetry must preserve native outcomes.
                    self.processor.veto(span)

    def __getattr__(self, key):
        return getattr(self.original, key)


def install_provider_guard(provider, processor):
    original = getattr(provider, "get_tracer", None)
    if not callable(original):
        return None
    previous = vars(provider).get("get_tracer")

    def guarded(name, *args, **kwargs):
        tracer = original(name, *args, **kwargs)
        return (
            TracerGuard(tracer, processor)
            if name == "openlit"
            or isinstance(name, str)
            and name.startswith("openlit.")
            else tracer
        )

    provider.get_tracer = guarded
    return provider, previous, guarded


def remove_provider_guard(hook):
    if hook is None:
        return
    provider, previous, installed = hook
    if vars(provider).get("get_tracer") is installed:
        if previous is None:
            del provider.get_tracer
        else:
            provider.get_tracer = previous


def install_context_guard(processor):
    original = context.detach

    def detached(token):
        try:
            if processor.active or processor._closing_provider is not None:
                processor.observe_detach(trace.get_current_span())
        except Exception as exc:  # noqa: BLE001 - checkpoints preserve native context APIs.
            logger.debug("OpenLIT detach checkpoint skipped: %s", type(exc).__name__)
        return original(token)

    context.detach = detached
    return original, detached


def remove_context_guard(hook):
    if hook is not None and context.detach is hook[1]:
        context.detach = hook[0]
