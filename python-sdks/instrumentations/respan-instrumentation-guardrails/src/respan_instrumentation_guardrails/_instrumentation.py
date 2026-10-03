"""Guardrails native OpenTelemetry integration for Respan."""

import importlib
import logging
import os
from threading import RLock
from typing import Any, ClassVar

from opentelemetry import context, trace
from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor
from respan_tracing.constants.context_constants import ENABLE_CONTENT_TRACING_KEY
from respan_tracing.core.tracer import RespanTracer
from respan_tracing.utils.span_factory import read_propagated_attributes

from respan_instrumentation_guardrails._translation import normalize

logger = logging.getLogger(__name__)
GUARDRAILS_RUNTIME_MODULE = "guardrails"


def _runtime_disabled(parent_context=None) -> bool:
    return (
        context.get_value(ENABLE_CONTENT_TRACING_KEY) is False
        or context.get_value(ENABLE_CONTENT_TRACING_KEY, parent_context) is False
    )


def _capture_content(parent_context=None) -> bool:
    if _runtime_disabled(parent_context):
        return False
    return os.getenv("TRACELOOP_TRACE_CONTENT", "true").lower() == "true" or bool(
        context.get_value("override_enable_content_tracing", parent_context)
    )


class GuardrailsSpanProcessor(SpanProcessor):
    """Normalize native spans before the active provider's exporters run."""

    def __init__(self) -> None:
        self.enabled = True
        self._lock = RLock()
        self._propagated_by_trace: dict[int, dict[str, Any]] = {}
        self._active_spans_by_trace: dict[int, int] = {}
        self._parents: dict[tuple[int, int], int | None] = {}
        self._content: dict[tuple[int, int], bool] = {}
        self._runtime_disabled_by_span: dict[tuple[int, int], bool] = {}
        self._llm_calls: dict[tuple[int, int], int] = {}

    @staticmethod
    def _identity(span: Any) -> tuple[int | None, int | None]:
        getter = getattr(span, "get_span_context", None)
        span_context = getter() if callable(getter) else None
        return getattr(span_context, "trace_id", None), getattr(
            span_context, "span_id", None
        )

    def on_start(self, span: Any, parent_context: Any = None) -> None:
        if not self.enabled:
            return
        trace_id, span_id = self._identity(span)
        with self._lock:
            if trace_id is not None:
                self._active_spans_by_trace[trace_id] = (
                    self._active_spans_by_trace.get(trace_id, 0) + 1
                )
            propagated = read_propagated_attributes()
            if trace_id is not None:
                if propagated:
                    self._propagated_by_trace.setdefault(trace_id, dict(propagated))
                else:
                    propagated = self._propagated_by_trace.get(trace_id, {})
            if trace_id is not None and span_id is not None:
                key = (trace_id, span_id)
                self._parents[key] = getattr(
                    getattr(span, "parent", None), "span_id", None
                )
                parent_key = (trace_id, self._parents[key])
                disabled = _runtime_disabled(
                    parent_context
                ) or self._runtime_disabled_by_span.get(parent_key, False)
                self._runtime_disabled_by_span[key] = disabled
                self._content[key] = not disabled and _capture_content(parent_context)
                self._llm_calls[key] = 0
        scope = getattr(getattr(span, "instrumentation_scope", None), "name", "")
        if scope != "guardrails-ai" and not scope.startswith("guardrails.telemetry."):
            return
        for key, value in propagated.items():
            if (getattr(span, "attributes", None) or {}).get(key) is None:
                span.set_attribute(key, value)

    def on_end(self, span: ReadableSpan) -> None:
        if not self.enabled:
            return
        trace_id, span_id = self._identity(span)
        key = (trace_id, span_id)
        with self._lock:
            try:
                is_llm = normalize(
                    span,
                    self._content.get(key, _capture_content()),
                    self._llm_calls.get(key),
                )
                if is_llm:
                    parent = self._parents.get(key)
                    seen = set()
                    while parent is not None and parent not in seen:
                        seen.add(parent)
                        parent_key = (trace_id, parent)
                        if parent_key not in self._llm_calls:
                            break
                        self._llm_calls[parent_key] += 1
                        parent = self._parents.get(parent_key)
            except Exception:
                logger.exception("Failed to normalize Guardrails telemetry")
            finally:
                self._parents.pop(key, None)
                self._content.pop(key, None)
                self._runtime_disabled_by_span.pop(key, None)
                self._llm_calls.pop(key, None)
                if trace_id is not None:
                    remaining = self._active_spans_by_trace.get(trace_id, 1) - 1
                    if remaining <= 0:
                        self._active_spans_by_trace.pop(trace_id, None)
                        self._propagated_by_trace.pop(trace_id, None)
                    else:
                        self._active_spans_by_trace[trace_id] = remaining

    def shutdown(self) -> None:
        with self._lock:
            self.enabled = False
            self._propagated_by_trace.clear()
            self._active_spans_by_trace.clear()
            self._parents.clear()
            self._content.clear()
            self._runtime_disabled_by_span.clear()
            self._llm_calls.clear()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


class GuardrailsInstrumentor:
    """Translate Guard/AsyncGuard native spans without wrapping application calls."""

    name = "guardrails"
    _lock: ClassVar[RLock] = RLock()
    _providers: ClassVar[dict[Any, tuple[GuardrailsSpanProcessor, set]]] = {}

    def __init__(self) -> None:
        self._provider = None
        self._is_instrumented = False

    @property
    def is_instrumented(self) -> bool:
        return self._is_instrumented

    def activate(self) -> None:
        if self._is_instrumented:
            return
        runtime = getattr(RespanTracer, "_instance", None)
        if runtime is not None and not getattr(runtime, "is_enabled", True):
            logger.info(
                "Guardrails instrumentation skipped because Respan tracing is disabled"
            )
            return
        try:
            _ = importlib.import_module(GUARDRAILS_RUNTIME_MODULE).Guard
        except (AttributeError, ImportError) as exc:
            logger.warning(
                "Failed to activate Guardrails instrumentation — missing runtime dependency: %s",
                exc,
            )
            return
        provider = trace.get_tracer_provider()
        if not callable(getattr(provider, "add_span_processor", None)):
            logger.warning(
                "Guardrails instrumentation requires an initialized tracer provider"
            )
            return
        with self._lock:
            if provider not in self._providers:
                processor = GuardrailsSpanProcessor()
                active = getattr(provider, "_active_span_processor", None)
                processors = getattr(active, "_span_processors", None)
                if processors is not None:
                    active._span_processors = (processor, *processors)
                else:
                    provider.add_span_processor(processor)
                self._providers[provider] = (processor, set())
            self._providers[provider][1].add(self)
            self._provider = provider
            self._is_instrumented = True

    def deactivate(self) -> None:
        with self._lock:
            if not self._is_instrumented:
                return
            state = self._providers.get(self._provider)
            if state is not None:
                processor, owners = state
                owners.discard(self)
                if not owners:
                    active = getattr(self._provider, "_active_span_processor", None)
                    processors = getattr(active, "_span_processors", None)
                    if processors is not None:
                        active._span_processors = tuple(
                            item for item in processors if item is not processor
                        )
                    processor.shutdown()
                    del self._providers[self._provider]
            self._provider = None
            self._is_instrumented = False
