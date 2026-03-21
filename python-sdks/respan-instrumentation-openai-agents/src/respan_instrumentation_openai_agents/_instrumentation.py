"""OpenAI Agents SDK instrumentation plugin for Respan."""

import logging
import threading
from typing import Any, Dict, List, Optional, Union

from agents.tracing.processor_interface import TracingProcessor
from agents.tracing.spans import Span
from agents.tracing.traces import Trace

from ._converter import convert_to_respan_log
from ._otel_emitter import emit_sdk_item

logger = logging.getLogger(__name__)


class _RespanTracingProcessor(TracingProcessor):
    """OpenAI Agents SDK TracingProcessor that emits spans into the OTEL pipeline."""

    def on_trace_start(self, trace: Trace) -> None:
        pass

    def on_trace_end(self, trace: Trace) -> None:
        emit_sdk_item(trace)

    def on_span_start(self, span: Span[Any]) -> None:
        pass

    def on_span_end(self, span: Span[Any]) -> None:
        emit_sdk_item(span)

    def shutdown(self) -> None:
        pass

    def force_flush(self) -> None:
        pass


class OpenAIAgentsInstrumentor:
    """Respan instrumentor for the OpenAI Agents SDK.

    Registers a ``TracingProcessor`` that converts OpenAI Agents SDK
    traces/spans to OTEL ``ReadableSpan`` objects and injects them into
    the single OTEL pipeline (``TracerProvider`` → ``RespanSpanExporter``
    → ``/v2/traces``).

    Usage::

        from respan import Respan
        from respan_instrumentation_openai_agents import OpenAIAgentsInstrumentor

        respan = Respan(instrumentations=[OpenAIAgentsInstrumentor()])
    """

    name = "openai-agents"

    def __init__(self) -> None:
        self._processor: Optional[_RespanTracingProcessor] = None

    def activate(self) -> None:
        """Register the tracing processor with the OpenAI Agents SDK.

        Replaces the default OpenAI backend processor so traces are only
        sent to Respan, not to OpenAI's tracing endpoint.
        """
        from agents.tracing import set_trace_processors

        self._processor = _RespanTracingProcessor()
        set_trace_processors([self._processor])
        logger.info("OpenAI Agents SDK instrumentation activated")

    def deactivate(self) -> None:
        """Deactivate the instrumentation."""
        self._processor = None
        logger.info("OpenAI Agents SDK instrumentation deactivated")


class LocalSpanCollector(TracingProcessor):
    """Thread-safe, in-process span collector for self-hosted deployments.

    Instead of sending spans over HTTP, this processor converts them using
    the same ``convert_to_respan_log`` logic and stores them in memory keyed
    by ``trace_id``.  After an agent run completes, call
    ``pop_trace(trace_id)`` to retrieve (and remove) the spans for that run.

    Register globally once at application startup::

        from agents import set_trace_processors
        collector = LocalSpanCollector()
        set_trace_processors([collector])

    Then after each ``Runner.run_streamed()``::

        spans = collector.pop_trace(trace_id)
        for span_data in spans:
            log_request(...)
    """

    def __init__(self) -> None:
        self._traces: Dict[str, List[Dict[str, Any]]] = {}
        self._lock = threading.Lock()

    def on_trace_start(self, trace: Trace) -> None:
        pass

    def on_trace_end(self, trace: Trace) -> None:
        data = convert_to_respan_log(trace)
        if data:
            with self._lock:
                self._traces.setdefault(trace.trace_id, []).insert(0, data)

    def on_span_start(self, span: Span[Any]) -> None:
        pass

    def on_span_end(self, span: Span[Any]) -> None:
        data = convert_to_respan_log(span)
        if data:
            trace_id = span.trace_id if hasattr(span, "trace_id") else None
            if trace_id:
                with self._lock:
                    self._traces.setdefault(trace_id, []).append(data)

    def shutdown(self) -> None:
        with self._lock:
            self._traces.clear()

    def force_flush(self) -> None:
        pass

    def pop_trace(self, trace_id: str) -> List[Dict[str, Any]]:
        """Retrieve and remove all collected spans for a trace.

        Returns an empty list if no spans were collected for *trace_id*.
        Thread-safe.
        """
        with self._lock:
            return self._traces.pop(trace_id, [])
