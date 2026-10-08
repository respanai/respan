"""Start-time content bounds and final privacy veto for processor delegates."""

from __future__ import annotations

import logging
import os
import threading
from collections import OrderedDict
from typing import Any

from openinference.semconv.trace import SpanAttributes as OISpanAttributes
from opentelemetry import context, trace
from opentelemetry.semconv._incubating.attributes import gen_ai_attributes as GenAI
from opentelemetry.semconv._incubating.attributes.error_attributes import ERROR_MESSAGE
from opentelemetry.semconv._incubating.attributes.exception_attributes import (
    EXCEPTION_MESSAGE,
    EXCEPTION_STACKTRACE,
)
from opentelemetry.semconv_ai import (
    SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY,
    SpanAttributes,
)
from respan_tracing.constants.context_constants import ENABLE_CONTENT_TRACING_KEY

logger = logging.getLogger(__name__)


def _allowed(ctx: Any = None) -> bool:
    return (
        context.get_value(ENABLE_CONTENT_TRACING_KEY, ctx) is not False
        and not context.get_value(context._SUPPRESS_INSTRUMENTATION_KEY, ctx)
        and not context.get_value(SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY, ctx)
        and os.getenv("TRACELOOP_TRACE_CONTENT", "true").strip().lower()
        not in {"0", "false", "off", "no"}
    )


def _key(span: Any) -> tuple[int, int] | None:
    value = (
        span.get_span_context()
        if hasattr(span, "get_span_context")
        else getattr(span, "context", None)
    )
    if value is not None and value.is_valid:
        return value.trace_id, value.span_id
    return None


def clear_content(attrs: Any) -> None:
    """Remove both source and canonical payload before any serializer runs."""
    exact = {
        ERROR_MESSAGE,
        EXCEPTION_MESSAGE,
        EXCEPTION_STACKTRACE,
        SpanAttributes.TRACELOOP_ENTITY_INPUT,
        SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
        SpanAttributes.LLM_REQUEST_FUNCTIONS,
        SpanAttributes.GEN_AI_REQUEST_STRUCTURED_OUTPUT_SCHEMA,
        OISpanAttributes.LLM_INVOCATION_PARAMETERS,
        OISpanAttributes.LLM_FUNCTION_CALL,
        OISpanAttributes.TOOL_PARAMETERS,
        OISpanAttributes.METADATA,
    }
    exact.update(
        getattr(GenAI, name, None)
        for name in (
            "GEN_AI_INPUT_MESSAGES",
            "GEN_AI_OUTPUT_MESSAGES",
            "GEN_AI_SYSTEM_INSTRUCTIONS",
            "GEN_AI_TOOL_DEFINITIONS",
            "GEN_AI_TOOL_CALL_ARGUMENTS",
            "GEN_AI_TOOL_CALL_RESULT",
            "GEN_AI_RETRIEVAL_QUERY_TEXT",
            "GEN_AI_RETRIEVAL_DOCUMENTS",
        )
    )
    prefixes = (
        f"{SpanAttributes.LLM_PROMPTS}.",
        f"{SpanAttributes.LLM_COMPLETIONS}.",
        "input.",
        "output.",
        f"{OISpanAttributes.LLM_INPUT_MESSAGES}.",
        f"{OISpanAttributes.LLM_OUTPUT_MESSAGES}.",
        f"{OISpanAttributes.LLM_TOOLS}",
        "embedding.embeddings.",
        "retrieval.documents.",
        "reranker.",
        "llm.prompt_template.",
        "llm.prompts",
        "llm.choices",
    )
    for key in tuple(attrs):
        if key in exact or key.startswith(prefixes):
            attrs.pop(key, None)


class ContentPolicy:
    """Keep booleans/IDs only; no prompt or response buffers are retained."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._active: dict[tuple[int, int], tuple[bool, tuple[int, int] | None]] = {}
        self._denied: OrderedDict[tuple[int, int], None] = OrderedDict()

    def _parent_allowed(self, parent: tuple[int, int] | None) -> bool:
        seen = set()
        while parent and parent not in seen:
            seen.add(parent)
            if parent in self._denied:
                return False
            state = self._active.get(parent)
            if state is None:
                break
            if not state[0]:
                return False
            parent = state[1]
        return True

    def start(self, span: Any, parent_context: Any = None) -> None:
        key = _key(span)
        if key is None:
            return
        parent = _key(trace.get_current_span(parent_context))
        with self._lock:
            try:
                allowed = (
                    _allowed()
                    and _allowed(parent_context)
                    and self._parent_allowed(parent)
                )
            except Exception:
                logger.debug("Content policy unavailable", exc_info=True)
                allowed = False
            self._active[key] = (allowed, parent)

    def end(self, span: Any) -> bool:
        key = _key(span)
        with self._lock:
            initial, parent = self._active.pop(key, (False, None))
            try:
                allowed = initial and _allowed() and self._parent_allowed(parent)
            except Exception:
                logger.debug("Content policy unavailable", exc_info=True)
                allowed = False
            if not allowed and key is not None:
                self._denied[key] = None
                # Finished private parents can still be linked by delayed SDK spans.
                while len(self._denied) > 8192:
                    self._denied.popitem(last=False)
            return allowed

    def observe_veto(self, span: Any) -> None:
        """Observe callback policy before native scope detachment restores it."""
        key = _key(span)
        with self._lock:
            state = self._active.get(key)
            if state is None:
                return
            try:
                allowed = _allowed() and self._parent_allowed(state[1])
            except Exception:
                logger.debug("Content veto unavailable", exc_info=True)
                allowed = False
            if not allowed:
                self._active[key] = (False, state[1])

    def clear(self) -> None:
        with self._lock:
            self._active.clear()
            self._denied.clear()
