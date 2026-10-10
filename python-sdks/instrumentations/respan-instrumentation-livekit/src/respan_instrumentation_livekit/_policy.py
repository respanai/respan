"""Bodyless native-parent capture bounds and irreversible content vetoes."""

from __future__ import annotations

import os
import sys
import threading
from collections import OrderedDict

from opentelemetry import context, trace
from opentelemetry.sdk.util import BoundedList
from opentelemetry.semconv._incubating.attributes import gen_ai_attributes as GenAI
from opentelemetry.semconv.attributes.error_attributes import ERROR_TYPE
from opentelemetry.semconv.attributes.http_attributes import HTTP_RESPONSE_STATUS_CODE
from opentelemetry.semconv_ai import SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY
from opentelemetry.semconv_ai import SpanAttributes as AI
from opentelemetry.trace import Status, StatusCode
from respan_sdk.constants.span_attributes import (
    RESPAN_LOG_METHOD,
    RESPAN_LOG_TYPE,
    RESPAN_METADATA,
)
from respan_tracing.constants.context_constants import ENABLE_CONTENT_TRACING_KEY


def suppressed(ctx=None):
    return bool(
        context.get_value(context._SUPPRESS_INSTRUMENTATION_KEY, ctx)
        or context.get_value(SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY, ctx)
    )


def content_allowed(setting=True, ctx=None):
    try:
        genai = sys.modules.get("livekit.agents.telemetry.gen_ai")
        if genai is not None and not genai.capture_content_enabled():
            return False
        native_utils = sys.modules.get("livekit.agents.telemetry.utils")
        redaction = getattr(native_utils, "redaction_enabled", None)
        if callable(redaction) and redaction(
            getattr(trace.get_current_span(ctx), "attributes", None)
        ):
            return False
        processors = getattr(
            getattr(trace.get_tracer_provider(), "_active_span_processor", None),
            "_span_processors",
            (),
        )
        native = [
            p
            for p in processors
            if type(p).__module__ == "livekit.agents.telemetry.pii"
        ]
        if any(getattr(p, "_allow_pii", True) is False for p in native):
            return False
        if not native and os.getenv(
            "LIVEKIT_TELEMETRY_ALLOW_PII", "true"
        ).strip().lower() in {"false", "0", "off", "no"}:
            return False
        return bool(
            setting
            and context.get_value(ENABLE_CONTENT_TRACING_KEY) is not False
            and context.get_value(ENABLE_CONTENT_TRACING_KEY, ctx) is not False
            and os.getenv("TRACELOOP_TRACE_CONTENT", "true").strip().lower()
            not in {"false", "0", "off", "no"}
            and os.getenv("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "true")
            .strip()
            .lower()
            not in {"false", "0", "off", "no"}
            and not suppressed()
            and not suppressed(ctx)
        )
    except Exception:  # noqa: BLE001 - telemetry observes without replacing native behavior.
        return False


def key(span):
    value = (
        span.get_span_context() if hasattr(span, "get_span_context") else span.context
    )
    return (value.trace_id, value.span_id) if value.is_valid else None


def private_attributes(attrs):
    structural = {
        RESPAN_LOG_METHOD,
        RESPAN_LOG_TYPE,
        AI.TRACELOOP_ENTITY_NAME,
        AI.TRACELOOP_ENTITY_PATH,
        AI.LLM_REQUEST_TYPE,
        AI.LLM_SYSTEM,
        AI.LLM_REQUEST_MODEL,
        AI.LLM_RESPONSE_MODEL,
        AI.LLM_IS_STREAMING,
        GenAI.GEN_AI_OPERATION_NAME,
        GenAI.GEN_AI_PROVIDER_NAME,
        GenAI.GEN_AI_TOOL_CALL_ID,
        ERROR_TYPE,
        HTTP_RESPONSE_STATUS_CODE,
    }
    markers = {
        "run_id",
        "example_run_id",
        "example",
        "scenario",
        "framework",
        "example_set",
        "integration",
        "client_mode",
    }
    result = {
        k: v
        for k, v in attrs.items()
        if k in structural
        or k.startswith(
            (
                "gen_ai.usage.",
                "llm.usage.",
                "respan.threads.",
                "respan.trace.",
                "respan.customer_params.",
            )
        )
        or (
            k.startswith(RESPAN_METADATA + ".")
            and k[len(RESPAN_METADATA) + 1 :] in markers
        )
    }
    value = attrs.get(RESPAN_METADATA)
    if isinstance(value, str):
        import json

        try:
            raw = json.loads(value)
        except (ValueError, TypeError):
            raw = {}
        if isinstance(raw, dict):
            result[RESPAN_METADATA] = json.dumps(
                {
                    k: v
                    for k, v in raw.items()
                    if k in markers and isinstance(v, (str, bool, int, float))
                }
            )
    return result


class CapturePolicy:
    def __init__(self, setting):
        self.setting = setting
        self.open = {}
        self.closed = OrderedDict()
        self.lock = threading.RLock()

    def owned(self, span):
        return getattr(getattr(span, "instrumentation_scope", None), "name", None) in {
            "livekit-agents",
            "respan.instrumentation.livekit",
        }

    def bound(self, k):
        seen = set()
        while k is not None and k not in seen:
            seen.add(k)
            state = self.open.get(k) or self.closed.get(k)
            if state is None:
                return True
            if not state[1]:
                return False
            k = state[2]
        return True

    def start(self, span, parent_context=None):
        with self.lock:
            k = key(span)
            parent = trace.get_current_span(parent_context)
            p = key(parent)
            if p is not None and p not in self.open and p not in self.closed:
                parent_parent = getattr(parent, "parent", None)
                pk = (
                    (parent_parent.trace_id, parent_parent.span_id)
                    if parent_parent is not None and parent_parent.is_valid
                    else None
                )
                self.open[p] = [
                    parent,
                    # A local recording parent predates policy observation. Its
                    # initial private bound cannot safely be reconstructed now.
                    content_allowed(self.setting, parent_context)
                    and not parent.is_recording(),
                    pk,
                ]
            if k is not None:
                self.open[k] = [
                    span,
                    content_allowed(self.setting, parent_context) and self.bound(p),
                    p,
                ]

    def clear_span(self, span):
        if not self.owned(span) or not span.is_recording():
            return
        attrs = getattr(span, "_attributes", None)
        if attrs is not None:
            kept = private_attributes(dict(attrs))
            attrs.clear()
            attrs.update(kept)
        if hasattr(span, "_events"):
            span._events = BoundedList(0)
        if span.status.status_code is StatusCode.ERROR:
            span._status = Status(StatusCode.ERROR)

    def veto(self, k):
        with self.lock:
            seen = set()
            while k is not None and k not in seen:
                seen.add(k)
                state = self.open.get(k) or self.closed.get(k)
                if state is None:
                    break
                state[1] = False
                k = state[2]
            for k, state in list(self.open.items()):
                if not self.bound(k):
                    state[1] = False
                    self.clear_span(state[0])

    def observe(self, span):
        with self.lock:
            k = key(span)
            if not content_allowed(self.setting) or not self.bound(k):
                self.veto(k)
            return span.is_recording() and self.bound(k)

    def end(self, span):
        with self.lock:
            k = key(span)
            state = self.open.get(k)
            if state is None:
                return False
            if not content_allowed(self.setting) or not self.bound(k):
                self.veto(k)
            state = self.open.pop(k)
            state[0] = None
            self.closed[k] = state
            while len(self.closed) > 4096:
                self.closed.popitem(last=False)
            return state[1] and self.bound(state[2])
