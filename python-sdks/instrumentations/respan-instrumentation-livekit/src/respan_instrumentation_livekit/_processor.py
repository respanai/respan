"""Normalize owned native LiveKit spans before downstream exporters."""

from __future__ import annotations

from livekit.agents.telemetry import trace_types
from opentelemetry import trace
from opentelemetry.sdk.trace import SpanProcessor
from opentelemetry.semconv._incubating.attributes import gen_ai_attributes as GenAI
from opentelemetry.semconv._incubating.attributes.error_attributes import ERROR_MESSAGE
from opentelemetry.semconv_ai import SpanAttributes as AI
from opentelemetry.trace import Status, StatusCode
from respan_sdk.constants.llm_logging import LogMethodChoices
from respan_sdk.constants.span_attributes import RESPAN_LOG_METHOD, RESPAN_LOG_TYPE

from ._policy import CapturePolicy, key, private_attributes
from ._serialization import text
from ._translator import (
    build_livekit_llm_attrs,
    build_native_tool_attrs,
    is_livekit_llm_span,
    usage_attributes,
)


class LiveKitSpanProcessor(SpanProcessor):
    def __init__(self, capture_content=True):
        self.policy = CapturePolicy(capture_content)
        self.active = True
        self.on_drained = None
        self.source_usage = {}

    def on_start(self, span, parent_context=None):
        if self.active:
            try:
                self.policy.start(span, parent_context)
            except Exception:  # noqa: BLE001 - privacy observation cannot replace native span creation.
                self.policy.open[key(span)] = [
                    span,
                    False,
                    key(trace.get_current_span(parent_context)),
                ]

    def on_end(self, span):
        if key(span) not in self.policy.open:
            return
        source_usage = self.source_usage.pop(key(span), {})
        try:
            capture = self.policy.end(span)
        except Exception:  # noqa: BLE001 - callbacks cannot replace native values/errors.
            state = self.policy.open.pop(key(span), [None, False, None])
            state[0] = None
            state[1] = False
            self.policy.closed[key(span)] = state
            capture = False
        if not self.policy.owned(span):
            return
        try:
            original = dict(span.attributes or {})
            attrs = dict(original)
            if is_livekit_llm_span(span.name, attrs):
                translated = build_livekit_llm_attrs(
                    span_name=span.name,
                    attrs=attrs,
                    events=span.events,
                    capture=capture,
                    source_usage=source_usage,
                )
                for k in list(attrs):
                    if k.startswith(("gen_ai.usage.", "llm.usage.")):
                        attrs.pop(k, None)
                attrs.update(translated)
            elif (
                span.name == "function_tool"
                and getattr(span.instrumentation_scope, "name", None)
                == "livekit-agents"
            ):
                attrs.update(build_native_tool_attrs(original, capture=capture))
                if original.get(trace_types.ATTR_FUNCTION_TOOL_IS_ERROR) is True:
                    span._status = Status(StatusCode.ERROR)
            else:
                attrs.setdefault(RESPAN_LOG_TYPE, "task")
                attrs.setdefault(
                    RESPAN_LOG_METHOD, LogMethodChoices.TRACING_INTEGRATION.value
                )
                attrs.setdefault(AI.TRACELOOP_ENTITY_NAME, text(span.name))
                attrs.setdefault(AI.TRACELOOP_ENTITY_PATH, "")
            for k in list(attrs):
                if k.startswith(("lk.", "langfuse.")) or k in {
                    "status_code",
                    ERROR_MESSAGE,
                    "tools",
                    "tool_calls",
                    "model",
                    "prompt_tokens",
                    "completion_tokens",
                    "total_request_tokens",
                }:
                    attrs.pop(k, None)
                elif isinstance(attrs[k], str):
                    attrs[k] = text(attrs[k])
                if attrs.get(RESPAN_LOG_TYPE) == "task" and k.startswith(
                    ("gen_ai.", "llm.")
                ):
                    attrs.pop(k, None)
                if (
                    attrs.get(RESPAN_LOG_TYPE) == "tool"
                    and k.startswith("gen_ai.tool.")
                    and k != GenAI.GEN_AI_TOOL_CALL_ID
                ):
                    attrs.pop(k, None)
            if span.status.status_code is StatusCode.ERROR:
                attrs.pop(AI.TRACELOOP_ENTITY_OUTPUT, None)
                for k in list(attrs):
                    if k.startswith(AI.LLM_COMPLETIONS + "."):
                        attrs.pop(k, None)
                span._status = Status(
                    StatusCode.ERROR, text(span.status.description) if capture else None
                )
            if not capture:
                attrs = private_attributes(attrs)
            span._attributes = attrs
            span._events = ()
        except Exception:  # noqa: BLE001 - telemetry observes without replacing native behavior.
            fallback = private_attributes(dict(span.attributes or {}))
            for name in list(fallback):
                if name.startswith(("gen_ai.usage.", "llm.usage.")):
                    fallback.pop(name, None)
            fallback.update(usage_attributes(source_usage))
            span._attributes = fallback
            span._events = ()
            if span.status.status_code is StatusCode.ERROR:
                span._status = Status(StatusCode.ERROR)
        if self.on_drained is not None:
            self.on_drained()

    def shutdown(self):
        self.policy.open.clear()
        self.policy.closed.clear()
        self.source_usage.clear()

    def force_flush(self, timeout_millis=30000):
        return True
