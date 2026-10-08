"""Scoped synchronous/asynchronous bridge around the upstream AgentSpec processor."""

from __future__ import annotations

import json
import re
import sys
from asyncio import CancelledError
from uuid import UUID

from openinference.semconv.trace import SpanAttributes as OISpanAttributes
from opentelemetry import context, trace
from opentelemetry.semconv._incubating.attributes.error_attributes import ERROR_MESSAGE
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import (
    GEN_AI_RESPONSE_FINISH_REASONS,
    GEN_AI_RESPONSE_ID,
    GEN_AI_TOOL_CALL_ID,
    GEN_AI_USAGE_INPUT_TOKENS,
    GEN_AI_USAGE_OUTPUT_TOKENS,
)
from opentelemetry.semconv.attributes.error_attributes import ERROR_TYPE
from opentelemetry.semconv.attributes.http_attributes import HTTP_RESPONSE_STATUS_CODE
from opentelemetry.semconv_ai import SpanAttributes
from respan_sdk.constants.span_attributes import RESPAN_LOG_TYPE
from respan_tracing.decorators.base import _should_send_prompts
from respan_tracing.utils.span_factory import read_propagated_attributes

from ._callbacks import _CAPTURE


def safe_json(value):
    def default(item):
        try:
            return item.model_dump(mode="json")
        except Exception:  # noqa: BLE001 - bounded telemetry serialization is best effort
            return {"type": type(item).__name__}

    try:
        encoded = json.dumps(
            value, default=default, separators=(",", ":"), ensure_ascii=False
        )
    except Exception:  # noqa: BLE001 - bounded telemetry serialization is best effort
        return json.dumps({"type": type(value).__name__, "unavailable": True})
    if len(encoded) > 32_000:
        encoded = json.dumps({"preview": encoded[:30_000], "truncated": True})
    return encoded


def safe_error(value):
    text = value if isinstance(value, str) else type(value).__name__
    text = re.sub(r"(?i)(bearer|basic)\s+[^\s,;]+", r"\1 [REDACTED]", text)
    return re.sub(
        r"""(?i)((?:["'])?(?:api[_-]?key|token|secret|password|authorization)(?:["'])?\s*[:=]\s*)(?:"[^"]*"|'[^']*'|[^\s,;}]+)""",
        r"\1[REDACTED]",
        text,
    )[:4000]


def make_processor(base, *, provider, **kwargs):
    class ScopedProcessor(base):
        def __init__(self):
            super().__init__(**kwargs)
            self._policies = {}
            self._native_spans = {}
            self._active = True

        def on_start(self, span):
            parent = getattr(span, "_parent_span", None)
            parent_policy = self._policies.get(getattr(parent, "id", None))
            allowed = self._active and not context.get_value(
                context._SUPPRESS_INSTRUMENTATION_KEY
            )
            if parent_policy is not None and not parent_policy["allowed"]:
                allowed = False
            sampler = getattr(provider, "sampler", None)
            if sampler is not None and allowed:
                sampling_context = context.get_current()
                if parent is not None:
                    parent_context = trace.SpanContext(
                        int(UUID(span._trace.id)),
                        (int(UUID(parent.id)) % (2**64 - 1)) + 1,
                        False,
                        trace.TraceFlags(1),
                    )
                    sampling_context = trace.set_span_in_context(
                        trace.NonRecordingSpan(parent_context)
                    )
                allowed = sampler.should_sample(
                    sampling_context, int(UUID(span._trace.id)), span.name
                ).decision.is_sampled()
            policy = {
                "allowed": allowed,
                "parent": getattr(parent, "id", None),
                "capture": allowed
                and not self.mask_sensitive_information
                and _should_send_prompts()
                and (parent_policy is None or parent_policy["capture"]),
                "extra": {},
                "propagated": read_propagated_attributes() if allowed else {},
            }
            self._policies[span.id] = policy
            if allowed:
                self._native_spans[span.id] = span
            if allowed:
                callback = _CAPTURE.get() or {}
                if policy["capture"] and callback.get("messages") is not None:
                    policy["messages"] = callback["messages"]
                super().on_start(span)

        def _create_otel_span_from_agentspec_span(self, span):
            otel = super()._create_otel_span_from_agentspec_span(span)
            policy = self._policies.get(
                span.id, {"capture": False, "extra": {}, "propagated": {}}
            )
            extra = {**policy["propagated"], **policy["extra"]}
            callback = _CAPTURE.get() or {}
            if type(span).__name__ == "NodeExecutionSpan":
                from respan_sdk.constants.llm_logging import LOG_TYPE_TASK

                extra[RESPAN_LOG_TYPE] = LOG_TYPE_TASK
            if type(span).__name__ == "LlmGenerationSpan":
                config = getattr(span, "llm_config", None)
                provider_name = getattr(config, "provider", None)
                if provider_name is None and type(config).__name__ == "OpenAiConfig":
                    provider_name = "openai"
                if isinstance(provider_name, str) and provider_name:
                    extra[SpanAttributes.LLM_SYSTEM] = provider_name
                completion = callback.get("completion", {})
                if policy["capture"] and completion.get("tool_calls"):
                    extra[f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls"] = safe_json(
                        completion["tool_calls"]
                    )
                usage = callback.get("usage", {})
                if callback.get("response_id"):
                    extra[GEN_AI_RESPONSE_ID] = callback["response_id"]
                if isinstance(callback.get("finish_reason"), str):
                    extra[GEN_AI_RESPONSE_FINISH_REASONS] = [callback["finish_reason"]]
                if any(
                    type(event).__name__
                    in {"LlmGenerationChunkReceived", "ToolCallChunkReceived"}
                    for event in span.events
                ):
                    extra[SpanAttributes.GEN_AI_IS_STREAMING] = True
                for source, target in [
                    ("input_tokens", GEN_AI_USAGE_INPUT_TOKENS),
                    ("output_tokens", GEN_AI_USAGE_OUTPUT_TOKENS),
                    ("cache", SpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS),
                    ("reasoning", SpanAttributes.LLM_USAGE_REASONING_TOKENS),
                ]:
                    if usage.get(source) is not None:
                        extra[target] = usage[source]
                if "input_tokens" in usage:
                    extra[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] = usage[
                        "input_tokens"
                    ]
                if "output_tokens" in usage:
                    extra[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS] = usage[
                        "output_tokens"
                    ]
                if "input_tokens" in usage and "output_tokens" in usage:
                    extra[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] = (
                        usage["input_tokens"] + usage["output_tokens"]
                    )
                for i, message in enumerate(policy.get("messages", [])):
                    for key, value in message.items():
                        extra[f"{SpanAttributes.LLM_PROMPTS}.{i}.{key}"] = (
                            safe_json(value) if not isinstance(value, str) else value
                        )
            if type(span).__name__ == "ToolExecutionSpan":
                for event in span.events:
                    if type(event).__name__ == "ToolExecutionRequest":
                        description = getattr(span, "description", "")
                        if description.startswith("tcid__"):
                            extra[GEN_AI_TOOL_CALL_ID] = description.removeprefix(
                                "tcid__"
                            )
                requests = [
                    e.request_id
                    for e in span.events
                    if type(e).__name__ == "ToolExecutionRequest"
                ]
                for event in span.events:
                    if (
                        type(event).__name__ == "ToolExecutionResponse"
                        and event.request_id not in requests
                    ):
                        extra[GEN_AI_TOOL_CALL_ID] = event.request_id
            for event in span.events:
                if type(event).__name__ == "ExceptionRaised":
                    detail = (
                        safe_error(event.exception_message)
                        if policy["capture"]
                        else event.exception_type
                    )
                    otel.set_status(trace.Status(trace.StatusCode.ERROR, detail))
                    extra[ERROR_TYPE] = event.exception_type
                    extra[ERROR_MESSAGE] = detail
            error = callback.get("error") or policy.get("error")
            if error is not None:
                detail = (
                    safe_error(
                        "; ".join(
                            arg
                            for arg in getattr(error, "args", ())
                            if isinstance(arg, str)
                        )
                    )
                    if policy["capture"]
                    else type(error).__name__
                )
                otel.set_status(trace.Status(trace.StatusCode.ERROR, detail))
                extra[ERROR_TYPE] = type(error).__name__
                extra[ERROR_MESSAGE] = detail
            if error is not None and type(span).__name__ == "ToolExecutionSpan":
                # The native error handler calls on_tool_end(output=None) while
                # unwinding. That is not a returned tool result.
                otel._attributes.pop(OISpanAttributes.OUTPUT_VALUE, None)
                otel._attributes.pop(OISpanAttributes.OUTPUT_MIME_TYPE, None)
            status = getattr(error, "status_code", None)
            if (
                isinstance(status, int)
                and not isinstance(status, bool)
                and 400 <= status <= 599
            ):
                extra[HTTP_RESPONSE_STATUS_CODE] = status
            otel._respan_agentspec_extra = extra
            otel._respan_agentspec_capture = policy["capture"]
            return otel

        def _close_children(self, parent_id):
            for child_id, policy in list(self._policies.items()):
                if policy["parent"] != parent_id:
                    continue
                child = self._native_spans.get(child_id)
                if child is not None:
                    self.on_end(child)
                else:
                    self._close_children(child_id)
                    self._policies.pop(child_id, None)

        def on_end(self, span):
            # Some native async generator wrappers close the enclosing agent
            # before dispatching a terminal model callback. Finalize the bridge
            # children using only events already observed, without changing the
            # runtime span stack or manufacturing a response/usage/error.
            self._close_children(span.id)
            policy = self._policies.get(span.id)
            if policy is not None:
                policy["capture"] = policy["capture"] and _should_send_prompts()
                if isinstance(sys.exception(), (Exception, CancelledError)) and not any(
                    type(event).__name__
                    in {
                        "LlmGenerationResponse",
                        "AgentExecutionEnd",
                        "FlowExecutionEnd",
                        "ToolExecutionResponse",
                        "NodeExecutionEnd",
                    }
                    for event in span.events
                ):
                    policy["error"] = sys.exception()
            try:
                if policy is not None and policy["allowed"] and self._active:
                    # End-time suppression must not change a span's start policy.
                    token = context.attach(
                        context.set_value(context._SUPPRESS_INSTRUMENTATION_KEY, False)
                    )
                    try:
                        super().on_end(span)
                    finally:
                        context.detach(token)
            finally:
                self._policies.pop(span.id, None)
                self._native_spans.pop(span.id, None)
                self._span_registry.pop(span.id, None)

        async def on_start_async(self, span):
            self.on_start(span)

        async def on_end_async(self, span):
            self.on_end(span)

        async def on_event_async(self, event, span):
            self.on_event(event, span)

        async def startup_async(self):
            self.startup()

        async def shutdown_async(self):
            self.shutdown()

        def shutdown(self):
            self._active = False
            self._policies.clear()
            self._native_spans.clear()
            super().shutdown()

    return ScopedProcessor()
