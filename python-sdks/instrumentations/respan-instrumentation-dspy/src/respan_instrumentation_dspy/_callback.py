"""DSPy native callbacks routed through actual sampled OpenTelemetry spans."""

from __future__ import annotations

import logging
import os
import threading
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from dspy.clients.base_lm import BaseLM
from dspy.utils.callback import ACTIVE_CALL_ID, BaseCallback
from opentelemetry import context, trace
from opentelemetry.semconv._incubating.attributes import gen_ai_attributes
from opentelemetry.semconv.attributes.exception_attributes import (
    EXCEPTION_MESSAGE,
    EXCEPTION_TYPE,
)
from opentelemetry.semconv_ai import (
    SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY,
    SpanAttributes,
)
from respan_sdk.constants.span_attributes import RESPAN_LOG_TYPE
from respan_tracing.constants.context_constants import ENABLE_CONTENT_TRACING_KEY

from respan_instrumentation_dspy._serialization import redact_text
from respan_instrumentation_dspy._utils import (
    add_lm_request_attributes,
    add_lm_usage_attributes,
    get,
    message,
    normalize_messages,
    plain,
    safe_json,
    set_messages,
    tool_definitions,
)

logger = logging.getLogger(__name__)
_CONTENT_BOUND = "dspy.capture_content_bound"
_TOOL_CALLS = ContextVar("respan_dspy_tool_calls", default=())


def allowed() -> bool:
    return (
        context.get_value(ENABLE_CONTENT_TRACING_KEY) is not False
        and context.get_value(_CONTENT_BOUND) is not False
        and os.getenv("TRACELOOP_TRACE_CONTENT", "true").strip().lower()
        not in {"false", "0", "no", "off"}
    )


@dataclass
class _CallState:
    span: Any
    kind: str
    instance: Any
    capture: bool
    token: Any
    content: dict = field(default_factory=dict)
    result: Any = None
    ended: bool = False


class DSPyInstrumentationCallback(BaseCallback):
    def __init__(self, *, include_content=True, tracer_provider=None, policy=None):
        self._include_content = include_content
        self._policy = policy
        self._tracer = (tracer_provider or trace.get_tracer_provider()).get_tracer(
            "respan.instrumentation.dspy"
        )
        self._active_calls: dict[str, _CallState | None] = {}
        self._lock = threading.RLock()
        self._closed = False

    def _start_call(self, call_id, call_kind, instance, inputs):
        try:
            with self._lock:
                if self._closed or call_id in self._active_calls:
                    return
                parent = self._active_calls.get(ACTIVE_CALL_ID.get())
                if (
                    context.get_value(context._SUPPRESS_INSTRUMENTATION_KEY)
                    or context.get_value(SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY)
                    or (ACTIVE_CALL_ID.get() in self._active_calls and parent is None)
                ):
                    self._active_calls[call_id] = None
                    return
                if call_kind == "module" and isinstance(instance, BaseLM):
                    # DSPy 3.0 sends custom BaseLM subclasses to module callbacks.
                    call_kind = "chat"
                if call_kind == "module":
                    call_kind = (
                        "agent"
                        if type(instance).__name__ in {"ReAct", "ReActV2", "RLM"}
                        or not type(instance).__module__.startswith("dspy.")
                        else "task"
                    )
                name = (
                    get(instance, "name", type(instance).__name__)
                    if call_kind == "tool"
                    else type(instance).__name__
                )
                span_name = (
                    f"{call_kind}.{name}"
                    if call_kind in {"agent", "tool"}
                    else "llm"
                    if call_kind == "chat"
                    else call_kind
                )
                parent_context = (
                    trace.set_span_in_context(parent.span)
                    if parent
                    else context.get_current()
                )
                span = self._tracer.start_span(
                    span_name,
                    context=parent_context,
                    attributes={
                        RESPAN_LOG_TYPE: call_kind,
                        SpanAttributes.TRACELOOP_ENTITY_NAME: name,
                        SpanAttributes.TRACELOOP_ENTITY_PATH: "",
                    },
                )
                capture = (
                    span.is_recording()
                    and self._include_content
                    and allowed()
                    and (self._policy(instance) if self._policy else True)
                )
                token = context.attach(
                    context.set_value(
                        _CONTENT_BOUND, capture, trace.set_span_in_context(span)
                    )
                )
                state = _CallState(span, call_kind, instance, capture, token)
                self._active_calls[call_id] = state
            if call_kind == "chat":
                attrs = {}
                add_lm_request_attributes(attrs, instance, inputs)
                from dspy import settings

                if settings.send_stream is not None:
                    attrs[SpanAttributes.GEN_AI_IS_STREAMING] = True
                span.set_attributes(attrs)
            elif call_kind == "embedding":
                span.set_attribute(SpanAttributes.LLM_REQUEST_TYPE, "embedding")
                if isinstance(get(instance, "model"), str):
                    span.set_attribute(SpanAttributes.LLM_REQUEST_MODEL, instance.model)
            if capture:
                if call_kind == "chat":
                    messages = normalize_messages(
                        inputs.get("prompt"), inputs.get("messages")
                    )
                    set_messages(state.content, SpanAttributes.LLM_PROMPTS, messages)
                    tools = get(
                        inputs.get("prompt"),
                        "tools",
                        inputs.get("kwargs", {}).get("tools"),
                    )
                    if tools:
                        state.content[SpanAttributes.LLM_REQUEST_FUNCTIONS] = safe_json(
                            tool_definitions(tools)
                        )
                elif call_kind == "tool":
                    arguments = dict(inputs.get("kwargs", {}))
                    arguments.update(
                        {k: v for k, v in inputs.items() if k not in {"args", "kwargs"}}
                    )
                    if inputs.get("args"):
                        arguments["args"] = inputs["args"]
                    candidates = [
                        call
                        for call in _TOOL_CALLS.get()
                        if get(call, "name") == name
                        and plain(get(call, "args") or {}) == plain(arguments)
                    ]
                    if len(candidates) == 1 and isinstance(
                        get(candidates[0], "id"), str
                    ):
                        state.content[gen_ai_attributes.GEN_AI_TOOL_CALL_ID] = get(
                            candidates[0], "id"
                        )
                    state.content[SpanAttributes.TRACELOOP_ENTITY_INPUT] = safe_json(
                        {"name": name, "arguments": arguments}
                    )
                elif call_kind == "embedding":
                    state.content[SpanAttributes.TRACELOOP_ENTITY_INPUT] = safe_json(
                        inputs.get("inputs")
                    )
                else:
                    state.content[SpanAttributes.TRACELOOP_ENTITY_INPUT] = safe_json(
                        inputs
                    )
        except Exception:  # noqa: BLE001 - tracing cannot alter SDK calls
            logger.debug("DSPy callback start failed open")

    def capture_result(self, instance, result):
        state = self._active_calls.get(ACTIVE_CALL_ID.get())
        if (
            state
            and state.instance is instance
            and state.kind == "chat"
            and state.span.is_recording()
        ):
            state.result = result

    def _end_call(self, call_id, outputs, exception=None):
        with self._lock:
            state = self._active_calls.pop(call_id, None)
        if state is None:
            return
        try:
            if state.ended:
                return
            if not allowed():
                state.capture = False
            if exception is not None:
                state.span.set_status(trace.StatusCode.ERROR)
                attrs = {EXCEPTION_TYPE: type(exception).__name__}
                if state.capture:
                    args = BaseException.args.__get__(exception)
                    attrs[EXCEPTION_MESSAGE] = (
                        redact_text(args[0])
                        if len(args) == 1 and isinstance(args[0], str)
                        else safe_json(args)
                    )
                state.span.add_event("exception", attrs)
            if state.kind == "chat":
                self._finish_lm(state, outputs, exception)
            elif state.capture and exception is None and outputs is not None:
                state.content[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = safe_json(
                    outputs
                )
            if state.capture:
                state.span.set_attributes(state.content)
        except Exception:  # noqa: BLE001 - tracing cannot alter SDK calls
            logger.debug("DSPy callback completion failed open")
        finally:
            state.content.clear()
            state.result = None
            try:
                if not state.ended:
                    state.span.end()
                    state.ended = True
            finally:
                context.detach(state.token)

    def _finish_lm(self, state, outputs, exception):
        result = state.result
        cached = get(result, "cache_hit", False)
        if cached:
            state.span.set_attribute("dspy.cache_hit", True)
        responses = get(result, "responses")
        raw = get(result, "raw", result)
        if not cached:
            usage = (
                get(responses[0], "usage")
                if responses and len(responses) == 1
                else get(raw, "usage", get(result, "usage", get(outputs, "usage")))
            )
            attrs = {}
            add_lm_usage_attributes(attrs, usage)
            state.span.set_attributes(attrs)
        response = responses[0] if responses and len(responses) == 1 else raw
        response_model = get(response, "model", get(result, "response_model"))
        response_id = get(response, "id")
        if isinstance(response_model, str) and response_model:
            state.span.set_attribute(SpanAttributes.LLM_RESPONSE_MODEL, response_model)
        if isinstance(response_id, str) and response_id:
            state.span.set_attribute(gen_ai_attributes.GEN_AI_RESPONSE_ID, response_id)
        reasons = [
            get(item, "finish_reason")
            for item in (responses or get(raw, "choices", ()))
        ]
        reasons = [reason for reason in reasons if isinstance(reason, str) and reason]
        if reasons:
            state.span.set_attribute(
                SpanAttributes.LLM_RESPONSE_FINISH_REASON, tuple(reasons)
            )
        if not state.capture or exception is not None or outputs is None:
            return
        if responses:
            messages = [message(get(response, "message")) for response in responses]
        elif get(outputs, "message") is not None:
            messages = [message(outputs.message)]
        elif get(raw, "choices") is not None:
            messages = [
                message(
                    get(
                        choice,
                        "message",
                        {"role": "assistant", "content": get(choice, "text")},
                    )
                )
                for choice in get(raw, "choices")
            ]
        else:
            messages = (
                [
                    message(value)
                    if isinstance(value, dict)
                    else {"role": "assistant", "content": value}
                    for value in outputs
                ]
                if isinstance(outputs, list)
                else [{"role": "assistant", "content": outputs}]
            )
        set_messages(state.content, SpanAttributes.LLM_COMPLETIONS, messages)
        if responses and len(responses) == 1 and get(responses[0], "finish_reason"):
            state.span.set_attribute(
                SpanAttributes.LLM_RESPONSE_FINISH_REASON,
                (get(responses[0], "finish_reason"),),
            )

    def close(self):
        self._closed = True
        with self._lock:
            for state in self._active_calls.values():
                if state:
                    state.capture = False
                    state.content.clear()
                    state.result = None
                    state.instance = None
                    state.ended = True
                    state.span.end()


def _start(kind):
    def callback(self, call_id, instance, inputs):
        self._start_call(call_id, kind, instance, inputs)

    return callback


def _end(self, call_id, outputs=None, exception=None):
    self._end_call(call_id, outputs, exception)


for _name, _kind in (
    ("module", "module"),
    ("lm", "chat"),
    ("tool", "tool"),
    ("adapter_format", "task"),
    ("adapter_parse", "task"),
    ("evaluate", "task"),
    ("compile", "task"),
    ("interpreter_execute", "task"),
    ("interpreter_startup", "task"),
    ("interpreter_tool_call", "tool"),
    ("interpreter_shutdown", "task"),
    ("embedding", "embedding"),
):
    setattr(DSPyInstrumentationCallback, f"on_{_name}_start", _start(_kind))
    setattr(DSPyInstrumentationCallback, f"on_{_name}_end", _end)
