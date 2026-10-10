"""Guard native observer spans and extraction without replacing Pipecat execution."""

import asyncio
import inspect
import time
import weakref
from contextlib import contextmanager
from contextvars import ContextVar

from openinference.semconv.trace import SpanAttributes as OI
from opentelemetry import context, trace
from opentelemetry.semconv._incubating.attributes.error_attributes import ERROR_MESSAGE
from opentelemetry.semconv.trace import SpanAttributes as OTel
from opentelemetry.semconv_ai import SpanAttributes as TL

from respan_instrumentation_pipecat._policy import (
    AncestorPolicy,
    content_allowed,
    span_key,
    suppressed,
)
from respan_instrumentation_pipecat._serialization import (
    json_dumps,
    parse_json,
    safe_text,
)

_SCOPE = ContextVar("respan_pipecat_observer", default=None)
_SERVICE = ContextVar("respan_pipecat_openai_service", default=None)
_ORIGIN = ContextVar("respan_pipecat_openai_stream_origin", default=None)


def is_content(key):
    return key in {
        OI.INPUT_VALUE,
        OI.OUTPUT_VALUE,
        OI.TOOL_PARAMETERS,
        OI.METADATA,
        OI.LLM_TOOLS,
        TL.TRACELOOP_ENTITY_INPUT,
        TL.TRACELOOP_ENTITY_OUTPUT,
        TL.LLM_REQUEST_FUNCTIONS,
        "tools.definitions",
        "tool.result",
        ERROR_MESSAGE,
    } or key.startswith(
        (
            OI.LLM_INPUT_MESSAGES + ".",
            OI.LLM_OUTPUT_MESSAGES + ".",
            TL.LLM_PROMPTS + ".",
            TL.LLM_COMPLETIONS + ".",
            "exception.",
        )
    )


class NativeSpan(trace.Span):
    def __init__(self, runtime, span):
        self.runtime = runtime
        self.span = span
        self.config = _SCOPE.get()[1]._config if _SCOPE.get() else None
        self.recording = span.is_recording()
        self.capture = content_allowed(runtime.capture) and runtime.policy.allowed(span)
        self.error = None
        self.child_end = 0
        runtime.spans[span_key(span)] = self

    def allowed(self):
        try:
            self.capture = (
                self.capture
                and self.runtime.active
                and self.recording
                and self.runtime.policy.observe(self.span)
            )
        except Exception:  # noqa: BLE001 - telemetry faults preserve native execution.
            self.capture = False
        return self.capture

    def set_attribute(self, key, value):
        if is_content(key):
            input_key = key in {
                OI.INPUT_VALUE,
                OI.TOOL_PARAMETERS,
                OI.LLM_TOOLS,
                OI.METADATA,
                TL.TRACELOOP_ENTITY_INPUT,
                TL.LLM_REQUEST_FUNCTIONS,
                "tools.definitions",
            } or key.startswith((OI.LLM_INPUT_MESSAGES + ".", TL.LLM_PROMPTS + "."))
            if self.config and (
                self.config.hide_inputs if input_key else self.config.hide_outputs
            ):
                return
            if not self.allowed():
                return
            if isinstance(value, str):
                value = (
                    json_dumps(
                        parse_json(value),
                        complete=True,
                        tool_definitions=key in {OI.LLM_TOOLS, "tools.definitions"},
                    )
                    if key
                    in {
                        OI.INPUT_VALUE,
                        OI.OUTPUT_VALUE,
                        OI.METADATA,
                        TL.TRACELOOP_ENTITY_INPUT,
                        TL.TRACELOOP_ENTITY_OUTPUT,
                        TL.LLM_REQUEST_FUNCTIONS,
                        OI.TOOL_PARAMETERS,
                        OI.LLM_TOOLS,
                        "tools.definitions",
                        "tool.result",
                    }
                    or key.endswith(".tool_calls")
                    else safe_text(value, complete=True)
                )
        return self.span.set_attribute(key, value)

    def set_attributes(self, attributes):
        for key, value in attributes.items():
            self.set_attribute(key, value)

    def get_span_context(self):
        return self.span.get_span_context()

    def is_recording(self):
        return self.span.is_recording()

    def add_event(self, name, attributes=None, timestamp=None):
        if not self.allowed():
            attributes = {
                k: v for k, v in (attributes or {}).items() if k == OTel.EXCEPTION_TYPE
            }
        return self.span.add_event(name, attributes, timestamp)

    def set_status(self, status, description=None):
        return self.span.set_status(status, description)

    def update_name(self, name):
        return self.span.update_name(name)

    def record_exception(self, *args, **kwargs):
        if self.allowed():
            return self.span.record_exception(*args, **kwargs)

    def end(self, end_time=None):
        self.allowed()
        finish = max(end_time or time.time_ns(), self.child_end)
        parent = self.runtime.parents.get(span_key(self))
        if parent in self.runtime.spans:
            self.runtime.spans[parent].child_end = max(
                self.runtime.spans[parent].child_end, finish
            )
        result = self.span.end(end_time=finish)
        self.runtime.spans.pop(span_key(self), None)
        self.runtime.parents.pop(span_key(self), None)
        self.runtime.raw_usage.pop(span_key(self), None)
        return result


class NativeTracer:
    def __init__(self, runtime, original):
        self.runtime = runtime
        self.original = original

    def start_span(self, *args, **kwargs):
        if not self.runtime.active:
            if self.runtime.native_owned:
                return trace.NonRecordingSpan(trace.INVALID_SPAN_CONTEXT)
            return self.original.start_span(*args, **kwargs)
        if suppressed():
            return trace.NonRecordingSpan(trace.INVALID_SPAN_CONTEXT)
        span = self.original.start_span(*args, **kwargs)
        proxy = NativeSpan(self.runtime, span)
        self.runtime.parents[span_key(span)] = span_key(
            trace.get_current_span(kwargs.get("context"))
        )
        return proxy

    def __getattr__(self, name):
        return getattr(self.original, name)


class Runtime:
    def __init__(self, provider, capture):
        self.provider = provider
        self.native_owned = True
        self.capture = capture
        self.policy = AncestorPolicy(capture)
        self.active = True
        self.spans = {}
        self.parents = {}
        self.raw_usage = {}
        self.observers = weakref.WeakValueDictionary()
        self.hooks = []

    def checkpoint(self, observer=None):
        try:
            if self.policy.knows(trace.get_current_span()) or (
                not content_allowed(self.capture)
                and span_key(trace.get_current_span()) is not None
            ):
                self.policy.observe(trace.get_current_span())
            for span in list(self.spans.values()):
                span.allowed()
            if observer is not None:
                if not self.allowed(observer):
                    self.clear_content(observer)
                else:
                    self.clear_content(
                        observer,
                        inputs=observer._config.hide_inputs,
                        outputs=observer._config.hide_outputs,
                    )
        except Exception:  # noqa: BLE001 - telemetry faults preserve native execution.
            for span in self.spans.values():
                span.capture = False
            for owned in list(self.observers.values()):
                self.clear_content(owned)

    def allowed(self, observer):
        if not self.active or suppressed() or not content_allowed(self.capture):
            return False
        spans = [
            i.get("span")
            for i in getattr(observer, "_active_spans", {}).values()
            if isinstance(i, dict)
        ]
        turn = getattr(observer, "_turn_span", None)
        if turn is not None:
            spans.append(turn)
        return all(s.allowed() for s in spans if isinstance(s, NativeSpan))

    def clear_content(self, observer, *, inputs=True, outputs=True):
        names = []
        if inputs:
            names.append("_turn_user_text")
        if outputs:
            names.extend(["_turn_bot_text", "_respan_llm_text_chunks"])
        for name in names:
            setattr(observer, name, [])
        for info in getattr(observer, "_active_spans", {}).values():
            if isinstance(info, dict):
                if inputs:
                    info["accumulated_input"] = ""
                if outputs:
                    info["accumulated_output"] = ""

    @contextmanager
    def scope(self, observer, data=None):
        self.observers[id(observer)] = observer
        if not isinstance(observer._tracer, NativeTracer):
            observer._tracer = NativeTracer(self, observer._tracer)
        self.checkpoint(observer)
        token = _SCOPE.set((self, observer, data))
        try:
            yield
        finally:
            self.checkpoint(observer)
            _SCOPE.reset(token)

    def install(self, module):
        from pipecat.processors.frame_processor import FrameProcessor

        oldpush = FrameProcessor.push_frame

        async def push(processor, *args, **kwargs):
            def checkpoint():
                if not self.active:
                    return
                self.checkpoint()
                for observer in list(self.observers.values()):
                    info = observer._active_spans.get(id(processor))
                    if isinstance(info, dict) and isinstance(
                        info.get("span"), NativeSpan
                    ):
                        info["span"].allowed()
                        if not self.allowed(observer):
                            self.clear_content(observer)

            checkpoint()
            try:
                return await oldpush(processor, *args, **kwargs)
            finally:
                checkpoint()

        self.patch(FrameProcessor, "push_frame", push)
        cls = module.OpenInferenceObserver
        original_init = cls.__init__

        def initialize(observer, *args, **kwargs):
            result = original_init(observer, *args, **kwargs)
            if (
                self.active
                and getattr(observer._tracer, "span_processor", None)
                is self.provider._active_span_processor
            ):
                self.observers[id(observer)] = observer
                observer._tracer = NativeTracer(self, observer._tracer)
            return result

        self.patch(cls, "__init__", initialize)
        original = cls.on_push_frame

        async def wrapped(observer, data):
            if not self.active or not (
                isinstance(observer._tracer, NativeTracer)
                and observer._tracer.runtime is self
                or getattr(observer._tracer, "span_processor", None)
                is self.provider._active_span_processor
            ):
                return await original(observer, data)
            with self.scope(observer, data):
                from pipecat.frames.frames import (
                    FunctionCallFromLLM,
                    FunctionCallsStartedFrame,
                )

                if (
                    isinstance(data.frame, FunctionCallsStartedFrame)
                    and self.allowed(observer)
                    and not observer._config.hide_outputs
                ):
                    info = observer._active_spans.get(id(data.source))
                    if isinstance(info, dict):
                        calls = [
                            {
                                "id": c.tool_call_id,
                                "type": "function",
                                "function": {
                                    "name": c.function_name,
                                    "arguments": json_dumps(c.arguments, complete=True),
                                },
                            }
                            for c in data.frame.function_calls
                            if isinstance(c, FunctionCallFromLLM)
                        ]
                        if calls:
                            info["span"].set_attribute(
                                TL.LLM_COMPLETIONS + ".0.tool_calls",
                                json_dumps(calls, complete=True),
                            )
                return await original(observer, data)

        self.patch(cls, "on_push_frame", wrapped)
        oldtool = cls._handle_tool_frame

        async def tool(observer, data):
            if not self.active:
                return await oldtool(observer, data)
            from copy import copy
            from dataclasses import replace

            from respan_instrumentation_pipecat._serialization import jsonable

            redacted = copy(data.frame)
            redacted.arguments = (
                jsonable(data.frame.arguments, complete=True)
                if self.allowed(observer) and not observer._config.hide_inputs
                else None
            )
            redacted.result = (
                jsonable(data.frame.result, complete=True)
                if self.allowed(observer) and not observer._config.hide_outputs
                else None
            )
            return await oldtool(observer, replace(data, frame=redacted))

        self.patch(cls, "_handle_tool_frame", tool)
        oldservice = module.extract_service_attributes

        def service_attributes(service):
            current = _SCOPE.get()
            if (
                current
                and current[0] is self
                and (not self.allowed(current[1]) or current[1]._config.hide_inputs)
            ):
                from openinference.instrumentation.pipecat._attributes import (
                    detect_service_type,
                )

                kind = detect_service_type(service)
                attrs = {
                    OI.OPENINFERENCE_SPAN_KIND: "LLM" if kind == "llm" else "CHAIN",
                    "service.type": kind,
                }
                try:
                    model = getattr(service, "model_name", None)
                except Exception:  # noqa: BLE001 - telemetry faults preserve native execution.
                    model = None
                if isinstance(model, str):
                    attrs[OI.LLM_MODEL_NAME] = safe_text(model)
                return attrs
            attrs = oldservice(service)
            from openinference.instrumentation.pipecat._attributes import (
                detect_service_type,
            )

            attrs["service.type"] = detect_service_type(service)
            return attrs

        self.patch(module, "extract_service_attributes", service_attributes)
        oldjson = module.safe_json_dumps

        def dumps(value, *args, **kwargs):
            current = _SCOPE.get()
            if current and current[0] is self:
                return (
                    json_dumps(value, complete=True) if self.allowed(current[1]) else ""
                )
            return oldjson(value, *args, **kwargs)

        self.patch(module, "safe_json_dumps", dumps)
        oldframe = module.extract_attributes_from_frame

        def extract(frame):
            current = _SCOPE.get()
            if not current or current[0] is not self:
                return oldframe(frame)
            return self.extract(frame, current[1], oldframe)

        self.patch(module, "extract_attributes_from_frame", extract)
        original_context = module.Context

        def observer_context(*args, **kwargs):
            current = _SCOPE.get()
            if current and current[0] is self:
                return context.get_current()
            return original_context(*args, **kwargs)

        self.patch(module, "Context", observer_context)
        from pipecat.services.openai.base_llm import BaseOpenAILLMService

        old_process = BaseOpenAILLMService._process_context

        async def process(service, *args, **kwargs):
            token = _SERVICE.set((service, weakref.ref(asyncio.current_task())))
            try:
                return await old_process(service, *args, **kwargs)
            finally:
                _SERVICE.reset(token)

        self.patch(BaseOpenAILLMService, "_process_context", process)
        from openai._streaming import AsyncStream, ServerSentEvent

        old_stream = AsyncStream.__stream__

        def stream(client_stream, *args, **kwargs):
            original = old_stream(client_stream, *args, **kwargs)
            source = _SERVICE.get()
            if (
                source is None
                or source[1]() is not asyncio.current_task()
                or client_stream._client is not source[0]._client
            ):
                return original
            return SourceStream(original, client_stream.response)

        self.patch(AsyncStream, "__stream__", stream)
        original_json = ServerSentEvent.json

        def observe_json(value, *args, **kwargs):
            result = original_json(value, *args, **kwargs)
            if _ORIGIN.get() is not None:
                try:
                    self.observe_usage(result)
                except Exception:  # noqa: BLE001, S110 - source observation preserves native SSE.
                    pass
            return result

        self.patch(ServerSentEvent, "json", observe_json)
        olddetach = context.detach

        def detach(token):
            if self.active:
                self.checkpoint()
            return olddetach(token)

        self.patch(context, "detach", detach)

    def patch(self, owner, name, wrapper):
        original = (
            inspect.getattr_static(owner, name)
            if isinstance(owner, type)
            else getattr(owner, name)
        )
        self.hooks.append((owner, name, original, wrapper))
        setattr(owner, name, wrapper)

    def extract(self, frame, observer, original):
        from openinference.semconv.trace import ToolCallAttributes
        from pipecat.frames.frames import (
            FunctionCallResultFrame,
            LLMContextFrame,
            MetricsFrame,
        )

        if isinstance(frame, LLMContextFrame):
            if not self.allowed(observer) or observer._config.hide_inputs:
                return {}
            messages = frame.context._messages
            attrs = {OI.INPUT_VALUE: json_dumps(messages, complete=True)}
            for i, message in enumerate(messages[:8]):
                if isinstance(message, dict):
                    for key in ("role", "content", "tool_calls", "tool_call_id"):
                        if key in message:
                            value = message[key]
                            attrs[f"{OI.LLM_INPUT_MESSAGES}.{i}.message.{key}"] = (
                                json_dumps(value, complete=True)
                                if not isinstance(value, str)
                                else safe_text(value, complete=True)
                            )
            tools = frame.context._tools
            if tools:
                from pipecat.adapters.schemas.tools_schema import ToolsSchema

                if isinstance(tools, ToolsSchema):
                    tools = [
                        {"type": "function", "function": t.to_default_dict()}
                        for t in tools.standard_tools
                    ]
                attrs[OI.LLM_TOOLS] = json_dumps(
                    tools, complete=True, tool_definitions=True
                )
            return attrs
        if isinstance(frame, FunctionCallResultFrame):
            attrs = {
                OI.TOOL_NAME: frame.function_name,
                ToolCallAttributes.TOOL_CALL_ID: frame.tool_call_id,
            }
            if self.allowed(observer):
                attrs.update(
                    {
                        OI.TOOL_PARAMETERS: json_dumps(frame.arguments, complete=True),
                        "tool.result": json_dumps(frame.result, complete=True),
                    }
                )
            return attrs
        if isinstance(frame, MetricsFrame):
            from pipecat.metrics.metrics import LLMUsageMetricsData

            attrs = original(frame)
            current = _SCOPE.get()
            source = (
                getattr(current[2], "source", None)
                if current and len(current) > 2
                else None
            )
            info = observer._active_spans.get(id(source))
            if isinstance(info, dict) and info.get("service_type") == "llm":
                attrs.pop("metrics.processor", None)
            raw = (
                self.raw_usage.get(span_key(info["span"]))
                if isinstance(info, dict) and isinstance(info.get("span"), NativeSpan)
                else None
            )
            for item in frame.data:
                if not isinstance(item, LLMUsageMetricsData):
                    continue
                value = item.value
                for field, key in [
                    ("prompt_tokens", OI.LLM_TOKEN_COUNT_PROMPT),
                    ("completion_tokens", OI.LLM_TOKEN_COUNT_COMPLETION),
                    ("total_tokens", OI.LLM_TOKEN_COUNT_TOTAL),
                    ("cache_read_input_tokens", TL.LLM_USAGE_CACHE_READ_INPUT_TOKENS),
                    (
                        "cache_creation_input_tokens",
                        TL.LLM_USAGE_CACHE_CREATION_INPUT_TOKENS,
                    ),
                    ("reasoning_tokens", TL.LLM_USAGE_REASONING_TOKENS),
                ]:
                    count = (
                        raw.get(field)
                        if raw is not None
                        else getattr(value, field, None)
                    )
                    if (
                        isinstance(count, int)
                        and not isinstance(count, bool)
                        and count >= 0
                    ):
                        attrs[key] = count
                    else:
                        attrs.pop(key, None)
                        detail = {
                            "cache_read_input_tokens": OI.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ,
                            "cache_creation_input_tokens": OI.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE,
                            "reasoning_tokens": OI.LLM_TOKEN_COUNT_COMPLETION_DETAILS_REASONING,
                        }.get(field)
                        if detail is not None:
                            attrs.pop(detail, None)
            return attrs
        if not self.allowed(observer):
            return {}
        return original(frame)

    def observe_usage(self, response):
        source = _SERVICE.get()
        if (
            not self.active
            or source is None
            or source[1]() is not asyncio.current_task()
            or not isinstance(response, dict)
        ):
            return
        service = source[0]
        usage = response.get("usage")
        if not isinstance(usage, dict):
            return
        for observer in list(self.observers.values()):
            info = observer._active_spans.get(id(service))
            if (
                not isinstance(info, dict)
                or not isinstance(info.get("span"), NativeSpan)
                or not info["span"].is_recording()
            ):
                continue
            prompt = usage.get("prompt_tokens_details", {})
            completion = usage.get("completion_tokens_details", {})
            values = {
                field: usage.get(field)
                for field in ("prompt_tokens", "completion_tokens", "total_tokens")
            }
            values.update(
                cache_read_input_tokens=prompt.get("cached_tokens")
                if isinstance(prompt, dict)
                else None,
                cache_creation_input_tokens=usage.get("cache_creation_input_tokens"),
                reasoning_tokens=completion.get("reasoning_tokens")
                if isinstance(completion, dict)
                else None,
            )
            self.raw_usage[span_key(info["span"])] = {
                k: v
                if isinstance(v, int) and not isinstance(v, bool) and v >= 0
                else None
                for k, v in values.items()
            }

    def close(self):
        try:
            self.checkpoint()
            for span in reversed(list(self.spans.values())):
                try:
                    span.end()
                except Exception:  # noqa: BLE001 - cleanup preserves native execution.
                    span.capture = False
        finally:
            self.active = False
            for observer in list(self.observers.values()):
                self.clear_content(observer)
                if (
                    not self.native_owned
                    and isinstance(observer._tracer, NativeTracer)
                    and observer._tracer.runtime is self
                ):
                    observer._tracer = observer._tracer.original
            for owner, name, original, wrapper in reversed(self.hooks):
                if getattr(owner, name) is wrapper:
                    setattr(owner, name, original)
            self.hooks.clear()
            self.spans.clear()
            self.parents.clear()
            self.raw_usage.clear()
            self.policy.clear()


class SourceStream:
    """Observe SDK source inside each native advance, resetting before yielding."""

    def __init__(self, original, response):
        self.original = original
        self.response = response

    def __aiter__(self):
        return self

    async def _call(self, name, *args):
        token = _ORIGIN.set(self.response)
        try:
            return await getattr(self.original, name)(*args)
        except BaseException:
            self.response = None
            raise
        finally:
            _ORIGIN.reset(token)

    async def __anext__(self):
        return await self._call("__anext__")

    async def asend(self, value):
        return await self._call("asend", value)

    async def athrow(self, *args):
        return await self._call("athrow", *args)

    async def aclose(self):
        try:
            return await self._call("aclose")
        finally:
            self.response = None
