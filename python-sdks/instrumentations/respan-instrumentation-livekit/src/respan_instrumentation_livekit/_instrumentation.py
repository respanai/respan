"""Preserve native LiveKit tracing while observing released stream/tool boundaries."""

from __future__ import annotations

import importlib
import logging
import threading
import weakref
from collections import OrderedDict
from contextvars import ContextVar
from functools import wraps

from opentelemetry import trace
from opentelemetry.semconv.attributes.error_attributes import ERROR_TYPE
from opentelemetry.semconv.attributes.http_attributes import HTTP_RESPONSE_STATUS_CODE
from opentelemetry.trace import NonRecordingSpan, Status, StatusCode
from respan_tracing.core.tracer import RespanTracer

from ._constants import (
    LIVEKIT_RESPAN_PROVIDER_NAME_ATTR,
    LIVEKIT_RESPAN_TOOL_DEFINITIONS_ATTR,
)
from ._guard import TracerGuard, install_detach
from ._policy import key, suppressed
from ._processor import LiveKitSpanProcessor
from ._serialization import get_value, safe_json
from ._translator import build_tool_span_attrs, normalize_livekit_tools

logger = logging.getLogger(__name__)
_LOCK = threading.RLock()
_RUNTIME = None
_OWNERS = 0
_STREAM = ContextVar("respan_livekit_native_stream", default=None)


def _raw_usage(values):
    if not isinstance(values, dict):
        return {}
    result = {}
    for target, names in {
        "input": ("prompt_tokens", "input_tokens"),
        "output": ("completion_tokens", "output_tokens"),
        "total": ("total_tokens",),
        "cache_read": ("prompt_cached_tokens", "cache_read_tokens"),
        "cache_creation": ("cache_creation_tokens",),
        "reasoning": ("reasoning_tokens",),
    }.items():
        for name in names:
            v = values.get(name)
            if type(v) is int and v >= 0:
                result[target] = v
                break
    for name, target, field in [
        ("prompt_tokens_details", "cache_read", "cached_tokens"),
        ("completion_tokens_details", "reasoning", "reasoning_tokens"),
    ]:
        details = values.get(name)
        v = details.get(field) if isinstance(details, dict) else None
        if type(v) is int and v >= 0:
            result[target] = v
    return result


class Runtime:
    def __init__(self, provider, capture):
        self.provider = provider
        self.capture = capture
        self.active = True
        self.hooks = []
        self.calls = OrderedDict()
        self.usages = OrderedDict()
        self.processor = LiveKitSpanProcessor(capture)
        self.processor.on_drained = self.drained
        self.tracer = provider.get_tracer("respan.instrumentation.livekit")
        self.native = None
        self.previous = None
        self.installed = None

    def patch(self, owner, name, replacement):
        original = getattr(owner, name)
        self.hooks.append((owner, name, original, replacement))
        setattr(owner, name, replacement)
        return original

    def permitted(self, span):
        if key(span) not in self.processor.policy.open:
            return False
        try:
            result = self.processor.policy.observe(span)
            self.apply_call_bounds()
            return result
        except Exception:  # noqa: BLE001 - telemetry observes without replacing native behavior.
            self.processor.policy.veto(key(span))
            return False

    def remember(self, mapping, value, *state):
        ident = id(value)

        def released(ref):
            existing = mapping.get(ident)
            if existing is not None and existing[0] is ref:
                mapping.pop(ident, None)

        mapping[ident] = (weakref.ref(value, released), *state)

    def apply_call_bounds(self):
        for ident, state in list(self.calls.items()):
            if state[3] and not self.processor.policy.bound(state[2]):
                self.calls[ident] = (*state[:3], False)

    def usage(self, state, values, raw=False):
        if not state["span"].is_recording():
            return
        if raw:
            state["raw_usage"] = True
        if raw or not state["raw_usage"]:
            self.processor.source_usage[key(state["span"])] = _raw_usage(values)

    def observe_chunk(self, state, chunk):
        span = state["span"]
        if not span.is_recording():
            return
        usage = get_value(chunk, "usage")
        if usage is not None and not state["raw_usage"]:
            saved = self.usages.get(id(usage))
            if saved is not None and saved[0]() is usage:
                values = saved[1]
            else:
                # Coerced DTO values cannot establish original source types.
                values = {}
            self.usage(state, values)
        # Correlate the actual SDK call object, never a globally ambiguous call-id.
        for call in get_value(get_value(chunk, "delta"), "tool_calls", []) or []:
            self.remember(
                self.calls,
                call,
                span.get_span_context(),
                key(span),
                self.processor.policy.bound(key(span)),
            )
        self.permitted(span)

    def restore(self):
        for owner, name, original, replacement in reversed(self.hooks):
            if getattr(owner, name, None) is replacement:
                setattr(owner, name, original)
        self.hooks.clear()
        if self.native is not None:
            owned_inner = next(
                (owned for name, _, owned in self.installed or () if name == "_tracer"),
                None,
            )
            for name, old, owned in self.installed or ():
                if (
                    name == "_tracer_provider"
                    and owned_inner is not None
                    and self.native._tracer is not owned_inner
                ):
                    continue
                if getattr(self.native, name, None) is owned:
                    setattr(self.native, name, old)

    def drained(self):
        global _RUNTIME
        self.apply_call_bounds()
        if self.active:
            return
        with self.processor.policy.lock:
            if any(
                self.processor.policy.owned(s[0])
                for s in self.processor.policy.open.values()
                if s[0] is not None
            ):
                return
        self.restore()
        _remove_processor(self.provider, self.processor)
        self.processor.shutdown()
        self.calls.clear()
        self.usages.clear()
        with _LOCK:
            if _RUNTIME is self:
                _RUNTIME = None

    def install(self):
        llm = importlib.import_module("livekit.agents.llm")
        module = importlib.import_module("livekit.agents.llm.llm")
        utils = importlib.import_module("livekit.agents.llm.utils")
        telemetry = importlib.import_module("livekit.agents.telemetry")
        self.native = telemetry.tracer
        previous_provider = self.native._tracer_provider
        previous_tracer = self.native._tracer
        self.installed = [("_tracer_provider", previous_provider, self.provider)]
        # Keep the SDK's DynamicTracer and any existing native redaction processor.
        self.native.set_provider(self.provider)
        native_inner = self.native._tracer
        guard = TracerGuard(native_inner, self)
        self.native._tracer = guard
        self.installed = [
            ("_tracer_provider", previous_provider, self.provider),
            ("_tracer", previous_tracer, guard),
        ]
        _add_processor(self.provider, self.processor)
        install_detach(self)
        original = llm.LLMStream._main_task

        @wraps(original)
        async def main(stream, *args, **kwargs):
            if not self.active:
                return await original(stream, *args, **kwargs)
            span = trace.get_current_span()
            if key(span) not in self.processor.policy.open:
                return await original(stream, *args, **kwargs)
            state = {"span": span, "raw_usage": False, "runtime": self}
            token = _STREAM.set(state)
            # No usage event is different from a provider-reported zero count.
            if span.is_recording():
                self.processor.source_usage[key(span)] = {}
                module_parts = type(stream._llm).__module__.split(".")
                if len(module_parts) > 2 and module_parts[:2] == ["livekit", "plugins"]:
                    span.set_attribute(
                        LIVEKIT_RESPAN_PROVIDER_NAME_ATTR, module_parts[2]
                    )
            try:
                if self.permitted(span):
                    try:
                        definitions = normalize_livekit_tools(stream._tools)
                        if definitions:
                            span.set_attribute(
                                LIVEKIT_RESPAN_TOOL_DEFINITIONS_ATTR,
                                safe_json(definitions, schema=True),
                            )
                    except Exception:  # noqa: BLE001 - telemetry observes without replacing native behavior.
                        self.processor.policy.veto(key(span))
                source_send = stream._event_ch.send_nowait

                def send(value):
                    result = source_send(value)
                    try:
                        self.observe_chunk(state, value)
                    except Exception:  # noqa: BLE001 - telemetry observes without replacing native behavior.
                        self.processor.policy.veto(key(span))
                    return result

                stream._event_ch.send_nowait = send
                try:
                    return await original(stream, *args, **kwargs)
                except BaseException as exc:
                    status = get_value(exc, "status_code")
                    if (
                        span.is_recording()
                        and type(status) is int
                        and 100 <= status <= 599
                    ):
                        span.set_attribute(HTTP_RESPONSE_STATUS_CODE, status)
                    raise
                finally:
                    if stream._event_ch.send_nowait is send:
                        stream._event_ch.send_nowait = source_send
                    self.permitted(span)
            finally:
                _STREAM.reset(token)

        self.patch(llm.LLMStream, "_main_task", main)
        if hasattr(llm.LLMStream, "_record_genai_request"):
            request = llm.LLMStream._record_genai_request

            @wraps(request)
            def record(stream, span):
                if (
                    self.active
                    and key(span) in self.processor.policy.open
                    and not self.permitted(span)
                ):
                    stream._record_content = False
                return request(stream, span)

            self.patch(llm.LLMStream, "_record_genai_request", record)
        if hasattr(module, "_chat_ctx_to_otel_events"):
            events = module._chat_ctx_to_otel_events

            @wraps(events)
            def messages(*args, **kwargs):
                if self.active and not self.permitted(trace.get_current_span()):
                    return []
                return events(*args, **kwargs)

            self.patch(module, "_chat_ctx_to_otel_events", messages)
        original_init = llm.CompletionUsage.__init__

        @wraps(original_init)
        def usage_init(value, *args, **kwargs):
            result = original_init(value, *args, **kwargs)
            try:
                state = _STREAM.get()
                if self.active or (state is not None and state["runtime"] is self):
                    names = {
                        "prompt_tokens",
                        "completion_tokens",
                        "total_tokens",
                        "prompt_cached_tokens",
                        "cache_read_tokens",
                        "cache_creation_tokens",
                        "reasoning_tokens",
                    }
                    self.remember(
                        self.usages,
                        value,
                        {
                            k: v
                            for k, v in kwargs.items()
                            if k in names
                            and k in type(value).model_fields
                            and type(v) is int
                            and v >= 0
                        },
                    )
            except Exception:  # noqa: BLE001 - telemetry observes without replacing native behavior.
                logger.debug("Skipped source usage observation")
            return result

        self.patch(llm.CompletionUsage, "__init__", usage_init)
        event_type = importlib.import_module("openai._streaming").ServerSentEvent
        event_json = event_type.json

        @wraps(event_json)
        def parsed(event, *args, **kwargs):
            result = event_json(event, *args, **kwargs)
            state = _STREAM.get()
            if (
                state is not None
                and state["runtime"] is self
                and isinstance(result, dict)
            ):
                try:
                    if ("choices" in result or "usage" in result) and (
                        not state["raw_usage"] or isinstance(result.get("usage"), dict)
                    ):
                        self.usage(state, result.get("usage"), raw=True)
                except Exception:  # noqa: BLE001 - telemetry observes without replacing native behavior.
                    state["raw_usage"] = True
                    if state["span"].is_recording():
                        self.processor.source_usage[key(state["span"])] = {}
            return result

        self.patch(event_type, "json", parsed)
        execute = utils.execute_function_call

        @wraps(execute)
        async def tool(*args, **kwargs):
            if not self.active or suppressed():
                return await execute(*args, **kwargs)
            call = args[0] if args else kwargs.get("tool_call")
            parent_context = None
            saved = self.calls.pop(id(call), None)
            if saved is not None and saved[0]() is call:
                parent_context = trace.set_span_in_context(NonRecordingSpan(saved[1]))
            span = self.tracer.start_span("livekit.tool", context=parent_context)
            if saved is not None and (
                not saved[3] or not self.processor.policy.bound(saved[2])
            ):
                self.processor.policy.veto(key(span))
            result = None
            error = None
            with trace.use_span(
                span,
                end_on_exit=False,
                record_exception=False,
                set_status_on_exception=False,
            ):
                try:
                    result = await execute(*args, **kwargs)
                    candidate = get_value(result, "raw_exception")
                    error = candidate if isinstance(candidate, BaseException) else None
                    return result
                except BaseException as exc:
                    error = exc
                    raise
                finally:
                    try:
                        capture = self.permitted(span)
                        if span.is_recording():
                            if error is not None:
                                span.set_attribute(ERROR_TYPE, type(error).__name__)
                                span.set_status(Status(StatusCode.ERROR))
                                status = get_value(error, "status_code")
                                if type(status) is int and 100 <= status <= 599:
                                    span.set_attribute(
                                        HTTP_RESPONSE_STATUS_CODE, status
                                    )
                            span.set_attributes(
                                build_tool_span_attrs(
                                    tool_name=get_value(call, "name"),
                                    arguments=get_value(call, "arguments"),
                                    output=get_value(result, "raw_output"),
                                    call_id=get_value(call, "call_id"),
                                    capture=capture,
                                    capture_output=error is None and result is not None,
                                )
                            )
                            self.permitted(span)
                    except Exception:  # noqa: BLE001 - telemetry observes without replacing native behavior.
                        self.processor.policy.veto(key(span))
                    finally:
                        span.end()

        self.patch(utils, "execute_function_call", tool)
        if hasattr(llm, "execute_function_call"):
            self.patch(llm, "execute_function_call", tool)


def _add_processor(provider, processor):
    provider.add_span_processor(processor)
    active = provider._active_span_processor
    with active._lock:
        active._span_processors = (
            processor,
            *[p for p in active._span_processors if p is not processor],
        )


def _remove_processor(provider, processor):
    active = getattr(provider, "_active_span_processor", None)
    if active is not None:
        with active._lock:
            active._span_processors = tuple(
                p for p in active._span_processors if p is not processor
            )


class LiveKitInstrumentor:
    name = "livekit"

    def __init__(self, *, capture_content=True):
        self.capture_content = capture_content
        self._is_instrumented = False

    def activate(self):
        global _RUNTIME, _OWNERS
        with _LOCK:
            if self._is_instrumented:
                return
            enabled = getattr(RespanTracer, "_instance", None)
            if enabled is not None and not getattr(enabled, "is_enabled", True):
                return
            provider = trace.get_tracer_provider()
            if _RUNTIME is not None:
                if (
                    _RUNTIME.provider is not provider
                    or _RUNTIME.capture != self.capture_content
                ):
                    raise ValueError(
                        "Concurrent LiveKit owners must share provider and capture configuration"
                    )
                if not _RUNTIME.active:
                    raise RuntimeError("Prior LiveKit spans are still finishing")
            else:
                runtime = Runtime(provider, self.capture_content)
                try:
                    runtime.install()
                except Exception:
                    runtime.active = False
                    runtime.restore()
                    _remove_processor(provider, runtime.processor)
                    runtime.processor.shutdown()
                    raise
                _RUNTIME = runtime
            _OWNERS += 1
            self._is_instrumented = True

    def deactivate(self):
        global _OWNERS
        with _LOCK:
            if not self._is_instrumented:
                return
            self._is_instrumented = False
            _OWNERS -= 1
            if _OWNERS == 0:
                runtime = _RUNTIME
                runtime.active = False
                runtime.drained()
                # The original runtime remains held by any in-flight hooks.
