"""Owned native AgentOps tracer/helper guards and processor registration."""

from __future__ import annotations

import functools
import importlib
import logging
from contextlib import contextmanager
from threading import RLock

from agentops.semconv.span_attributes import SpanAttributes as AgentOpsSpanAttributes
from opentelemetry import trace
from opentelemetry.semconv_ai import SpanAttributes
from respan_tracing.core.tracer import RespanTracer

from ._constants import AGENTOPS_INSTRUMENTATION_NAME
from ._policy import content_allowed, suppressed
from ._processor import AgentOpsSpanProcessor, _structural_field
from ._serialization import data, json_value

logger = logging.getLogger(__name__)
_LOCK = RLock()
_RUNTIME = None
_MISSING = object()


class _ProviderProxy:
    def __init__(self, provider):
        self._provider = provider

    def force_flush(self, *a, **kw):
        return self._provider.force_flush(*a, **kw)

    def shutdown(self, *a, **kw):
        return None


class _TracerProxy:
    def __init__(self, tracer, runtime):
        self._tracer = tracer
        self._runtime = runtime

    def start_span(self, name, *args, **kwargs):
        if not self._runtime.active:
            return self._tracer.start_span(name, *args, **kwargs)
        if suppressed():
            return trace.NonRecordingSpan(trace.INVALID_SPAN_CONTEXT)
        attrs = dict(kwargs.get("attributes") or {})
        attrs.setdefault(
            AgentOpsSpanAttributes.AGENTOPS_SPAN_KIND,
            "embedding"
            if name in {"openai.embeddings", "openai.embedding"}
            else "task",
        )
        parent = (
            trace.get_current_span(kwargs.get("context")).get_span_context().span_id
        )
        processor = self._runtime.processor
        permitted = content_allowed(
            processor.capture_content
        ) and processor._captures.get(parent, processor._parents.get(parent, True))
        if not permitted:
            attrs = {k: v for k, v in attrs.items() if _structural_field(k, v)}
        pending = {
            k: v
            for k, v in attrs.items()
            if not isinstance(v, (str, int, float, bool, type(None)))
            and not isinstance(v, (list, tuple))
            or isinstance(v, (list, tuple))
            and not all(isinstance(x, (str, int, float, bool)) for x in v)
        }
        kwargs["attributes"] = {k: v for k, v in attrs.items() if k not in pending}
        span = self._tracer.start_span(name, *args, **kwargs)
        try:
            if span.is_recording() and processor.allowed(span):
                for key, value in pending.items():
                    span.set_attribute(key, json_value(value))
        except Exception:  # noqa: BLE001 - never replace native execution on telemetry failure
            processor._veto(span.get_span_context().span_id)
            logger.debug("Could not normalize AgentOps start attributes")
        return span

    @contextmanager
    def start_as_current_span(self, name, *args, **kwargs):
        end = kwargs.pop("end_on_exit", True)
        record = kwargs.pop("record_exception", True)
        status = kwargs.pop("set_status_on_exception", True)
        span = self.start_span(name, *args, **kwargs)
        with trace.use_span(
            span,
            end_on_exit=end,
            record_exception=record,
            set_status_on_exception=status,
        ):
            try:
                yield span
            finally:
                if self._runtime.active and span.is_recording():
                    try:
                        self._runtime.processor.allowed(span)
                    except Exception:  # noqa: BLE001 - policy observation cannot replace native errors
                        self._runtime.processor._veto(span.get_span_context().span_id)


class _Runtime:
    def __init__(self, provider, capture):
        self.provider = provider
        self.processor = AgentOpsSpanProcessor(capture_content=capture)
        self.active = True
        self.owners = 1
        self.patches = []
        self.core = None
        self.core_snapshot = None
        self.core_proxy = None

    def patch(self, owner, name, value):
        original = owner.__dict__.get(name, _MISSING)
        self.patches.append((owner, name, original, value))
        setattr(owner, name, value)

    def install(self):
        active = self.provider._active_span_processor
        with active._lock:
            active._span_processors = (self.processor, *active._span_processors)
        core_module = importlib.import_module("agentops.sdk.core")
        self.core = core_module.tracer
        if not self.core.initialized:
            self.core_snapshot = (
                self.core._initialized,
                self.core.provider,
                self.core._meter_provider,
            )
            self.core_proxy = _ProviderProxy(self.provider)
            self.core.provider = self.core_proxy
            self.core._meter_provider = None
            self.core._initialized = True
        original = self.core.get_tracer

        def get_tracer(*a, __original=original, **kw):
            value = __original(*a, **kw)
            return _TracerProxy(value, self) if self.active else value

        self.patch(self.core, "get_tracer", get_tracer)
        common = importlib.import_module("agentops.instrumentation.common.instrumentor")
        original_common = common.get_tracer

        def common_tracer(*a, **kw):
            value = original_common(*a, **kw)
            return _TracerProxy(value, self) if self.active else value

        self.patch(common, "get_tracer", common_tracer)
        utility = importlib.import_module("agentops.sdk.decorators.utility")
        factory = importlib.import_module("agentops.sdk.decorators.factory")
        for name, direction in (
            ("_record_entity_input", "input"),
            ("_record_entity_output", "output"),
        ):
            for module in (utility, factory):
                original = module.__dict__[name]

                @functools.wraps(original)
                def record(
                    span, *args, __original=original, __direction=direction, **kwargs
                ):
                    if not self.active:
                        return __original(span, *args, **kwargs)
                    try:
                        if not self.processor.allowed(span):
                            return None
                        if __original.__module__ != "agentops.sdk.decorators.utility":
                            __original(span, *args, **kwargs)
                        kind = (
                            kwargs.get(
                                "entity_kind", args[2] if len(args) > 2 else "entity"
                            )
                            if __direction == "input"
                            else kwargs.get(
                                "entity_kind", args[1] if len(args) > 1 else "entity"
                            )
                        )
                        value = (
                            {"args": args[0], "kwargs": args[1]}
                            if __direction == "input"
                            else args[0]
                        )
                        key = (
                            AgentOpsSpanAttributes.AGENTOPS_DECORATOR_INPUT.format(
                                entity_kind=kind
                            )
                            if __direction == "input"
                            else AgentOpsSpanAttributes.AGENTOPS_DECORATOR_OUTPUT.format(
                                entity_kind=kind
                            )
                        )
                        span.set_attribute(
                            key, json_value(value, complete=kind == "tool")
                        )

                    except Exception:  # noqa: BLE001 - observer cannot change SDK results
                        logger.debug("Could not observe AgentOps payload")
                        return None

                self.patch(module, name, record)
        self._observe_openai_embeddings()
        self._capture_declared_calls()

    def _capture_declared_calls(self):
        # Released AgentOps makes synthetic execution spans for model declarations.
        # Record only the actual declaration on its native model span instead.
        from agentops.instrumentation.providers.openai import stream_wrapper
        from agentops.instrumentation.providers.openai.wrappers import chat

        for module in (chat, stream_wrapper):
            original = module._create_tool_span

            @functools.wraps(original)
            def record(parent, call, __original=original):
                if not self.active:
                    return __original(parent, call)
                try:
                    if not self.processor.allowed(parent) or not isinstance(call, dict):
                        return None
                    key = f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls"
                    if self.processor.has_current_calls(parent):
                        return None
                    old = (parent.attributes or {}).get(key)
                    import json

                    values = json.loads(old) if isinstance(old, str) else []
                    value = data(call, complete=True)
                    function = value.get("function")
                    if isinstance(function, dict) and isinstance(
                        function.get("arguments"), dict
                    ):
                        function["arguments"] = json_value(
                            function["arguments"], complete=True
                        )
                    values.append(value)
                    parent.set_attribute(key, json_value(values, complete=True))
                    return None

                except Exception:  # noqa: BLE001 - observer cannot change SDK results
                    logger.debug("Could not observe AgentOps payload")
                    return None

            self.patch(module, "_create_tool_span", record)

    def _observe_openai_embeddings(self):
        try:
            from openai.resources.chat.completions import AsyncCompletions, Completions
            from openai.resources.embeddings import AsyncEmbeddings, Embeddings
        except ImportError:
            return

        def current():
            span = trace.get_current_span()
            if (
                not self.active
                or not span.is_recording()
                or (span.attributes or {}).get(AgentOpsSpanAttributes.LLM_REQUEST_TYPE)
                not in {"embedding", "chat", "completion"}
            ):
                return None
            return span

        def values(value):
            return (
                value
                if isinstance(value, dict)
                else vars(value)
                if type(value).__module__.startswith("openai.")
                else {}
            )

        def before(kwargs, embedding):
            span = current()
            if span is None or not self.processor.allowed(span):
                return
            value = kwargs.get("input") if embedding else kwargs.get("messages")
            if value is not None:
                span.set_attribute(
                    SpanAttributes.TRACELOOP_ENTITY_INPUT,
                    json_value(value, complete=embedding),
                )
                if not embedding and isinstance(value, list):
                    for index, message in enumerate(value):
                        if isinstance(message, dict) and message.get("tool_calls"):
                            span.set_attribute(
                                f"{SpanAttributes.LLM_PROMPTS}.{index}.tool_calls",
                                json_value(message["tool_calls"], complete=True),
                            )
            if not embedding and kwargs.get("tools") is not None:
                span.set_attribute(
                    SpanAttributes.LLM_REQUEST_FUNCTIONS,
                    json_value(kwargs["tools"], complete=True),
                )

        def observe(result, embedding):
            span = current()
            if span is None:
                return
            response = values(result)
            if response.get("usage") is not None:
                self.processor.source_usage(span, values(response["usage"]))
            if not self.processor.allowed(span):
                return
            if embedding:
                vectors = [
                    values(item).get("embedding")
                    for item in response.get("data", []) or []
                ]
                if vectors and all(isinstance(v, list) for v in vectors):
                    span.set_attribute(
                        SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
                        json_value(vectors, complete=True),
                    )
            else:
                for index, choice in enumerate(response.get("choices", []) or []):
                    message = data(values(choice).get("message"), complete=True)
                    if isinstance(message, dict) and message.get("tool_calls"):
                        span.set_attribute(
                            f"{SpanAttributes.LLM_COMPLETIONS}.{index}.tool_calls",
                            json_value(message["tool_calls"], complete=True),
                        )
                        self.processor.mark_current_calls(span)

        for owner, asynchronous, embedding in (
            (Embeddings, False, True),
            (AsyncEmbeddings, True, True),
            (Completions, False, False),
            (AsyncCompletions, True, False),
        ):
            original = owner.create
            if asynchronous:

                @functools.wraps(original)
                async def call(
                    instance, *a, __original=original, __embedding=embedding, **kw
                ):
                    try:
                        before(kw, __embedding)
                    except Exception:  # noqa: BLE001 - observation cannot alter native results
                        logger.debug("Could not observe AgentOps request")
                    result = await __original(instance, *a, **kw)
                    try:
                        observe(result, __embedding)
                    except Exception:  # noqa: BLE001 - observation cannot alter native results
                        logger.debug("Could not observe AgentOps response")
                    return result
            else:

                @functools.wraps(original)
                def call(
                    instance, *a, __original=original, __embedding=embedding, **kw
                ):
                    try:
                        before(kw, __embedding)
                    except Exception:  # noqa: BLE001 - observation cannot alter native results
                        logger.debug("Could not observe AgentOps request")
                    result = __original(instance, *a, **kw)
                    try:
                        observe(result, __embedding)
                    except Exception:  # noqa: BLE001 - observation cannot alter native results
                        logger.debug("Could not observe AgentOps response")
                    return result

            self.patch(owner, "create", call)

    def restore(self):
        self.active = False
        try:
            for owner, name, original, wrapper in reversed(self.patches):
                if owner.__dict__.get(name) is wrapper:
                    if original is _MISSING:
                        delattr(owner, name)
                    else:
                        setattr(owner, name, original)
            active = self.provider._active_span_processor
            with active._lock:
                active._span_processors = tuple(
                    p for p in active._span_processors if p is not self.processor
                )
            if self.core_snapshot and self.core.provider is self.core_proxy:
                initialized, provider, meter = self.core_snapshot
                self.core.provider = provider
                if self.core._initialized is True:
                    self.core._initialized = initialized
                if self.core._meter_provider is None:
                    self.core._meter_provider = meter
        finally:
            self.processor.shutdown()
            self.patches.clear()


class AgentOpsInstrumentor:
    name = AGENTOPS_INSTRUMENTATION_NAME

    def __init__(self, *, capture_content=True):
        self._capture_content = capture_content
        self._is_instrumented = False
        self._runtime = None

    def activate(self):
        global _RUNTIME
        with _LOCK:
            if self._is_instrumented:
                return
            tracer = getattr(RespanTracer, "_instance", None)
            if tracer is not None and not getattr(tracer, "is_enabled", True):
                return
            provider = trace.get_tracer_provider()
            if _RUNTIME:
                if _RUNTIME.provider is not provider:
                    raise RuntimeError("AgentOps is active on another provider")
                if _RUNTIME.processor.capture_content != self._capture_content:
                    raise ValueError(
                        "AgentOps is active with different privacy settings"
                    )
                _RUNTIME.owners += 1
            else:
                runtime = _Runtime(provider, self._capture_content)
                try:
                    runtime.install()
                except BaseException:
                    runtime.restore()
                    raise
                _RUNTIME = runtime
            self._runtime = _RUNTIME
            self._is_instrumented = True

    def deactivate(self):
        global _RUNTIME
        with _LOCK:
            if not self._is_instrumented:
                return
            self._is_instrumented = False
            runtime, self._runtime = self._runtime, None
            runtime.owners -= 1
            if runtime.owners:
                return
            _RUNTIME = None
            runtime.restore()
