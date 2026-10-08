"""AgentScope adapters using the active OpenTelemetry provider."""

from __future__ import annotations

import functools
import importlib
import inspect
import json
import logging
from collections.abc import Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from threading import RLock
from types import MethodType
from typing import Any

from opentelemetry import context as context_api
from opentelemetry import trace
from opentelemetry.context import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import LLMRequestTypeValues, SpanAttributes
from opentelemetry.trace import Status, StatusCode
from respan_sdk.constants.llm_logging import LogMethodChoices
from respan_sdk.constants.span_attributes import (
    RESPAN_LOG_METHOD,
    RESPAN_LOG_TYPE,
    RESPAN_METADATA,
)
from respan_tracing.core.tracer import RespanTracer
from respan_tracing.decorators.base import _should_send_prompts

from . import _attributes as attrs

logger = logging.getLogger(__name__)
AGENTSCOPE_INSTRUMENTATION_NAME = "agentscope"
AGENTSCOPE_AGENT_MODULE = "agentscope.agent"
AGENTSCOPE_MODEL_MODULE = "agentscope.model"
AGENTSCOPE_TOOL_MODULE = "agentscope.tool"
_CALLS: ContextVar[frozenset] = ContextVar(
    "respan_agentscope_calls", default=frozenset()
)
_CAPTURE: ContextVar[bool] = ContextVar("respan_agentscope_capture", default=True)
_LOCK = RLock()
_PATCHES: dict[tuple[int, str], Any] = {}


def _enabled():
    runtime = getattr(RespanTracer, "_instance", None)
    return bool(getattr(runtime, "is_enabled", True)) and not context_api.get_value(
        _SUPPRESS_INSTRUMENTATION_KEY
    )


def _tracer():
    runtime = getattr(RespanTracer, "_instance", None)
    if runtime is not None:
        return runtime.get_tracer("respan.instrumentation.agentscope")
    return trace.get_tracer("respan.instrumentation.agentscope")


def _arg(args, kwargs, position, name):
    return args[position] if len(args) > position else kwargs.get(name)


def _strip_content(attributes):
    return {
        key: value
        for key, value in attributes.items()
        if key
        not in (
            SpanAttributes.TRACELOOP_ENTITY_INPUT,
            SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
            SpanAttributes.LLM_REQUEST_FUNCTIONS,
            RESPAN_METADATA,
        )
        and not key.startswith(
            (f"{SpanAttributes.LLM_PROMPTS}.", f"{SpanAttributes.LLM_COMPLETIONS}.")
        )
    }


class _Operation:
    def __init__(self, kind, instance, args, kwargs, capture, parent, active):
        self.active = active
        self.kind, self.instance = kind, instance
        self.args, self.kwargs = args, kwargs
        self.capture, self.parent = capture, parent
        self.span = None
        self.response = None
        self.partial = {"content": [], "usage": None}
        self.ended = False
        self.reason = ""
        self.streaming = False
        self.key = (kind, id(instance))

    def start(self):
        if self.span is None:
            self.span = _tracer().start_span(
                f"agentscope.{self.kind}", context=self.parent
            )

    @contextmanager
    def scope(self):
        self.start()
        token = context_api.attach(trace.set_span_in_context(self.span, self.parent))
        active = _CALLS.set(_CALLS.get() | {self.key})
        policy = _CAPTURE.set(self.capture)
        try:
            yield
        finally:
            _CAPTURE.reset(policy)
            _CALLS.reset(active)
            context_api.detach(token)

    def observe(self, result):
        try:
            self._observe(result)
        except Exception:
            logger.debug("AgentScope result capture failed", exc_info=True)

    def _observe(self, result):
        reason = attrs._object_value(result, "finished_reason")
        if reason is not None:
            self.reason = str(reason)
        if self.kind == "model_call":
            if attrs._object_value(result, "is_last", True):
                self.response = result
                return
            usage = attrs._object_value(result, "usage")
            if usage is not None:
                self.partial["usage"] = usage
            # SDKs yield deltas followed by a cumulative is_last response. Retain
            # only the observed partial output if callers stop before that result.
            for block in attrs._content_blocks(result):
                item = attrs._object_to_dict(block)
                previous = next(
                    (
                        b
                        for b in self.partial["content"]
                        if b.get("id") == item.get("id")
                        and b.get("type") == item.get("type")
                    ),
                    None,
                )
                key = {
                    "text": "text",
                    "thinking": "thinking",
                    "tool_call": "input",
                }.get(item.get("type"))
                if previous is not None and key and isinstance(item.get(key), str):
                    previous[key] = previous.get(key, "") + item[key]
                else:
                    self.partial["content"].append(item)
            self.response = self.partial
        elif self.kind in ("agent", "workflow"):
            # reply_stream defaults to events, not Msg. Content deltas are real
            # output; a final Msg (if requested) supersedes them.
            if attrs._object_value(
                result, "role"
            ) is not None and attrs._object_has_attr(result, "content"):
                self.response = result
            elif str(attrs._object_value(result, "type")).lower() == "text_block_delta":
                delta = attrs._object_value(result, "delta", "")
                old = self.response or {"content": ""}
                if isinstance(old, dict):
                    old["content"] += delta
                    self.response = old
        else:
            self.response = result

    def attributes(self):
        if self.kind in ("agent", "workflow"):
            result = attrs._agent_attributes(
                agent=self.instance,
                input_value=_arg(self.args, self.kwargs, 0, "inputs"),
                output_value=self.response,
            )
            result[RESPAN_LOG_TYPE] = self.kind
            if self.response is None:
                result.pop(SpanAttributes.TRACELOOP_ENTITY_OUTPUT, None)
        elif self.kind == "tool":
            result = attrs._tool_attributes(
                tool_call=_arg(self.args, self.kwargs, 0, "tool_call"),
                chunks=[] if self.response is None else [self.response],
            )
            if self.response is None:
                result.pop(SpanAttributes.TRACELOOP_ENTITY_OUTPUT, None)
        elif self.kind == "embedding":
            result = {
                RESPAN_LOG_METHOD: LogMethodChoices.TRACING_INTEGRATION.value,
                RESPAN_LOG_TYPE: "embedding",
                SpanAttributes.TRACELOOP_ENTITY_NAME: "embedding",
                SpanAttributes.LLM_REQUEST_TYPE: LLMRequestTypeValues.EMBEDDING.value,
                SpanAttributes.LLM_REQUEST_MODEL: attrs._model_name(self.instance),
                SpanAttributes.TRACELOOP_ENTITY_INPUT: attrs._json_string(
                    _arg(self.args, self.kwargs, 0, "inputs")
                ),
            }
            provider = attrs._model_provider(self.instance)
            if provider is not None:
                result[SpanAttributes.LLM_SYSTEM] = provider
            if self.response is not None:
                # Vector data is never truncated by the generic SDK serializer.
                result[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = json.dumps(
                    attrs._object_value(self.response, "embeddings"),
                    separators=(",", ":"),
                )
                usage = attrs._object_value(self.response, "usage")
                tokens = attrs._coerce_int(attrs._object_value(usage, "tokens"))
                if (
                    attrs._object_value(self.response, "source") != "cache"
                    and tokens is not None
                ):
                    result[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] = tokens
                    result[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] = tokens
                    result[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] = tokens
        else:
            result = attrs._model_attributes(
                model=self.instance,
                messages=_arg(self.args, self.kwargs, 0, "messages"),
                tools=_arg(self.args, self.kwargs, 1, "tools")
                if self.kind == "model_call"
                else None,
                response=self.response,
            )
        if self.kind in ("model_call", "structured"):
            result[SpanAttributes.GEN_AI_IS_STREAMING] = self.streaming
        return result if self.capture else _strip_content(result)

    def finish(self, error=None):
        if self.ended:
            return
        self.ended = True
        self.start()
        try:
            if self.span.is_recording():
                try:
                    self.span.set_attributes(self.attributes())
                except Exception:
                    logger.debug(
                        "AgentScope telemetry serialization failed", exc_info=True
                    )
                state = str(attrs._object_value(self.response, "state", ""))
                reason = self.reason or str(
                    attrs._object_value(self.response, "finished_reason", "")
                )
                if error is not None:
                    # No fabricated provider response or HTTP status.
                    self.span.set_status(Status(StatusCode.ERROR, str(error)))
                    self.span.record_exception(error)
                elif any(
                    word in (state + reason).lower()
                    for word in ("error", "denied", "interrupted")
                ):
                    self.span.set_status(Status(StatusCode.ERROR, state or reason))
        finally:
            self.span.end()


class _Iterator:
    """Demand-driven proxy: attach only while the SDK advances or closes."""

    def __init__(self, source, operation):
        self.source, self.operation = source, operation
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.closed:
            raise StopAsyncIteration
        try:
            with self.operation.scope():
                result = await self.source.__anext__()
            self.operation.observe(result)
            return result
        except StopAsyncIteration:
            self.closed = True
            self.operation.finish()
            raise
        except BaseException as error:
            self.closed = True
            try:
                with self.operation.scope():
                    close = getattr(self.source, "aclose", None)
                    if close:
                        await close()
            except BaseException:
                logger.debug(
                    "AgentScope cleanup failed after an iteration error", exc_info=True
                )
            self.operation.finish(error)
            raise

    async def aclose(self):
        if self.closed:
            return
        self.closed = True
        try:
            with self.operation.scope():
                close = getattr(self.source, "aclose", None)
                if close:
                    await close()
        except BaseException as error:
            self.operation.finish(error)
            raise
        else:
            self.operation.finish()


async def _stream(source, operation):
    # AgentScope checks inspect.isasyncgen(), so retain the native generator API.
    if operation.span is None and not operation.active():
        try:
            async for item in source:
                yield item
        finally:
            close = getattr(source, "aclose", None)
            if close:
                await close()
        return
    operation.streaming = True
    iterator = _Iterator(source, operation)
    try:
        async for item in iterator:
            yield item
    finally:
        await iterator.aclose()


@dataclass
class _Patch:
    target: Any
    name: str
    original: Any
    raw: Any
    had_own: bool
    kind: str
    bound: bool
    capture: bool
    owners: set = field(default_factory=set)
    wrapper: Any = None

    def make_wrapper(self):
        @functools.wraps(self.original)
        def call(instance, *args, **kwargs):
            result = (
                self.original(*args, **kwargs)
                if self.bound
                else self.original(instance, *args, **kwargs)
            )
            if (
                not self.owners
                or not _enabled()
                or (self.kind, id(instance)) in _CALLS.get()
            ):
                return result
            capture = self.capture and _CAPTURE.get() and _should_send_prompts()
            operation = _Operation(
                self.kind,
                instance,
                args,
                kwargs,
                capture,
                context_api.get_current(),
                lambda: bool(self.owners),
            )
            if inspect.isawaitable(result):

                async def wait():
                    # A retained wrapper must become inert after final teardown.
                    if not self.owners:
                        return await result
                    try:
                        with operation.scope():
                            response = await result
                    except BaseException as error:
                        operation.finish(error)
                        raise
                    if attrs._object_has_attr(response, "__anext__"):
                        return _stream(response, operation)
                    operation.observe(response)
                    operation.finish()
                    return response

                return wait()
            if attrs._object_has_attr(result, "__anext__"):
                return _stream(result, operation)
            return result

        return MethodType(call, self.target) if self.bound else call


class AgentScopeInstrumentor:
    """Trace AgentScope agents, chat/structured models, embeddings and tools.

    Explicit custom model instances patch their classes (Python resolves special
    methods on the type). Shared patches stay active until their final owner ends.
    """

    name = AGENTSCOPE_INSTRUMENTATION_NAME

    def __init__(
        self,
        *,
        agent=None,
        model=None,
        models: Sequence[Any] | None = None,
        toolkit=None,
        instrument_models=True,
        instrument_tools=True,
        instrument_embeddings=True,
        embedding_models: Sequence[Any] | None = None,
        capture_content=True,
    ):
        if model is not None and models is not None:
            raise ValueError("Pass either model or models, not both")
        self._agent, self._toolkit = agent, toolkit
        self._models = (
            tuple(models) if models is not None else (() if model is None else (model,))
        )
        self._embedding_models = tuple(embedding_models or ())
        self._instrument_models, self._instrument_tools = (
            instrument_models,
            instrument_tools,
        )
        self._instrument_embeddings = instrument_embeddings
        self._capture_content = capture_content
        self._patches = []
        self._is_instrumented = False

    def _patch(self, target, name, kind):
        original = attrs._object_value(target, name)
        if not callable(original):
            return
        # Reuse inherited patches too, including explicit instance owners.
        underlying = getattr(original, "__func__", original)
        existing = next(
            (
                p
                for p in _PATCHES.values()
                if getattr(p.wrapper, "__func__", p.wrapper) is underlying
            ),
            None,
        )
        key = (id(target), name)
        patch = existing or _PATCHES.get(key)
        if patch is not None:
            if patch.capture != self._capture_content:
                raise ValueError(
                    "AgentScope owners must use the same capture_content setting"
                )
        else:
            patch = _Patch(
                target,
                name,
                original,
                inspect.getattr_static(target, name),
                name in vars(target),
                kind,
                not inspect.isclass(target),
                self._capture_content,
            )
            patch.wrapper = patch.make_wrapper()
            setattr(target, name, patch.wrapper)
            _PATCHES[key] = patch
        patch.owners.add(self)
        if not any(p is patch for p in self._patches):
            self._patches.append(patch)

    def activate(self):
        with _LOCK:
            if self._is_instrumented:
                return
            if not bool(
                getattr(getattr(RespanTracer, "_instance", None), "is_enabled", True)
            ):
                logger.info(
                    "AgentScope instrumentation skipped because tracing is disabled"
                )
                return
            try:
                agent_module = importlib.import_module(AGENTSCOPE_AGENT_MODULE)
                agents = (
                    [self._agent] if self._agent is not None else [agent_module.Agent]
                )
                for agent in agents:
                    for method in ("reply", "reply_stream"):
                        self._patch(agent, method, "agent")
                if self._instrument_models:
                    module = importlib.import_module(AGENTSCOPE_MODEL_MODULE)
                    targets = (
                        [type(m) for m in self._models]
                        if self._models
                        else [
                            value
                            for value in vars(module).values()
                            if inspect.isclass(value)
                            and (
                                value.__name__.endswith(("ChatModel", "ChatModelBase"))
                                or value.__name__
                                in ("ChatModelBase", "OpenAIResponseModel")
                            )
                        ]
                    )
                    for target in dict.fromkeys(targets):
                        self._patch(target, "__call__", "model_call")
                        self._patch(target, "generate_structured_output", "structured")
                if self._instrument_embeddings:
                    module = importlib.import_module("agentscope.embedding")
                    targets = (
                        [type(m) for m in self._embedding_models]
                        if self._embedding_models
                        else [
                            value
                            for value in vars(module).values()
                            if inspect.isclass(value)
                            and value.__name__.endswith("EmbeddingModel")
                        ]
                    )
                    for target in dict.fromkeys(targets):
                        # Instrument each real/cache batch before SDK aggregation
                        # can synthesize a zero usage on missing provider fields.
                        self._patch(target, "_call_api", "embedding")
                if self._instrument_tools:
                    target = (
                        self._toolkit
                        if self._toolkit is not None
                        else importlib.import_module(AGENTSCOPE_TOOL_MODULE).Toolkit
                    )
                    self._patch(target, "call_tool", "tool")
                if self._agent is None:
                    try:
                        pipeline = importlib.import_module("agentscope.pipeline")
                    except ImportError:
                        pipeline = None
                    if pipeline is not None:
                        for name in ("TeamPipeline", "GoalPipeline"):
                            target = getattr(pipeline, name, None)
                            if target is not None:
                                self._patch(target, "reply_stream", "workflow")
                self._is_instrumented = bool(self._patches)
            except BaseException:
                self.deactivate()
                raise

    def deactivate(self):
        with _LOCK:
            for patch in reversed(self._patches):
                patch.owners.discard(self)
                if patch.owners:
                    continue
                if (
                    inspect.getattr_static(patch.target, patch.name, None)
                    is patch.wrapper
                ):
                    if patch.had_own:
                        setattr(patch.target, patch.name, patch.raw)
                    else:
                        delattr(patch.target, patch.name)
                _PATCHES.pop((id(patch.target), patch.name), None)
            self._patches.clear()
            self._is_instrumented = False
