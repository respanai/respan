"""Register a native AI telemetry adapter and capture non-chat operation data."""

from __future__ import annotations

import contextvars
import functools
import importlib
import inspect
import os
from collections import OrderedDict
from threading import RLock
from types import SimpleNamespace
from typing import Any, ClassVar

from ai import experimental_telemetry as telemetry
from ai.experimental_telemetry.otel import OtelAdapter
from opentelemetry import context, trace
from opentelemetry.semconv_ai import SpanAttributes
from respan_sdk.constants.llm_logging import (
    LOG_TYPE_EMBEDDING,
    LOG_TYPE_TASK,
)
from respan_sdk.constants.span_attributes import (
    RESPAN_INTERNAL_SPAN_NAME_KIND,
    RESPAN_LOG_TYPE,
    RESPAN_METADATA,
)
from respan_tracing.constants.context_constants import ENABLE_CONTENT_TRACING_KEY

from ._operation_sink import _OperationSink
from ._translator import json_value, span_attributes, usage_attributes

_operation: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "respan_vercel_operation", default=None
)
_OPERATIONS = (
    ("embeddings", "embed", "embed", LOG_TYPE_EMBEDDING),
    ("images", "generate_image", "generate_image", LOG_TYPE_TASK),
    ("videos", "generate_video", "generate_video", LOG_TYPE_TASK),
    ("audio", "generate_audio", "generate_audio", LOG_TYPE_TASK),
    ("transcriptions", "transcribe", "transcribe", LOG_TYPE_TASK),
    ("reranking", "rerank", "rerank", LOG_TYPE_TASK),
    ("evaluation", "experimental_evaluate", "evaluate", LOG_TYPE_TASK),
)


def _content_allowed() -> bool:
    return context.get_value(ENABLE_CONTENT_TRACING_KEY) is not False and os.getenv(
        "TRACELOOP_TRACE_CONTENT", "true"
    ).strip().lower() not in {"0", "false", "no", "off"}


_DATA_MODELS = {
    model.model_fields["kind"].default: model
    for model in (
        telemetry.RunSpanData,
        telemetry.LoopTurnSpanData,
        telemetry.AiStreamSpanData,
        telemetry.AiGenerateSpanData,
        telemetry.ToolExecutionSpanData,
        telemetry.HookSpanData,
        telemetry.EmbedSpanData,
        telemetry.EvaluateSpanData,
        telemetry.GenerateAudioSpanData,
        telemetry.GenerateImageSpanData,
        telemetry.GenerateVideoSpanData,
        telemetry.RerankSpanData,
        telemetry.TranscribeSpanData,
        telemetry.CustomSpanData,
    )
}


def _typed_data(data: Any) -> Any:
    if not isinstance(data, dict):
        return data
    model = _DATA_MODELS.get(data.get("kind"))
    return model.model_validate(data) if model else SimpleNamespace(**data)


class _SpanView:
    """Read updated SDK snapshots without changing data seen by other adapters."""

    def __init__(self, span: Any) -> None:
        self._span = span
        self.capture_content = _content_allowed()

    @property
    def data(self) -> Any:
        return _typed_data(self._span.data)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._span, name)


class _Adapter(OtelAdapter):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # Finished replay parents may have inherited an outer OTel trace ID,
        # which differs from the SDK's durable trace ID. Keep only contexts,
        # never payloads, and bound retention for long-running processes.
        self._contexts: OrderedDict[tuple[str, str], Any] = OrderedDict()

    def __call__(self, span: Any, /) -> Any:
        view = _SpanView(span)
        if _operation.get() == view.data.kind:
            return None  # The operation wrapper captures the full result instead.
        return super().__call__(view)

    def _parent_context(self, span: Any) -> Any:
        if span.parent_id in self._live:
            # Keep intervening application spans in a live call's ancestry.
            return super()._parent_context(span)
        key = (span.trace_id, span.parent_id)
        parent = self._contexts.get(key)
        if parent is not None:
            self._contexts.move_to_end(key)
            return trace.set_span_in_context(
                trace.NonRecordingSpan(parent), context.Context()
            )
        return super()._parent_context(span)

    async def on_span_start(self, span: Any, /) -> None:
        await super().on_span_start(span)
        current = self._live.get(span.id)
        if current is not None:
            key = (span.trace_id, span.id)
            self._contexts[key] = current.get_span_context()
            self._contexts.move_to_end(key)
            while len(self._contexts) > 4096:
                self._contexts.popitem(last=False)

    def span_attrs(self, span: Any, /) -> dict[str, Any]:
        return span_attributes(
            span,
            self._is_capturing_content and span.capture_content and _content_allowed(),
        )

    async def on_span_end(self, span: Any, /) -> None:
        # Tool failures can be returned to the model without raising an exception.
        current = self._live.get(span.id)
        if current is not None and getattr(_typed_data(span.data), "is_error", False):
            current.set_status(trace.StatusCode.ERROR, "Tool execution failed")
        await super().on_span_end(span)


class VercelInstrumentor:
    """Trace Vercel's ``ai`` Python SDK into the active OpenTelemetry provider.

    Activate after initializing Respan, before importing operation functions.
    Deactivate after in-flight SDK calls have completed. The upstream telemetry
    adapter owns durable span identifiers; this plugin never shuts down the
    application's tracer provider.
    """

    name = "vercel"
    _lock: ClassVar[RLock] = RLock()
    _shared: ClassVar[_Adapter | None] = None
    _users: ClassVar[int] = 0
    _generation: ClassVar[int] = 0
    _active_operations: ClassVar[int] = 0
    _current_wrappers: ClassVar[dict[str, Any]] = {}
    _patches: ClassVar[list[tuple[Any, str, Any, Any]]] = []
    _previous_ids: ClassVar[Any] = None
    _installed_ids: ClassVar[Any] = None

    def __init__(
        self, *, capture_content: bool = True, tracer_provider: Any = None
    ) -> None:
        self._capture_content = capture_content
        self._provider = tracer_provider
        self._active = False

    def activate(self) -> None:
        cls = type(self)
        provider = self._provider or trace.get_tracer_provider()
        with cls._lock:
            if self._active:
                return
            if cls._shared is not None:
                if (
                    cls._shared._provider is not provider
                    or cls._shared._is_capturing_content != self._capture_content
                ):
                    raise ValueError(
                        "Active Vercel instrumentors must share provider "
                        "and content settings"
                    )
            else:
                cls._generation += 1
                cls._previous_ids = getattr(provider, "id_generator", None)
                adapter = _Adapter(
                    tracer_provider=provider, capture_content=self._capture_content
                )
                cls._installed_ids = getattr(provider, "id_generator", None)
                cls._shared = adapter
                try:
                    self._patch_operations(provider)
                    telemetry.register(adapter)
                except BaseException:
                    cls._restore()
                    raise
            cls._users += 1
            self._active = True

    def _patch_operations(self, provider: Any) -> None:
        import ai.ops

        cls = type(self)
        tracer = trace.get_tracer(
            "respan.instrumentation.vercel", tracer_provider=provider
        )
        for module_name, function_name, kind, log_type in _OPERATIONS:
            module = importlib.import_module(f"ai.ops.{module_name}")
            original = getattr(module, function_name)
            wrapper = self._wrap_operation(original, tracer, kind, log_type)
            cls._current_wrappers[kind] = wrapper
            for owner in (module, ai.ops):
                previous = getattr(owner, function_name)
                setattr(owner, function_name, wrapper)
                cls._patches.append((owner, function_name, previous, wrapper))

    def _wrap_operation(
        self, original: Any, tracer: Any, kind: str, log_type: str
    ) -> Any:
        signature = inspect.signature(original)
        capture_setting = self._capture_content
        cls = type(self)
        generation = cls._generation

        @functools.wraps(original)
        async def wrapped(*args: Any, **kwargs: Any) -> Any:
            with cls._lock:
                current = cls._current_wrappers.get(kind)
                is_current = generation == cls._generation and current is wrapped
                if is_current:
                    cls._active_operations += 1
            if not is_current:
                # Imported aliases cannot be unpatched. Route to the current
                # activation (including its privacy/provider settings), or the
                # original callable after the final deactivate.
                return await (current or original)(*args, **kwargs)
            try:
                capture = capture_setting and _content_allowed()
                bound = signature.bind(*args, **kwargs)
                input_payload = None
                if capture:
                    inputs = {
                        key: value
                        for key, value in bound.arguments.items()
                        if key != "model"
                    }
                    input_payload = json_value(
                        inputs["values"] if kind == "embed" else inputs
                    )
                # No public current-sink accessor exists in this experimental
                # SDK version. Only inspect it; routing stays on public use_sink.
                sink = importlib.import_module(
                    "ai.experimental_telemetry.span"
                )._current_sink.get()
                if sink is not None:
                    deferred = _OperationSink(sink, kind)
                    result = None
                    try:
                        async with telemetry.use_sink(deferred):
                            result = await original(*args, **kwargs)
                        return result
                    finally:
                        await deferred.finish(
                            capture_content=capture and _content_allowed(),
                            input_payload=input_payload,
                            result=result,
                        )
                model = bound.arguments.get("model")
                attrs = {
                    RESPAN_LOG_TYPE: log_type,
                    SpanAttributes.TRACELOOP_ENTITY_NAME: kind,
                }
                metadata = {"operation": kind}
                if model is not None and log_type == LOG_TYPE_EMBEDDING:
                    attrs[SpanAttributes.LLM_REQUEST_MODEL] = model.id
                    attrs[SpanAttributes.LLM_SYSTEM] = model.provider.name
                elif model is not None:
                    metadata.update(
                        {"model": model.id, "provider": model.provider.name}
                    )
                if kind in {"generate_audio", "transcribe"}:
                    attrs[RESPAN_INTERNAL_SPAN_NAME_KIND] = (
                        "speech" if kind == "generate_audio" else "transcribe"
                    )
                if log_type == LOG_TYPE_EMBEDDING:
                    attrs[SpanAttributes.LLM_REQUEST_TYPE] = "embedding"
                token = _operation.set(kind)
                try:
                    with tracer.start_as_current_span(
                        kind, attributes=attrs, kind=trace.SpanKind.CLIENT
                    ) as span:
                        try:
                            result = await original(*args, **kwargs)
                            if log_type == LOG_TYPE_EMBEDDING:
                                span.set_attributes(usage_attributes(result.usage))
                            elif result.usage is not None:
                                metadata["usage"] = result.usage.model_dump(
                                    exclude_none=True
                                )
                            span.set_attribute(RESPAN_METADATA, json_value(metadata))
                            if capture and _content_allowed():
                                span.set_attribute(
                                    SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
                                    json_value(result.value),
                                )
                            return result
                        finally:
                            if capture and _content_allowed():
                                span.set_attribute(
                                    SpanAttributes.TRACELOOP_ENTITY_INPUT, input_payload
                                )
                finally:
                    _operation.reset(token)
            finally:
                with cls._lock:
                    cls._active_operations -= 1

        return wrapped

    @classmethod
    def _restore(cls) -> None:
        for owner, name, previous, wrapper in reversed(cls._patches):
            if getattr(owner, name) is wrapper:
                setattr(owner, name, previous)
        cls._patches = []
        cls._current_wrappers = {}
        cls._generation += 1
        if cls._shared is not None:
            provider = cls._shared._provider
            if getattr(provider, "id_generator", None) is cls._installed_ids:
                provider.id_generator = cls._previous_ids
        cls._shared = None
        cls._previous_ids = cls._installed_ids = None

    def deactivate(self) -> None:
        cls = type(self)
        with cls._lock:
            if not self._active:
                return
            if cls._users == 1:
                if cls._shared._live or cls._active_operations:
                    raise RuntimeError(
                        "Wait for in-flight AI SDK calls before deactivating"
                    )
                telemetry.unregister(cls._shared)
                cls._restore()
            cls._users -= 1
            self._active = False
