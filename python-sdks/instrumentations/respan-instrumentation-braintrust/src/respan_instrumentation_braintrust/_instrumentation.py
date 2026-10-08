"""Observe native Braintrust lifecycle and export records without replacing its sink."""

from __future__ import annotations

import inspect
import logging
import os
import threading
from collections import OrderedDict
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
from typing import Any

from opentelemetry import context, trace
from opentelemetry.semconv._incubating.attributes.error_attributes import ERROR_MESSAGE
from opentelemetry.semconv.attributes.error_attributes import ERROR_TYPE
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
from respan_tracing.core.tracer import RespanTracer

from ._constants import BRAINTRUST_SPAN_TYPE_TO_LOG_TYPE
from ._mapping import attributes, get, plain, private_usage
from ._serialization import json_string, redact_text

logger = logging.getLogger(__name__)
_LOCK = threading.RLock()
_RUNTIME = None
_WRITE = ContextVar("respan_braintrust_write", default=None)
_REQUEST = ContextVar("respan_braintrust_request", default=None)
_ADVANCE = ContextVar("respan_braintrust_advance", default=False)
_CONTENT_BOUND = "respan_braintrust_content_bound"
_MISSING = object()


def _allowed():
    return (
        context.get_value(ENABLE_CONTENT_TRACING_KEY) is not False
        and context.get_value(_CONTENT_BOUND) is not False
        and os.getenv("TRACELOOP_TRACE_CONTENT", "true").strip().lower()
        not in {"0", "false", "off", "no"}
    )


def _safe(fn, *args):
    try:
        return fn(*args)
    except Exception:  # noqa: BLE001 - telemetry must preserve native behavior
        logger.debug("Braintrust telemetry observation failed open")
        return None


@dataclass
class _State:
    span: Any
    capture: bool
    rows: list = field(default_factory=list)
    pending: int = 0
    next_sequence: int = 0
    sdk_ended: bool = False
    finished: bool = False
    source_id: str | None = None
    token: Any = None
    error: BaseException | None = None
    parent: Any = None
    embedding: dict | None = None
    native_sink: Any = None
    masking: Any = None


class _ObservedLazy:
    def __init__(self, native, runtime, state, sequence):
        self.native, self.runtime, self.state = native, runtime, state
        self.sequence = sequence
        self.resolved = False

    def __getattr__(self, name):
        return getattr(self.native, name)

    def get(self, *args, **kwargs):
        value = self.native.get(*args, **kwargs)
        with self.runtime.lock:
            if not self.resolved:
                self.resolved = True
                _safe(self.runtime.record, self.state, value, self.sequence)
                self.state.pending -= 1
                _safe(self.runtime.maybe_finish, self.state)
        return value


class _ObservedLogger:
    def __init__(self, native, runtime):
        self.native, self.runtime = native, runtime

    def __getattr__(self, name):
        return getattr(self.native, name)

    def log(self, *items):
        state = _WRITE.get()
        if state is not None:
            state.native_sink = self.native
            _safe(self.runtime.veto, state)
        if not self.runtime.active or state is None or state.finished:
            return self.native.log(*items)
        wrappers = []
        with self.runtime.lock:
            for item in items:
                state.pending += 1
                wrappers.append(
                    _ObservedLazy(item, self.runtime, state, state.next_sequence)
                )
                state.next_sequence += 1
        return self.native.log(*wrappers)


class _Iterator:
    """Detach only the OTel context between native generator advances."""

    def __init__(self, native):
        self.native, self.saved = native, None

    def __iter__(self):
        return self

    def __getattr__(self, name):
        return getattr(self.native, name)

    def advance(self, method, *args):
        advancing = _ADVANCE.set(True)
        token = context.attach(self.saved or context.get_current())
        try:
            value = method(*args)
            self.saved = context.get_current()
            return value
        except BaseException:
            self.saved = None
            raise
        finally:
            context.detach(token)
            _ADVANCE.reset(advancing)

    def __next__(self):
        return self.advance(self.native.__next__)

    def send(self, value):
        return self.advance(self.native.send, value)

    def throw(self, *args):
        return self.advance(self.native.throw, *args)

    def close(self):
        try:
            return self.advance(self.native.close)
        finally:
            self.saved = None


class _AsyncIterator:
    def __init__(self, native):
        self.native, self.saved = native, None

    def __aiter__(self):
        return self

    def __getattr__(self, name):
        return getattr(self.native, name)

    async def advance(self, method, *args):
        advancing = _ADVANCE.set(True)
        token = context.attach(self.saved or context.get_current())
        try:
            value = await method(*args)
            self.saved = context.get_current()
            return value
        except BaseException:
            self.saved = None
            raise
        finally:
            context.detach(token)
            _ADVANCE.reset(advancing)

    async def __anext__(self):
        return await self.advance(self.native.__anext__)

    async def asend(self, value):
        return await self.advance(self.native.asend, value)

    async def athrow(self, *args):
        return await self.advance(self.native.athrow, *args)

    async def aclose(self):
        try:
            return await self.advance(self.native.aclose)
        finally:
            self.saved = None


class _Runtime:
    def __init__(self, braintrust, provider, content, masking):
        self.braintrust, self.provider, self.content, self.masking = (
            braintrust,
            provider,
            content,
            masking,
        )
        self.active = True
        self.count = 1
        self.patches = []
        self.states = {}
        self.source = {}
        self.parents = OrderedDict()
        self.lock = threading.RLock()
        self.tracer = provider.get_tracer("respan.instrumentation.braintrust")

    def patch(self, owner, name, factory):
        original = getattr(owner, name, None)
        if original is None:
            return
        replacement = factory(original)
        previous = vars(owner).get(name, _MISSING)
        self.patches.append((owner, name, previous, replacement))
        setattr(owner, name, replacement)

    def begin(self, instance, arguments):
        parent_ids = arguments.get("parent_span_ids")
        parent_id = get(parent_ids, "span_id")
        ids = get(parent_ids, "span_parents")
        if not parent_id and ids:
            parent_id = ids[0]
        parent = self.source.get(parent_id)
        cached = self.parents.get(parent_id)
        if (
            context.get_value(context._SUPPRESS_INSTRUMENTATION_KEY)
            or context.get_value(SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY)
            or (parent_id in self.source and parent is None)
        ):
            self.states[id(instance)] = None
            return
        native = arguments.get("span_attributes") or {}
        kind = (
            get(native, "type")
            or arguments.get("type")
            or arguments.get("default_root_type")
        )
        kind = BRAINTRUST_SPAN_TYPE_TO_LOG_TYPE.get(kind, "task")
        name = arguments.get("name") or get(native, "name") or kind
        name = redact_text(name) if isinstance(name, str) else kind
        parent_span = (
            parent.span
            if parent
            else trace.NonRecordingSpan(cached[0])
            if cached
            else None
        )
        parent_context = (
            trace.set_span_in_context(parent_span)
            if parent_span
            else context.get_current()
        )
        span = self.tracer.start_span(
            f"{kind}.{name}"
            if kind in {"tool", "agent"}
            else "llm"
            if kind == "chat"
            else kind,
            context=parent_context,
            attributes={
                RESPAN_LOG_TYPE: kind,
                SpanAttributes.TRACELOOP_ENTITY_NAME: name,
                SpanAttributes.TRACELOOP_ENTITY_PATH: "",
            },
        )
        state = _State(span, False, parent=parent, masking=self.masking)
        self.states[id(instance)] = state
        request = _REQUEST.get()
        if request is not None:
            span.set_attribute(SpanAttributes.GEN_AI_IS_STREAMING, request)
        state.capture = (
            span.is_recording()
            and self.content
            and _allowed()
            and (parent.capture if parent else cached[1] if cached else True)
        )

    def bind(self, instance):
        state = self.states.get(id(instance))
        identifier = get(instance, "span_id")
        if isinstance(identifier, str):
            self.source[identifier] = state
            if state:
                state.source_id = identifier
                state.span.set_attribute("braintrust.span_id", identifier)

    def veto(self, state):
        if state is None or state.finished:
            return
        ancestor = state.parent
        denied = False
        while ancestor is not None:
            if not ancestor.capture:
                denied = True
                break
            ancestor = ancestor.parent
        sink = state.native_sink
        masked = sink is not None and bool(
            get(sink, "_export_customizers") or get(sink, "_masking_function")
        )
        if state.capture and (denied or masked or not _allowed()):
            state.capture = False
            state.rows.clear()
            state.embedding = None

    def enter(self, instance):
        state = self.states.get(id(instance))
        self.veto(state)
        if not state or state.finished or not get(instance, "can_set_current"):
            return
        if not _ADVANCE.get():
            frame = inspect.currentframe()
            try:
                while frame is not None:
                    if frame.f_globals.get(
                        "__name__"
                    ) == "braintrust.logger" and frame.f_code.co_flags & (
                        inspect.CO_GENERATOR | inspect.CO_ASYNC_GENERATOR
                    ):
                        # Generators decorated before activation cannot be wrapped
                        # retroactively. Leave their native caller context intact.
                        return
                    frame = frame.f_back
            finally:
                del frame
        current = context.get_current()
        # Braintrust's compatibility context must retain its native NonRecordingSpan
        # so current_span/traceparent APIs continue returning the native SDK span.
        if type(instance.state.context_manager).__module__.startswith(
            "braintrust.otel"
        ):
            state.token = context.attach(
                context.set_value(_CONTENT_BOUND, state.capture, current)
            )
        else:
            state.token = context.attach(
                context.set_value(
                    _CONTENT_BOUND,
                    state.capture,
                    trace.set_span_in_context(state.span, current),
                )
            )

    def leave(self, instance):
        state = self.states.get(id(instance))
        self.veto(state)
        if state and state.token is not None:
            token, state.token = state.token, None
            context.detach(token)
        if state and state.finished:
            self.states.pop(id(instance), None)
        if not self.active and not any(
            value and value.token is not None for value in self.states.values()
        ):
            self.restore_deferred()

    def record(self, state, record, sequence):
        if (
            not self.active
            or state.finished
            or not state.span.is_recording()
            or not isinstance(record, dict)
        ):
            return
        self.veto(state)
        # Source records are already copied/customized by Braintrust. Never resolve
        # extra lazy values or upload attachments to construct Respan payloads.
        keys = {
            "id",
            "span_id",
            "root_span_id",
            "span_parents",
            "span_attributes",
            "metrics",
            "error",
            "_is_merge",
            "_merge_paths",
        }
        if state.capture:
            keys |= {"input", "output", "metadata", "scores", "tags", "expected"}
        snapshot = {
            k: plain(v)
            for k, v in record.items()
            if k in keys and (state.capture or k not in {"span_attributes", "metrics"})
        }
        if not state.capture:
            snapshot.pop("error", None)
            snapshot["metrics"] = private_usage(record.get("metrics"))
            native = record.get("span_attributes")
            if isinstance(native, dict):
                snapshot["span_attributes"] = {
                    k: get(native, k)
                    for k in ("type", "name", "model", "provider")
                    if isinstance(get(native, k), str)
                }
            metadata = record.get("metadata")
            if isinstance(metadata, dict):
                safe = {
                    k: get(metadata, k)
                    for k in ("model", "model_name", "provider", "system")
                    if isinstance(get(metadata, k), str)
                }
                for key in ("usage", "token_usage"):
                    counters = private_usage(get(metadata, key))
                    if counters:
                        safe[key] = counters
                snapshot["metadata"] = safe
        state.rows.append((sequence, snapshot))

    def maybe_finish(self, state):
        if state.sdk_ended and state.pending == 0:
            self.finish(state)

    def finish(self, state, error=None):
        if state.finished:
            return
        state.finished = True
        try:
            if state.rows:
                from braintrust.merge_row_batch import merge_row_batch

                merged = merge_row_batch(
                    [row for _, row in sorted(state.rows, key=lambda value: value[0])]
                )
                record = (
                    merged[0][0]
                    if merged and isinstance(merged[0], list) and merged[0]
                    else merged[0]
                    if merged
                    else {}
                )
            else:
                record = {}
            if state.embedding is not None:
                record.setdefault("span_attributes", {})["type"] = "embedding"
                record.setdefault("metadata", {})["model"] = state.embedding.get(
                    "model"
                )
                record["metrics"] = state.embedding.get("usage") or record.get(
                    "metrics", {}
                )
                if (
                    state.capture
                    and isinstance(record.get("output"), dict)
                    and "embedding_length" in record["output"]
                ):
                    record["output"] = state.embedding.get("vectors")
            masker = state.masking or self.masking
            if (
                state.masking is not None
                and self.masking is not None
                and state.masking is not self.masking
            ):

                def masker(value):
                    return self.masking(state.masking(value))

            attrs = (
                attributes(record, capture=state.capture, masking=masker)
                if record
                else {}
            )
            state.span.set_attributes(attrs)
            actual = error or state.error
            if actual is not None or record.get("error"):
                state.span.set_status(trace.StatusCode.ERROR)
                event = {}
                if actual is not None:
                    event[EXCEPTION_TYPE] = type(actual).__name__
                    state.span.set_attribute(ERROR_TYPE, type(actual).__name__)
                    args = BaseException.args.__get__(actual) if state.capture else ()
                    message = (
                        redact_text(args[0])
                        if len(args) == 1 and isinstance(args[0], str)
                        else json_string(args)
                    )
                else:
                    message = record.get("error")
                if state.capture and isinstance(message, str):
                    event[EXCEPTION_MESSAGE] = redact_text(message)
                    state.span.set_attribute(ERROR_MESSAGE, event[EXCEPTION_MESSAGE])
                state.span.add_event("exception", event)
            if state.source_id:
                self.parents[state.source_id] = (
                    state.span.get_span_context(),
                    state.capture,
                )
                self.parents.move_to_end(state.source_id)
                while len(self.parents) > 4096:
                    self.parents.popitem(last=False)
        finally:
            state.rows.clear()
            state.embedding = None
            state.masking = None
            state.error = None
            state.span.end()
            if state.source_id:
                self.source.pop(state.source_id, None)
            for key, value in tuple(self.states.items()):
                if value is state and state.token is None:
                    self.states.pop(key, None)

    def install(self):
        runtime = self

        def constructor(original):
            signature = inspect.signature(original)

            @wraps(original)
            def wrapped(instance, *args, **kwargs):
                if not runtime.active:
                    return original(instance, *args, **kwargs)
                bound = signature.bind(instance, *args, **kwargs)
                _safe(runtime.begin, instance, bound.arguments)
                try:
                    result = original(instance, *args, **kwargs)
                except BaseException as error:
                    state = runtime.states.get(id(instance))
                    if state:
                        _safe(runtime.finish, state, error)
                    raise
                _safe(runtime.bind, instance)
                return result

            return wrapped

        self.patch(self.braintrust.logger.SpanImpl, "__init__", constructor)

        def sink(original):
            @wraps(original)
            def wrapped(instance, *args, **kwargs):
                native = original(instance, *args, **kwargs)
                return _ObservedLogger(native, runtime) if runtime.active else native

            return wrapped

        self.patch(self.braintrust.logger.BraintrustState, "global_bg_logger", sink)

        def writer(original):
            @wraps(original)
            def wrapped(instance, *args, **kwargs):
                if not runtime.active:
                    return original(instance, *args, **kwargs)
                state = runtime.states.get(id(instance))
                _safe(runtime.veto, state)
                token = _WRITE.set(state)
                try:
                    return original(instance, *args, **kwargs)
                finally:
                    _WRITE.reset(token)

            return wrapped

        self.patch(self.braintrust.logger.SpanImpl, "log_internal", writer)

        def current(original):
            @wraps(original)
            def wrapped(instance, *args, **kwargs):
                result = original(instance, *args, **kwargs)
                if runtime.active:
                    _safe(runtime.enter, instance)
                return result

            return wrapped

        self.patch(self.braintrust.logger.SpanImpl, "set_current", current)

        def uncurrent(original):
            @wraps(original)
            def wrapped(instance, *args, **kwargs):
                _safe(runtime.leave, instance)
                return original(instance, *args, **kwargs)

            return wrapped

        self.patch(self.braintrust.logger.SpanImpl, "unset_current", uncurrent)

        def ending(original):
            @wraps(original)
            def wrapped(instance, *args, **kwargs):
                state = runtime.states.get(id(instance))
                _safe(runtime.veto, state)
                try:
                    return original(instance, *args, **kwargs)
                finally:
                    if state:
                        state.sdk_ended = True
                        _safe(runtime.maybe_finish, state)
                    else:
                        runtime.states.pop(id(instance), None)

            return wrapped

        self.patch(self.braintrust.logger.SpanImpl, "end", ending)

        def exit_factory(original):
            @wraps(original)
            def wrapped(instance, exc_type, exc_value, tb):
                state = runtime.states.get(id(instance))
                if state:
                    state.error = exc_value
                    _safe(runtime.veto, state)
                return original(instance, exc_type, exc_value, tb)

            return wrapped

        self.patch(self.braintrust.logger.SpanImpl, "__exit__", exit_factory)

        def traced_factory(original):
            @wraps(original)
            def wrapped(*args, **kwargs):
                result = original(*args, **kwargs)

                def decorate(function):
                    if inspect.isgeneratorfunction(function):

                        @wraps(function)
                        def wrapper(*a, **kw):
                            return (
                                yield from (
                                    _Iterator(function(*a, **kw))
                                    if runtime.active
                                    else function(*a, **kw)
                                )
                            )

                        return wrapper
                    if inspect.isasyncgenfunction(function):

                        @wraps(function)
                        async def wrapper(*a, **kw):
                            iterator = (
                                _AsyncIterator(function(*a, **kw))
                                if runtime.active
                                else function(*a, **kw)
                            )
                            method, args = iterator.__anext__, ()
                            while True:
                                try:
                                    value = await method(*args)
                                except StopAsyncIteration:
                                    return
                                try:
                                    sent = yield value
                                except GeneratorExit:
                                    await iterator.aclose()
                                    raise
                                except BaseException as error:  # noqa: BLE001 - forward native async generator throw
                                    method, args = (
                                        iterator.athrow,
                                        (type(error), error, error.__traceback__),
                                    )
                                else:
                                    method, args = iterator.asend, (sent,)

                        return wrapper
                    return function

                if args and callable(args[0]) and len(args) == 1 and not kwargs:
                    return decorate(result)

                @wraps(result)
                def decorator(function):
                    return decorate(result(function))

                return decorator

            return wrapped

        # The released native OpenAI embedding wrapper logs only vector length.
        # Observe its real response while preserving its own record and return.
        def embedding_factory(original):
            @wraps(original)
            def wrapped(instance, response, span, *args, **kwargs):
                result = original(instance, response, span, *args, **kwargs)
                if runtime.active:
                    state = runtime.states.get(id(span))
                    if state and state.span.is_recording():

                        def capture():
                            runtime.veto(state)
                            state.embedding = {
                                "model": get(response, "model"),
                                "usage": private_usage(get(response, "usage")),
                            }
                            if state.capture:
                                state.embedding["vectors"] = [
                                    plain(get(item, "embedding"))
                                    for item in get(response, "data", ())
                                ]

                        _safe(capture)
                return result

            return wrapped

        import importlib

        for module_name in ("braintrust.integrations.openai.tracing", "braintrust.oai"):
            try:
                module = importlib.import_module(module_name)
            except ImportError:
                continue

            def request_factory(original):
                if inspect.iscoroutinefunction(original):

                    @wraps(original)
                    async def wrapped(instance, *args, **kwargs):
                        token = _REQUEST.set(
                            kwargs.get("stream")
                            if type(kwargs.get("stream")) is bool
                            else None
                        )
                        try:
                            return await original(instance, *args, **kwargs)
                        finally:
                            _REQUEST.reset(token)

                    return wrapped

                @wraps(original)
                def wrapped(instance, *args, **kwargs):
                    token = _REQUEST.set(
                        kwargs.get("stream")
                        if type(kwargs.get("stream")) is bool
                        else None
                    )
                    try:
                        return original(instance, *args, **kwargs)
                    finally:
                        _REQUEST.reset(token)

                return wrapped

            self.patch(module.ChatCompletionWrapper, "create", request_factory)
            self.patch(module.ChatCompletionWrapper, "acreate", request_factory)
            self.patch(module.EmbeddingWrapper, "process_output", embedding_factory)
            break
        self.patch(self.braintrust.logger, "traced", traced_factory)
        self.patch(self.braintrust, "traced", traced_factory)

    def close(self):
        self.active = False
        for state in tuple(self.states.values()):
            if state:
                state.capture = False
                state.rows.clear()
                _safe(self.finish, state)
        self.states = {
            key: state
            for key, state in self.states.items()
            if state and state.token is not None
        }
        self.source.clear()
        self.parents.clear()
        retained = []
        for owner, name, previous, replacement in reversed(self.patches):
            if name == "unset_current" and self.states:
                retained.append((owner, name, previous, replacement))
                continue
            if getattr(owner, name, None) is replacement:
                if previous is _MISSING:
                    delattr(owner, name)
                else:
                    setattr(owner, name, previous)
        self.patches = retained

    def restore_deferred(self):
        for owner, name, previous, replacement in self.patches:
            if getattr(owner, name, None) is replacement:
                if previous is _MISSING:
                    delattr(owner, name)
                else:
                    setattr(owner, name, previous)
        self.patches.clear()


class BraintrustInstrumentor:
    name = "braintrust"

    def __init__(self, *, tracer_provider=None, include_content=True):
        self._provider, self._content = tracer_provider, bool(include_content)
        self._runtime = None
        self._masking = None

    @property
    def is_instrumented(self):
        return self._runtime is not None

    def __enter__(self):
        self.activate()
        return self

    def __exit__(self, *args):
        self.deactivate()

    def activate(self):
        global _RUNTIME
        with _LOCK:
            if self._runtime:
                return
            tracer = getattr(RespanTracer, "_instance", None)
            if tracer is not None and not getattr(tracer, "is_enabled", True):
                return
            import braintrust

            provider = self._provider or trace.get_tracer_provider()
            if _RUNTIME:
                if (
                    _RUNTIME.provider is not provider
                    or _RUNTIME.content != self._content
                    or _RUNTIME.masking is not self._masking
                ):
                    raise ValueError(
                        "Braintrust instrumentation already has different provider/privacy settings"
                    )
                _RUNTIME.count += 1
            else:
                runtime = _Runtime(braintrust, provider, self._content, self._masking)
                try:
                    runtime.install()
                except BaseException:
                    runtime.close()
                    raise
                _RUNTIME = runtime
            self._runtime = _RUNTIME

    def deactivate(self):
        global _RUNTIME
        with _LOCK:
            if not self._runtime:
                return
            runtime, self._runtime = self._runtime, None
            runtime.count -= 1
            if not runtime.count:
                runtime.close()
                _RUNTIME = None

    def set_masking_function(self, masking_function):
        if (
            self._runtime
            and self._runtime.count > 1
            and self._runtime.masking is not masking_function
        ):
            raise ValueError(
                "Shared Braintrust owners must use the same masking function"
            )
        self._masking = masking_function
        if self._runtime:
            self._runtime.masking = masking_function

    def enforce_queue_size_limit(self, enforce):
        if self._runtime:
            self._runtime.braintrust._internal_get_global_state().global_bg_logger().enforce_queue_size_limit(
                enforce
            )

    def log(self, *items):
        # Retain the old logger-facing interface without replacing Braintrust's sink.
        if not self._runtime:
            return
        self._runtime.braintrust._internal_get_global_state().global_bg_logger().log(
            *items
        )

    def flush(self, batch_size=None):
        if self._runtime:
            self._runtime.braintrust._internal_get_global_state().global_bg_logger().flush(
                batch_size
            )


RespanBraintrustInstrumentor = BraintrustInstrumentor
