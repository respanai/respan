"""Observe official Watsonx HTTP/NDJSON calls without changing native outcomes."""

# ruff: noqa: BLE001 -- telemetry failures preserve native results and cleanup.
from __future__ import annotations

import builtins
import functools
import importlib
import inspect
import json
import logging
import threading
import types
import weakref
from contextvars import ContextVar

from opentelemetry import context, trace
from opentelemetry.sdk.util import BoundedList
from opentelemetry.semconv._incubating.attributes.error_attributes import ERROR_MESSAGE
from opentelemetry.semconv.attributes.error_attributes import ERROR_TYPE
from opentelemetry.semconv.attributes.http_attributes import HTTP_RESPONSE_STATUS_CODE
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.trace import SpanKind, Status, StatusCode
from respan_sdk.constants.span_attributes import (
    RESPAN_METADATA,
    RESPAN_SPAN_ATTRIBUTES_MAP,
)
from respan_tracing.core.tracer import RespanTracer
from respan_tracing.utils.span_factory import _PROPAGATED_ATTRIBUTES

from ._otel_emitter import base_attributes, build_attributes
from ._privacy import (
    _STARTING,
    PolicyObserver,
    content_allowed,
    json_text,
    suppressed,
    text,
    value,
)
from ._translator import native_value

logger = logging.getLogger(__name__)
_LOCK = threading.RLock()
_OWNERS = set()
_CONFIG = None
_PATCHES = []
_OBSERVERS = []
_HOOKS = []
_PENDING = weakref.WeakSet()
_ACTIVE = ContextVar("respan_watsonx_call", default=None)


def _provider(config):
    return config[1] if config[1] is not None else RespanTracer().tracer_provider


def _observer(provider):
    with _LOCK:
        for existing, observer in _OBSERVERS:
            if existing is provider:
                return observer
        observer = PolicyObserver()
        _OBSERVERS.append((provider, observer))
        provider.add_span_processor(observer)
        # Observe a partially started span before a foreign processor can fail.
        processors = provider._active_span_processor
        processors._span_processors = (observer,) + tuple(
            item for item in processors._span_processors if item is not observer
        )
        return observer


def _restore_context(ambient):
    try:
        if context.get_current() is ambient:
            return
        runtime = context._RUNTIME_CONTEXT
        current = getattr(runtime, "_current_context", None)
        if current is not None:
            current.set(ambient)
        else:
            runtime.attach(ambient)
    except BaseException:
        logger.debug("Watsonx context restoration failed")


def _detach(token, ambient):
    try:
        context.detach(token)
    except BaseException:
        logger.debug("Watsonx context detach failed")
    _restore_context(ambient)


def _model_only(request):
    if type(request) is dict and type(request.get("model_id")) is str:
        return {"model_id": text(request["model_id"])}
    return {}


def _propagated():
    source = _PROPAGATED_ATTRIBUTES.get()
    result = {}
    if type(source) is not dict:
        return result
    for key, item in source.items():
        if type(key) is str and key in RESPAN_SPAN_ATTRIBUTES_MAP:
            target = RESPAN_SPAN_ATTRIBUTES_MAP[key]
            if target == RESPAN_METADATA:
                if type(item) is dict:
                    result[target] = json_text(item)
            elif any(type(item) is kind for kind in (str, bool, int, float)):
                result[target] = value(item)
    return result


class _Call:
    def __init__(self, mode, request, stream, config):
        self.ctx = context.get_current()
        self.carrier = trace.get_current_span(self.ctx)
        provider = _provider(config)
        self.observer = _observer(provider)
        self.mode, self.stream = mode, stream
        self.span = None
        starting = _STARTING.set(self)
        try:
            self.span = trace.get_tracer(__name__, tracer_provider=provider).start_span(
                "watsonx." + mode, context=self.ctx, kind=SpanKind.CLIENT
            )
        finally:
            _STARTING.reset(starting)
        self.recording = self.span.is_recording()
        self.allowed = bool(
            config[0]
            and self.recording
            and content_allowed(self.ctx)
            and self.observer.allowed(self.carrier)
        )
        request = request() if self.recording else {}
        self.request = _model_only(request) if self.recording else {}
        self.propagated = {}
        self.payload = None
        self.frames = []
        self.pending = None
        self.http_code = None
        self.undo = []
        self.done = False
        _PENDING.add(self)
        try:
            if self.allowed:
                self.request = native_value(request)
                self.propagated = _propagated()
        except BaseException:
            self.allowed = False
            self.request = _model_only(request) if self.recording else {}

    def policy(self):
        if self.done:
            return False
        self.allowed = bool(
            self.allowed
            and content_allowed(self.ctx)
            and content_allowed()
            and not suppressed(self.ctx)
            and self.observer.allowed(self.carrier)
            and self.observer.allowed(trace.get_current_span())
        )
        if not self.allowed:
            self.observer.deny(self.span)
            self.request = _model_only(self.request) if self.recording else {}
            self.propagated.clear()
            self.payload = self.pending = None
            self.frames.clear()
        return self.allowed

    def explicit_policy(self):
        # OTel SimpleSpanProcessor suppresses instrumentation while exporting.
        # That transient guard must not veto unrelated pending siblings.
        if not content_allowed(self.ctx) or not content_allowed():
            self.allowed = False
            self.policy()

    def tap(self, owner, name, replacement):
        original = getattr(owner, name)
        owned = replacement(original)
        had_own = name in owner.__dict__
        self.undo.append((weakref.ref(owner), name, original, owned, had_own))
        setattr(owner, name, owned)

    def response(self, response):
        import httpx

        if type(response) is not httpx.Response or self.done:
            return
        self.http_code = response.status_code
        if not self.recording:
            return

        def json_tap(original):
            def decoded(*args, **kwargs):
                result = original(*args, **kwargs)
                try:
                    if self.policy() and type(result) is dict:
                        self.payload = value(result)
                except BaseException:
                    self.allowed = False
                    self.policy()
                return result

            return decoded

        def lines_tap(original):
            def lines(*args, **kwargs):
                for line in original(*args, **kwargs):
                    self.line(line)
                    yield line

            return lines

        def async_lines_tap(original):
            async def lines(*args, **kwargs):
                async for line in original(*args, **kwargs):
                    self.line(line)
                    yield line

            return lines

        self.tap(response, "json", json_tap)
        if self.stream:
            self.tap(response, "iter_lines", lines_tap)
            self.tap(response, "aiter_lines", async_lines_tap)

    def line(self, line):
        try:
            self.pending = None
            if self.policy() and type(line) is str and line.startswith("data:"):
                decoded = json.loads(line.partition(":")[2])
                if type(decoded) is dict and not decoded.get("error"):
                    self.pending = decoded
        except BaseException:
            # Native JSON parsing still runs and raises its original error.
            self.pending = None

    def chunk(self, chunk):
        if self.done:
            return
        try:
            if self.policy():
                self.frames.append(
                    self.pending if self.pending is not None else native_value(chunk)
                )
            self.pending = None
        except BaseException:
            self.allowed = False
            self.policy()

    def finish(self, response=None, error=None):
        if self.done:
            return
        ambient = context.get_current()
        try:
            allowed = self.policy()
            if self.recording:
                payload = (self.frames or None) if self.stream else self.payload
                if (
                    not self.stream
                    and allowed
                    and (
                        type(response) is dict
                        or (
                            type(response) is list
                            and all(type(item) is dict for item in response)
                        )
                    )
                ):
                    payload = native_value(response)
                if payload is None and response is not None and allowed:
                    payload = native_value(response)
                attrs = build_attributes(
                    mode=self.mode,
                    request=self.request,
                    payload=payload,
                    stream=self.stream,
                    capture_content=allowed,
                )
                if allowed:
                    attrs.update(self.propagated)
                if self.http_code is not None:
                    attrs[HTTP_RESPONSE_STATUS_CODE] = self.http_code
                if error is not None:
                    cls = type(error)
                    attrs[ERROR_TYPE] = type.__getattribute__(cls, "__name__")
                    # Exact installed exceptions only; subclasses may override hooks.
                    import httpx
                    import ibm_watsonx_ai.wml_client_error as errors

                    known = any(
                        cls is candidate
                        for module in (builtins, httpx, errors)
                        for candidate in vars(module).values()
                        if type(candidate) is type
                        and issubclass(candidate, BaseException)
                    )
                    args = BaseException.args.__get__(error)
                    message = (
                        text(args[0])
                        if allowed
                        and known
                        and type(args) is tuple
                        and args
                        and type(args[0]) is str
                        else None
                    )
                    if message is not None:
                        attrs[ERROR_MESSAGE] = message
                    self.span.set_status(Status(StatusCode.ERROR, message))
                self.span.set_attributes(attrs)
        except BaseException:
            logger.debug("Watsonx telemetry mapping failed")
        finally:
            try:
                if self.recording and not self.policy():
                    self.scrub()
            except BaseException:
                if self.recording:
                    self.scrub()
            # Mark done before SDK processors enter transient export suppression.
            self.done = True
            try:
                self.span.end()
            except BaseException:
                logger.debug("Watsonx telemetry end failed")
            _restore_context(ambient)
            for ref, name, original, owned, had_own in self.undo:
                owner = ref()
                if owner is not None and getattr(owner, name, None) is owned:
                    try:
                        if had_own:
                            setattr(owner, name, original)
                        else:
                            delattr(owner, name)
                    except BaseException:
                        logger.debug("Watsonx response tap restoration failed")
            self.undo.clear()
            self.request = None
            self.payload = self.pending = self.ctx = self.carrier = None
            self.frames.clear()
            self.propagated.clear()
            _PENDING.discard(self)

    def scrub(self):
        structural = set(base_attributes(self.mode)) | {
            SpanAttributes.TRACELOOP_ENTITY_NAME,
            SpanAttributes.TRACELOOP_ENTITY_PATH,
            SpanAttributes.LLM_REQUEST_MODEL,
            SpanAttributes.LLM_IS_STREAMING,
            HTTP_RESPONSE_STATUS_CODE,
            ERROR_TYPE,
        }
        for key in tuple(self.span.attributes or {}):
            if key not in structural:
                self.span._attributes.pop(key, None)
        self.span._events = BoundedList(0)
        if self.span.status.status_code is StatusCode.ERROR:
            self.span._status = Status(StatusCode.ERROR)


def _hook(response):
    state = _ACTIVE.get()
    if state is not None:
        try:
            state.response(response)
        except BaseException:
            logger.debug("Watsonx response observation failed")


def _request_hook(request):
    state = _ACTIVE.get()
    if state is None or state.done or not state.recording:
        return
    try:
        import httpx

        if type(request) is httpx.Request and state.policy():
            decoded = json.loads(request.content)
            if type(decoded) is dict:
                native = value(decoded)
                state.request.setdefault("native_requests", []).append(native)
                state.request["native_request"] = native
                for key, item in native.items():
                    if (
                        key not in ("inputs", "input", "messages")
                        or key not in state.request
                    ):
                        state.request[key] = item
    except BaseException:
        logger.debug("Watsonx native request observation failed")


async def _async_hook(response):
    _hook(response)


async def _async_request_hook(request):
    _request_hook(request)


def _transport(instance):
    import httpx

    sdk = object.__getattribute__(instance, "__dict__").get("_client")
    state = object.__getattribute__(sdk, "__dict__")
    with _LOCK:
        for name in ("_httpx_client", "_async_httpx_client"):
            owner = state.get(name)
            if not any(
                type(owner) is kind for kind in (httpx.Client, httpx.AsyncClient)
            ):
                continue
            if any(ref() is owner for ref, _ in _HOOKS):
                continue
            hooks = {
                "request": _async_request_hook
                if type(owner) is httpx.AsyncClient
                else _request_hook,
                "response": _async_hook if type(owner) is httpx.AsyncClient else _hook,
            }
            _HOOKS.append((weakref.ref(owner), hooks))
            for kind, hook in hooks.items():
                owner.event_hooks[kind].append(hook)


def _enabled():
    instance = getattr(RespanTracer, "_instance", None)
    return instance is None or bool(getattr(instance, "is_enabled", True))


def _start(mode, request, stream, client):
    state = None
    ambient = context.get_current()
    try:
        state = _Call.__new__(_Call)
        state.__init__(mode, request, stream, _CONFIG)
        _transport(client)
        return state
    except BaseException:
        if state is not None and getattr(state, "span", None) is not None:
            try:
                state.scrub()
                state.span.end()
            except BaseException:
                logger.debug("Watsonx startup span end failed")
            _PENDING.discard(state)
        logger.debug("Watsonx telemetry startup failed")
        return None
    finally:
        _restore_context(ambient)


def _run(state, operation, finish_errors=True):
    if state.done:
        return operation()
    ambient = context.get_current()
    token = None
    active = _ACTIVE.set(state)
    try:
        try:
            token = context.attach(trace.set_span_in_context(state.span))
        except BaseException:
            _restore_context(ambient)
        try:
            return operation()
        except (StopIteration, StopAsyncIteration, GeneratorExit):
            raise
        except BaseException as error:
            if finish_errors:
                state.finish(error=error)
            raise
    finally:
        _ACTIVE.reset(active)
        if token is not None:
            _detach(token, ambient)
        else:
            _restore_context(ambient)


async def _arun(state, operation, finish_errors=True):
    if state.done:
        return await operation()
    ambient = context.get_current()
    token = None
    active = _ACTIVE.set(state)
    try:
        try:
            token = context.attach(trace.set_span_in_context(state.span))
        except BaseException:
            _restore_context(ambient)
        try:
            return await operation()
        except (StopIteration, StopAsyncIteration, GeneratorExit):
            raise
        except BaseException as error:
            if finish_errors:
                state.finish(error=error)
            raise
    finally:
        _ACTIVE.reset(active)
        if token is not None:
            _detach(token, ambient)
        else:
            _restore_context(ambient)


class _Stream:
    """Delegate every native generator operation, including pre-first close."""

    def __init__(self, native, state):
        self.native, self.state = native, state

    def __iter__(self):
        return self

    def __next__(self):
        return self._advance(lambda: next(self.native))

    def _advance(self, operation):
        try:
            result = _run(self.state, operation, False)
        except StopIteration:
            self.state.finish()
            raise
        except BaseException as error:
            if self.native.gi_frame is None:
                self.state.finish(error=error)
            raise
        else:
            self.state.chunk(result)
            return result

    def send(self, value):
        return self._advance(lambda: self.native.send(value))

    def throw(self, *args):
        return self._advance(lambda: self.native.throw(*args))

    def close(self):
        try:
            return _run(self.state, self.native.close)
        finally:
            self.state.finish()

    def __del__(self):
        try:
            self.close()
        except BaseException:
            logger.debug("Watsonx garbage collection cleanup failed")

    def __getattr__(self, name):
        return getattr(self.native, name)


class _AsyncStream:
    def __init__(self, native, state):
        self.native, self.state = native, state

    def __aiter__(self):
        return self

    async def __anext__(self):
        return await self._advance(self.native.__anext__)

    async def _advance(self, operation):
        try:
            result = await _arun(self.state, operation, False)
        except StopAsyncIteration:
            self.state.finish()
            raise
        except BaseException as error:
            if self.native.ag_frame is None:
                self.state.finish(error=error)
            raise
        else:
            self.state.chunk(result)
            return result

    async def asend(self, value):
        return await self._advance(lambda: self.native.asend(value))

    async def athrow(self, *args):
        return await self._advance(lambda: self.native.athrow(*args))

    async def aclose(self):
        try:
            return await _arun(self.state, self.native.aclose)
        finally:
            self.state.finish()

    def __del__(self):
        try:
            self.state.finish()
        except BaseException:
            logger.debug("Watsonx garbage collection cleanup failed")

    def __getattr__(self, name):
        return getattr(self.native, name)


def _snapshot(original, instance, args, kwargs):
    bound = inspect.signature(original).bind(instance, *args, **kwargs)
    bound.apply_defaults()
    request = {key: item for key, item in bound.arguments.items() if key != "self"}
    fields = object.__getattribute__(instance, "__dict__")
    model = fields.get("_model_id", fields.get("model_id"))
    if type(model) is str:
        request["model_id"] = model
    if request.get("params") is None and "params" in fields:
        request["params"] = fields["params"]
    return request


def _wrap(original, mode, method, asynchronous=False):
    if asynchronous:

        @functools.wraps(original)
        async def call(instance, *args, **kwargs):
            if (
                _CONFIG is None
                or _ACTIVE.get() is not None
                or suppressed()
                or not _enabled()
            ):
                return await original(instance, *args, **kwargs)
            state = _start(
                mode,
                lambda: _snapshot(original, instance, args, kwargs),
                "stream" in method,
                instance,
            )
            if state is None:
                return await original(instance, *args, **kwargs)
            response = await _arun(state, lambda: original(instance, *args, **kwargs))
            if type(response) is types.AsyncGeneratorType:
                state.stream = True
                return _AsyncStream(response, state)
            state.finish(response=response)
            return response

        return call

    @functools.wraps(original)
    def call(instance, *args, **kwargs):
        if (
            _CONFIG is None
            or _ACTIVE.get() is not None
            or suppressed()
            or not _enabled()
        ):
            return original(instance, *args, **kwargs)
        state = _start(
            mode,
            lambda: _snapshot(original, instance, args, kwargs),
            "stream" in method,
            instance,
        )
        if state is None:
            return original(instance, *args, **kwargs)
        response = _run(state, lambda: original(instance, *args, **kwargs))
        if type(response) is types.GeneratorType:
            state.stream = True
            return _Stream(response, state)
        state.finish(response=response)
        return response

    return call


def _detach_guard(original):
    @functools.wraps(original)
    def detach(token):
        for state in tuple(_PENDING):
            if not state.done:
                try:
                    state.explicit_policy()
                except BaseException:
                    logger.debug("Watsonx content policy observation failed")
        return original(token)

    return detach


class WatsonxInstrumentor:
    """Instrument the four inference methods of the official Watsonx SDK."""

    name = "watsonx"

    def __init__(self, *, capture_content=True, tracer_provider=None):
        self.config = (capture_content, tracer_provider)
        self._is_instrumented = False

    def activate(self):
        global _CONFIG
        with _LOCK:
            if self in _OWNERS:
                return
            if _OWNERS and self.config != _CONFIG:
                raise ValueError(
                    "Watsonx is active with a different content/provider configuration"
                )
            if not _OWNERS:
                try:
                    sdk = importlib.import_module("ibm_watsonx_ai.foundation_models")
                    _observer(_provider(self.config))
                    for owner, mode, names in (
                        (
                            sdk.ModelInference,
                            "generate",
                            (
                                "generate",
                                "generate_text",
                                "generate_text_stream",
                                "agenerate",
                                "agenerate_stream",
                            ),
                        ),
                        (
                            sdk.ModelInference,
                            "chat",
                            ("chat", "chat_stream", "achat", "achat_stream"),
                        ),
                        (
                            sdk.Embeddings,
                            "embed",
                            (
                                "generate",
                                "embed_documents",
                                "embed_query",
                                "agenerate",
                                "aembed_documents",
                                "aembed_query",
                            ),
                        ),
                    ):
                        for method in names:
                            original = inspect.getattr_static(owner, method, None)
                            if original is None:
                                continue
                            owned = _wrap(
                                original,
                                mode,
                                method,
                                inspect.iscoroutinefunction(original),
                            )
                            _PATCHES.append((owner, method, original, owned))
                            setattr(owner, method, owned)
                    original = context.detach
                    owned = _detach_guard(original)
                    _PATCHES.append((context, "detach", original, owned))
                    context.detach = owned
                    _CONFIG = self.config
                except ImportError:
                    _cleanup()
                    return
                except BaseException:
                    _cleanup()
                    logger.debug("Watsonx activation failed")
                    return
            _OWNERS.add(self)
            self._is_instrumented = True

    def deactivate(self):
        with _LOCK:
            _OWNERS.discard(self)
            self._is_instrumented = False
            if not _OWNERS:
                _cleanup()


def _cleanup():
    global _CONFIG
    for state in tuple(_PENDING):
        state.finish()
    for owner, name, original, owned in reversed(_PATCHES):
        if getattr(owner, name, None) is owned:
            setattr(owner, name, original)
    _PATCHES.clear()
    for ref, hooks in _HOOKS:
        owner = ref()
        if owner is not None:
            for kind, hook in hooks.items():
                owner.event_hooks[kind][:] = [
                    item for item in owner.event_hooks[kind] if item is not hook
                ]
    _HOOKS.clear()
    for provider, observer in _OBSERVERS:
        processor = getattr(provider, "_active_span_processor", None)
        processors = getattr(processor, "_span_processors", ())
        if any(item is observer for item in processors):
            processor._span_processors = tuple(
                item for item in processors if item is not observer
            )
    _OBSERVERS.clear()
    _CONFIG = None
