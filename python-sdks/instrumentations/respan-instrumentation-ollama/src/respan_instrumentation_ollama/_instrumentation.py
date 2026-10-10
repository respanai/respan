"""Observe official Ollama HTTP/NDJSON calls without changing native outcomes."""

# ruff: noqa: BLE001 -- telemetry failures preserve native results and cleanup.
from __future__ import annotations

import builtins
import functools
import importlib
import json
import logging
import threading
import weakref
from contextvars import ContextVar

from opentelemetry import context, trace
from opentelemetry.sdk.util import BoundedList
from opentelemetry.semconv.attributes.error_attributes import ERROR_TYPE
from opentelemetry.semconv.attributes.http_attributes import HTTP_RESPONSE_STATUS_CODE
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.trace import SpanKind, Status, StatusCode
from respan_sdk.constants.otlp_constants import ERROR_MESSAGE_ATTR
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
_ACTIVE = ContextVar("respan_ollama_call", default=None)


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
    except Exception:
        logger.debug("Ollama context restoration failed")


def _detach(token, ambient):
    try:
        context.detach(token)
    except Exception:
        logger.debug("Ollama context detach failed")
    _restore_context(ambient)


def _model_only(request):
    if type(request) is dict and type(request.get("model")) is str:
        return {"model": text(request["model"])}
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
            elif type(item) in (str, bool, int, float):
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
                "ollama." + mode, context=self.ctx, kind=SpanKind.CLIENT
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
                self.request = value(request)
                self.propagated = _propagated()
        except Exception:
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
        setattr(owner, name, owned)
        self.undo.append((weakref.ref(owner), name, original, owned, had_own))

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
                except Exception:
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
            if self.policy() and type(line) is str:
                decoded = json.loads(line)
                if type(decoded) is dict and not decoded.get("error"):
                    self.pending = decoded
        except Exception:
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
        except Exception:
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
                    import ollama._types as types

                    known = any(
                        cls is candidate
                        for module in (builtins, httpx, types)
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
                        attrs[ERROR_MESSAGE_ATTR] = message
                    self.span.set_status(Status(StatusCode.ERROR, message))
                self.span.set_attributes(attrs)
        except Exception:
            logger.debug("Ollama telemetry mapping failed")
        finally:
            try:
                if self.recording and not self.policy():
                    self.scrub()
            except Exception:
                if self.recording:
                    self.scrub()
            # Mark done before SDK processors enter transient export suppression.
            self.done = True
            try:
                self.span.end()
            except Exception:
                logger.debug("Ollama telemetry end failed")
            _restore_context(ambient)
            for ref, name, original, owned, had_own in self.undo:
                owner = ref()
                if owner is not None and getattr(owner, name, None) is owned:
                    try:
                        if had_own:
                            setattr(owner, name, original)
                        else:
                            delattr(owner, name)
                    except Exception:
                        logger.debug("Ollama response tap restoration failed")
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
        except Exception:
            logger.debug("Ollama response observation failed")


async def _async_hook(response):
    _hook(response)


def _transport(client):
    import httpx

    owner = client._client
    if type(owner) not in (httpx.Client, httpx.AsyncClient):
        return
    with _LOCK:
        for ref, hook in _HOOKS:
            if ref() is owner:
                return
        hook = _async_hook if type(owner) is httpx.AsyncClient else _hook
        owner.event_hooks["response"].append(hook)
        _HOOKS.append((weakref.ref(owner), hook))


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
    except Exception:
        if state is not None and getattr(state, "span", None) is not None:
            try:
                state.span.end()
            except Exception:
                logger.debug("Ollama startup span end failed")
            _PENDING.discard(state)
        logger.debug("Ollama telemetry startup failed")
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
        except Exception:
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
        except Exception:
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

    def __getattr__(self, name):
        return getattr(self.native, name)


def _mode(cls):
    import ollama

    return {
        ollama.ChatResponse: "chat",
        ollama.GenerateResponse: "generate",
        ollama.EmbedResponse: "embed",
        ollama.EmbeddingsResponse: "embeddings",
    }.get(cls)


def _wrap(original, asynchronous=False):
    if asynchronous:

        @functools.wraps(original)
        async def call(client, cls, *args, stream=False, **kwargs):
            mode = _mode(cls)
            if _CONFIG is None or mode is None or suppressed() or not _enabled():
                return await original(client, cls, *args, stream=stream, **kwargs)
            state = _start(mode, kwargs.get("json", {}), stream, client)
            if state is None:
                return await original(client, cls, *args, stream=stream, **kwargs)
            response = await _arun(
                state, lambda: original(client, cls, *args, stream=stream, **kwargs)
            )
            if stream:
                return _AsyncStream(response, state)
            state.finish(response=response)
            return response

        return call

    @functools.wraps(original)
    def call(client, cls, *args, stream=False, **kwargs):
        mode = _mode(cls)
        if _CONFIG is None or mode is None or suppressed() or not _enabled():
            return original(client, cls, *args, stream=stream, **kwargs)
        state = _start(mode, kwargs.get("json", {}), stream, client)
        if state is None:
            return original(client, cls, *args, stream=stream, **kwargs)
        response = _run(
            state, lambda: original(client, cls, *args, stream=stream, **kwargs)
        )
        if stream:
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
                except Exception:
                    logger.debug("Ollama content policy observation failed")
        return original(token)

    return detach


class OllamaInstrumentor:
    """Instrument the four inference methods of the official Ollama SDK."""

    name = "ollama"

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
                    "Ollama is active with a different content/provider configuration"
                )
            if not _OWNERS:
                try:
                    sdk = importlib.import_module("ollama")
                    _observer(_provider(self.config))
                    for owner, name, factory in (
                        (sdk.Client, "_request", lambda fn: _wrap(fn)),
                        (sdk.AsyncClient, "_request", lambda fn: _wrap(fn, True)),
                        (context, "detach", _detach_guard),
                    ):
                        original = getattr(owner, name)
                        owned = factory(original)
                        setattr(owner, name, owned)
                        _PATCHES.append((owner, name, original, owned))
                    _CONFIG = self.config
                except ImportError:
                    _cleanup()
                    return
                except Exception:
                    _cleanup()
                    logger.debug("Ollama activation failed")
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
    for ref, hook in _HOOKS:
        owner = ref()
        if owner is not None:
            owner.event_hooks["response"][:] = [
                item for item in owner.event_hooks["response"] if item is not hook
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
