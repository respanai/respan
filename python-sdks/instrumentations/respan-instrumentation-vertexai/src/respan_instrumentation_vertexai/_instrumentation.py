"""Observe released native Vertex calls and streams with real OTel sampling."""

# ruff: noqa: BLE001 -- telemetry failures never log native payloads.
from __future__ import annotations

import contextvars
import functools
import importlib
import importlib.metadata
import inspect
import threading
import types
import weakref
from dataclasses import dataclass

from opentelemetry import context, trace
from opentelemetry.sdk.util import BoundedList
from opentelemetry.semconv._incubating.attributes.error_attributes import ERROR_MESSAGE
from opentelemetry.semconv.attributes.error_attributes import ERROR_TYPE
from opentelemetry.semconv.attributes.http_attributes import HTTP_RESPONSE_STATUS_CODE
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.trace import Status, StatusCode
from respan_sdk.constants.span_attributes import (
    RESPAN_LOG_TYPE,
    RESPAN_METADATA,
    RESPAN_PROMPT,
    RESPAN_SPAN_ATTRIBUTES_MAP,
)
from respan_tracing.core.tracer import RespanTracer
from respan_tracing.utils.span_factory import _PROPAGATED_ATTRIBUTES

from respan_instrumentation_vertexai._otel_emitter import (
    base_attributes,
    request_attributes,
    response_attributes,
)
from respan_instrumentation_vertexai._policy import AncestorPolicy, suppressed
from respan_instrumentation_vertexai._serialization import (
    REDACTED,
    json_dumps,
    provider_status_code,
    safe_exception_message,
    safe_text,
    safe_type_name,
    sensitive_key,
    to_jsonable,
)
from respan_instrumentation_vertexai._translator import request_payload_from_call

_LOCK = threading.RLock()
_ACTIVATION_COUNT = 0
_ENABLED = False
_CAPTURE_CONTENT = True
_PROVIDER = None
_POLICIES = weakref.WeakKeyDictionary()
_PATCHES = []


def _provider():
    return _PROVIDER or trace.get_tracer_provider()


def _policy():
    provider = _provider()
    with _LOCK:
        policy = _POLICIES.get(provider)
        if policy is None:
            if not callable(getattr(provider, "add_span_processor", None)):
                return None
            policy = AncestorPolicy(_CAPTURE_CONTENT)
            provider.add_span_processor(policy)
            _POLICIES[provider] = policy
        policy.setting = _CAPTURE_CONTENT
        policy.enabled = True
        return policy


def _attempt(fn, default=None):
    try:
        return fn()
    except BaseException:
        return default


def _propagated_attributes():
    # The released bridge calls str() on metadata. Read the same canonical
    # ContextVar without invoking arbitrary customer formatting hooks.
    values = _PROPAGATED_ATTRIBUTES.get()
    result = {}
    if type(values) is not dict:
        return result
    for key, value in values.items():
        if type(key) is not str or key not in RESPAN_SPAN_ATTRIBUTES_MAP:
            continue
        target = RESPAN_SPAN_ATTRIBUTES_MAP[key]
        if target == RESPAN_METADATA and type(value) is dict:
            for name, item in value.items():
                if type(name) is str:
                    cleaned = REDACTED if sensitive_key(name) else to_jsonable(item)
                    result[f"{target}.{safe_text(name)}"] = (
                        cleaned if type(cleaned) is str else json_dumps(cleaned)
                    )
        elif target == RESPAN_PROMPT:
            result[target] = json_dumps(value)
        elif type(value) in {str, bool, int, float}:
            result[target] = to_jsonable(value)
    return result


def _restore_context(ambient):
    if context.get_current() is ambient:
        return
    _attempt(lambda: context._RUNTIME_CONTEXT.attach(ambient))
    if context.get_current() is not ambient:
        state = object.__getattribute__(context._RUNTIME_CONTEXT, "__dict__")
        for candidate in state.values():
            if type(candidate) is contextvars.ContextVar:
                candidate.set(ambient)
                break


class _Call:
    def __init__(self, instance, args, kwargs, *, name, embedding):
        self.span = None
        self.policy = None
        self.finished = False
        self.failed = False
        self.embedding = embedding
        self.chunks = []
        self.content = set()
        self.token = None
        self.policy = _policy()
        self.span = (
            _provider()
            .get_tracer(
                "vertexai",
                importlib.metadata.version("respan-instrumentation-vertexai"),
            )
            .start_span(
                name, kind=trace.SpanKind.CLIENT, attributes=base_attributes(embedding)
            )
        )
        if not self.recording():
            return
        if self.allowed():
            payload = request_payload_from_call(
                instance=instance, args=args, kwargs=kwargs, embedding=embedding
            )
            self.set_attributes(request_attributes(payload))
            self.set_attributes(_propagated_attributes())
        self.span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_NAME, name)
        self.span.set_attribute(
            SpanAttributes.TRACELOOP_ENTITY_PATH,
            "" if not trace.get_current_span().get_span_context().is_valid else name,
        )

    def recording(self):
        return self.span is not None and bool(_attempt(self.span.is_recording, False))

    def scrub(self):
        self.chunks.clear()
        attributes = getattr(self.span, "_attributes", None)
        if attributes is not None:
            structural = {
                RESPAN_LOG_TYPE,
                SpanAttributes.LLM_SYSTEM,
                SpanAttributes.LLM_REQUEST_TYPE,
                SpanAttributes.TRACELOOP_ENTITY_NAME,
                SpanAttributes.TRACELOOP_ENTITY_PATH,
                ERROR_TYPE,
                HTTP_RESPONSE_STATUS_CODE,
            }
            for key in list(attributes):
                if key not in structural:
                    _attempt(lambda key=key: attributes.pop(key, None))
        if getattr(self.span, "_events", None) is not None:
            self.span._events = BoundedList(0)
        status = getattr(self.span, "status", None)
        if status is not None:
            self.span._status = Status(status.status_code)

    def allowed(self):
        allowed = (
            not self.failed
            and self.policy is not None
            and self.policy.observe(self.span)
        )
        if not allowed:
            self.scrub()
        return allowed

    def set_attributes(self, values):
        for key, value in values.items():
            self.content.add(key)
            self.span.set_attribute(key, value)

    def capture(self, response):
        if self.recording() and self.allowed():
            self.chunks.append(response)

    def output(self, response):
        if self.recording() and self.allowed():
            self.set_attributes(response_attributes(response, embedding=self.embedding))

    def error(self, exc):
        if self.recording():
            allowed = self.allowed()
            message = safe_exception_message(exc) if allowed else None
            self.span.set_status(Status(StatusCode.ERROR, message))
            self.span.set_attribute(ERROR_TYPE, safe_type_name(exc))
            # Only native exceptions carry HTTP-like codes. Never infer 500/200.
            code = provider_status_code(exc)
            if code is not None:
                self.span.set_attribute(HTTP_RESPONSE_STATUS_CODE, code)
            if message is not None:
                self.content.add(ERROR_MESSAGE)
                self.span.set_attribute(ERROR_MESSAGE, message)

    def attach(self):
        ambient = context.get_current()
        try:
            token = (
                context.attach(trace.set_span_in_context(self.span))
                if self.recording()
                else None
            )
        except BaseException:
            _restore_context(ambient)
            raise
        return token, ambient

    def detach(self, token):
        if token is None:
            return
        token, ambient = token
        if not self.finished:
            _attempt(self.allowed)
        if token is not None:
            try:
                context.detach(token)
            except BaseException:
                _attempt(lambda: context._RUNTIME_CONTEXT.detach(token))
        _attempt(lambda: _restore_context(ambient))

    def finish(self, error=None):
        if self.finished:
            return
        self.finished = True
        if self.recording():
            if error is not None:
                _attempt(lambda: self.error(error))
            else:
                _attempt(lambda: self.span.set_status(Status(StatusCode.OK)))
            if not _attempt(self.allowed, False):
                _attempt(self.scrub)
            ambient = context.get_current()
            try:
                _attempt(self.span.end)
            finally:
                _attempt(lambda: _restore_context(ambient))
        self.chunks.clear()
        self.content.clear()
        self.policy = None


def _observe(call, fn):
    try:
        return fn()
    except BaseException:
        call.failed = True
        _attempt(call.scrub)
        return None


class _Iterator:
    def __init__(self, source, call):
        self.source = source
        self.call = call

    def __iter__(self):
        return self

    def __next__(self):
        return self._step(lambda: next(self.source))

    def send(self, value):
        return self._step(lambda: self.source.send(value))

    def throw(self, *args):
        return self._step(lambda: self.source.throw(*args))

    def _step(self, fn):
        token = _observe(self.call, self.call.attach)
        try:
            chunk = fn()
        except StopIteration:
            _observe(self.call, lambda: self.call.output(self.call.chunks))
            self.call.finish()
            raise
        except BaseException as exc:
            _observe(self.call, lambda: self.call.output(self.call.chunks))
            self.call.finish(exc)
            raise
        else:
            _observe(self.call, lambda: self.call.capture(chunk))
            return chunk
        finally:
            self.call.detach(token)

    def close(self):
        try:
            result = self.source.close()
        except BaseException as exc:
            self.call.finish(exc)
            raise
        else:
            _observe(self.call, lambda: self.call.output(self.call.chunks))
            self.call.finish()
            return result

    def __getattr__(self, name):
        return getattr(self.source, name)


class _AsyncIterator:
    def __init__(self, source, call):
        self.source = source
        self.call = call

    def __aiter__(self):
        return self

    async def __anext__(self):
        return await self._step(self.source.__anext__)

    async def asend(self, value):
        return await self._step(lambda: self.source.asend(value))

    async def athrow(self, *args):
        return await self._step(lambda: self.source.athrow(*args))

    async def _step(self, fn):
        token = _observe(self.call, self.call.attach)
        try:
            chunk = await fn()
        except StopAsyncIteration:
            _observe(self.call, lambda: self.call.output(self.call.chunks))
            self.call.finish()
            raise
        except BaseException as exc:
            _observe(self.call, lambda: self.call.output(self.call.chunks))
            self.call.finish(exc)
            raise
        else:
            _observe(self.call, lambda: self.call.capture(chunk))
            return chunk
        finally:
            self.call.detach(token)

    async def aclose(self):
        try:
            result = await self.source.aclose()
        except BaseException as exc:
            self.call.finish(exc)
            raise
        else:
            _observe(self.call, lambda: self.call.output(self.call.chunks))
            self.call.finish()
            return result

    def __getattr__(self, name):
        return getattr(self.source, name)


def _start(instance, args, kwargs, *, name, embedding):
    if (
        not _ENABLED
        or suppressed()
        or not VertexAIInstrumentor._is_respan_tracing_enabled()
    ):
        return None
    ambient = context.get_current()
    call = None
    try:
        call = _Call.__new__(_Call)
        call.__init__(instance, args, kwargs, name=name, embedding=embedding)
        return call
    except BaseException:
        if call is not None and getattr(call, "span", None) is not None:
            call.failed = True
            _attempt(call.scrub)
            call.finished = True
            _attempt(call.span.end)
            call.chunks.clear()
            call.policy = None
        return None
    finally:
        _attempt(lambda: _restore_context(ambient))


def _wrap(original, name, embedding, asynchronous):
    if asynchronous:

        @functools.wraps(original)
        async def async_call(instance, *args, **kwargs):
            call = _start(instance, args, kwargs, name=name, embedding=embedding)
            token = _observe(call, call.attach) if call is not None else None
            try:
                response = await original(instance, *args, **kwargs)
            except BaseException as exc:
                if call is not None:
                    call.finish(exc)
                raise
            finally:
                if call is not None:
                    call.detach(token)
            if call is None:
                return response
            if call.recording() and type(response) is types.AsyncGeneratorType:
                return _AsyncIterator(response, call)
            _observe(call, lambda: call.output(response))
            call.finish()
            return response

        return async_call

    @functools.wraps(original)
    def sync_call(instance, *args, **kwargs):
        call = _start(instance, args, kwargs, name=name, embedding=embedding)
        token = _observe(call, call.attach) if call is not None else None
        try:
            response = original(instance, *args, **kwargs)
        except BaseException as exc:
            if call is not None:
                call.finish(exc)
            raise
        finally:
            if call is not None:
                call.detach(token)
        if call is None:
            return response
        if call.recording() and type(response) is types.GeneratorType:
            return _Iterator(response, call)
        _observe(call, lambda: call.output(response))
        call.finish()
        return response

    return sync_call


@dataclass
class _Patch:
    cls: object
    method_name: str
    original: object
    wrapper: object


def _load_targets():
    module = importlib.import_module("vertexai.generative_models")
    language = importlib.import_module("vertexai.language_models")
    return [
        (
            module.GenerativeModel,
            "generate_content",
            "vertexai.generate_content",
            False,
            False,
        ),
        (
            module.GenerativeModel,
            "generate_content_async",
            "vertexai.generate_content",
            False,
            True,
        ),
        (
            module.ChatSession,
            "send_message",
            "vertexai.chat.send_message",
            False,
            False,
        ),
        (
            module.ChatSession,
            "send_message_async",
            "vertexai.chat.send_message",
            False,
            True,
        ),
        (
            language.TextEmbeddingModel,
            "get_embeddings",
            "vertexai.embedding",
            True,
            False,
        ),
        (
            language.TextEmbeddingModel,
            "get_embeddings_async",
            "vertexai.embedding",
            True,
            True,
        ),
    ]


def _restore():
    for patch in reversed(_PATCHES):
        if inspect.getattr_static(patch.cls, patch.method_name, None) is patch.wrapper:
            setattr(patch.cls, patch.method_name, patch.original)
    _PATCHES.clear()


def _remove_policies():
    for provider, policy in list(_POLICIES.items()):
        policy.enabled = False
        policy.clear()
        processor = getattr(provider, "_active_span_processor", None)
        processors = getattr(processor, "_span_processors", ())
        if processor is not None:
            processor._span_processors = tuple(
                item for item in processors if item is not policy
            )
    _POLICIES.clear()


class VertexAIInstrumentor:
    name = "vertexai"

    def __init__(self, *, capture_content=True, tracer_provider=None):
        self._capture_content = capture_content
        self._provider = tracer_provider
        self._is_instrumented = False

    @staticmethod
    def _is_respan_tracing_enabled():
        tracer = getattr(RespanTracer, "_instance", None)
        return tracer is None or bool(getattr(tracer, "is_enabled", True))

    def activate(self):
        global _ACTIVATION_COUNT, _CAPTURE_CONTENT, _PROVIDER, _ENABLED
        with _LOCK:
            if self._is_instrumented or not self._is_respan_tracing_enabled():
                return
            if _ACTIVATION_COUNT:
                if (
                    self._capture_content != _CAPTURE_CONTENT
                    or self._provider is not _PROVIDER
                ):
                    raise ValueError(
                        "Vertex capture_content/tracer_provider must match the active instrumentor"
                    )
            else:
                try:
                    targets = _load_targets()
                except ImportError:
                    return
                _CAPTURE_CONTENT = self._capture_content
                _PROVIDER = self._provider
                try:
                    for cls, method, name, embedding, asynchronous in targets:
                        original = inspect.getattr_static(cls, method)
                        wrapped = _wrap(original, name, embedding, asynchronous)
                        setattr(cls, method, wrapped)
                        _PATCHES.append(_Patch(cls, method, original, wrapped))
                    _policy()
                except BaseException:
                    _restore()
                    _remove_policies()
                    raise
                _ENABLED = True
            _ACTIVATION_COUNT += 1
            self._is_instrumented = True

    def deactivate(self):
        global _ACTIVATION_COUNT, _ENABLED
        with _LOCK:
            if not self._is_instrumented:
                return
            self._is_instrumented = False
            _ACTIVATION_COUNT = max(0, _ACTIVATION_COUNT - 1)
            if _ACTIVATION_COUNT == 0:
                _ENABLED = False
                _restore()
                _remove_policies()
