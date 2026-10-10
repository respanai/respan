"""Native finite Marqo boundaries with sampled, irreversible content policy."""

from __future__ import annotations

import contextvars
import functools
import importlib
import inspect
import json
import logging
import threading
import types
import weakref
from contextlib import contextmanager
from dataclasses import dataclass

from opentelemetry import context, trace
from opentelemetry.attributes import BoundedAttributes
from opentelemetry.sdk.util import BoundedList
from opentelemetry.semconv._incubating.attributes.error_attributes import ERROR_MESSAGE
from opentelemetry.semconv.attributes.error_attributes import ERROR_TYPE
from opentelemetry.semconv.attributes.http_attributes import HTTP_RESPONSE_STATUS_CODE
from opentelemetry.semconv.trace import SpanAttributes as DB
from opentelemetry.semconv_ai import SpanAttributes as AI
from opentelemetry.trace import Status, StatusCode
from respan_sdk.constants.span_attributes import RESPAN_LOG_TYPE, RESPAN_METADATA
from respan_tracing.constants.context_constants import ENABLE_CONTENT_TRACING_KEY

from ._policy import Policy, key, permitted, suppressed
from ._serialization import json_string, native_storage, redact_text

logger = logging.getLogger(__name__)
_LOCK = threading.RLock()
_MANAGER = None
_OWNERS = set()
_CURRENT = contextvars.ContextVar("respan_marqo_call", default=None)
_MISSING = object()


@dataclass(frozen=True)
class PatchSpec:
    module: str
    class_name: str
    methods: tuple[str, ...]
    label: str = "client"


class _Call:
    def __init__(self, manager, kind, args, kwargs):
        self.manager = manager
        self.kind = kind
        self.family = kind
        self.name = "marqo." + kind
        self.done = False
        self.finishing = False
        self.allowed = False
        self.input = self.output = None
        self.has_output = False
        self.keys = set()
        self.error_type = None
        self.span = None
        self.policy = None
        self.requests = []
        self.http_statuses = []
        self.structural = {}
        try:
            provider = manager.refresh()
            self.policy = manager.policy
            initial = (
                manager.capture
                and permitted()
                and self.policy.enroll(trace.get_current_span())
            )
            self.span = provider.get_tracer("marqo").start_span(
                self.name, attributes={RESPAN_LOG_TYPE: "task"}
            )
            self.allowed = bool(
                initial
                and self.span.is_recording()
                and self.policy.bound(key(self.span))
            )
            if self.span.is_recording():
                self.structural = {
                    k: v
                    for k, v in self.span.attributes.items()
                    if k.startswith(RESPAN_METADATA)
                }
                manager.instrumentor._set_start_attributes(
                    self.span, kind, args, kwargs
                )
                self.span.set_attribute(AI.TRACELOOP_ENTITY_NAME, self.name)
                self.span.set_attribute(AI.TRACELOOP_ENTITY_PATH, "")
            manager.states.add(self)
            if self.allowed:
                positional = args if kind == "index.create" else args[1:]
                self.input = json_string({"args": positional, "kwargs": kwargs})
                self.check()
        except Exception:
            self.discard()
            raise

    def request(self, method, kwargs):
        if self.check():
            body = kwargs.get("data")
            if type(body) is bytes:
                body = bytes.decode(body, "utf-8")
            if type(body) is str:
                try:
                    body = json.loads(body)
                except ValueError:
                    pass
            self.requests.append(
                json_string({"method": method, "url": kwargs.get("url"), "body": body})
            )
            self.check()

    def response(self, response):
        data = native_storage(response, self.manager.response_class) or {}
        status = data.get("status_code")
        if type(status) is int:
            self.http_statuses.append(status)

    def check(self):
        if self.done:
            return self.allowed
        self.allowed = bool(
            self.allowed and permitted() and self.policy.bound(key(self.span))
        )
        if not self.allowed:
            self.scrub()
        return self.allowed

    def capture(self, k, value):
        if self.check():
            self.keys.add(k)
            try:
                self.span.set_attribute(k, value)
            finally:
                self.keys.add(k)
            self.check()

    def scrub(self):
        self.input = self.output = None
        self.requests.clear()
        self.has_output = False
        if self.span and (self.span.is_recording() or self.finishing):
            for k in self.keys:
                self.span._attributes._dict.pop(k, None)
            self.span._attributes._dict.pop(ERROR_MESSAGE, None)
            self.span._events._dq.clear()
            if self.span.status.status_code == StatusCode.ERROR:
                self.span._status = Status(StatusCode.ERROR)

    def observe(self, result):
        if self.check():
            output = result
            if self.kind == "index.embed" and type(result) is dict:
                embeddings = result.get("embeddings")
                if type(embeddings) is list and all(
                    type(r) is list
                    or (type(r) is dict and type(r.get("embedding")) is list)
                    for r in embeddings
                ):
                    output = [
                        r.get("embedding") if type(r) is dict else r for r in embeddings
                    ]
                    extras = {k: v for k, v in result.items() if k != "embeddings"}
                    extras["records"] = [
                        {k: v for k, v in r.items() if k != "embedding"}
                        if type(r) is dict
                        else None
                        for r in embeddings
                    ]
                    self.capture(RESPAN_METADATA + ".marqo.result", json_string(extras))
                model = result.get("model")
                if type(model) is str:
                    self.span.set_attribute(AI.LLM_REQUEST_MODEL, redact_text(model))
            self.output = json_string(output)
            self.has_output = True
            self.check()

    def error(self, error):
        if not self.span.is_recording():
            return
        self.error_type = type.__dict__["__name__"].__get__(
            type(error), type(type(error))
        )
        self.span.set_attribute(ERROR_TYPE, self.error_type)
        self.span.set_status(Status(StatusCode.ERROR))
        if self.check():
            for value in BaseException.args.__get__(error):
                if type(value) is str or type(value) is dict:
                    text = (
                        redact_text(value) if type(value) is str else json_string(value)
                    )
                    self.capture(ERROR_MESSAGE, text)
                    self.span.set_status(Status(StatusCode.ERROR, text))
                    self.check()
                    break

    def finish(self):
        if self.done or self.finishing:
            return
        self.finishing = True
        try:
            if self.check():
                if self.input is not None:
                    data = json.loads(self.input)
                    if self.requests:
                        data["native_requests"] = [json.loads(r) for r in self.requests]
                    self.capture(AI.TRACELOOP_ENTITY_INPUT, json_string(data))
                if self.has_output:
                    self.capture(AI.TRACELOOP_ENTITY_OUTPUT, self.output)
            if self.span.is_recording():
                if (
                    self.error_type is None
                    and self.span.status.status_code != StatusCode.ERROR
                ):
                    self.span.set_status(Status(StatusCode.OK))
                self.span.set_attributes(self.structural)
                self.span.set_attribute(
                    RESPAN_LOG_TYPE,
                    "embedding" if self.kind == "index.embed" else "task",
                )
                if self.kind == "index.embed":
                    self.span.set_attribute(AI.LLM_REQUEST_TYPE, "embedding")
                if self.http_statuses:
                    # A final observed request status, not an aggregate guessed code.
                    self.span.set_attribute(
                        HTTP_RESPONSE_STATUS_CODE, self.http_statuses[-1]
                    )
                if self.error_type:
                    self.span.set_attribute(ERROR_TYPE, self.error_type)
                self.check()
        except Exception:  # noqa: BLE001 - discard telemetry faults without replacing results.
            self.discard()
            return
        try:
            self.span.end()
        except Exception:  # noqa: BLE001 - observer span end cannot replace the native result.
            logger.debug("Marqo telemetry span end failed")
        finally:
            self.done = True
            self.finishing = False
            self.input = self.output = None
            self.requests.clear()
            self.has_output = False
            self.manager.states.discard(self)
            if self.policy:
                self.policy.on_end(self.span)

    def discard(self):
        self.allowed = False
        self.requests.clear()
        self.input = self.output = None
        self.has_output = False
        if self.done:
            return
        self.done = True
        try:
            if self.span and (self.span.is_recording() or self.finishing):
                for k in self.keys:
                    self.span._attributes._dict.pop(k, None)
                self.span._attributes._dict.pop(ERROR_MESSAGE, None)
                self.span._attributes = BoundedAttributes(
                    attributes={
                        **self.structural,
                        RESPAN_LOG_TYPE: "embedding"
                        if self.kind == "index.embed"
                        else "task",
                        AI.TRACELOOP_ENTITY_NAME: self.name,
                        AI.TRACELOOP_ENTITY_PATH: "",
                        DB.DB_SYSTEM: "marqo",
                        DB.DB_OPERATION: self.kind,
                    },
                    immutable=False,
                )
                self.span._events._dq.clear()
                self.span._status = (
                    Status(StatusCode.ERROR)
                    if self.error_type
                    or self.span.status.status_code == StatusCode.ERROR
                    else Status(StatusCode.UNSET)
                )
                if self.error_type:
                    self.span._attributes[ERROR_TYPE] = self.error_type
        except Exception:  # noqa: BLE001 - guaranteed observer cleanup best effort.
            logger.debug("Marqo telemetry discard failed")
        try:
            if self.span:
                self.span.end()
        except Exception:  # noqa: BLE001 - native SDK behavior has priority.
            logger.debug("Marqo telemetry discard end failed")
        finally:
            self.manager.states.discard(self)
            if self.policy and self.span:
                self.policy.on_end(self.span)

    def safe(self, method, *args):
        if self.done:
            return None
        try:
            return method(*args)
        except Exception:  # noqa: BLE001 - all telemetry faults are isolated.
            logger.debug("Marqo telemetry observation failed")
            self.discard()
            return None


@contextmanager
def _scope(state):
    current = span_token = private_token = None
    ambient = context.get_current()
    try:
        if state and not state.done:
            state.safe(state.check)
            if not state.done:
                current = _CURRENT.set(state)
                try:
                    if not state.allowed:
                        private_token = context.attach(
                            context.set_value(ENABLE_CONTENT_TRACING_KEY, False)
                        )
                    span_token = context.attach(trace.set_span_in_context(state.span))
                except Exception:  # noqa: BLE001 - native SDK scopes survive observer attach failure.
                    state.discard()
                    try:
                        state.policy.original_attach(ambient)
                    except Exception:  # noqa: BLE001 - native result has priority.
                        logger.debug("Marqo telemetry attach restoration failed")
        yield
    finally:
        if state and not state.done:
            state.safe(state.check)
        for token in (span_token, private_token):
            if token is not None:
                try:
                    context.detach(token)
                except Exception:  # noqa: BLE001 - context cleanup cannot replace the native outcome.
                    try:
                        state.policy.original_detach(token)
                    except Exception:  # noqa: BLE001 - preserve native outcomes after both detach attempts.
                        logger.debug("Marqo telemetry detach failed")
        if current is not None:
            _CURRENT.reset(current)


def _wrapper(original, operation):
    @functools.wraps(original)
    def wrapper(*args, **kwargs):
        manager = _MANAGER
        active = _CURRENT.get()
        aliases = {
            "client.create_index": "index.create",
            "client.delete_index": "index.delete",
        }
        if (
            not manager
            or not manager.enabled
            or suppressed()
            or (active and aliases.get(active.kind) == operation)
        ):
            return original(*args, **kwargs)
        try:
            state = _Call(manager, operation, args, kwargs)
        except Exception:  # noqa: BLE001 - observer startup cannot alter native calls.
            state = None
            logger.debug("Marqo observer startup failed")
        try:
            with _scope(state):
                result = original(*args, **kwargs)
        except BaseException as error:
            if state:
                state.safe(state.error, error)
                state.safe(state.finish)
            raise
        if state:
            state.safe(state.observe, result)
            state.safe(state.finish)
        return result

    return wrapper


class _Manager:
    def __init__(self, instrumentor):
        from requests import Response

        self.instrumentor = instrumentor
        self.provider = instrumentor.tracer_provider
        self.capture = instrumentor.capture_content
        self.enabled = True
        self.policy = None
        self.bound_provider = None
        self.states = weakref.WeakSet()
        self.patches = []
        self.response_class = Response

    def refresh(self):
        provider = self.provider or trace.get_tracer_provider()
        if provider is not self.bound_provider:
            if self.policy:
                self.policy.close()
            self.policy = Policy(provider, self.scrub)
            self.bound_provider = provider
        return provider

    def scrub(self, readable=None):
        for state in list(self.states):
            if not state.done:
                state.safe(state.check)
                if (
                    readable is not None
                    and state.span is not None
                    and key(state.span) == key(readable)
                    and not state.allowed
                ):
                    for k in state.keys:
                        readable._attributes._dict.pop(k, None)
                    readable._attributes._dict.pop(ERROR_MESSAGE, None)
                    readable._events = BoundedList(maxlen=0)
                    if readable.status.status_code == StatusCode.ERROR:
                        readable._status = Status(StatusCode.ERROR)

    def patch(self, obj, name, replacement):
        namespace = type.__dict__["__dict__"].__get__(obj, type(obj))
        present = name in namespace
        stored = namespace.get(name)
        self.patches.append((obj, name, replacement, present, stored))
        setattr(obj, name, replacement)

    def install(self):
        for spec in self.instrumentor.patches:
            cls = getattr(importlib.import_module(spec.module), spec.class_name)
            for name in spec.methods:
                original = inspect.getattr_static(cls, name, None)
                if type(original) is staticmethod:
                    fn = staticmethod.__get__(original, None, cls)
                    replacement = staticmethod(_wrapper(fn, spec.label + "." + name))
                elif type(original) is types.FunctionType:
                    replacement = _wrapper(original, spec.label + "." + name)
                else:
                    continue
                self.patch(cls, name, replacement)
        from marqo._httprequests import HttpRequests

        original = inspect.getattr_static(HttpRequests, "_operation")

        @functools.wraps(original)
        def operation(instance, method):
            native = original(instance, method)

            @functools.wraps(native)
            def observed(*args, **kwargs):
                state = _CURRENT.get()
                if state and not state.done:
                    state.safe(state.request, method, kwargs)
                result = native(*args, **kwargs)
                if state and not state.done:
                    state.safe(state.response, result)
                return result

            return observed

        self.patch(HttpRequests, "_operation", operation)

    def close(self):
        self.enabled = False
        for state in list(self.states):
            state.discard()
        for obj, name, replacement, present, stored in reversed(self.patches):
            if inspect.getattr_static(obj, name, None) is replacement:
                if present:
                    setattr(obj, name, stored)
                else:
                    delattr(obj, name)
        self.patches.clear()
        if self.policy:
            self.policy.close()


class NativeClientInstrumentor:
    name = "marqo"
    vendor = "marqo"
    patches = ()

    def __init__(self, *, capture_content=True, tracer_provider=None):
        self.capture_content = capture_content is True
        self.tracer_provider = tracer_provider
        self._active = False

    @classmethod
    def _set_start_attributes(cls, span, operation, args, kwargs):
        span.set_attribute(DB.DB_SYSTEM, cls.vendor)
        span.set_attribute(DB.DB_OPERATION, operation)
        if operation == "index.embed":
            span.set_attribute(RESPAN_LOG_TYPE, "embedding")
            span.set_attribute(AI.LLM_REQUEST_TYPE, "embedding")

    def activate(self):
        global _MANAGER
        if self._active:
            return
        with _LOCK:
            if _MANAGER:
                if (
                    _MANAGER.provider is not self.tracer_provider
                    or _MANAGER.capture is not self.capture_content
                ):
                    raise ValueError("Marqo instrumentation configuration conflict")
            else:
                manager = _Manager(self)
                try:
                    manager.install()
                    provider = self.tracer_provider or trace.get_tracer_provider()
                    if hasattr(provider, "add_span_processor"):
                        manager.refresh()
                except BaseException:
                    manager.close()
                    raise
                _MANAGER = manager
            _OWNERS.add(self)
            self._active = True

    def deactivate(self):
        global _MANAGER
        if not self._active:
            return
        with _LOCK:
            self._active = False
            _OWNERS.discard(self)
            if not _OWNERS and _MANAGER:
                manager = _MANAGER
                _MANAGER = None
                manager.close()
