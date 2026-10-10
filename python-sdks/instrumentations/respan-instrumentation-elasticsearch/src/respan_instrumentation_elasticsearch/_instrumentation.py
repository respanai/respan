"""Augment the official native Elasticsearch tracer without changing API results."""

# ruff: noqa: BLE001 -- all telemetry faults fail closed.
from __future__ import annotations

import contextvars
import functools
import importlib
import json
import logging
import threading
import types
import weakref
from contextlib import contextmanager

import elastic_transport
from elastic_transport.client_utils import DEFAULT
from elasticsearch._otel import OpenTelemetry
from opentelemetry import context, trace
from opentelemetry.sdk.util import BoundedList
from opentelemetry.semconv._incubating.attributes.db_attributes import (
    DB_COLLECTION_NAME,
    DB_OPERATION_NAME,
    DB_QUERY_TEXT,
    DB_RESPONSE_STATUS_CODE,
    DB_SYSTEM_NAME,
)
from opentelemetry.semconv._incubating.attributes.error_attributes import (
    ERROR_MESSAGE,
    ERROR_TYPE,
)
from opentelemetry.semconv._incubating.attributes.http_attributes import (
    HTTP_REQUEST_METHOD,
    HTTP_RESPONSE_STATUS_CODE,
)
from opentelemetry.semconv._incubating.attributes.server_attributes import (
    SERVER_ADDRESS,
    SERVER_PORT,
)
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.trace import Status, StatusCode
from respan_sdk.constants.llm_logging import LOG_TYPE_TASK
from respan_sdk.constants.span_attributes import RESPAN_LOG_TYPE, RESPAN_METADATA
from respan_tracing.core.tracer import RespanTracer

from ._native import OMIT, namespace, status, storage, value
from ._policy import (
    CREATING_CALL,
    AncestorPolicy,
    content_allowed,
    span_key,
    suppressed,
)
from ._serialization import (
    json_dumps,
    safe_exception_message,
    safe_text,
    safe_type_name,
)

logger = logging.getLogger(__name__)
_LOCK = threading.RLock()
_ACTIVE = contextvars.ContextVar("respan_elasticsearch_call", default=None)
_PATCHES = []
_POLICIES = weakref.WeakKeyDictionary()
_PENDING = weakref.WeakSet()
_CONFIG = None
_OWNERS = 0


def _attempt(fn, default=None):
    try:
        return fn()
    except BaseException:
        logger.debug("Elasticsearch telemetry operation failed")
        return default


def _restore_context(ambient):
    if context.get_current() is ambient:
        return
    _attempt(lambda: context._RUNTIME_CONTEXT.attach(ambient))
    if context.get_current() is not ambient:
        for candidate in storage(context._RUNTIME_CONTEXT).values():
            if type(candidate) is contextvars.ContextVar:
                candidate.set(ambient)
                break


def _enabled():
    current = RespanTracer._instance
    return current is None or storage(current).get("is_enabled", True) is not False


def _provider():
    provider = _CONFIG[0] if _CONFIG is not None else None
    provider = provider if provider is not None else trace.get_tracer_provider()
    # ProxyTracerProvider resolves lazily, after application configuration.
    if type(provider) is trace.ProxyTracerProvider:
        provider = trace._TRACER_PROVIDER
    return provider


def _policy(provider):
    if provider is None or not hasattr(provider, "add_span_processor"):
        return None
    found = _POLICIES.get(provider)
    if found is None:
        found = AncestorPolicy(_CONFIG[1])
        _POLICIES[provider] = found
        provider.add_span_processor(found)
        processor = provider._active_span_processor
        entries = processor._span_processors
        processor._span_processors = (found,) + tuple(
            p for p in entries if p is not found
        )
    return found


def _remove_policies():
    for provider, observer in list(_POLICIES.items()):
        observer.enabled = False
        processor = provider._active_span_processor
        _attempt(
            lambda processor=processor, observer=observer: setattr(
                processor,
                "_span_processors",
                tuple(p for p in processor._span_processors if p is not observer),
            )
        )
        observer.clear()
    _POLICIES.clear()


def _observe(call, fn):
    ambient = context.get_current()
    try:
        return fn()
    except BaseException:
        call.failed = True
        _attempt(call.scrub)
        return None
    finally:
        if not call.finished and call.span is not None:
            _attempt(lambda: call.allowed(honor_suppression=False))
        _restore_context(ambient)


class _Call:
    def __init__(self, name):
        self.span = None
        self.creation_name = name
        self.policy = None
        self.finished = False
        self.failed = False
        self.request = None
        self.native_configuration = None
        self.output = OMIT
        self.response_meta = None
        self.error_value = None
        self.cleanups = []
        self.base = {
            RESPAN_LOG_TYPE: LOG_TYPE_TASK,
            DB_SYSTEM_NAME: "elasticsearch",
            SpanAttributes.TRACELOOP_ENTITY_NAME: name,
            SpanAttributes.TRACELOOP_ENTITY_PATH: name
            if trace.get_current_span().get_span_context().is_valid
            else "",
        }
        self.allowed_initially = content_allowed(_CONFIG[1])
        _PENDING.add(self)

    def recording(self):
        return self.span is not None and _attempt(self.span.is_recording, False)

    def allowed(self, *, honor_suppression=True):
        allowed = (
            self.allowed_initially
            and not self.failed
            and self.recording()
            and self.policy is not None
            and self.policy.observe(self.span, honor_suppression=honor_suppression)
        )
        if not allowed:
            if self.policy is not None:
                self.policy._deny_chain(span_key(self.span))
            self.scrub()
        return allowed

    def scrub(self, readable=None):
        self.request = None
        self.native_configuration = None
        self.output = OMIT
        self.response_meta = None
        self.error_value = None
        structural = set(self.base) | {
            DB_OPERATION_NAME,
            HTTP_REQUEST_METHOD,
            HTTP_RESPONSE_STATUS_CODE,
            DB_RESPONSE_STATUS_CODE,
            SERVER_ADDRESS,
            SERVER_PORT,
            ERROR_TYPE,
        }
        if self.span is not None:
            attributes = getattr(self.span, "_attributes", None)
            if attributes is not None:
                for key in list(attributes):
                    if key not in structural:
                        _attempt(lambda key=key: attributes.pop(key, None))
            if hasattr(self.span, "_events"):
                self.span._events = BoundedList(0)
            if self.recording():
                self.span._status = Status(self.span.status.status_code)
        if readable is not None:
            readable._attributes = types.MappingProxyType(
                {
                    key: item
                    for key, item in (readable.attributes or {}).items()
                    if key in structural
                }
            )
            readable._events = ()
            readable._status = Status(readable.status.status_code)

    def set(self, key, item):
        self.span.set_attribute(key, item)

    def request_values(self, original, args, kwargs, *, transport=False):
        if not self.allowed():
            return
        if transport:
            method = kwargs.get("method", args[0] if args else None)
            target = kwargs.get("target", args[1] if len(args) > 1 else None)
            if self.request is None:
                self.request = {"method": value(method), "target": value(target)}
            self.request["transport"] = value(
                {k: v for k, v in kwargs.items() if k != "otel_span"}
            )
        else:
            # The released native signature is stable across 8.13 and 9.5.
            self.request = {
                "method": value(args[0] if args else kwargs.get("method")),
                "target": value(args[1] if len(args) > 1 else kwargs.get("path")),
            }
            for key, item in kwargs.items():
                if key != "otel_span" and item is not DEFAULT:
                    self.request[key] = value(item)
            if self.native_configuration is not None:
                self.request["native"] = self.native_configuration
            attrs = self.span.attributes or {}
            index = attrs.get(
                "db.operation.parameter.index",
                attrs.get("db.elasticsearch.path_parts.index"),
            )
            if type(index) is str:
                self.set(DB_COLLECTION_NAME, safe_text(index))
            parts = kwargs.get("path_parts")
            if type(parts) is dict and type(parts.get("index")) is str:
                self.set(DB_COLLECTION_NAME, safe_text(parts["index"]))
            operation = kwargs.get("endpoint_id")
            if type(operation) is str:
                self.set(DB_OPERATION_NAME, operation)
        hook = _CONFIG[3] if _CONFIG is not None else None
        if hook is not None and not transport:
            hook(
                self.span,
                self.request.get("method"),
                self.request.get("target"),
                kwargs,
            )

    def response(self, response, *, transport=False):
        actual_status = status(response)
        if type(actual_status) is int:
            self.set(HTTP_RESPONSE_STATUS_CODE, actual_status)
            self.set(DB_RESPONSE_STATUS_CODE, str(actual_status))
        if self.allowed():
            if not transport or self.output is OMIT:
                self.output = value(response)
                from ._native import meta

                self.response_meta = value(meta(response))
            hook = _CONFIG[4] if _CONFIG is not None else None
            if hook is not None and not transport:
                from ._native import body

                hook(self.span, body(response))

    def error(self, error):
        if not self.recording():
            return
        self.output = OMIT
        self.span.set_status(Status(StatusCode.ERROR))
        self.set(ERROR_TYPE, safe_type_name(error))
        actual_status = status(error)
        if type(actual_status) is int:
            self.set(HTTP_RESPONSE_STATUS_CODE, actual_status)
            self.set(DB_RESPONSE_STATUS_CODE, str(actual_status))
        if self.allowed():
            # Exact released native classes and exact builtin exceptions only.
            cls = type(error)
            module = namespace(cls).get("__module__", "")
            native = type(module) is str and module.startswith(
                ("elastic_transport", "elasticsearch")
            )
            installed = (
                _attempt(
                    lambda: getattr(
                        importlib.import_module(module), safe_type_name(error)
                    )
                )
                if native
                else None
            )
            if (native and installed is cls) or any(
                cls is k for k in (ValueError, TypeError, RuntimeError, OSError)
            ):
                state = BaseException.__dict__["__dict__"].__get__(error)
                message = state.get("message")
                message = (
                    safe_text(message)
                    if type(message) is str
                    else safe_exception_message(error)
                )
                if message is not None:
                    self.set(ERROR_MESSAGE, message)

    def finish(self):
        if self.finished:
            return
        if self.recording():
            _observe(self, self._finish_attributes)
        self.finished = True
        if self.span is not None:
            _attempt(self.span.end)
        for undo in reversed(self.cleanups):
            _attempt(undo)
        self.cleanups.clear()
        self.request = None
        self.output = OMIT
        self.response_meta = None
        if self.policy is not None:
            self.policy.calls.pop(span_key(self.span), None)
        _PENDING.discard(self)

    def _finish_attributes(self):
        # Normalize actual legacy SDK fields into their upstream semantic names.
        attrs = getattr(self.span, "_attributes", None)
        if attrs is not None:
            for old, new in (
                ("db.system", DB_SYSTEM_NAME),
                ("db.operation", DB_OPERATION_NAME),
                ("db.statement", DB_QUERY_TEXT),
            ):
                if old in attrs:
                    self.set(new, attrs.pop(old))
        for key, item in self.base.items():
            self.set(key, item)
        if not self.allowed(honor_suppression=False):
            return
        if attrs is not None:
            for key, item in list(attrs.items()):
                if type(item) is str:
                    self.set(key, safe_text(item))
        if self.response_meta is not None:
            self.set(
                f"{RESPAN_METADATA}.elasticsearch.response",
                json_dumps(self.response_meta),
            )
        if self.request is not None:
            self.set(SpanAttributes.TRACELOOP_ENTITY_INPUT, self.dumps(self.request))
        if self.output is not OMIT:
            self.set(SpanAttributes.TRACELOOP_ENTITY_OUTPUT, self.dumps(self.output))

    def dumps(self, item):
        result = json_dumps(item)
        limit = _CONFIG[2] if _CONFIG is not None else None
        if limit is not None and len(result) > limit:
            return json_dumps(
                {
                    "truncated": True,
                    "original_characters": len(result),
                    "preview": result[:limit],
                }
            )
        return result


def _protect_setter(call):
    span = call.span
    if span is None or not call.recording():
        return
    for key in (
        f"{RESPAN_METADATA}.run_id",
        f"{RESPAN_METADATA}.scenario",
        f"{RESPAN_METADATA}.example_set",
    ):
        item = (span.attributes or {}).get(key)
        if type(item) is str:
            call.base[key] = safe_text(item)
    marker = (span.attributes or {}).get(RESPAN_METADATA)
    if type(marker) is str:
        decoded = _attempt(lambda: json.loads(marker), {})
        if type(decoded) is dict:
            keep = {
                key: item
                for key, item in decoded.items()
                if type(key) is str
                and key in {"run_id", "scenario", "example_set"}
                and type(item) is str
            }
            if keep:
                call.base[RESPAN_METADATA] = json_dumps(keep)
    state = storage(span)
    present = "set_attribute" in state
    prior = state.get("set_attribute")
    original = span.set_attribute

    def setter(key, item):
        try:
            return original(key, item)
        except BaseException:
            call.failed = True
            _attempt(call.scrub)
            return None

    def undo():
        if storage(span).get("set_attribute") is setter:
            if present:
                span.set_attribute = prior
            else:
                delattr(span, "set_attribute")

    call.cleanups.append(undo)
    span.set_attribute = setter


@contextmanager
def _scope(name, native_tracer=None):
    ambient = context.get_current()
    call = _Call(name)
    active = _ACTIVE.set(call)
    token = None
    try:
        if _enabled() and not suppressed():

            def startup():
                provider = _provider()
                call.policy = _policy(provider)
                tracer = (
                    provider.get_tracer("elasticsearch-api")
                    if provider is not None
                    else native_tracer
                )
                if tracer is None:
                    return
                creating = CREATING_CALL.set(call)
                try:
                    call.span = tracer.start_span(name)
                finally:
                    CREATING_CALL.reset(creating)
                _protect_setter(call)

            _observe(call, startup)
        if call.span is None or call.failed:
            if call.span is not None:
                _attempt(call.scrub)
                _attempt(call.span.end)
            call.span = trace.NonRecordingSpan(trace.INVALID_SPAN_CONTEXT)
        token = _attempt(lambda: context.attach(trace.set_span_in_context(call.span)))
        if token is None:
            call.failed = True
            call.scrub()
            _restore_context(ambient)
        try:
            yield call.span
        except BaseException as exc:
            _observe(call, functools.partial(call.error, exc))
            raise
        finally:
            if call.recording():
                _observe(call, lambda: call.allowed(honor_suppression=False))
            if token is not None:
                _attempt(lambda: context.detach(token))
            _restore_context(ambient)
            call.finish()
    finally:
        _ACTIVE.reset(active)
        _restore_context(ambient)


class _TracerShim:
    def __init__(self, tracer):
        self.tracer = tracer

    def start_as_current_span(self, name, *args, **kwargs):
        return _scope(name, self.tracer)


def _helper_wrapper(original):
    @functools.wraps(original)
    @contextmanager
    def wrapped(instance, *args, **kwargs):
        state = storage(instance)
        if _CONFIG is None or type(instance) is not OpenTelemetry:
            with original(instance, *args, **kwargs) as span:
                yield span
            return
        if not _enabled() or suppressed() or state.get("enabled") is not True:
            active = _ACTIVE.set(False)
            try:
                yield elastic_transport.OpenTelemetrySpan(None)
            finally:
                _ACTIVE.reset(active)
            return
        clone = OpenTelemetry(
            enabled=True,
            tracer=_TracerShim(state.get("tracer")),
            body_strategy=state.get("body_strategy"),
        )
        with original(clone, *args, **kwargs) as span:
            call = _ACTIVE.get()
            if call is not None and call is not False and call.recording():

                def capture_configuration():
                    if call.allowed():
                        call.native_configuration = value(
                            {"arguments": args, "parameters": kwargs}
                        )

                _observe(call, capture_configuration)
            yield span

    return wrapped


def _use_span_wrapper(original):
    @functools.wraps(original)
    @contextmanager
    def wrapped(instance, span):
        if _CONFIG is None or type(instance) is not OpenTelemetry:
            with original(instance, span):
                yield
            return
        state = storage(instance)
        native = (
            storage(span).get("otel_span")
            if type(span) is elastic_transport.OpenTelemetrySpan
            else None
        )
        ambient = context.get_current()
        token = None
        try:
            if state.get("enabled") and native is not None and not suppressed():
                token = _attempt(
                    lambda: context.attach(trace.set_span_in_context(native))
                )
            yield
        finally:
            if token is not None:
                _attempt(lambda: context.detach(token))
            _restore_context(ambient)

    return wrapped


def _api_wrapper(original, asynchronous, transport=False):
    def before(args, kwargs):
        call = _ACTIVE.get()
        if call is not None and call is not False and call.recording():
            _observe(
                call,
                lambda: call.request_values(
                    original, args, kwargs, transport=transport
                ),
            )
        return call

    if asynchronous:

        @functools.wraps(original)
        async def wrapped(instance, *args, **kwargs):
            call = before(args, kwargs)
            if call is None and transport and _CONFIG is not None:
                with _scope("elasticsearch.transport"):
                    return await wrapped(instance, *args, **kwargs)
            try:
                response = await original(instance, *args, **kwargs)
            except BaseException as exc:
                if call is not None and call is not False:
                    _observe(call, functools.partial(call.error, exc))
                raise
            if call is not None and call is not False and call.recording():
                _observe(call, lambda: call.response(response, transport=transport))
            return response
    else:

        @functools.wraps(original)
        def wrapped(instance, *args, **kwargs):
            call = before(args, kwargs)
            if call is None and transport and _CONFIG is not None:
                with _scope("elasticsearch.transport"):
                    return wrapped(instance, *args, **kwargs)
            try:
                response = original(instance, *args, **kwargs)
            except BaseException as exc:
                if call is not None and call is not False:
                    _observe(call, functools.partial(call.error, exc))
                raise
            if call is not None and call is not False and call.recording():
                _observe(call, lambda: call.response(response, transport=transport))
            return response

    return wrapped


def _restore():
    for owner, name, original, wrapper in reversed(_PATCHES):
        if namespace(owner).get(name) is wrapper:
            _attempt(
                lambda owner=owner, name=name, original=original: setattr(
                    owner, name, original
                )
            )
    _PATCHES.clear()
    _remove_policies()


class ElasticsearchInstrumentor:
    """Add canonical Respan TASK capture to native SDK sync/async spans."""

    name = "elasticsearch"

    def __init__(
        self,
        *,
        capture_content=True,
        max_attribute_chars=None,
        request_hook=None,
        response_hook=None,
        tracer_provider=None,
    ):
        if max_attribute_chars is not None and (
            type(max_attribute_chars) is not int or max_attribute_chars <= 0
        ):
            raise ValueError("max_attribute_chars must be a positive integer or None")
        self.config = (
            tracer_provider,
            capture_content is True,
            max_attribute_chars,
            request_hook,
            response_hook,
        )
        self._is_instrumented = False

    def activate(self, *, tracer_provider=None):
        global _CONFIG, _OWNERS
        with _LOCK:
            if self._is_instrumented:
                return
            config = (
                self.config
                if tracer_provider is None
                else (tracer_provider, *self.config[1:])
            )
            if _OWNERS:
                if any(a is not b for a, b in zip(config, _CONFIG)):
                    raise ValueError(
                        "Elasticsearch instrumentation configuration conflict"
                    )
                _OWNERS += 1
                self._is_instrumented = True
                return
            _CONFIG = config
            try:
                _policy(_provider())
                targets = [
                    (OpenTelemetry, "span", _helper_wrapper),
                    (OpenTelemetry, "helpers_span", _helper_wrapper),
                    (OpenTelemetry, "use_span", _use_span_wrapper),
                ]
                for module, asynchronous in (
                    ("elasticsearch._sync.client._base", False),
                    ("elasticsearch._async.client._base", True),
                ):
                    cls = importlib.import_module(module).BaseClient
                    targets.append(
                        (
                            cls,
                            "_perform_request",
                            lambda fn, asynchronous=asynchronous: _api_wrapper(
                                fn, asynchronous
                            ),
                        )
                    )
                for cls, asynchronous in (
                    (elastic_transport.Transport, False),
                    (elastic_transport.AsyncTransport, True),
                ):
                    targets.append(
                        (
                            cls,
                            "perform_request",
                            lambda fn, asynchronous=asynchronous: _api_wrapper(
                                fn, asynchronous, True
                            ),
                        )
                    )
                for owner, name, factory in targets:
                    original = namespace(owner).get(name)
                    if original is None:
                        continue
                    wrapper = factory(original)
                    _PATCHES.append((owner, name, original, wrapper))
                    setattr(owner, name, wrapper)
            except BaseException:
                _restore()
                _CONFIG = None
                raise
            _OWNERS = 1
            self._is_instrumented = True

    def deactivate(self):
        global _CONFIG, _OWNERS
        with _LOCK:
            if not self._is_instrumented:
                return
            self._is_instrumented = False
            _OWNERS -= 1
            if _OWNERS:
                return
            for call in list(_PENDING):
                call.failed = True
                _attempt(call.scrub)
            _restore()
            _CONFIG = None
