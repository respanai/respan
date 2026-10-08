"""Observe released native LanceDB operations and Arrow values with real OTel."""

# ruff: noqa: BLE001 -- telemetry faults never alter native behavior.
from __future__ import annotations

import contextvars
import functools
import importlib
import importlib.metadata
import inspect
import threading
import types
import weakref

from opentelemetry import context, trace
from opentelemetry.attributes import BoundedAttributes
from opentelemetry.sdk.util import BoundedList
from opentelemetry.semconv._incubating.attributes.error_attributes import ERROR_MESSAGE
from opentelemetry.semconv.attributes.db_attributes import (
    DB_COLLECTION_NAME,
    DB_NAMESPACE,
    DB_OPERATION_NAME,
    DB_SYSTEM_NAME,
)
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
from weaviate import exceptions as native_exceptions

from ._constants import WEAVIATE_PATCH_SPECS
from ._policy import (
    CREATING_CALL,
    AncestorPolicy,
    span_key,
    suppressed,
)
from ._serialization import (
    REDACTED,
    json_dumps,
    native_storage,
    register_native_types,
    safe_exception_message,
    safe_text,
    safe_type_name,
    sensitive_key,
    to_jsonable,
)

_NATIVE_ERRORS = tuple(
    value
    for value in vars(native_exceptions).values()
    if isinstance(value, type) and issubclass(value, BaseException)
)

_LOCK = threading.RLock()
_ACTIVATION_COUNT = 0
_ENABLED = False
_CAPTURE_CONTENT = True
_PROVIDER = None
_POLICIES = weakref.WeakKeyDictionary()
_PATCHES = []
_PENDING = weakref.WeakSet()
_ACTIVE_CALL = contextvars.ContextVar("respan_weaviate_active_call", default=False)


def _base(operation):
    return {
        RESPAN_LOG_TYPE: "task",
        DB_SYSTEM_NAME: "weaviate",
        DB_OPERATION_NAME: operation.rsplit(".", 1)[-1],
    }


def _tracing_enabled():
    instance = RespanTracer._instance
    if instance is None:
        return True
    return (
        type(instance) is RespanTracer
        and object.__getattribute__(instance, "__dict__").get("is_enabled") is True
    )


def _provider():
    return _PROVIDER if _PROVIDER is not None else trace.get_tracer_provider()


def _policy():
    provider = _provider()
    with _LOCK:
        policy = _POLICIES.get(provider)
        if policy is None:
            if not callable(getattr(provider, "add_span_processor", None)):
                return None
            policy = AncestorPolicy(_CAPTURE_CONTENT)
            _POLICIES[provider] = policy
            try:
                provider.add_span_processor(policy)
                processor = getattr(provider, "_active_span_processor", None)
                if processor is not None:
                    processor._span_processors = (
                        policy,
                        *(
                            item
                            for item in processor._span_processors
                            if item is not policy
                        ),
                    )
            except BaseException:
                _remove_policies()
                raise
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
        elif any(type(value) is kind for kind in (str, bool, int, float)):
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
    def __init__(self, kwargs, *, name, operation):
        self.span = None
        self.creation_name = name
        self.policy = None
        self.finished = False
        self.failed = False
        self.operation = operation
        self.chunks = []
        self.content = set()
        self.cleanups = []
        self.propagated = {}
        self.priority = {}
        self.base = _base(operation)
        self.base[SpanAttributes.TRACELOOP_ENTITY_NAME] = name
        self.base[SpanAttributes.TRACELOOP_ENTITY_PATH] = (
            "" if not trace.get_current_span().get_span_context().is_valid else name
        )
        _PENDING.add(self)
        self.policy = _policy()
        creation_token = CREATING_CALL.set(self)
        try:
            self.span = (
                _provider()
                .get_tracer(
                    "weaviate",
                    importlib.metadata.version("respan-instrumentation-weaviate"),
                )
                .start_span(name, kind=trace.SpanKind.CLIENT, attributes=self.base)
            )
        finally:
            CREATING_CALL.reset(creation_token)
        if self.recording() and self.allowed():
            self.propagated = _propagated_attributes()
            self.set_attributes(
                {SpanAttributes.TRACELOOP_ENTITY_INPUT: json_dumps(kwargs)}
            )

    def __del__(self):
        try:
            if not getattr(self, "finished", True):
                if self.recording() and self.chunks:
                    _observe(self, lambda: self.output(self.chunks))
                self.finish(completed=False)
        except BaseException:  # noqa: S110 - GC telemetry must not affect native resources.
            pass

    def recording(self):
        return self.span is not None and bool(_attempt(self.span.is_recording, False))

    def scrub(self, readable=None):
        self.chunks.clear()
        self.propagated.clear()
        self.priority.clear()
        attributes = getattr(self.span, "_attributes", None)
        structural = set(self.base) | {ERROR_TYPE}
        if type(attributes) is BoundedAttributes:
            attributes = object.__getattribute__(attributes, "_dict")
        if attributes is not None:
            for key in list(attributes):
                if key not in structural:
                    _attempt(lambda key=key: attributes.pop(key, None))
        if getattr(self.span, "_events", None) is not None:
            self.span._events = BoundedList(0)
        if readable is not None:
            readable._attributes = types.MappingProxyType(
                {
                    key: value
                    for key, value in (readable.attributes or {}).items()
                    if key in structural
                }
            )
            readable._events = ()
            readable._status = Status(readable.status.status_code)
        status = getattr(self.span, "status", None)
        if status is not None:
            self.span._status = Status(status.status_code)

    def allowed(self, *, honor_suppression=True):
        if self.failed and self.policy is not None:
            self.policy._deny_chain(span_key(self.span))
        value = (
            not self.failed
            and self.policy is not None
            and self.policy.observe(self.span, honor_suppression=honor_suppression)
        )
        if not value:
            self.scrub()
        return value

    def set_attributes(self, values):
        priority = {
            SpanAttributes.TRACELOOP_ENTITY_INPUT,
            SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
            RESPAN_LOG_TYPE,
            f"{RESPAN_METADATA}.weaviate.request",
            f"{RESPAN_METADATA}.weaviate.result",
        }
        for key, value in values.items():
            if key not in priority:
                self.content.add(key)
                self.span.set_attribute(key, value)
        # Reassert structure/propagation after indexed convenience fields. Full
        # native JSON is last under the SDK's own attribute-count bounds.
        for key, value in {**self.base, **self.propagated}.items():
            self.content.add(key)
            self.span.set_attribute(key, value)
        self.priority.update(
            {key: value for key, value in values.items() if key in priority}
        )
        for key, value in self.priority.items():
            self.content.add(key)
            self.span.set_attribute(key, value)

    def capture(self, response):
        if self.recording() and self.allowed():
            self.chunks.append(response)

    def output(self, response):
        if self.recording() and self.allowed():
            values = {SpanAttributes.TRACELOOP_ENTITY_OUTPUT: json_dumps(response)}
            self.set_attributes(values)

    def error(self, exc):
        if not self.recording():
            return
        allowed = self.allowed()
        message = (
            safe_exception_message(exc)
            if allowed
            and any(
                type(exc) is kind
                for kind in (
                    ValueError,
                    RuntimeError,
                    TypeError,
                    OSError,
                    KeyError,
                    *_NATIVE_ERRORS,
                )
            )
            else None
        )
        self.span.set_status(Status(StatusCode.ERROR, message))
        self.span.set_attribute(ERROR_TYPE, safe_type_name(exc))
        if any(type(exc) is kind for kind in _NATIVE_ERRORS):
            raw = BaseException.__dict__["__dict__"].__get__(exc)
            code = raw.get("_status_code")
            if type(code) is int and 400 <= code <= 599:
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
            _attempt(lambda: _restore_context(ambient))
            raise
        return token, ambient

    def detach(self, state):
        if state is None:
            return
        token, ambient = state
        if not self.finished:
            _attempt(lambda: self.allowed(honor_suppression=False))
        if token is not None:
            try:
                context.detach(token)
            except BaseException:
                _attempt(lambda: context._RUNTIME_CONTEXT.detach(token))
        _attempt(lambda: _restore_context(ambient))

    def finish(self, error=None, *, completed=True):
        if self.finished:
            return
        self.finished = True
        if self.recording():
            if (
                error is None
                and completed
                and self.span.status.status_code is StatusCode.UNSET
            ):
                _attempt(lambda: self.span.set_status(Status(StatusCode.OK)))
            elif error is not None:
                _attempt(lambda: self.error(error))
            if not _attempt(lambda: self.allowed(honor_suppression=False), False):
                _attempt(self.scrub)
            ambient = context.get_current()
            try:
                _attempt(self.span.end)
            finally:
                _attempt(lambda: _restore_context(ambient))
        for cleanup in reversed(self.cleanups):
            _attempt(cleanup)
        self.cleanups.clear()
        self.chunks.clear()
        self.content.clear()
        self.propagated.clear()
        self.priority.clear()
        if self.policy is not None and self.span is not None:
            self.policy.calls.pop(span_key(self.span), None)
        self.policy = None
        _PENDING.discard(self)


def _observe(call, fn, *, keep_context=False):
    ambient = context.get_current()
    try:
        return fn()
    except BaseException:
        call.failed = True
        _attempt(call.scrub)
        _attempt(lambda: _restore_context(ambient))
        return None
    finally:
        if not keep_context:
            if not call.finished and call.policy is not None and call.span is not None:
                _attempt(
                    lambda: call.policy.observe(call.span, honor_suppression=False)
                )
            _attempt(lambda: _restore_context(ambient))


def _remove_policies():
    for provider, policy in list(_POLICIES.items()):
        policy.enabled = False
        processor = getattr(provider, "_active_span_processor", None)
        if processor is not None:
            _attempt(
                lambda processor=processor, policy=policy: setattr(
                    processor,
                    "_span_processors",
                    tuple(p for p in processor._span_processors if p is not policy),
                )
            )
        policy.clear()
    _POLICIES.clear()


def _identity(instance):
    raw = native_storage(instance)
    if type(raw) is not dict:
        return {}
    result = {}
    for source, target in (
        ("name", "collection"),
        ("_name", "collection"),
        ("_tenant", "tenant"),
    ):
        value = raw.get(source)
        if type(value) is str and target not in result:
            result[target] = safe_text(value)
    return result


def _start(instance, args, kwargs, operation):
    ambient = context.get_current()
    if not _ENABLED or _ACTIVE_CALL.get() or not _tracing_enabled() or suppressed():
        return None
    call = object.__new__(_Call)
    try:
        # Arguments are converted only after recording and privacy eligibility.
        _Call.__init__(call, {}, name=f"weaviate.{operation}", operation=operation)
        if call.recording() and call.allowed():
            identity = _identity(instance)
            if "collection" in identity:
                call.base[DB_COLLECTION_NAME] = identity["collection"]
            if "tenant" in identity:
                call.base[DB_NAMESPACE] = identity["tenant"]
            call.set_attributes(
                {
                    SpanAttributes.TRACELOOP_ENTITY_INPUT: json_dumps(
                        {
                            "operation": operation,
                            **identity,
                            "args": args,
                            "kwargs": kwargs,
                        }
                    )
                }
            )
        return call
    except BaseException:
        call.failed = True
        _attempt(lambda: call.scrub())
        _attempt(lambda: call.finish(completed=False))
        return None
    finally:
        _attempt(lambda: _restore_context(ambient))


def _wrap(original, operation, asynchronous):
    if asynchronous:

        @functools.wraps(original)
        async def wrapped(instance, *args, **kwargs):
            call = _start(instance, args, kwargs, operation)
            if call is None:
                return await original(instance, *args, **kwargs)
            token = _ACTIVE_CALL.set(True)
            state = _observe(call, call.attach, keep_context=True)
            try:
                result = await original(instance, *args, **kwargs)
            except BaseException as error:
                _observe(call, lambda error=error: call.finish(error))
                raise
            else:
                _observe(call, lambda: call.output(result))
                _observe(call, call.finish)
                return result
            finally:
                _observe(call, lambda: call.detach(state), keep_context=True)
                _ACTIVE_CALL.reset(token)

        return wrapped

    @functools.wraps(original)
    def wrapped(instance, *args, **kwargs):
        call = _start(instance, args, kwargs, operation)
        if call is None:
            return original(instance, *args, **kwargs)
        token = _ACTIVE_CALL.set(True)
        state = _observe(call, call.attach, keep_context=True)
        try:
            result = original(instance, *args, **kwargs)
        except BaseException as error:
            _observe(call, lambda error=error: call.finish(error))
            raise
        else:
            _observe(call, lambda: call.output(result))
            _observe(call, call.finish)
            return result
        finally:
            _observe(call, lambda: call.detach(state), keep_context=True)
            _ACTIVE_CALL.reset(token)

    return wrapped


def _targets():
    found = set()
    for spec in WEAVIATE_PATCH_SPECS:
        module = importlib.import_module(spec.module)
        owner = getattr(module, spec.class_name, None)
        if owner is None:
            continue
        for name in spec.methods:
            key = owner, name
            if key in found:
                continue
            original = getattr(owner, name, None)
            if not callable(original):
                continue
            found.add(key)
            present = name in vars(owner)
            yield owner, name, original, present, f"{spec.label}.{name}", spec.is_async


def _restore():
    for owner, name, original, wrapper, present in reversed(_PATCHES):
        if inspect.getattr_static(owner, name, None) is wrapper:
            if present:
                _attempt(
                    lambda owner=owner, name=name, original=original: setattr(
                        owner, name, original
                    )
                )
            else:
                _attempt(lambda owner=owner, name=name: delattr(owner, name))
    _PATCHES.clear()
    _remove_policies()


class WeaviateInstrumentor:
    """Trace native Weaviate v4 sync and async manager operations."""

    name = "weaviate"

    def __init__(self, *, capture_content=True):
        self._is_instrumented = False
        self._capture_content = capture_content is True

    def activate(self, *, tracer_provider=None, capture_content=None):
        global _ENABLED, _CAPTURE_CONTENT, _PROVIDER, _ACTIVATION_COUNT
        setting = (
            self._capture_content
            if capture_content is None
            else capture_content is True
        )
        with _LOCK:
            if self._is_instrumented or not _tracing_enabled():
                return
            if _ACTIVATION_COUNT:
                if tracer_provider is not _PROVIDER or setting is not _CAPTURE_CONTENT:
                    raise ValueError("Weaviate instrumentation configuration conflict")
                _ACTIVATION_COUNT += 1
                self._is_instrumented = True
                return
            _PROVIDER = tracer_provider
            _CAPTURE_CONTENT = setting
            try:
                targets = list(_targets())
                register_native_types()
                _policy()
                for owner, name, original, present, operation, asynchronous in targets:
                    wrapper = _wrap(original, operation, asynchronous)
                    _PATCHES.append((owner, name, original, wrapper, present))
                    setattr(owner, name, wrapper)
            except BaseException:
                _restore()
                _PROVIDER = None
                raise
            _ENABLED = True
            _ACTIVATION_COUNT = 1
            self._is_instrumented = True

    def deactivate(self):
        global _ENABLED, _ACTIVATION_COUNT, _PROVIDER
        with _LOCK:
            if not self._is_instrumented:
                return
            self._is_instrumented = False
            _ACTIVATION_COUNT -= 1
            if _ACTIVATION_COUNT:
                return
            _ENABLED = False
            for call in list(_PENDING):
                _observe(call, lambda call=call: call.finish(completed=False))
            _restore()
            _PROVIDER = None

    instrument = activate
    uninstrument = deactivate
