"""Retain upstream sync Chroma spans and observe complete native payloads."""

from __future__ import annotations

import importlib
import inspect
import logging
import threading
from contextvars import ContextVar
from functools import wraps

from opentelemetry import context, trace
from opentelemetry.attributes import BoundedAttributes
from opentelemetry.sdk.util import BoundedList
from opentelemetry.semconv._incubating.attributes.error_attributes import ERROR_MESSAGE
from opentelemetry.semconv.attributes.error_attributes import ERROR_TYPE
from opentelemetry.semconv.trace import SpanAttributes as DB
from opentelemetry.semconv_ai import SpanAttributes as AI
from opentelemetry.trace import Status, StatusCode
from respan_sdk.constants.span_attributes import RESPAN_LOG_TYPE, RESPAN_METADATA
from respan_tracing.core.tracer import RespanTracer

from ._constants import CLIENT_METHODS, COLLECTION_METHODS
from ._policy import CapturePolicy, PrivacyObserver, explicit_capture, suppressed
from ._serialization import (
    initialize_native_types,
    json_string,
    json_value,
    native_dict,
    native_storage,
    redact_text,
)

logger = logging.getLogger(__name__)
_LOCK = threading.RLock()
_OWNERS = set()
_MANAGER = None
_REQUEST = ContextVar("respan_chroma_native_request", default=None)
_CURRENT = ContextVar("respan_chroma_native_call", default=None)


def _register(provider, observer):
    active = getattr(provider, "_active_span_processor", None)
    if active is None:
        provider.add_span_processor(observer)
    else:
        with active._lock:
            active._span_processors = (
                observer,
                *(p for p in active._span_processors if p is not observer),
            )


def _remove(provider, observer):
    active = getattr(provider, "_active_span_processor", None)
    if active is not None:
        with active._lock:
            active._span_processors = tuple(
                p for p in active._span_processors if p is not observer
            )


class _Call:
    def __init__(self, manager, operation, instance, args, kwargs, signature, options):
        self.manager = manager
        self.operation = operation
        self.kind = "task"
        self.done = False
        self.finishing = False
        self.error = False
        self.success = False
        self.span = None
        self.keys = set()
        self.input = self.output = None
        self.has_output = False
        self.marker = {}
        self.policy = CapturePolicy(
            manager.observer, manager.capture, options.get("context", manager.context)
        )
        self.was_recording = False
        native_options = dict(options)
        native_name = native_options.pop("native_name", "chroma." + operation)
        for key in ("record_exception", "set_status_on_exception", "end_on_exit"):
            native_options.pop(key, None)
        native_options.setdefault(
            "context",
            manager.context if manager.context is not None else self.policy.context,
        )
        attrs = dict(native_options.pop("attributes", None) or {})
        attrs.update(
            {
                RESPAN_LOG_TYPE: "task",
                AI.TRACELOOP_ENTITY_NAME: "chroma." + operation,
                AI.TRACELOOP_ENTITY_PATH: "",
                DB.DB_SYSTEM: "chroma",
                DB.DB_OPERATION: operation.rsplit(".", 1)[-1],
            }
        )
        self.span = (
            manager.observe()
            .get_tracer("opentelemetry.instrumentation.chromadb", manager.version)
            .start_span(native_name, attributes=attrs, **native_options)
        )
        self.was_recording = self.span.is_recording()
        self.marker = {
            key: value
            for key, value in (getattr(self.span, "attributes", None) or {}).items()
            if key.startswith(RESPAN_METADATA)
        }
        manager.observer.states.add(self)
        if self.check():
            try:
                bound = signature.bind_partial(instance, *args, **kwargs)
                values = {
                    key: value
                    for key, value in bound.arguments.items()
                    if key != "self"
                }
            except TypeError:
                values = {"args": args, "kwargs": kwargs}
            self.input = {"arguments": json_value(values)}
            data = native_dict(instance)
            if data is not None:
                self.input["collection"] = json_value(data)
                if type(data.get("name")) is str:
                    self.capture(DB.DB_NAME, redact_text(data["name"]))

    def safe(self, method, *args):
        if self.done:
            return None
        try:
            return getattr(self, method)(*args)
        except Exception:  # noqa: BLE001
            self.abort()
            return None

    def drop(self):
        self.input = self.output = None
        self.has_output = False

    def scrub(self):
        self.drop()
        if self.span is not None:
            attrs = getattr(self.span, "_attributes", None)
            if attrs is not None:
                for key in self.keys | {
                    AI.TRACELOOP_ENTITY_INPUT,
                    AI.TRACELOOP_ENTITY_OUTPUT,
                    ERROR_MESSAGE,
                }:
                    if type(attrs) is BoundedAttributes:
                        attrs._dict.pop(key, None)
                    else:
                        attrs.pop(key, None)
            self.span._events = BoundedList(maxlen=0)
            if self.span.status.status_code == StatusCode.ERROR:
                self.span._status = Status(StatusCode.ERROR)

    def clean(self):
        try:
            self.scrub()
        except Exception:  # noqa: BLE001
            self.drop()
            if self.span is not None:
                self.span._attributes = BoundedAttributes(
                    attributes={
                        **self.marker,
                        RESPAN_LOG_TYPE: "task",
                        AI.TRACELOOP_ENTITY_NAME: "chroma." + self.operation,
                        AI.TRACELOOP_ENTITY_PATH: "",
                        DB.DB_SYSTEM: "chroma",
                        DB.DB_OPERATION: self.operation.rsplit(".", 1)[-1],
                    },
                    immutable=False,
                )
                self.span._events = BoundedList(maxlen=0)
                self.span._status = Status(
                    StatusCode.ERROR
                    if self.error or self.span.status.status_code == StatusCode.ERROR
                    else StatusCode.UNSET
                )

    def release(self):
        self.drop()
        self.keys.clear()
        self.policy.allowed = False
        self.policy.context = self.policy.supplied = None
        self.policy.parent = trace.INVALID_SPAN
        self.policy.supplied_parent = None

    def abort(self):
        if self.done:
            return
        self.policy.allowed = False
        self.finishing = True
        try:
            self.clean()
        except Exception:  # noqa: BLE001
            self.drop()
        try:
            if self.span is not None:
                self.span.end()
        except Exception:  # noqa: BLE001
            logger.debug("Chroma observer discard failed")
        finally:
            self.done = True
            self.finishing = False
            self.manager.observer.states.discard(self)
            self.release()

    def check(self):
        allowed = self.policy.check() and (
            self.span.is_recording() or (self.finishing and self.was_recording)
        )
        if not allowed:
            self.scrub()
        return allowed

    def capture(self, key, value):
        if self.check():
            self.keys.add(key)
            try:
                self.span.set_attribute(key, value)
            finally:
                self.keys.add(key)
            self.check()

    def result(self, value):
        self.success = True
        if self.check():
            self.output = json_value(value)
            self.has_output = True

    def failure(self, error):
        self.error = True
        self.span.set_status(Status(StatusCode.ERROR))
        self.span.set_attribute(
            ERROR_TYPE,
            type.__dict__["__name__"].__get__(type(error), type(type(error))),
        )
        if self.check():
            for argument in BaseException.args.__get__(error):
                if type(argument) is str:
                    self.capture(ERROR_MESSAGE, redact_text(argument))
                    break

    def finish(self):
        if self.done or self.finishing:
            return
        self.finishing = True
        try:
            if (
                self.success
                and not self.error
                and self.span.status.status_code != StatusCode.ERROR
            ):
                self.span.set_status(Status(StatusCode.OK))
            if self.check():
                if self.input is not None:
                    self.capture(AI.TRACELOOP_ENTITY_INPUT, json_string(self.input))
                if self.has_output:
                    self.capture(AI.TRACELOOP_ENTITY_OUTPUT, json_string(self.output))
            self.span.set_attribute(RESPAN_LOG_TYPE, "task")
            self.span.set_attribute(
                AI.TRACELOOP_ENTITY_NAME, "chroma." + self.operation
            )
            self.span.set_attribute(DB.DB_SYSTEM, "chroma")
            self.span.set_attribute(DB.DB_OPERATION, self.operation.rsplit(".", 1)[-1])
            self.span.set_attributes(self.marker)
            self.check()
        except Exception:  # noqa: BLE001
            self.policy.allowed = False
            try:
                self.clean()
            except Exception:  # noqa: BLE001
                self.drop()
        finally:
            try:
                self.span.end()
            except Exception:  # noqa: BLE001
                self.policy.allowed = False
                try:
                    self.clean()
                except Exception:  # noqa: BLE001
                    self.drop()
                logger.debug("Chroma observer end failed")
            finally:
                self.done = True
                self.finishing = False
                self.manager.observer.states.discard(self)
                self.release()


class _Scope:
    def __init__(self, manager, operation, instance, args, kwargs, signature, options):
        self.manager = manager
        self.values = (operation, instance, args, kwargs, signature, options)
        self.state = None
        self.token = self.current = None
        self.ambient = None

    def __enter__(self):
        self.ambient = context.get_current()
        state = _Call.__new__(_Call)
        try:
            state.__init__(self.manager, *self.values)
            self.state = state
            state.safe("check")
            self.token = context.attach(trace.set_span_in_context(state.span))
        except Exception:  # noqa: BLE001
            if hasattr(state, "policy"):
                state.abort()
            try:
                self.manager.original_attach(self.ambient)
            except Exception:  # noqa: BLE001
                logger.debug("Chroma observer attach restore failed")
        self.current = _CURRENT.set(self.state)
        return self.state.span if self.state is not None else trace.INVALID_SPAN

    def __exit__(self, kind, error, tb):
        if self.state is not None:
            if error is not None:
                self.state.safe("failure", error)
            self.state.safe("check")
            self.state.safe("finish")
        try:
            if self.token is not None:
                context.detach(self.token)
        except Exception:  # noqa: BLE001
            try:
                self.manager.original_detach(self.token)
            except Exception:  # noqa: BLE001
                logger.debug("Chroma observer detach restore failed")
        finally:
            if self.current is not None:
                _CURRENT.reset(self.current)
        return False


class _Tracer:
    def __init__(self, manager):
        self.manager = manager

    def start_as_current_span(self, name, **options):
        request = _REQUEST.get()
        if request is None:
            raise RuntimeError("Chroma delegate span without native request")
        return _Scope(self.manager, *request, {**options, "native_name": name})


class _Provider:
    def __init__(self, manager):
        self.manager = manager

    def get_tracer(self, *args, **kwargs):
        return _Tracer(self.manager)


def _native_wrapper(manager, original, operation, signature):
    if inspect.iscoroutinefunction(original):

        @wraps(original)
        async def asynchronous(instance, *args, **kwargs):
            if not manager.enabled or manager.suppressed():
                return await original(instance, *args, **kwargs)
            with _Scope(manager, operation, instance, args, kwargs, signature, {}):
                state = _CURRENT.get()
                try:
                    result = await original(instance, *args, **kwargs)
                except BaseException as error:
                    if state is not None:
                        state.safe("failure", error)
                    raise
                if state is not None:
                    state.safe("result", result)
                return result

        return asynchronous

    @wraps(original)
    def synchronous(instance, *args, **kwargs):
        if not manager.enabled or manager.suppressed():
            return original(instance, *args, **kwargs)
        with _Scope(manager, operation, instance, args, kwargs, signature, {}):
            state = _CURRENT.get()
            try:
                result = original(instance, *args, **kwargs)
            except BaseException as error:
                if state is not None:
                    state.safe("failure", error)
                raise
            if state is not None:
                state.safe("result", result)
            return result

    return synchronous


class _Manager:
    def __init__(self, capture, provider, supplied, callback):
        self.capture = capture
        self.provider = provider
        self.context = supplied
        self.callback = callback
        self.enabled = True
        self.observer = PrivacyObserver()
        self.providers = []
        self.patches = []
        self.delegate = None
        self.version = None
        runtime = context._RUNTIME_CONTEXT
        self.original_attach = runtime.attach
        self.original_detach = runtime.detach

    def suppressed(self):
        return suppressed() or (self.context is not None and suppressed(self.context))

    def observe(self):
        provider = (
            self.provider if self.provider is not None else trace.get_tracer_provider()
        )
        if not any(provider is item for item in self.providers) and hasattr(
            provider, "add_span_processor"
        ):
            _register(provider, self.observer)
            self.providers.append(provider)
        return provider

    def patch(self, obj, name, replacement):
        namespace = (
            type.__dict__["__dict__"].__get__(obj, type(obj))
            if isinstance(obj, type)
            else native_storage(obj, type(obj))
        )
        present = namespace is not None and name in namespace
        before = namespace.get(name) if present else None
        self.patches.append((obj, name, replacement, present, before))
        setattr(obj, name, replacement)

    def close(self):
        self.enabled = False
        for state in list(self.observer.states):
            state.policy.allowed = False
            state.safe("finish")
        for obj, name, replacement, present, before in reversed(self.patches):
            if inspect.getattr_static(obj, name, None) is replacement:
                if present:
                    setattr(obj, name, before)
                else:
                    delattr(obj, name)
        for provider in self.providers:
            _remove(provider, self.observer)
        self.providers.clear()
        if self.delegate is not None:
            self.delegate._is_instrumented_by_opentelemetry = False
        if hasattr(self, "config") and self.config.exception_logger is self.callback:
            self.config.exception_logger = self.previous_callback


class ChromaInstrumentor:
    name = "chroma"

    def __init__(
        self,
        *,
        capture_content=True,
        tracer_provider=None,
        context=None,
        exception_logger=None,
    ):
        self.capture_content = capture_content is True
        self.provider = tracer_provider
        self.context = context
        self.callback = exception_logger
        self._is_instrumented = False

    def activate(self):
        global _MANAGER
        with _LOCK:
            if self._is_instrumented:
                return
            tracer = getattr(RespanTracer, "_instance", None)
            if tracer is not None and not getattr(tracer, "is_enabled", True):
                return
            if _OWNERS:
                if (
                    _MANAGER.capture is not self.capture_content
                    or _MANAGER.provider is not self.provider
                    or _MANAGER.context is not self.context
                    or _MANAGER.callback is not self.callback
                ):
                    raise RuntimeError("Chroma instrumentation configuration conflict")
                _OWNERS.add(self)
                self._is_instrumented = True
                return
            upstream = importlib.import_module("opentelemetry.instrumentation.chromadb")
            cls = upstream.ChromaInstrumentor
            prior = cls.__new__(cls)
            if prior.is_instrumented_by_opentelemetry:
                logger.warning("Foreign Chroma instrumentation is active")
                return
            manager = _Manager(
                self.capture_content, self.provider, self.context, self.callback
            )
            try:
                initialize_native_types()
                manager.observe()
                config = importlib.import_module(
                    "opentelemetry.instrumentation.chromadb.config"
                ).Config
                manager.config = config
                manager.previous_callback = config.exception_logger
                manager.version = importlib.import_module(
                    "opentelemetry.instrumentation.chromadb.version"
                ).__version__
                specifications = {}
                before = []
                for spec in upstream.WRAPPED_METHODS:
                    target = getattr(spec["package"], spec["object"], None)
                    if target is not None:
                        original = inspect.getattr_static(target, spec["method"])
                        signature = inspect.signature(original)
                        specifications[id(spec)] = (signature, spec["method"])
                        before.append((target, spec["method"], original))
                factory = upstream._wrap

                def delegate_factory(tracer, spec):
                    base = factory(tracer, spec)
                    signature, method = specifications[id(spec)]

                    def wrapper(wrapped, instance, args, kwargs):
                        if not manager.enabled or manager.suppressed():
                            return wrapped(*args, **kwargs)
                        operation = (
                            "collection." + method
                            if spec["object"] == "Collection"
                            else "segment." + method
                        )
                        token = _REQUEST.set(
                            (operation, instance, args, kwargs, signature)
                        )
                        held = []
                        failed = []

                        def observed(*ignored_args, **ignored_kwargs):
                            state = _CURRENT.get()
                            try:
                                result = wrapped(*args, **kwargs)
                            except BaseException as error:  # noqa: BLE001
                                failed.append(
                                    (error, BaseException.__traceback__.__get__(error))
                                )
                                if state is not None:
                                    state.safe("failure", error)
                                # The native error is raised after the delegate has
                                # exited. Its own diagnostic coercion never runs.
                                return {}
                            held.append(result)
                            if state is not None:
                                state.safe("result", result)
                            # Upstream event summaries are replaced by complete
                            # capture-gated canonical payloads on its native span.
                            return {}

                        try:
                            base(observed, instance, (), {})
                        finally:
                            _REQUEST.reset(token)
                        if failed:
                            error, tb = failed[0]
                            raise BaseException.with_traceback(error, tb)
                        return held[0]

                    return wrapper

                manager.patch(upstream, "_wrap", delegate_factory)
                delegate = cls(exception_logger=self.callback)
                manager.delegate = delegate
                delegate.instrument(tracer_provider=_Provider(manager))
                if not delegate.is_instrumented_by_opentelemetry:
                    raise RuntimeError("Released Chroma delegate did not activate")
                for target, name, original in before:
                    owned = inspect.getattr_static(target, name)
                    manager.patches.append((target, name, owned, True, original))
                from chromadb.api.client import Client
                from chromadb.api.models.Collection import Collection

                targets = [
                    (Client, CLIENT_METHODS, "client."),
                    (Collection, COLLECTION_METHODS, "collection."),
                ]
                try:
                    from chromadb.api.async_client import AsyncClient
                    from chromadb.api.models.AsyncCollection import AsyncCollection
                except ImportError:
                    pass  # Native0.5.0 exposes synchronous clients only.
                else:
                    targets.extend(
                        [
                            (AsyncClient, CLIENT_METHODS, "client."),
                            (AsyncCollection, COLLECTION_METHODS, "collection."),
                        ]
                    )
                for target, methods, prefix in targets:
                    for name in methods:
                        if target is Collection and any(
                            t is target and n == name for t, n, _ in before
                        ):
                            continue
                        original = inspect.getattr_static(target, name, None)
                        if original is not None:
                            manager.patch(
                                target,
                                name,
                                _native_wrapper(
                                    manager,
                                    original,
                                    prefix + name,
                                    inspect.signature(original),
                                ),
                            )
                runtime = context._RUNTIME_CONTEXT
                original_detach = runtime.detach

                def detach(token):
                    if not explicit_capture():
                        manager.observer.notice()
                    return original_detach(token)

                manager.patch(runtime, "detach", detach)
            except BaseException:
                # Capture partial upstream patches even when its installer raises.
                if "before" in locals():
                    for target, name, original in before:
                        now = inspect.getattr_static(target, name)
                        if now is not original and not any(
                            obj is target and field == name
                            for obj, field, *_ in manager.patches
                        ):
                            manager.patches.append((target, name, now, True, original))
                manager.close()
                raise
            _MANAGER = manager
            _OWNERS.add(self)
            self._is_instrumented = True

    def deactivate(self):
        global _MANAGER
        with _LOCK:
            if not self._is_instrumented:
                return
            _OWNERS.discard(self)
            self._is_instrumented = False
            if not _OWNERS and _MANAGER is not None:
                _MANAGER.close()
                _MANAGER = None
