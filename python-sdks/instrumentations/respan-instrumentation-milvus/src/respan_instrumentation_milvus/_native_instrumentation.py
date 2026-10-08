"""Retain upstream sync Milvus spans and observe complete native payloads."""

from __future__ import annotations

import importlib
import inspect
import logging
import threading
import weakref
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from types import FunctionType

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

from ._constants import CLIENT_METHODS
from ._policy import CapturePolicy, PrivacyObserver, explicit_capture, suppressed
from ._serialization import (
    initialize_native_types,
    json_string,
    json_value,
    native_extra,
    native_storage,
    redact_text,
)

logger = logging.getLogger(__name__)
_LOCK = threading.RLock()
_OWNERS = set()
_MANAGER = None
_REQUEST = ContextVar("respan_milvus_native_request", default=None)
_CURRENT = ContextVar("respan_milvus_native_call", default=None)


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
        self.error_type = None
        self.success = False
        self.span = None
        self.keys = set()
        self.input = self.output = None
        self.has_output = False
        self.marker = {}
        self.deferred = False
        self.batches = []
        self.instance = instance
        self.policy = CapturePolicy(
            manager.observer, manager.capture, options.get("context", manager.context)
        )
        self.was_recording = False
        native_options = dict(options)
        native_name = native_options.pop("native_name", "milvus." + operation)
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
                AI.TRACELOOP_ENTITY_NAME: "milvus." + operation,
                AI.TRACELOOP_ENTITY_PATH: "",
                DB.DB_SYSTEM: "milvus",
                DB.DB_OPERATION: operation.rsplit(".", 1)[-1],
            }
        )
        self.span = (
            manager.observe()
            .get_tracer("opentelemetry.instrumentation.milvus", manager.version)
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
            collection = values.get("collection_name")
            if type(collection) is str:
                self.capture(DB.DB_NAME, redact_text(collection))

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
        self.batches.clear()

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
                        AI.TRACELOOP_ENTITY_NAME: "milvus." + self.operation,
                        AI.TRACELOOP_ENTITY_PATH: "",
                        DB.DB_SYSTEM: "milvus",
                        DB.DB_OPERATION: self.operation.rsplit(".", 1)[-1],
                    },
                    immutable=False,
                )
                if self.error_type is not None:
                    self.span._attributes[ERROR_TYPE] = self.error_type
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
        self.manager.forget(self)
        self.instance = None

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
            logger.debug("Milvus observer discard failed")
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
        if self.manager.defer(self, value):
            return
        self.success = True
        if self.check():
            self.output = json_value(value)
            extra = native_extra(value)
            if extra is not None:
                self.capture(RESPAN_METADATA + ".milvus.result", json_string(extra))
            self.has_output = True

    def handle_result(self, method, kind, result):
        _handle_result(self, method, kind, result)

    def defer_handle(self, value):
        self.manager.defer(self, value)

    def failure(self, error):
        self.error = True
        self.error_type = type.__dict__["__name__"].__get__(
            type(error), type(type(error))
        )
        self.span.set_status(Status(StatusCode.ERROR))
        self.span.set_attribute(ERROR_TYPE, self.error_type)
        if self.check():
            message = None
            for argument in BaseException.args.__get__(error):
                if type(argument) is str:
                    message = argument
                    break
            if message is None:
                from pymilvus.exceptions import MilvusException

                data = native_storage(error, MilvusException)
                if data is not None and type(data.get("_message")) is str:
                    message = data["_message"]
            if message is not None:
                self.capture(ERROR_MESSAGE, redact_text(message))

    def finish(self, force=False):
        if self.done or self.finishing or (self.deferred and not force):
            return
        self.finishing = True
        try:
            if (
                self.success
                and not self.error
                and self.span.status.status_code != StatusCode.ERROR
            ):
                self.span.set_status(Status(StatusCode.OK))
            self.span.set_attributes(self.marker)
            self.span.set_attribute(RESPAN_LOG_TYPE, "task")
            self.span.set_attribute(
                AI.TRACELOOP_ENTITY_NAME, "milvus." + self.operation
            )
            self.span.set_attribute(DB.DB_SYSTEM, "milvus")
            self.span.set_attribute(DB.DB_OPERATION, self.operation.rsplit(".", 1)[-1])
            self.span.set_attribute(AI.TRACELOOP_ENTITY_PATH, "")
            if self.error_type is not None:
                self.span.set_attribute(ERROR_TYPE, self.error_type)
            if self.check():
                if self.input is not None:
                    self.capture(AI.TRACELOOP_ENTITY_INPUT, json_string(self.input))
                if self.has_output:
                    self.capture(AI.TRACELOOP_ENTITY_OUTPUT, json_string(self.output))
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
                logger.debug("Milvus observer end failed")
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
                logger.debug("Milvus observer attach restore failed")
        self.current = _CURRENT.set(self.state)
        return self.state.span if self.state is not None else trace.INVALID_SPAN

    def __exit__(self, kind, error, tb):
        if self.state is not None:
            self.state.safe("check")
            self.state.safe("finish")
        try:
            if self.token is not None:
                context.detach(self.token)
        except Exception:  # noqa: BLE001
            try:
                self.manager.original_detach(self.token)
            except Exception:  # noqa: BLE001
                logger.debug("Milvus observer detach restore failed")
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
            raise RuntimeError("Milvus delegate span without native request")
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


@contextmanager
def _consume(state):
    token = current = None
    ambient = context.get_current()
    try:
        state.safe("check")
        if not state.done:
            current = _CURRENT.set(state)
            try:
                token = context.attach(trace.set_span_in_context(state.span))
            except Exception:  # noqa: BLE001 - native consumption must survive observer faults.
                state.abort()
                try:
                    state.manager.original_attach(ambient)
                except Exception:  # noqa: BLE001 - native consumption takes precedence.
                    logger.debug("Milvus consumption context restoration failed")
        yield
    finally:
        state.safe("check")
        if token is not None:
            try:
                context.detach(token)
            except Exception:  # noqa: BLE001 - restore the native ambient context.
                try:
                    state.manager.original_detach(token)
                except Exception:  # noqa: BLE001 - native outcomes take precedence.
                    logger.debug("Milvus consumption detach restoration failed")
        if current is not None:
            _CURRENT.reset(current)


def _handle_result(state, method, kind, result):
    if method in ("close", "cancel"):
        state.safe("finish", True)
    elif kind == "iterator":
        if state.check():
            state.batches.append(json_value(result))
            state.output = state.batches
            state.has_output = True
        mro = type.__dict__["__mro__"].__get__(type(result), type(type(result)))
        if any(base is list for base in mro) and list.__len__(result) == 0:
            state.success = True
            state.safe("finish", True)
    elif result is not None:
        state.deferred = False
        state.safe("result", result)
        state.safe("finish", True)


def _handle_wrapper(manager, original, method, kind):
    if inspect.iscoroutinefunction(original):

        @wraps(original)
        async def asynchronous(instance, *args, **kwargs):
            state = manager.handle(instance)
            if state is None or state.done:
                return await original(instance, *args, **kwargs)
            try:
                with _consume(state):
                    result = await original(instance, *args, **kwargs)
            except BaseException as error:
                state.safe("failure", error)
                state.safe("finish", True)
                raise
            state.safe("handle_result", method, kind, result)
            return result

        return asynchronous

    @wraps(original)
    def synchronous(instance, *args, **kwargs):
        state = manager.handle(instance)
        if (
            state is None
            or state.done
            or (method == "close" and _CURRENT.get() is state)
        ):
            return original(instance, *args, **kwargs)
        try:
            with _consume(state):
                result = original(instance, *args, **kwargs)
        except BaseException as error:
            state.safe("failure", error)
            state.safe("finish", True)
            raise
        state.safe("handle_result", method, kind, result)
        return result

    return synchronous


def _worker_wrapper(manager, original, method):
    @wraps(original)
    def worker(instance, *args, **kwargs):
        state = manager.handle(instance) if method == "run" else _CURRENT.get()
        if state is None or state.done:
            if method == "start" and manager.suppressed():
                manager.suppressed_workers[id(instance)] = weakref.ref(instance)
            if (
                method == "run"
                and manager.suppressed_workers.get(id(instance), lambda: None)()
                is instance
            ):
                token = None
                try:
                    token = manager.original_attach(
                        context.set_value(context._SUPPRESS_INSTRUMENTATION_KEY, True)
                    )
                    return original(instance, *args, **kwargs)
                finally:
                    manager.suppressed_workers.pop(id(instance), None)
                    if token is not None:
                        manager.original_detach(token)
            return original(instance, *args, **kwargs)
        if method == "start":
            state.safe("defer_handle", instance)
            return original(instance, *args, **kwargs)
        with _consume(state):
            return original(instance, *args, **kwargs)

    return worker


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
        self.handles = {}
        self.suppressed_workers = {}
        self.handle_types = []
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

    def forget(self, state):
        for identity, (_, owned) in list(self.handles.items()):
            if owned is state:
                self.handles.pop(identity, None)

    def defer(self, state, value):
        mro = type.__dict__["__mro__"].__get__(type(value), type(type(value)))
        if not any(base is cls for cls, _ in self.handle_types for base in mro):
            return False
        state.deferred = True
        state.success = False
        identity = id(value)

        def abandoned(reference):
            entry = self.handles.get(identity)
            if entry is not None and entry[0] is reference:
                self.handles.pop(identity, None)
                state.safe("finish", True)

        self.handles[identity] = (weakref.ref(value, abandoned), state)
        return True

    def handle(self, value):
        entry = self.handles.get(id(value))
        return entry[1] if entry is not None and entry[0]() is value else None

    def install_client_close(self, target):
        original = inspect.getattr_static(target, "close", None)
        if type(original) is not FunctionType:
            return
        if inspect.iscoroutinefunction(original):

            @wraps(original)
            async def close(client, *args, **kwargs):
                try:
                    return await original(client, *args, **kwargs)
                finally:
                    self.close_client_states(client)
        else:

            @wraps(original)
            def close(client, *args, **kwargs):
                try:
                    return original(client, *args, **kwargs)
                finally:
                    self.close_client_states(client)

        self.patch(target, "close", close)

    def close_client_states(self, client):
        for state in list(self.observer.states):
            if state.instance is client:
                state.safe("finish", True)

    def install_handles(self):
        from pymilvus.orm.iterator import QueryIterator, SearchIterator

        classes = [(QueryIterator, "iterator"), (SearchIterator, "iterator")]
        for module, name in (
            ("pymilvus.milvus_client.optimize_task", "OptimizeTask"),
            ("pymilvus.milvus_client.async_optimize_task", "AsyncOptimizeTask"),
        ):
            try:
                classes.append((getattr(importlib.import_module(module), name), "task"))
            except ImportError:
                pass
        self.handle_types = classes
        for target, kind in classes:
            for method in (
                ("next", "close") if kind == "iterator" else ("result", "cancel")
            ):
                original = inspect.getattr_static(target, method, None)
                if type(original) is FunctionType:
                    self.patch(
                        target, method, _handle_wrapper(self, original, method, kind)
                    )
            if kind == "task":
                for method in ("start", "run"):
                    original = inspect.getattr_static(target, method, None)
                    if type(original) is FunctionType:
                        self.patch(
                            target, method, _worker_wrapper(self, original, method)
                        )

    def close(self):
        self.enabled = False
        for state in list(self.observer.states):
            state.policy.allowed = False
            state.safe("finish", True)
        self.handles.clear()
        self.suppressed_workers.clear()
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


class MilvusInstrumentor:
    name = "milvus"

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
                    raise RuntimeError("Milvus instrumentation configuration conflict")
                _OWNERS.add(self)
                self._is_instrumented = True
                return
            upstream = importlib.import_module("opentelemetry.instrumentation.milvus")
            cls = upstream.MilvusInstrumentor
            prior = cls.__new__(cls)
            if prior.is_instrumented_by_opentelemetry:
                logger.warning("Foreign Milvus instrumentation is active")
                return
            manager = _Manager(
                self.capture_content, self.provider, self.context, self.callback
            )
            try:
                initialize_native_types()
                manager.observe()
                config = importlib.import_module(
                    "opentelemetry.instrumentation.milvus.config"
                ).Config
                manager.config = config
                manager.previous_callback = config.exception_logger
                manager.version = importlib.import_module(
                    "opentelemetry.instrumentation.milvus.version"
                ).__version__
                specifications = {}
                before = []
                available = []
                for spec in upstream.WRAPPED_METHODS:
                    target = getattr(spec["package"], spec["object"], None)
                    if target is not None:
                        original = inspect.getattr_static(target, spec["method"], None)
                        if original is None:
                            continue
                        signature = inspect.signature(original)
                        specifications[id(spec)] = (signature, spec["method"])
                        before.append((target, spec["method"], original))
                        available.append(spec)
                if len(available) != len(upstream.WRAPPED_METHODS):
                    manager.patch(upstream, "WRAPPED_METHODS", available)
                factory = upstream._wrap

                def delegate_factory(tracer, *factory_args):
                    *metrics, spec = factory_args
                    base = factory(tracer, *metrics, spec)
                    signature, method = specifications[id(spec)]

                    def wrapper(wrapped, instance, args, kwargs):
                        if not manager.enabled or manager.suppressed():
                            return wrapped(*args, **kwargs)
                        operation = "client." + method
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
                        except Exception:  # noqa: BLE001 - delegate observer faults preserve native outcomes.
                            if not held and not failed:
                                observed()
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
                    raise RuntimeError("Released Milvus delegate did not activate")
                for target, name, original in before:
                    owned = inspect.getattr_static(target, name)
                    manager.patches.append((target, name, owned, True, original))
                import pymilvus

                targets = [pymilvus.MilvusClient]
                asynchronous = getattr(pymilvus, "AsyncMilvusClient", None)
                if asynchronous is not None:
                    targets.append(asynchronous)
                for target in targets:
                    for name in CLIENT_METHODS:
                        if any(t is target and n == name for t, n, _ in before):
                            continue
                        original = inspect.getattr_static(target, name, None)
                        if type(original) is FunctionType:
                            manager.patch(
                                target,
                                name,
                                _native_wrapper(
                                    manager,
                                    original,
                                    "client." + name,
                                    inspect.signature(original),
                                ),
                            )
                    manager.install_client_close(target)
                manager.install_handles()
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
