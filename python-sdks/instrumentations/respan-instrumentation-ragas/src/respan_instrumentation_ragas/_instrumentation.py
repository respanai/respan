"""Sampled native Ragas evaluations, metrics and experiment lifecycles."""

from __future__ import annotations

import contextvars
import functools
import importlib
import inspect
import logging
import threading
import types
import weakref
from contextlib import contextmanager

from opentelemetry import context, trace
from opentelemetry.attributes import BoundedAttributes
from opentelemetry.sdk.util import BoundedList
from opentelemetry.semconv._incubating.attributes.error_attributes import ERROR_MESSAGE
from opentelemetry.semconv.attributes.error_attributes import ERROR_TYPE
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
_CURRENT = contextvars.ContextVar("respan_ragas_call", default=None)
_MISSING = object()


def _family(kind, args, kwargs):
    if kind == "evaluation":
        return kind, id(args[0] if args else kwargs.get("dataset"))
    if kind.startswith("metric"):
        return "metric", id(args[0]) if args else None
    return kind, id(args[0]) if args else None


class _Call:
    def __init__(self, manager, kind, args, kwargs):
        self.manager = manager
        self.kind = kind
        self.family = _family(kind, args, kwargs)
        self.name = "ragas." + kind
        self.done = False
        self.finishing = False
        self.allowed = False
        self.input = self.output = None
        self.has_output = False
        self.keys = set()
        self.error_type = None
        self.span = None
        self.policy = None
        self.structural = {}
        try:
            provider = manager.refresh()
            self.policy = manager.policy
            initial = (
                manager.capture
                and permitted()
                and self.policy.enroll(trace.get_current_span())
            )
            self.span = provider.get_tracer("ragas").start_span(
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
                self.entity(args, kwargs)
                self.span.set_attribute(AI.TRACELOOP_ENTITY_NAME, self.name)
                self.span.set_attribute(AI.TRACELOOP_ENTITY_PATH, "")
            manager.states.add(self)
            if self.allowed:
                positional = (
                    args[1:] if kind.startswith(("metric", "experiment")) else args
                )
                self.input = json_string({"args": positional, "kwargs": kwargs})
                self.check()
        except Exception:
            self.discard()
            raise

    def entity(self, args, kwargs):
        if self.kind.startswith("metric") and args:
            for base in self.manager.metric_bases:
                data = native_storage(args[0], base)
                if data is not None:
                    name = data.get("name")
                    if type(name) is str:
                        self.name = redact_text(name) + (
                            ".batch" if self.kind == "metric_batch" else ""
                        )
                    break
        elif self.kind.startswith("experiment") and args:
            data = native_storage(args[0], self.manager.experiment_class) or {}
            name = data.get("__name__")
            if type(name) is str:
                self.name = "ragas." + self.kind + "." + redact_text(name)

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
            self.output = json_string(result)
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
                if type(value) is str:
                    text = redact_text(value)
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
                    self.capture(AI.TRACELOOP_ENTITY_INPUT, self.input)
                if self.has_output:
                    self.capture(AI.TRACELOOP_ENTITY_OUTPUT, self.output)
            if self.span.is_recording():
                if self.error_type is None:
                    self.span.set_status(Status(StatusCode.OK))
                self.span.set_attributes(self.structural)
                self.span.set_attribute(RESPAN_LOG_TYPE, "task")
                if self.error_type:
                    self.span.set_attribute(ERROR_TYPE, self.error_type)
                self.check()
        except Exception:  # noqa: BLE001 - discard telemetry faults without replacing results.
            self.discard()
            return
        try:
            self.span.end()
        except Exception:  # noqa: BLE001 - observer span end cannot replace the native result.
            logger.debug("Ragas telemetry span end failed")
        finally:
            self.done = True
            self.finishing = False
            self.input = self.output = None
            self.has_output = False
            self.manager.states.discard(self)
            if self.policy:
                self.policy.on_end(self.span)

    def discard(self):
        self.allowed = False
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
                        RESPAN_LOG_TYPE: "task",
                        AI.TRACELOOP_ENTITY_NAME: self.name,
                    },
                    immutable=False,
                )
                self.span._events._dq.clear()
                self.span._status = (
                    Status(StatusCode.ERROR)
                    if self.error_type
                    else Status(StatusCode.UNSET)
                )
                if self.error_type:
                    self.span._attributes[ERROR_TYPE] = self.error_type
        except Exception:  # noqa: BLE001 - guaranteed observer cleanup best effort.
            logger.debug("Ragas telemetry discard failed")
        try:
            if self.span:
                self.span.end()
        except Exception:  # noqa: BLE001 - native SDK behavior has priority.
            logger.debug("Ragas telemetry discard end failed")
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
            logger.debug("Ragas telemetry observation failed")
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
                        logger.debug("Ragas telemetry attach restoration failed")
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
                        logger.debug("Ragas telemetry detach failed")
        if current is not None:
            _CURRENT.reset(current)


def _start(manager, kind, args, kwargs):
    try:
        return _Call(manager, kind, args, kwargs)
    except Exception:  # noqa: BLE001 - observer startup errors cannot alter vendor calls.
        logger.debug("Ragas telemetry startup failed")
        return None


def _wrapper(original, kind):
    if inspect.iscoroutinefunction(original):

        @functools.wraps(original)
        async def wrapper(*args, **kwargs):
            manager = _MANAGER
            active = _CURRENT.get()
            if (
                not manager
                or not manager.enabled
                or suppressed()
                or (active and active.family == _family(kind, args, kwargs))
            ):
                return await original(*args, **kwargs)
            state = _start(manager, kind, args, kwargs)
            try:
                with _scope(state):
                    result = await original(*args, **kwargs)
            except BaseException as error:
                if state:
                    state.safe(state.error, error)
                    state.safe(state.finish)
                raise
            if state:
                if kind == "evaluation" and type(result) is manager.executor_class:
                    state.safe(manager.defer, result, state)
                else:
                    state.safe(state.observe, result)
                    state.safe(state.finish)
            return result
    else:

        @functools.wraps(original)
        def wrapper(*args, **kwargs):
            manager = _MANAGER
            active = _CURRENT.get()
            if (
                not manager
                or not manager.enabled
                or suppressed()
                or (active and active.family == _family(kind, args, kwargs))
            ):
                return original(*args, **kwargs)
            state = _start(manager, kind, args, kwargs)
            try:
                with _scope(state):
                    result = original(*args, **kwargs)
            except BaseException as error:
                if state:
                    state.safe(state.error, error)
                    state.safe(state.finish)
                raise
            if state:
                if kind == "evaluation" and type(result) is manager.executor_class:
                    state.safe(manager.defer, result, state)
                else:
                    state.safe(state.observe, result)
                    state.safe(state.finish)
            return result

    return wrapper


class _Manager:
    def __init__(self, provider, capture):
        self.provider = provider
        self.capture = capture
        self.policy = None
        self.bound_provider = None
        self.enabled = True
        self.states = weakref.WeakSet()
        self.patches = []
        self.pending = {}
        self.metric_bases = []
        self.experiment_class = self.executor_class = None

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
        namespace = (
            type.__dict__["__dict__"].__get__(obj, type(obj))
            if isinstance(obj, type)
            else vars(obj)
        )
        present = name in namespace
        stored = namespace.get(name)
        original = inspect.getattr_static(obj, name, _MISSING)
        self.patches.append((obj, name, original, replacement, present, stored))
        setattr(obj, name, replacement)

    def wrap(self, obj, name, kind):
        if any(p[0] is obj and p[1] == name for p in self.patches):
            return
        original = inspect.getattr_static(obj, name, None)
        # Class access is telemetry: only known native function storage is read.
        # Unknown descriptors must not execute user getters during activation.
        if type(original) is types.FunctionType:
            self.patch(obj, name, _wrapper(original, kind))

    def metric_class(self, cls):
        names = type.__dict__["__dict__"].__get__(cls, type(cls))
        for name in (
            "single_turn_score",
            "single_turn_ascore",
            "multi_turn_score",
            "multi_turn_ascore",
            "score",
            "ascore",
            "batch_score",
            "abatch_score",
        ):
            if name in names:
                self.wrap(cls, name, "metric_batch" if "batch" in name else "metric")

    def metric_hook(self, base):
        original = inspect.getattr_static(base, "__init_subclass__")
        manager = self

        def hook(cls, **kwargs):
            original.__get__(None, cls)(**kwargs)
            if manager.enabled:
                manager.metric_class(cls)

        self.patch(base, "__init_subclass__", classmethod(hook))

    def install(self):
        ragas = importlib.import_module("ragas")
        evaluation = importlib.import_module("ragas.evaluation")
        metrics = importlib.import_module("ragas.metrics.base")
        importlib.import_module("ragas.metrics.collections")
        experiment = importlib.import_module("ragas.experiment")
        executor = importlib.import_module("ragas.executor")
        self.experiment_class = experiment.ExperimentWrapper
        self.executor_class = executor.Executor
        for obj in (ragas, evaluation):
            for name in ("evaluate", "aevaluate"):
                self.wrap(obj, name, "evaluation")
        self.metric_bases = [
            metrics.SingleTurnMetric,
            metrics.MultiTurnMetric,
            metrics.SimpleBaseMetric,
        ]
        for base in self.metric_bases:
            pending = [base]
            seen = set()
            while pending:
                cls = pending.pop()
                if id(cls) in seen:
                    continue
                seen.add(id(cls))
                self.metric_class(cls)
                pending.extend(type.__subclasses__(cls))
            self.metric_hook(base)
        self.wrap(self.experiment_class, "__call__", "experiment_row")
        self.wrap(self.experiment_class, "arun", "experiment_run")
        for name in ("results", "aresults", "cancel"):
            original = getattr(self.executor_class, name)
            self.patch(self.executor_class, name, self.executor_wrapper(original, name))

    def defer(self, executor, state):
        ident = id(executor)

        def abandoned(ref):
            entry = self.pending.pop(ident, None)
            if entry:
                entry[1].discard()

        self.pending[ident] = (weakref.ref(executor, abandoned), state)

    def executor_state(self, executor):
        entry = self.pending.get(id(executor))
        return entry[1] if entry and entry[0]() is executor else None

    def executor_wrapper(self, original, name):
        manager = self
        if inspect.iscoroutinefunction(original):

            @functools.wraps(original)
            async def wrapper(executor, *args, **kwargs):
                state = manager.executor_state(executor)
                try:
                    with _scope(state):
                        result = await original(executor, *args, **kwargs)
                except BaseException as error:
                    if state:
                        state.safe(state.error, error)
                        state.safe(state.finish)
                    raise
                if state:
                    state.safe(state.observe, result)
                    state.safe(state.finish)
                    manager.pending.pop(id(executor), None)
                return result
        else:

            @functools.wraps(original)
            def wrapper(executor, *args, **kwargs):
                state = manager.executor_state(executor)
                try:
                    with _scope(state):
                        result = original(executor, *args, **kwargs)
                except BaseException as error:
                    if state:
                        state.safe(state.error, error)
                        state.safe(state.finish)
                    raise
                if state and name == "cancel":
                    state.discard()
                    manager.pending.pop(id(executor), None)
                elif state:
                    # results delegates native aresults, which owns completion.
                    if not state.done:
                        state.safe(state.observe, result)
                        state.safe(state.finish)
                    manager.pending.pop(id(executor), None)
                return result

        return wrapper

    def close(self):
        self.enabled = False
        for state in list(self.states):
            state.discard()
        self.pending.clear()
        for obj, name, original, replacement, present, stored in reversed(self.patches):
            if inspect.getattr_static(obj, name, None) is replacement:
                if present:
                    setattr(obj, name, stored)
                else:
                    delattr(obj, name)
        self.patches.clear()
        if self.policy:
            self.policy.close()


class RagasInstrumentor:
    """Instrument native Ragas evaluation, metric and experiment APIs."""

    name = "ragas"

    def __init__(self, *, capture_content=True, tracer_provider=None):
        self.capture_content = capture_content
        self.tracer_provider = tracer_provider
        self._active = False

    def activate(self):
        global _MANAGER
        if self._active:
            return
        with _LOCK:
            if _MANAGER:
                if (
                    _MANAGER.capture is not self.capture_content
                    or _MANAGER.provider is not self.tracer_provider
                ):
                    raise ValueError("Ragas instrumentation configuration conflict")
            else:
                try:
                    importlib.import_module("ragas")
                except ImportError:
                    logger.warning("Ragas SDK unavailable")
                    return
                manager = _Manager(self.tracer_provider, self.capture_content)
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
            _OWNERS.discard(self)
            self._active = False
            if not _OWNERS and _MANAGER:
                manager = _MANAGER
                _MANAGER = None
                manager.close()
