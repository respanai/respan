"""Own public Arize SDK wrappers without replacing native results."""

from __future__ import annotations

import concurrent.futures
import importlib
import inspect
import logging
from functools import wraps
from threading import RLock

from opentelemetry import trace
from opentelemetry.semconv_ai import SpanAttributes
from respan_tracing.core.tracer import RespanTracer

from respan_instrumentation_arize._constants import (
    ARIZE_CLIENT_SPECS,
    ARIZE_INSTRUMENTATION_NAME,
    ArizeClientSpec,
)
from respan_instrumentation_arize._policy import content_allowed, suppressed
from respan_instrumentation_arize._serialization import safe_json_dumps
from respan_instrumentation_arize._span_emitter import (
    Operation,
    build_arize_span_attributes,
)

logger = logging.getLogger(__name__)
_PATCH_LOCK = RLock()
_ORIGINAL_METHODS = {}
_ACTIVE_INSTANCES = 0
_ACTIVE_CONFIG = None
_GENERATION = None
_OPEN_OPERATIONS = {}
_ENDED_POLICIES = {}
_FLOW_LOCK = RLock()


def _load_client_class(spec):
    return getattr(importlib.import_module(spec.module_name), spec.class_name)


def _finish_registration(operation):
    with _FLOW_LOCK:
        identifier = operation.span.get_span_context().span_id
        _OPEN_OPERATIONS.pop(identifier, None)
        _ENDED_POLICIES[identifier] = operation
        while len(_ENDED_POLICIES) > 2048:
            _ENDED_POLICIES.pop(next(iter(_ENDED_POLICIES)))


def _begin_operation(resource, method_name, args, kwargs, setting, generation):
    with _PATCH_LOCK:
        if _ACTIVE_INSTANCES == 0 or _GENERATION is not generation or suppressed():
            return None
        current = trace.get_current_span().get_span_context()
        with _FLOW_LOCK:
            parent = (
                (
                    _OPEN_OPERATIONS.get(current.span_id)
                    or _ENDED_POLICIES.get(current.span_id)
                )
                if current.is_valid
                else None
            )
        allowed = content_allowed(setting)
        if not allowed and parent is not None:
            parent.veto()
        span = trace.get_tracer(__name__).start_span(
            f"arize.{resource}.{method_name}",
            attributes=build_arize_span_attributes(
                resource=resource, method_name=method_name
            ),
        )
        operation = Operation(
            span,
            span.is_recording() and allowed and (parent is None or parent.capture),
            setting,
            parent,
            on_finished=_finish_registration,
        )
        with _FLOW_LOCK:
            _OPEN_OPERATIONS[span.get_span_context().span_id] = operation
        if operation.capture:
            try:
                span.set_attribute(
                    SpanAttributes.TRACELOOP_ENTITY_INPUT,
                    safe_json_dumps({"args": args, "kwargs": kwargs}),
                )
            except Exception:  # noqa: BLE001 - telemetry must not alter native SDK behavior
                operation.veto()
        return operation


def _finish_native(operation, result=None, error=None):
    if operation is None:
        return
    try:
        if error is None and isinstance(result, concurrent.futures.Future):
            operation.observe_future(result)
        else:
            operation.finish(result, error)
    except BaseException:  # noqa: BLE001 - telemetry must not alter native SDK behavior
        logger.debug("Could not observe native Arize completion")


def _safe_begin(*args):
    try:
        return _begin_operation(*args)
    except Exception:  # noqa: BLE001 - telemetry must not alter native SDK behavior
        logger.debug("Could not start native Arize observation")
        return None


def _wrap_method(*, resource, method_name, original, setting, generation):
    if inspect.iscoroutinefunction(original):

        @wraps(original)
        async def call(self, *args, **kwargs):
            operation = _safe_begin(
                resource, method_name, args, kwargs, setting, generation
            )
            if operation is None:
                return await original(self, *args, **kwargs)
            try:
                with trace.use_span(
                    operation.span,
                    end_on_exit=False,
                    record_exception=False,
                    set_status_on_exception=False,
                ):
                    try:
                        result = await original(self, *args, **kwargs)
                    finally:
                        if not content_allowed(setting):
                            operation.veto()
            except BaseException as error:
                _finish_native(operation, error=error)
                raise
            _finish_native(operation, result)
            return result
    else:

        @wraps(original)
        def call(self, *args, **kwargs):
            operation = _safe_begin(
                resource, method_name, args, kwargs, setting, generation
            )
            if operation is None:
                return original(self, *args, **kwargs)
            try:
                with trace.use_span(
                    operation.span,
                    end_on_exit=False,
                    record_exception=False,
                    set_status_on_exception=False,
                ):
                    try:
                        result = original(self, *args, **kwargs)
                    finally:
                        if not content_allowed(setting):
                            operation.veto()
            except BaseException as error:
                _finish_native(operation, error=error)
                raise
            _finish_native(operation, result)
            return result

    return call


def _patch_client_class(spec, client_class, setting, generation):
    patched = False
    for method_name in spec.methods:
        original = getattr(client_class, method_name, None)
        if not callable(original):
            continue
        key = (client_class, method_name)
        if key not in _ORIGINAL_METHODS:
            wrapper = _wrap_method(
                resource=spec.resource,
                method_name=method_name,
                original=original,
                setting=setting,
                generation=generation,
            )
            _ORIGINAL_METHODS[key] = (original, wrapper)
            setattr(client_class, method_name, wrapper)
        patched = True
    return patched


def _restore_arize_clients():
    global _ACTIVE_CONFIG, _GENERATION
    for (client_class, method_name), (original, wrapper) in reversed(
        list(_ORIGINAL_METHODS.items())
    ):
        if getattr(client_class, method_name, None) is wrapper:
            setattr(client_class, method_name, original)
    _ORIGINAL_METHODS.clear()
    _GENERATION = None
    _ACTIVE_CONFIG = None
    for operation in reversed(list(_OPEN_OPERATIONS.values())):
        operation.close()
    _OPEN_OPERATIONS.clear()
    _ENDED_POLICIES.clear()


class ArizeInstrumentor:
    """One refcounted patch set shared by compatible Respan owners."""

    name = ARIZE_INSTRUMENTATION_NAME

    def __init__(
        self,
        *,
        client_specs: tuple[ArizeClientSpec, ...] = ARIZE_CLIENT_SPECS,
        capture_content: bool = True,
    ):
        self._client_specs = client_specs
        self._capture_content = capture_content
        self._is_instrumented = False

    @staticmethod
    def _is_respan_tracing_enabled():
        instance = getattr(RespanTracer, "_instance", None)
        return instance is None or bool(getattr(instance, "is_enabled", True))

    def activate(self):
        global _ACTIVE_INSTANCES, _ACTIVE_CONFIG, _GENERATION
        with _PATCH_LOCK:
            if self._is_instrumented:
                return
            if not self._is_respan_tracing_enabled():
                return
            config = (
                self._client_specs,
                self._capture_content,
                trace.get_tracer_provider(),
            )
            if _ACTIVE_INSTANCES:
                if _ACTIVE_CONFIG != config:
                    raise ValueError(
                        "Arize instrumentation is active with different settings or provider"
                    )
                _ACTIVE_INSTANCES += 1
                self._is_instrumented = True
                return
            generation = object()
            try:
                patched = 0
                for spec in self._client_specs:
                    try:
                        client_class = _load_client_class(spec)
                    except (ImportError, AttributeError):
                        continue
                    patched += bool(
                        _patch_client_class(
                            spec, client_class, self._capture_content, generation
                        )
                    )
                if not patched:
                    return
            except BaseException:
                _restore_arize_clients()
                raise
            _GENERATION = generation
            _ACTIVE_CONFIG = config
            _ACTIVE_INSTANCES = 1
            self._is_instrumented = True

    def deactivate(self):
        global _ACTIVE_INSTANCES
        with _PATCH_LOCK:
            if not self._is_instrumented:
                return
            self._is_instrumented = False
            _ACTIVE_INSTANCES -= 1
            if not _ACTIVE_INSTANCES:
                _restore_arize_clients()
