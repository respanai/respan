"""Small reusable base for native client-library instrumentation plugins."""

from __future__ import annotations

import importlib
import inspect
import logging
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
from threading import RLock
from typing import Any, ClassVar

from opentelemetry import trace
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.semconv.attributes.http_attributes import HTTP_RESPONSE_STATUS_CODE
from opentelemetry.semconv.trace import SpanAttributes as OTelSpanAttributes
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.trace import SpanKind, Status, StatusCode
from respan_sdk.constants import ERROR_MESSAGE_ATTR
from respan_sdk.constants.llm_logging import LOG_TYPE_TASK
from respan_sdk.constants.span_attributes import RESPAN_LOG_TYPE
from respan_tracing.core.tracer import RespanTracer
from wrapt import wrap_function_wrapper

from respan_instrumentation_pinecone._serialization import (
    MAX_ITEMS,
    exception_message,
    exception_status_code,
    is_sensitive_key,
    json_dumps,
    safe_text,
    safe_type_name,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PatchSpec:
    """A vendor class and the public operations that should be wrapped."""

    module: str
    class_name: str
    methods: tuple[str, ...] | None = None
    is_async: bool = False
    is_paginator: bool = False
    label: str = "client"
    exclude: frozenset[str] = field(default_factory=frozenset)


def _call_input(
    wrapped: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    try:
        bound = inspect.signature(wrapped).bind_partial(*args, **kwargs)
        return {key: value for key, value in bound.arguments.items() if key != "self"}
    except Exception:  # noqa: BLE001 - provider call signatures may be hostile
        return {"args": list(args), "kwargs": kwargs}


def _instance_identity(instance: Any) -> dict[str, str]:
    identity: dict[str, str] = {}
    for key in (
        "name",
        "_name",
        "table_name",
        "_table_name",
        "index_name",
        "_index_name",
        "collection_name",
        "_collection_name",
        "uri",
        "_uri",
    ):
        try:
            value = getattr(instance, key, None)
        except Exception:  # noqa: BLE001 - provider descriptors must not break calls
            value = None
        if isinstance(value, (str, int)):
            identity_key = key.lstrip("_")
            identity[identity_key] = (
                "<redacted>"
                if is_sensitive_key(identity_key)
                else safe_text(str(value), endpoint=identity_key in {"uri", "host"})
            )
    try:
        config = getattr(instance, "_config", None)
        host = getattr(config, "host", None)
    except Exception:  # noqa: BLE001 - provider config descriptors may raise
        host = None
    if isinstance(host, str):
        identity["host"] = safe_text(host, endpoint=True)
    return identity


@dataclass(frozen=True)
class AppliedPatch:
    target: type
    attribute: str
    installed_wrapper: Any


class NativeClientInstrumentor:
    """Base lifecycle and span mapping for vendor client adapters."""

    name = "native-client"
    vendor = "native-client"
    patches: tuple[PatchSpec, ...] = ()
    _patches_applied: ClassVar[bool] = False
    _activation_count: ClassVar[int] = 0
    _patched_targets: ClassVar[list[AppliedPatch]] = []
    _capture_content_config: ClassVar[bool | None] = None
    _lifecycle_lock: ClassVar[RLock] = RLock()
    _active_call: ClassVar[ContextVar[bool]] = ContextVar(
        "respan_native_client_active",
        default=False,
    )

    def __init__(self, *, capture_content: bool = True) -> None:
        self._capture_content = capture_content
        self._is_instrumented = False

    @staticmethod
    def _tracing_enabled() -> bool:
        tracer = getattr(RespanTracer, "_instance", None)
        return tracer is None or bool(getattr(tracer, "is_enabled", True))

    @classmethod
    def _operation_name(cls, patch: PatchSpec, method: str) -> str:
        return f"{patch.label}.{method}" if patch.label else method

    @classmethod
    def _span_name(cls, operation: str) -> str:
        return f"{cls.vendor}.{operation}"

    @classmethod
    def _set_start_attributes(
        cls,
        span: Any,
        operation: str,
        instance: Any,
        wrapped: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        entity_name = cls._span_name(operation)
        payload = {
            "operation": operation,
            **_instance_identity(instance),
            **_call_input(wrapped, args, kwargs),
        }
        span.set_attribute(RESPAN_LOG_TYPE, LOG_TYPE_TASK)
        span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_NAME, entity_name)
        span.set_attribute(
            SpanAttributes.TRACELOOP_ENTITY_PATH,
            "" if getattr(span, "parent", None) is None else entity_name,
        )
        if getattr(cls, "_capture_content_config", True):
            span.set_attribute(
                SpanAttributes.TRACELOOP_ENTITY_INPUT,
                json_dumps(payload),
            )
        span.set_attribute(OTelSpanAttributes.DB_SYSTEM, cls.vendor)
        span.set_attribute(OTelSpanAttributes.DB_OPERATION, operation)

    @classmethod
    def _set_error(cls, span: Any, exc: Exception) -> None:
        message = exception_message(exc)
        exception_type = safe_type_name(exc)
        status_code = exception_status_code(exc)
        add_event = getattr(span, "add_event", None)
        if callable(add_event):
            add_event(
                "exception",
                {
                    OTelSpanAttributes.EXCEPTION_MESSAGE: message,
                    OTelSpanAttributes.EXCEPTION_TYPE: exception_type,
                },
            )
        span.set_status(Status(StatusCode.ERROR, message))
        span.set_attribute(HTTP_RESPONSE_STATUS_CODE, status_code)
        span.set_attribute(ERROR_MESSAGE_ATTR, message)
        if cls._capture_content_config:
            span.set_attribute(
                SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
                json_dumps({"error": exception_type, "message": message}),
            )

    def _trace_sync(
        self,
        operation: str,
        wrapped: Any,
        instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        active_call = type(self)._active_call
        if active_call.get():
            return wrapped(*args, **kwargs)
        token = active_call.set(True)
        try:
            tracer = trace.get_tracer(self.name, "0.1.0")
            with tracer.start_as_current_span(
                self._span_name(operation),
                kind=SpanKind.CLIENT,
                record_exception=False,
                set_status_on_exception=False,
            ) as span:
                self._set_start_attributes(
                    span,
                    operation,
                    instance,
                    wrapped,
                    args,
                    kwargs,
                )
                try:
                    result = wrapped(*args, **kwargs)
                except Exception as exc:
                    self._set_error(span, exc)
                    raise
                span.set_status(Status(StatusCode.OK))
                if self._capture_content:
                    span.set_attribute(
                        SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
                        json_dumps(result),
                    )
                return result
        finally:
            active_call.reset(token)

    def _trace_async_generator(
        self,
        operation: str,
        wrapped: Any,
        instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        """Trace a lazy async generator without leaking span context to callers."""
        source = wrapped(*args, **kwargs)

        async def iterate():
            active_call = type(self)._active_call
            if active_call.get() or not type(self)._patches_applied:
                try:
                    async for item in source:
                        yield item
                finally:
                    await source.aclose()
                return
            span = trace.get_tracer(self.name, "0.1.0").start_span(
                self._span_name(operation), kind=SpanKind.CLIENT
            )
            self._set_start_attributes(span, operation, instance, wrapped, args, kwargs)
            captured = []
            count = 0
            failed = False
            try:
                while True:
                    with trace.use_span(
                        span,
                        end_on_exit=False,
                        record_exception=False,
                        set_status_on_exception=False,
                    ):
                        token = active_call.set(True)
                        try:
                            item = await anext(source)
                        finally:
                            active_call.reset(token)
                    count += 1
                    if self._capture_content and len(captured) < MAX_ITEMS:
                        captured.append(item)
                    yield item
            except StopAsyncIteration:
                pass
            except GeneratorExit:
                raise
            except BaseException as exception:
                failed = True
                self._set_error(span, exception)
                raise
            finally:
                try:
                    await source.aclose()
                except BaseException as exception:
                    if not failed:
                        failed = True
                        self._set_error(span, exception)
                        raise
                finally:
                    if not failed:
                        span.set_status(Status(StatusCode.OK))
                        if self._capture_content:
                            span.set_attribute(
                                SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
                                json_dumps(
                                    {
                                        "items": captured,
                                        "count": count,
                                        "truncated": count > len(captured),
                                    }
                                ),
                            )
                    span.end()

        return iterate()

    def _trace_paginator(
        self,
        operation: str,
        wrapped: Any,
        instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        """Keep the native paginator and trace only pages actually requested."""
        try:
            paginator = wrapped(*args, **kwargs)
        except Exception as exception:  # noqa: BLE001 - preserve SDK validation errors

            def failed_call(
                *_args: Any, _error: Exception = exception, **_kwargs: Any
            ) -> Any:
                raise _error

            return self._trace_sync(operation, failed_call, instance, args, kwargs)
        fetch_page = paginator._fetch_page

        if inspect.iscoroutinefunction(fetch_page):

            async def traced_page(token: str | None) -> Any:
                if not type(self)._patches_applied:
                    return await fetch_page(token)

                @wraps(wrapped)
                async def call(*_args: Any, **_kwargs: Any) -> Any:
                    return await fetch_page(token)

                return await self._trace_async(
                    operation,
                    call,
                    instance,
                    args,
                    {**kwargs, "pagination_token": token},
                )
        else:

            def traced_page(token: str | None) -> Any:
                if not type(self)._patches_applied:
                    return fetch_page(token)

                @wraps(wrapped)
                def call(*_args: Any, **_kwargs: Any) -> Any:
                    return fetch_page(token)

                return self._trace_sync(
                    operation,
                    call,
                    instance,
                    args,
                    {**kwargs, "pagination_token": token},
                )

        paginator._fetch_page = traced_page
        return paginator

    async def _trace_async(
        self,
        operation: str,
        wrapped: Any,
        instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        active_call = type(self)._active_call
        if active_call.get():
            return await wrapped(*args, **kwargs)
        token = active_call.set(True)
        try:
            tracer = trace.get_tracer(self.name, "0.1.0")
            with tracer.start_as_current_span(
                self._span_name(operation),
                kind=SpanKind.CLIENT,
                record_exception=False,
                set_status_on_exception=False,
            ) as span:
                self._set_start_attributes(
                    span,
                    operation,
                    instance,
                    wrapped,
                    args,
                    kwargs,
                )
                try:
                    result = await wrapped(*args, **kwargs)
                except Exception as exc:
                    self._set_error(span, exc)
                    raise
                span.set_status(Status(StatusCode.OK))
                if self._capture_content:
                    span.set_attribute(
                        SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
                        json_dumps(result),
                    )
                return result
        finally:
            active_call.reset(token)

    @staticmethod
    def _methods_for(target_class: type, patch: PatchSpec) -> tuple[str, ...]:
        if patch.methods is not None:
            return patch.methods
        return tuple(
            name
            for name in dir(target_class)
            if not name.startswith("_")
            and name not in patch.exclude
            and callable(getattr(target_class, name, None))
        )

    @staticmethod
    def _wrapper_chain_contains(current: Any, expected: Any) -> bool:
        seen: set[int] = set()
        while current is not None and id(current) not in seen:
            if current is expected:
                return True
            seen.add(id(current))
            current = getattr(current, "__wrapped__", None)
        return False

    @classmethod
    def _remove_owned_patches(cls, patches: list[AppliedPatch]) -> list[AppliedPatch]:
        retained: list[AppliedPatch] = []
        for patch in reversed(patches):
            try:
                current = inspect.getattr_static(patch.target, patch.attribute)
                if current is patch.installed_wrapper:
                    unwrap(patch.target, patch.attribute)
                    current = inspect.getattr_static(patch.target, patch.attribute)
                if cls._wrapper_chain_contains(current, patch.installed_wrapper):
                    retained.append(patch)
            except Exception:
                retained.append(patch)
                logger.debug(
                    "Failed to remove %s.%s wrapper",
                    getattr(patch.target, "__name__", safe_type_name(patch.target)),
                    patch.attribute,
                    exc_info=True,
                )
        retained.reverse()
        return retained

    def activate(self) -> None:
        cls = type(self)
        with cls._lifecycle_lock:
            if self._is_instrumented or not self._tracing_enabled():
                return
            if cls._patches_applied:
                if cls._capture_content_config != self._capture_content:
                    raise ValueError(
                        "all active PineconeInstrumentor instances must use the "
                        "same capture_content setting"
                    )
                cls._activation_count += 1
                self._is_instrumented = True
                return
            if cls._patched_targets and all(
                cls._wrapper_chain_contains(
                    inspect.getattr_static(item.target, item.attribute),
                    item.installed_wrapper,
                )
                for item in cls._patched_targets
            ):
                cls._patches_applied = True
                cls._activation_count = 1
                cls._capture_content_config = self._capture_content
                self._is_instrumented = True
                return

            cls._patched_targets = cls._remove_owned_patches(cls._patched_targets)
            if cls._patched_targets:
                raise RuntimeError("Pinecone instrumentation has stale wrappers")

            patched_targets: list[AppliedPatch] = []
            try:
                for patch in cls.patches:
                    try:
                        module = importlib.import_module(patch.module)
                        target_class = getattr(module, patch.class_name)
                    except (ImportError, AttributeError):
                        continue
                    for method in self._methods_for(target_class, patch):
                        if not callable(getattr(target_class, method, None)):
                            continue
                        if any(
                            item.target is target_class and item.attribute == method
                            for item in patched_targets
                        ):
                            continue
                        operation = self._operation_name(patch, method)
                        target_method = getattr(target_class, method)
                        try:
                            annotation = inspect.signature(
                                target_method
                            ).return_annotation
                        except (TypeError, ValueError):
                            # Optional extension-backed clients may not expose a
                            # signature. Retain their existing wrapper path.
                            annotation = None
                        is_paginator = patch.is_paginator or (
                            isinstance(annotation, str) and "Paginator" in annotation
                        )

                        def traced(
                            wrapped: Any,
                            instance: Any,
                            args: tuple[Any, ...],
                            kwargs: dict[str, Any],
                            *,
                            _operation: str = operation,
                            _async: bool = patch.is_async,
                            _paginator: bool = is_paginator,
                            _async_generator: bool = inspect.isasyncgenfunction(
                                inspect.unwrap(target_method)
                            ),
                        ) -> Any:
                            instrumentor_class = type(self)
                            if (
                                not instrumentor_class._patches_applied
                                or not instrumentor_class._activation_count
                            ):
                                return wrapped(*args, **kwargs)
                            if _async_generator:
                                return self._trace_async_generator(
                                    _operation, wrapped, instance, args, kwargs
                                )
                            if _paginator:
                                return self._trace_paginator(
                                    _operation, wrapped, instance, args, kwargs
                                )
                            if _async:
                                return self._trace_async(
                                    _operation, wrapped, instance, args, kwargs
                                )
                            return self._trace_sync(
                                _operation, wrapped, instance, args, kwargs
                            )

                        target = f"{patch.class_name}.{method}"
                        wrap_function_wrapper(patch.module, target, traced)
                        patched_targets.append(
                            AppliedPatch(
                                target_class,
                                method,
                                inspect.getattr_static(target_class, method),
                            )
                        )
            except Exception:
                cls._patched_targets = cls._remove_owned_patches(patched_targets)
                cls._patches_applied = False
                cls._activation_count = 0
                cls._capture_content_config = None
                self._is_instrumented = False
                raise

            self._is_instrumented = bool(patched_targets)
            cls._patches_applied = self._is_instrumented
            cls._activation_count = int(self._is_instrumented)
            cls._patched_targets = patched_targets
            cls._capture_content_config = (
                self._capture_content if self._is_instrumented else None
            )
            if not self._is_instrumented:
                logger.warning(
                    "%s instrumentation found no supported methods", cls.vendor
                )

    def deactivate(self) -> None:
        cls = type(self)
        with cls._lifecycle_lock:
            if not self._is_instrumented:
                return
            self._is_instrumented = False
            cls._activation_count = max(cls._activation_count - 1, 0)
            if cls._activation_count:
                return
            cls._patches_applied = False
            cls._patched_targets = cls._remove_owned_patches(cls._patched_targets)
            if not cls._patched_targets:
                cls._capture_content_config = None

    def instrument(self) -> None:
        self.activate()

    def uninstrument(self) -> None:
        self.deactivate()
