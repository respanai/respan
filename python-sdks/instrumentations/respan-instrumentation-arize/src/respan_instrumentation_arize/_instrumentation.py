"""Arize Python SDK instrumentation plugin for Respan."""

from __future__ import annotations

import importlib
import inspect
import logging
import time
from collections.abc import Callable
from threading import Lock
from typing import Any

from respan_instrumentation_arize._constants import (
    ARIZE_CLIENT_SPECS,
    ARIZE_INSTRUMENTATION_NAME,
    ArizeClientSpec,
)
from respan_instrumentation_arize._span_emitter import emit_arize_span
from respan_tracing.core.tracer import RespanTracer

logger = logging.getLogger(__name__)

_PATCH_LOCK = Lock()
_ORIGINAL_METHODS: dict[tuple[type[Any], str], Callable[..., Any]] = {}
_ACTIVE_INSTANCES = 0


def _load_client_class(spec: ArizeClientSpec) -> type[Any]:
    module = importlib.import_module(spec.module_name)
    client_class = getattr(module, spec.class_name, None)
    if client_class is None:
        raise AttributeError(f"{spec.module_name}.{spec.class_name}")
    return client_class


def _wrap_method(
    *,
    resource: str,
    method_name: str,
    original: Callable[..., Any],
) -> Callable[..., Any]:
    if inspect.iscoroutinefunction(original):

        async def async_wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            start_time_ns = time.time_ns()
            try:
                result = await original(self, *args, **kwargs)
            except Exception as exc:
                emit_arize_span(
                    resource=resource,
                    method_name=method_name,
                    args=args,
                    kwargs=kwargs,
                    result=None,
                    start_time_ns=start_time_ns,
                    end_time_ns=time.time_ns(),
                    error=exc,
                )
                raise

            emit_arize_span(
                resource=resource,
                method_name=method_name,
                args=args,
                kwargs=kwargs,
                result=result,
                start_time_ns=start_time_ns,
                end_time_ns=time.time_ns(),
            )
            return result

        return async_wrapper

    def sync_wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        start_time_ns = time.time_ns()
        try:
            result = original(self, *args, **kwargs)
        except Exception as exc:
            emit_arize_span(
                resource=resource,
                method_name=method_name,
                args=args,
                kwargs=kwargs,
                result=None,
                start_time_ns=start_time_ns,
                end_time_ns=time.time_ns(),
                error=exc,
            )
            raise

        emit_arize_span(
            resource=resource,
            method_name=method_name,
            args=args,
            kwargs=kwargs,
            result=result,
            start_time_ns=start_time_ns,
            end_time_ns=time.time_ns(),
        )
        return result

    return sync_wrapper


def _patch_client_class(spec: ArizeClientSpec, client_class: type[Any]) -> bool:
    patched_any = False
    for method_name in spec.methods:
        original = getattr(client_class, method_name, None)
        if original is None or not callable(original):
            logger.debug(
                "Skipping Arize %s.%s; method is not available",
                spec.class_name,
                method_name,
            )
            continue

        key = (client_class, method_name)
        if key not in _ORIGINAL_METHODS:
            _ORIGINAL_METHODS[key] = original
            setattr(
                client_class,
                method_name,
                _wrap_method(
                    resource=spec.resource,
                    method_name=method_name,
                    original=original,
                ),
            )

        patched_any = True

    return patched_any


def _patch_arize_clients(specs: tuple[ArizeClientSpec, ...]) -> int:
    patched_count = 0
    for spec in specs:
        try:
            client_class = _load_client_class(spec)
        except (AttributeError, ImportError) as exc:
            logger.debug(
                "Skipping Arize instrumentation target %s.%s: %s",
                spec.module_name,
                spec.class_name,
                exc,
            )
            continue
        if _patch_client_class(spec, client_class):
            patched_count += 1
    return patched_count


def _restore_arize_clients() -> None:
    for (client_class, method_name), original in _ORIGINAL_METHODS.items():
        setattr(client_class, method_name, original)
    _ORIGINAL_METHODS.clear()


class ArizeInstrumentor:
    """Respan instrumentor for the Arize Python SDK."""

    name = ARIZE_INSTRUMENTATION_NAME

    def __init__(
        self,
        *,
        client_specs: tuple[ArizeClientSpec, ...] = ARIZE_CLIENT_SPECS,
    ) -> None:
        self._client_specs = client_specs
        self._is_instrumented = False

    @staticmethod
    def _is_respan_tracing_enabled() -> bool:
        tracer = getattr(RespanTracer, "_instance", None)
        if tracer is None:
            return True
        return bool(getattr(tracer, "is_enabled", True))

    def activate(self) -> None:
        """Monkey-patch Arize SDK public client operation methods."""
        global _ACTIVE_INSTANCES

        if self._is_instrumented:
            return

        if not self._is_respan_tracing_enabled():
            logger.info("Arize instrumentation skipped because Respan tracing is disabled")
            return

        with _PATCH_LOCK:
            patched_count = _patch_arize_clients(self._client_specs)
            if patched_count:
                _ACTIVE_INSTANCES += 1
                self._is_instrumented = True
                logger.info("Arize instrumentation activated")
                return

        logger.warning("Failed to activate Arize instrumentation - no compatible clients found")

    def deactivate(self) -> None:
        """Restore original Arize SDK client operation methods."""
        global _ACTIVE_INSTANCES

        if not self._is_instrumented:
            return

        with _PATCH_LOCK:
            _ACTIVE_INSTANCES = max(0, _ACTIVE_INSTANCES - 1)
            if _ACTIVE_INSTANCES == 0:
                _restore_arize_clients()

        self._is_instrumented = False
        logger.info("Arize instrumentation deactivated")
