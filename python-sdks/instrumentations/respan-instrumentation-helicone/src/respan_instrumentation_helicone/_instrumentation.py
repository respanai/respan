"""Lifecycle-safe patching for Helicone's Python manual logger."""

from __future__ import annotations

import importlib
import inspect
import logging
import threading
import time
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from typing import Any

from respan_tracing.core.tracer import RespanTracer

from respan_instrumentation_helicone._constants import (
    HELICONE_BUILDER_CONTEXT_ATTR,
    HELICONE_HELPERS_MODULE,
    HELICONE_INSTRUMENTATION_NAME,
    HELICONE_LOG_BUILDER_CLASS,
    HELICONE_MANUAL_LOGGER_CLASS,
)
from respan_instrumentation_helicone._emitter import (
    HeliconeEmissionContext,
    capture_emission_context,
    emit_helicone_log,
)
from respan_instrumentation_helicone._serialization import safe_text, safe_type_name

logger = logging.getLogger(__name__)

_LOCK = threading.RLock()
_REFCOUNT = 0
_CONFIG: tuple[bool] | None = None
_ORIGINAL_METHODS: dict[tuple[type[Any], str], Any] = {}
_INSTALLED_METHODS: dict[tuple[type[Any], str], Any] = {}


@dataclass
class _PatchGeneration:
    capture_content: bool
    enabled: bool = True


@dataclass
class _LogRequestState:
    start_time: float
    terminal_sink_calls: int = 0
    in_operation: bool = False


@dataclass(frozen=True)
class _BuilderState:
    error: BaseException | None
    status_code: int
    is_streaming: bool
    context_snapshot: HeliconeEmissionContext | None


_GENERATION: _PatchGeneration | None = None
_CURRENT_LOG_REQUEST: ContextVar[_LogRequestState | None] = ContextVar(
    "respan_helicone_log_request", default=None
)
_CURRENT_BUILDER: ContextVar[_BuilderState | None] = ContextVar(
    "respan_helicone_builder", default=None
)


def _tracing_enabled() -> bool:
    tracer = getattr(RespanTracer, "_instance", None)
    return tracer is None or bool(getattr(tracer, "is_enabled", True))


def _load_classes() -> tuple[type[Any], type[Any]]:
    module = importlib.import_module(HELICONE_HELPERS_MODULE)
    return (
        getattr(module, HELICONE_MANUAL_LOGGER_CLASS),
        getattr(module, HELICONE_LOG_BUILDER_CLASS),
    )


def _builder_error(value: Any) -> BaseException | None:
    if value is None:
        return None
    if isinstance(value, BaseException):
        return value
    if isinstance(value, str):
        return RuntimeError(safe_text(value))
    return RuntimeError(safe_type_name(value))


def _builder_status(builder: Any, error: BaseException | None) -> int:
    if bool(getattr(builder, "was_cancelled", False)):
        return 499
    status = getattr(builder, "status", None)
    if (
        isinstance(status, int)
        and not isinstance(status, bool)
        and 400 <= status <= 599
    ):
        return status
    return 500 if error is not None else 200


def _make_send_log_wrapper(original: Any, generation: _PatchGeneration) -> Any:
    @wraps(original)
    def wrapper(
        instance: Any,
        provider: str | None,
        request: dict,
        response: dict | str,
        options: Any,
    ) -> Any:
        if not generation.enabled:
            return original(instance, provider, request, response, options)
        request_state = _CURRENT_LOG_REQUEST.get()
        is_terminal_sink = request_state is not None and not request_state.in_operation
        if is_terminal_sink:
            if request_state.terminal_sink_calls:
                return original(instance, provider, request, response, options)
            request_state.terminal_sink_calls += 1
        builder_state = _CURRENT_BUILDER.get()
        constructor_headers = getattr(instance, "headers", None)
        try:
            result = original(instance, provider, request, response, options)
        except BaseException as exc:
            emit_helicone_log(
                provider=provider,
                request=request,
                response=response,
                options=options,
                capture_content=generation.capture_content,
                error=exc,
                status_code=500,
                constructor_headers=constructor_headers,
            )
            raise
        emit_helicone_log(
            provider=provider,
            request=request,
            response=response,
            options=options,
            capture_content=generation.capture_content,
            error=builder_state.error if builder_state is not None else None,
            status_code=(
                builder_state.status_code if builder_state is not None else None
            ),
            is_streaming=(
                builder_state.is_streaming if builder_state is not None else None
            ),
            context_snapshot=(
                builder_state.context_snapshot if builder_state is not None else None
            ),
            constructor_headers=constructor_headers,
        )
        return result

    wrapper.__respan_helicone_wrapper__ = True
    return wrapper


def _make_log_request_wrapper(original: Any, generation: _PatchGeneration) -> Any:
    @wraps(original)
    def wrapper(
        instance: Any,
        request: dict,
        operation: Any,
        additional_headers: dict | None = None,
        provider: str | None = None,
    ) -> Any:
        resolved_headers = additional_headers or {}
        if not generation.enabled:
            return original(
                instance,
                request,
                operation,
                additional_headers=resolved_headers,
                provider=provider,
            )
        state = _LogRequestState(start_time=time.time())

        @wraps(operation)
        def observed_operation(recorder: Any) -> Any:
            state.in_operation = True
            try:
                return operation(recorder)
            finally:
                state.in_operation = False

        token = _CURRENT_LOG_REQUEST.set(state)
        try:
            return original(
                instance,
                request,
                observed_operation,
                additional_headers=resolved_headers,
                provider=provider,
            )
        except BaseException as exc:
            if state.terminal_sink_calls == 0:
                emit_helicone_log(
                    provider=provider,
                    request=request,
                    response={
                        "status": "error",
                        "error": {"type": type(exc).__name__},
                    },
                    options={
                        "start_time": state.start_time,
                        "end_time": time.time(),
                        "additional_headers": resolved_headers,
                    },
                    capture_content=generation.capture_content,
                    error=exc,
                    status_code=500,
                    constructor_headers=getattr(instance, "headers", None),
                )
            raise
        finally:
            _CURRENT_LOG_REQUEST.reset(token)

    wrapper.__respan_helicone_wrapper__ = True
    return wrapper


def _make_builder_send_log_wrapper(original: Any, generation: _PatchGeneration) -> Any:
    @wraps(original)
    async def wrapper(instance: Any, *args: Any, **kwargs: Any) -> Any:
        if not generation.enabled:
            result = original(instance, *args, **kwargs)
            return await result if inspect.isawaitable(result) else result
        error = _builder_error(getattr(instance, "error", None))
        state = _BuilderState(
            error=error,
            status_code=_builder_status(instance, error),
            is_streaming=(
                bool(getattr(instance, "stream_chunks", None))
                or bool(getattr(instance, "request", {}).get("stream"))
            ),
            context_snapshot=getattr(instance, HELICONE_BUILDER_CONTEXT_ATTR, None),
        )
        token = _CURRENT_BUILDER.set(state)
        try:
            result = original(instance, *args, **kwargs)
            return await result if inspect.isawaitable(result) else result
        finally:
            _CURRENT_BUILDER.reset(token)

    wrapper.__respan_helicone_wrapper__ = True
    return wrapper


def _make_log_builder_wrapper(original: Any, generation: _PatchGeneration) -> Any:
    @wraps(original)
    def wrapper(instance: Any, *args: Any, **kwargs: Any) -> Any:
        if not generation.enabled:
            return original(instance, *args, **kwargs)
        snapshot = capture_emission_context()
        builder = original(instance, *args, **kwargs)
        try:
            setattr(builder, HELICONE_BUILDER_CONTEXT_ATTR, snapshot)
        except BaseException:
            logger.debug("Failed to attach Helicone builder context", exc_info=True)
        return builder

    wrapper.__respan_helicone_wrapper__ = True
    return wrapper


def _patch(
    target: type[Any],
    method_name: str,
    factory: Any,
    generation: _PatchGeneration,
) -> None:
    key = (target, method_name)
    if key in _ORIGINAL_METHODS:
        return
    original = getattr(target, method_name)
    installed = factory(original, generation)
    _ORIGINAL_METHODS[key] = original
    try:
        setattr(target, method_name, installed)
    except BaseException:
        _ORIGINAL_METHODS.pop(key, None)
        raise
    _INSTALLED_METHODS[key] = installed


def _restore_all() -> None:
    for key, original in list(_ORIGINAL_METHODS.items())[::-1]:
        target, method_name = key
        installed = _INSTALLED_METHODS.get(key)
        try:
            current = getattr(target, method_name)
        except BaseException:
            logger.debug("Failed to inspect Helicone patch", exc_info=True)
            continue
        if current is installed:
            try:
                setattr(target, method_name, original)
            except BaseException:
                logger.debug("Failed to restore Helicone patch", exc_info=True)
                continue
        _ORIGINAL_METHODS.pop(key, None)
        _INSTALLED_METHODS.pop(key, None)


def _install_all(generation: _PatchGeneration) -> None:
    logger_class, builder_class = _load_classes()
    installed_before = set(_ORIGINAL_METHODS)
    try:
        _patch(logger_class, "send_log", _make_send_log_wrapper, generation)
        _patch(logger_class, "log_request", _make_log_request_wrapper, generation)
        _patch(logger_class, "log_builder", _make_log_builder_wrapper, generation)
        _patch(builder_class, "send_log", _make_builder_send_log_wrapper, generation)
    except BaseException:
        for key in set(_ORIGINAL_METHODS) - installed_before:
            target, method_name = key
            original = _ORIGINAL_METHODS.get(key)
            installed = _INSTALLED_METHODS.get(key)
            if original is not None and getattr(target, method_name, None) is installed:
                try:
                    setattr(target, method_name, original)
                except BaseException:
                    logger.debug("Failed to roll back Helicone patch", exc_info=True)
            _ORIGINAL_METHODS.pop(key, None)
            _INSTALLED_METHODS.pop(key, None)
        raise


class HeliconeInstrumentor:
    """Instrument one shared `helicone-helpers` manual logger runtime."""

    name = HELICONE_INSTRUMENTATION_NAME

    def __init__(self, *, capture_content: bool = True) -> None:
        self._capture_content = bool(capture_content)
        self._is_instrumented = False

    def activate(self) -> None:
        global _CONFIG, _GENERATION, _REFCOUNT

        if self._is_instrumented:
            return
        if not _tracing_enabled():
            logger.info(
                "Helicone instrumentation skipped because Respan tracing is disabled"
            )
            return
        with _LOCK:
            if self._is_instrumented:
                return
            requested_config = (self._capture_content,)
            if _REFCOUNT:
                if _CONFIG != requested_config:
                    logger.error(
                        "Helicone instrumentation is already active with a different "
                        "capture_content setting"
                    )
                    return
                _REFCOUNT += 1
                self._is_instrumented = True
                return
            generation = _PatchGeneration(capture_content=self._capture_content)
            try:
                _install_all(generation)
            except ImportError as exc:
                logger.warning(
                    "Failed to activate Helicone instrumentation - missing dependency: %s",
                    exc,
                )
                return
            except BaseException:
                generation.enabled = False
                logger.exception("Failed to activate Helicone instrumentation")
                return
            _CONFIG = requested_config
            _GENERATION = generation
            _REFCOUNT = 1
            self._is_instrumented = True
            logger.info("Helicone instrumentation activated")

    def deactivate(self) -> None:
        global _CONFIG, _GENERATION, _REFCOUNT

        if not self._is_instrumented:
            return
        with _LOCK:
            if not self._is_instrumented:
                return
            self._is_instrumented = False
            _REFCOUNT = max(0, _REFCOUNT - 1)
            if _REFCOUNT:
                return
            if _GENERATION is not None:
                _GENERATION.enabled = False
            _restore_all()
            _CONFIG = None
            _GENERATION = None
            logger.info("Helicone instrumentation deactivated")

    def instrument(self) -> None:
        self.activate()

    def uninstrument(self) -> None:
        self.deactivate()
