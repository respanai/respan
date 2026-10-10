"""Pipecat instrumentation plugin for Respan."""

from __future__ import annotations

import importlib
import inspect
import logging
import threading
from typing import Any

from opentelemetry import context as context_api
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from respan_tracing.core.tracer import RespanTracer

from respan_instrumentation_pipecat._observer_hooks import (
    ObserverHook,
    install_observer_hook,
    remove_observer_hook,
)
from respan_instrumentation_pipecat._runtime import Runtime
from respan_instrumentation_pipecat._translator import PipecatOpenInferenceTranslator

logger = logging.getLogger(__name__)

PIPECAT_INSTRUMENTATION_NAME = "pipecat"
OPENINFERENCE_PIPECAT_MODULE = "openinference.instrumentation.pipecat"
OPENINFERENCE_PIPECAT_OBSERVER_MODULE = (
    "openinference.instrumentation.pipecat._observer"
)

_LOCK = threading.RLock()
_REFCOUNT = 0
_PROCESSOR: PipecatOpenInferenceTranslator | None = None
_PROVIDER: Any = None
_UPSTREAM: Any = None
_OBSERVER_MODULE: Any = None
_OBSERVER_CONTEXT_ORIGINAL: Any = None
_OBSERVER_HOOK: ObserverHook | None = None
_CONFIG: dict[str, Any] | None = None
_RUNTIME = None
_LEASE = None
_MISSING = object()


def _native_snapshot(upstream):
    try:
        cls = importlib.import_module("pipecat.pipeline.worker").PipelineWorker
        field = "_original_worker_init"
    except ImportError:
        cls = importlib.import_module("pipecat.pipeline.task").PipelineTask
        field = "_original_task_init"
    fields = {
        name: getattr(upstream, name, _MISSING)
        for name in (field, "_tracer", "_config", "_debug_log_filename")
    }
    return cls, field, inspect.getattr_static(cls, "__init__"), fields


def _restore_native(upstream, lease):
    if not lease["owned"]:
        return
    cls, field, original, fields = lease["before"]
    _, _, wrapper, installed = lease["installed"]
    current = inspect.getattr_static(cls, "__init__")
    foreign = {k: getattr(upstream, k, _MISSING) for k in fields}
    setattr(upstream, field, original if current is wrapper else current)
    try:
        if upstream.is_instrumented_by_opentelemetry:
            upstream.uninstrument()
        else:
            upstream._uninstrument()
    finally:
        if inspect.getattr_static(cls, "__init__") is wrapper:
            cls.__init__ = original
        for k, before in fields.items():
            expected = (
                (original if current is wrapper else current)
                if k == field
                else installed[k]
            )
            if getattr(upstream, k, _MISSING) is not expected:
                continue
            replacement = before if foreign[k] is installed[k] else foreign[k]
            if replacement is _MISSING:
                try:
                    delattr(upstream, k)
                except AttributeError:
                    pass
            else:
                setattr(upstream, k, replacement)
        upstream._is_instrumented_by_opentelemetry = False


def _load_openinference_pipecat_class() -> type:
    pipecat_module = importlib.import_module(OPENINFERENCE_PIPECAT_MODULE)
    return pipecat_module.PipecatInstrumentor


def _load_openinference_pipecat_observer_module() -> Any:
    return importlib.import_module(OPENINFERENCE_PIPECAT_OBSERVER_MODULE)


def _is_respan_tracing_enabled() -> bool:
    tracer = getattr(RespanTracer, "_instance", None)
    if tracer is None:
        return True
    return bool(getattr(tracer, "is_enabled", True))


def _active_processors(provider: Any) -> tuple[Any, tuple[Any, ...] | None]:
    active = getattr(provider, "_active_span_processor", None)
    processors = getattr(active, "_span_processors", None) if active else None
    return active, processors


def _register_processor(
    provider: Any, processor: PipecatOpenInferenceTranslator
) -> None:
    active, processors = _active_processors(provider)
    if active is not None and processors is not None:
        remaining = tuple(
            existing for existing in processors if existing is not processor
        )
        active._span_processors = (processor, *remaining)
    elif hasattr(provider, "add_span_processor"):
        provider.add_span_processor(processor)


def _unregister_processor(
    provider: Any, processor: PipecatOpenInferenceTranslator
) -> None:
    active, processors = _active_processors(provider)
    if active is not None and processors is not None:
        active._span_processors = tuple(
            existing for existing in processors if existing is not processor
        )


def _same_config(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if left.keys() != right.keys():
        return False
    for key, value in left.items():
        if key == "provider" and value is not right[key]:
            return False
        if value is right[key]:
            continue
        try:
            if value == right[key]:
                continue
        except Exception:  # noqa: BLE001 - hostile configuration equality is rejected
            return False
        return False
    return True


def _patch_observer(observer_module: Any) -> tuple[Any, ObserverHook]:
    original_context = observer_module.Context
    try:
        hook = install_observer_hook(observer_module)
    except Exception:
        if observer_module.Context is context_api.get_current:
            observer_module.Context = original_context
        raise
    return original_context, hook


def _restore_observer(
    observer_module: Any,
    original_context: Any,
    hook: ObserverHook | None,
) -> None:
    remove_observer_hook(hook)
    if (
        observer_module is not None
        and observer_module.Context is context_api.get_current
    ):
        observer_module.Context = original_context


class PipecatInstrumentor:
    """Activate OpenInference Pipecat once and normalize its spans for Respan."""

    name = PIPECAT_INSTRUMENTATION_NAME

    def __init__(
        self, *, capture_content=True, tracer_provider=None, **instrumentor_kwargs: Any
    ) -> None:
        self.capture_content = bool(capture_content)
        self.provider = tracer_provider
        self._instrumentor_kwargs = dict(instrumentor_kwargs)
        self._is_instrumented = False

    def activate(self) -> None:
        """Instrument Pipecat with shared, transactional lifecycle ownership."""
        global _CONFIG, _OBSERVER_CONTEXT_ORIGINAL, _OBSERVER_HOOK, _RUNTIME, _LEASE
        global _OBSERVER_MODULE, _PROCESSOR, _PROVIDER, _REFCOUNT, _UPSTREAM

        if self._is_instrumented:
            return
        if not _is_respan_tracing_enabled():
            logger.info(
                "Pipecat instrumentation skipped because Respan tracing is disabled"
            )
            return
        with _LOCK:
            if self._is_instrumented:
                return
            provider = self.provider or trace.get_tracer_provider()
            if not isinstance(provider, TracerProvider):
                logger.warning(
                    "Pipecat instrumentation requires an initialized OTel SDK TracerProvider"
                )
                return
            config = {
                "provider": provider,
                "capture_content": self.capture_content,
                **self._instrumentor_kwargs,
            }
            if _REFCOUNT:
                if _CONFIG is None or not _same_config(_CONFIG, config):
                    raise ValueError("Pipecat owners must share provider and settings")
                _REFCOUNT += 1
                self._is_instrumented = True
                return

            try:
                instrumentor_class = _load_openinference_pipecat_class()
                observer_module = _load_openinference_pipecat_observer_module()
            except ImportError as exc:
                logger.warning(
                    "Failed to activate Pipecat instrumentation — missing dependency: %s",
                    exc,
                )
                return

            runtime = Runtime(provider, self.capture_content)
            processor = PipecatOpenInferenceTranslator(runtime)
            upstream = instrumentor_class()
            if (
                upstream.is_instrumented_by_opentelemetry
                and getattr(upstream._tracer, "span_processor", None)
                is not provider._active_span_processor
            ):
                raise ValueError("Foreign Pipecat delegate uses a different provider")
            before = _native_snapshot(upstream)
            lease = {
                "before": before,
                "installed": before,
                "owned": not upstream.is_instrumented_by_opentelemetry,
            }
            runtime.native_owned = lease["owned"]
            original_context: Any = None
            hook: ObserverHook | None = None
            registered = False
            try:
                registered = True
                _register_processor(provider, runtime.policy)
                _register_processor(provider, processor)
                original_context, hook = _patch_observer(observer_module)
                try:
                    if lease["owned"]:
                        upstream.instrument(
                            tracer_provider=provider, **self._instrumentor_kwargs
                        )
                finally:
                    lease["installed"] = _native_snapshot(upstream)
                if lease["owned"] and (
                    not upstream.is_instrumented_by_opentelemetry
                    or lease["installed"][2] is lease["before"][2]
                ):
                    raise RuntimeError(
                        "Native Pipecat delegate did not install its SDK hook; pair Pipecat<1.3 with delegate1.x, Pipecat>=1.3 with delegate2.x"
                    )
                runtime.install(observer_module)
            except Exception:
                try:
                    try:
                        runtime.close()
                    finally:
                        _restore_native(upstream, lease)
                except Exception:
                    logger.exception("Failed to roll back Pipecat instrumentation")
                if hook is not None or original_context is not None:
                    _restore_observer(observer_module, original_context, hook)
                if registered:
                    _unregister_processor(provider, processor)
                    _unregister_processor(provider, runtime.policy)
                logger.exception("Failed to activate Pipecat instrumentation")
                return

            _CONFIG = config
            _RUNTIME = runtime
            _LEASE = lease
            _OBSERVER_CONTEXT_ORIGINAL = original_context
            _OBSERVER_HOOK = hook
            _OBSERVER_MODULE = observer_module
            _PROCESSOR = processor
            _PROVIDER = provider
            _UPSTREAM = upstream
            _REFCOUNT = 1
            self._is_instrumented = True
            logger.info("Pipecat instrumentation activated")

    def deactivate(self) -> None:
        """Release one owner and remove only the final shared activation."""
        global _CONFIG, _OBSERVER_CONTEXT_ORIGINAL, _OBSERVER_HOOK, _RUNTIME, _LEASE
        global _OBSERVER_MODULE, _PROCESSOR, _PROVIDER, _REFCOUNT, _UPSTREAM

        if not self._is_instrumented:
            return
        with _LOCK:
            if not self._is_instrumented:
                return
            self._is_instrumented = False
            _REFCOUNT = max(0, _REFCOUNT - 1)
            if _REFCOUNT:
                return
            if _UPSTREAM is not None:
                try:
                    try:
                        _RUNTIME.close()
                    finally:
                        _restore_native(_UPSTREAM, _LEASE)
                except Exception:
                    logger.exception("Failed to deactivate Pipecat instrumentation")
            if _PROCESSOR is not None and _PROVIDER is not None:
                _unregister_processor(_PROVIDER, _PROCESSOR)
                _unregister_processor(_PROVIDER, _RUNTIME.policy)
            _restore_observer(
                _OBSERVER_MODULE,
                _OBSERVER_CONTEXT_ORIGINAL,
                _OBSERVER_HOOK,
            )
            _CONFIG = None
            _OBSERVER_CONTEXT_ORIGINAL = None
            _OBSERVER_HOOK = None
            _OBSERVER_MODULE = None
            _PROCESSOR = None
            _PROVIDER = None
            _UPSTREAM = None
            _RUNTIME = None
            _LEASE = None
            logger.info("Pipecat instrumentation deactivated")

    def instrument(self) -> None:
        """Alias used by direct OpenTelemetry-style integrations."""
        self.activate()

    def uninstrument(self) -> None:
        """Alias used by direct OpenTelemetry-style integrations."""
        self.deactivate()
