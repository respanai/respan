"""OpenRouter instrumentation plugin for Respan."""

from __future__ import annotations

import inspect
import logging
import threading
from dataclasses import dataclass, field
from typing import Any

from opentelemetry import trace
from respan_instrumentation_openai import OpenAIInstrumentor
from respan_instrumentation_openai import _instrumentation as openai_instrumentation
from respan_tracing.core.tracer import RespanTracer

from respan_instrumentation_openrouter._constants import (
    OPENROUTER_INSTRUMENTATION_NAME,
)
from respan_instrumentation_openrouter._processor import (
    OpenRouterSpanProcessor,
)

logger = logging.getLogger(__name__)

_BRIDGE_LOCK = threading.RLock()
_ACTIVE_BRIDGE_OWNER: OpenRouterInstrumentor | None = None
_ACTIVE_RUNTIME: _OpenRouterRuntime | None = None


@dataclass(frozen=True)
class _DelegateMethodPatch:
    target: type[Any]
    attribute: str
    previous: Any
    installed: Any


@dataclass(eq=False)
class _OpenRouterRuntime:
    tracer_provider: Any
    delegate: Any
    processor: OpenRouterSpanProcessor
    owns_delegate_patches: bool
    normalize_all_openai_spans: bool
    capture_content: bool
    members: set[OpenRouterInstrumentor] = field(default_factory=set)
    delegate_method_patches: list[_DelegateMethodPatch] = field(default_factory=list)
    delegate_class: type[Any] | None = None
    delegate_activate: Any = None
    delegate_deactivate: Any = None
    coordinated_activate: Any = None
    coordinated_deactivate: Any = None
    external_delegate_members: set[Any] = field(default_factory=set)
    preexisting_delegate_active: bool = False
    lifecycle_active: bool = False
    native_patches: list[Any] = field(default_factory=list)
    refcounted_delegate: bool = False
    active: bool = True
    calls: set[Any] = field(default_factory=set)

    @property
    def config(self) -> tuple[bool, bool]:
        return self.normalize_all_openai_spans, self.capture_content


def _active_span_processors(tracer_provider: Any) -> tuple[Any, tuple[Any, ...] | None]:
    active_span_processor = getattr(tracer_provider, "_active_span_processor", None)
    processors = (
        getattr(active_span_processor, "_span_processors", None)
        if active_span_processor is not None
        else None
    )
    if processors is None:
        return active_span_processor, None
    return active_span_processor, tuple(processors)


def _register_processor_before_exporters(
    tracer_provider: Any,
    processor: OpenRouterSpanProcessor,
) -> None:
    active_span_processor, processors = _active_span_processors(tracer_provider)
    if active_span_processor is None or processors is None:
        if hasattr(tracer_provider, "add_span_processor"):
            tracer_provider.add_span_processor(processor)
        return

    if any(existing_processor is processor for existing_processor in processors):
        return
    active_span_processor._span_processors = (processor, *processors)


def _unregister_processor(
    tracer_provider: Any,
    processor: OpenRouterSpanProcessor | None,
) -> None:
    if processor is None:
        return

    active_span_processor, processors = _active_span_processors(tracer_provider)
    if active_span_processor is None or processors is None:
        return
    active_span_processor._span_processors = tuple(
        existing_processor
        for existing_processor in processors
        if existing_processor is not processor
    )


def _delegate_instrumentation_suppressed() -> bool:
    """Read optional delegate suppression state across supported releases."""

    callback = getattr(
        openai_instrumentation,
        "_is_openai_instrumentation_suppressed",
        None,
    )
    if not callable(callback):
        return False
    try:
        return bool(callback())
    except Exception:
        logger.debug("Failed to read OpenAI delegate suppression state", exc_info=True)
        return False


def _install_safe_delegate_methods(runtime: _OpenRouterRuntime) -> None:
    from respan_instrumentation_openrouter._observer import wrapper

    original_methods = getattr(openai_instrumentation, "_original_methods", {})
    for entry in getattr(openai_instrumentation, "_TARGETS", ()):
        if len(entry) == 4:
            module_path, class_name, kind, is_async = entry
            attribute = "create"
        else:
            module_path, class_name, attribute, kind, is_async = entry
        if kind not in {"chat", "completion", "response", "embedding"}:
            continue
        target = openai_instrumentation._load_class(module_path, class_name)
        original = original_methods.get((target, attribute))
        if original is None:
            continue
        current = inspect.getattr_static(target, attribute)
        installed_methods = getattr(openai_instrumentation, "_INSTALLED_METHODS", {})
        owned = installed_methods.get((target, attribute)) is current
        if not owned and inspect.isfunction(current):
            owned = (
                inspect.getclosurevars(current).nonlocals.get("original") is original
            )
        if not owned:
            continue
        installed = wrapper(
            original, kind=kind, runtime=runtime, is_async=is_async, fallback=current
        )
        setattr(target, attribute, installed)
        runtime.delegate_method_patches.append(
            _DelegateMethodPatch(target, attribute, current, installed)
        )


def _restore_safe_delegate_methods(runtime: _OpenRouterRuntime) -> None:
    for patch in reversed(runtime.delegate_method_patches):
        if inspect.getattr_static(patch.target, patch.attribute) is patch.installed:
            setattr(patch.target, patch.attribute, patch.previous)
    runtime.delegate_method_patches.clear()


def _install_delegate_lifecycle(runtime: _OpenRouterRuntime) -> None:
    delegate_class = type(runtime.delegate)
    original_activate = delegate_class.activate
    original_deactivate = delegate_class.deactivate
    runtime.delegate_class = delegate_class
    runtime.delegate_activate = original_activate
    runtime.delegate_deactivate = original_deactivate
    runtime.preexisting_delegate_active = not runtime.owns_delegate_patches
    runtime.refcounted_delegate = (
        delegate_class.__module__ == openai_instrumentation.__name__
        and hasattr(openai_instrumentation, "_REFCOUNT")
    )
    if runtime.refcounted_delegate:
        # New released delegates own their leases through their public lifecycle.
        return
    runtime.lifecycle_active = True

    def coordinated_activate(delegate_self: Any) -> None:
        with _BRIDGE_LOCK:
            if runtime.lifecycle_active:
                if delegate_self is not runtime.delegate:
                    runtime.external_delegate_members.add(delegate_self)
                delegate_self._is_instrumented = True
                return
        original_activate(delegate_self)

    def coordinated_deactivate(delegate_self: Any) -> None:
        with _BRIDGE_LOCK:
            if runtime.lifecycle_active:
                if delegate_self in runtime.external_delegate_members:
                    runtime.external_delegate_members.discard(delegate_self)
                    delegate_self._is_instrumented = False
                    return
                if delegate_self is not runtime.delegate:
                    runtime.preexisting_delegate_active = False
                    delegate_self._is_instrumented = False
                    return
                return
        original_deactivate(delegate_self)

    runtime.coordinated_activate = coordinated_activate
    runtime.coordinated_deactivate = coordinated_deactivate
    delegate_class.activate = coordinated_activate
    delegate_class.deactivate = coordinated_deactivate


def _restore_delegate_lifecycle(runtime: _OpenRouterRuntime) -> None:
    runtime.lifecycle_active = False
    delegate_class = runtime.delegate_class
    if delegate_class is not None:
        if delegate_class.activate is runtime.coordinated_activate:
            delegate_class.activate = runtime.delegate_activate
        if delegate_class.deactivate is runtime.coordinated_deactivate:
            delegate_class.deactivate = runtime.delegate_deactivate
    runtime.coordinated_activate = None
    runtime.coordinated_deactivate = None


def _has_external_delegate_owner(runtime: _OpenRouterRuntime) -> bool:
    return runtime.preexisting_delegate_active or any(
        bool(getattr(member, "_is_instrumented", False))
        for member in runtime.external_delegate_members
    )


class OpenRouterInstrumentor:
    """Respan instrumentor for native and OpenAI-compatible OpenRouter SDKs."""

    name = OPENROUTER_INSTRUMENTATION_NAME

    def __init__(
        self,
        *,
        normalize_all_openai_spans: bool = True,
        capture_content: bool = True,
    ) -> None:
        self._normalize_all_openai_spans = normalize_all_openai_spans
        self._capture_content = capture_content
        self._delegate = None
        self._processor: OpenRouterSpanProcessor | None = None
        self._is_instrumented = False
        self._runtime: _OpenRouterRuntime | None = None

    @staticmethod
    def _is_respan_tracing_enabled() -> bool:
        tracer = getattr(RespanTracer, "_instance", None)
        if tracer is None:
            return True
        return bool(getattr(tracer, "is_enabled", True))

    def activate(self) -> None:
        """Instrument OpenRouter calls made through the OpenAI Python client."""
        global _ACTIVE_BRIDGE_OWNER, _ACTIVE_RUNTIME

        if self._is_instrumented:
            return

        if not self._is_respan_tracing_enabled():
            logger.info(
                "OpenRouter instrumentation skipped because Respan tracing is disabled"
            )
            return

        requested_config = (
            self._normalize_all_openai_spans,
            self._capture_content,
        )
        with _BRIDGE_LOCK:
            if _ACTIVE_RUNTIME is not None:
                if _ACTIVE_RUNTIME.tracer_provider is not trace.get_tracer_provider():
                    logger.error(
                        "OpenRouter runtime already belongs to a different tracer provider"
                    )
                    return
                if _ACTIVE_RUNTIME.config != requested_config:
                    logger.error(
                        "OpenRouter instrumentation config mismatch: active "
                        "normalize_all_openai_spans=%s capture_content=%s; "
                        "requested normalize_all_openai_spans=%s capture_content=%s",
                        *_ACTIVE_RUNTIME.config,
                        *requested_config,
                    )
                    return
                _ACTIVE_RUNTIME.members.add(self)
                self._runtime = _ACTIVE_RUNTIME
                self._delegate = _ACTIVE_RUNTIME.delegate
                self._processor = _ACTIVE_RUNTIME.processor
                self._is_instrumented = True
                if _ACTIVE_BRIDGE_OWNER is None:
                    _ACTIVE_BRIDGE_OWNER = self
                logger.info("OpenRouter instrumentation joined the active runtime")
                return

            tracer_provider = trace.get_tracer_provider()
            delegate = None
            processor = None
            runtime = None
            delegate_methods = getattr(
                openai_instrumentation,
                "_original_methods",
                {},
            )
            delegate_was_active = bool(delegate_methods)
            try:
                delegate = OpenAIInstrumentor()
                processor = OpenRouterSpanProcessor(
                    normalize_all_openai_spans=self._normalize_all_openai_spans,
                    capture_content=self._capture_content,
                )
                runtime = _OpenRouterRuntime(
                    tracer_provider=tracer_provider,
                    delegate=delegate,
                    processor=processor,
                    owns_delegate_patches=not delegate_was_active,
                    normalize_all_openai_spans=self._normalize_all_openai_spans,
                    capture_content=self._capture_content,
                )
                _register_processor_before_exporters(
                    tracer_provider=tracer_provider,
                    processor=processor,
                )
                delegate.activate()
                if (
                    getattr(delegate, "_is_instrumented", True) is False
                    and not delegate_was_active
                ):
                    _unregister_processor(tracer_provider, processor)
                    return
                _install_safe_delegate_methods(runtime)
                from respan_instrumentation_openrouter._observer import (
                    install_native,
                    install_source_observers,
                )

                install_native(runtime)
                install_source_observers(runtime)
                _install_delegate_lifecycle(runtime)
            except Exception:
                if runtime is not None:
                    runtime.active = False
                    from respan_instrumentation_openrouter._observer import (
                        restore_native,
                    )

                    restore_native(runtime)
                    _restore_delegate_lifecycle(runtime)
                    _restore_safe_delegate_methods(runtime)
                _unregister_processor(
                    tracer_provider=tracer_provider,
                    processor=processor,
                )
                if delegate is not None and (
                    not delegate_was_active
                    or hasattr(openai_instrumentation, "_REFCOUNT")
                ):
                    try:
                        delegate.deactivate()
                    except Exception:
                        logger.exception(
                            "Failed to clean up OpenRouter instrumentation"
                        )
                logger.exception("Failed to activate OpenRouter instrumentation")
                return

            runtime.members.add(self)
            _ACTIVE_RUNTIME = runtime
            _ACTIVE_BRIDGE_OWNER = self
            self._runtime = runtime
            self._delegate = delegate
            self._processor = processor
            self._is_instrumented = True
            logger.info("OpenRouter instrumentation activated")

    def deactivate(self) -> None:
        """Release this instance and tear down the shared runtime when last."""
        global _ACTIVE_BRIDGE_OWNER, _ACTIVE_RUNTIME

        with _BRIDGE_LOCK:
            runtime = self._runtime
            if runtime is None or not self._is_instrumented:
                self._delegate = None
                self._processor = None
                self._runtime = None
                self._is_instrumented = False
                return

            runtime.members.discard(self)
            self._delegate = None
            self._processor = None
            self._runtime = None
            self._is_instrumented = False

            if runtime.members:
                if _ACTIVE_BRIDGE_OWNER is self:
                    _ACTIVE_BRIDGE_OWNER = next(iter(runtime.members))
                logger.info(
                    "OpenRouter instrumentation instance released; shared runtime retained"
                )
                return

            if _ACTIVE_RUNTIME is runtime:
                _ACTIVE_RUNTIME = None
            if _ACTIVE_BRIDGE_OWNER is self or not runtime.members:
                _ACTIVE_BRIDGE_OWNER = None

            runtime.active = False
            for call in tuple(runtime.calls):
                call.allowed = False
                call.finish()
            external_delegate_owner = _has_external_delegate_owner(runtime)
            _restore_delegate_lifecycle(runtime)
            from respan_instrumentation_openrouter._observer import restore_native

            restore_native(runtime)
            _unregister_processor(
                tracer_provider=runtime.tracer_provider,
                processor=runtime.processor,
            )
            if external_delegate_owner and not runtime.refcounted_delegate:
                # Restore the exact delegate wrappers that the independent owner
                # joined; its eventual deactivate call still owns final teardown.
                _restore_safe_delegate_methods(runtime)
            else:
                _restore_safe_delegate_methods(runtime)
                try:
                    runtime.delegate_deactivate(runtime.delegate)
                except Exception:
                    logger.exception(
                        "Failed to deactivate OpenRouter delegate instrumentation"
                    )
                finally:
                    _restore_safe_delegate_methods(runtime)
            logger.info("OpenRouter instrumentation deactivated")
