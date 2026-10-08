"""Lifecycle management for generic OpenInference instrumentors."""

from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from typing import Any, ClassVar

from opentelemetry import trace

from respan_instrumentation_openinference._translator import OpenInferenceTranslator

logger = logging.getLogger(__name__)


@dataclass
class _SharedRegistration:
    instrumentor: Any
    kwargs: dict[str, Any]
    is_span_processor: bool
    owned_activation: bool
    refcount: int = 1


class OpenInferenceInstrumentor:
    """Adapt any OpenInference instrumentor to Respan's lifecycle protocol.

    Standard ``instrument``/``uninstrument`` delegates and processor-style
    delegates are shared process-wide per instrumentor class. The first active
    wrapper owns activation and the last wrapper owns teardown. A single
    translator is kept before export processors while any delegate is active.
    """

    _lock: ClassVar[threading.RLock] = threading.RLock()
    _translator: ClassVar[OpenInferenceTranslator] = OpenInferenceTranslator()
    _translator_registered: ClassVar[bool] = False
    _active_span_processors: ClassVar[list[Any]] = []
    _registrations: ClassVar[dict[type, _SharedRegistration]] = {}
    _provider: ClassVar[Any] = None
    _original_order: ClassVar[tuple[Any, ...]] = ()
    _scope_patch: ClassVar[tuple[Any, Any, Any] | None] = None

    @classmethod
    def _observe_native_scope(cls) -> None:
        if cls._scope_patch is not None:
            return
        try:
            from openinference.instrumentation import OITracer
        except ImportError:
            return
        original = OITracer.start_as_current_span

        @wraps(original)
        @contextmanager
        def scope(instance, *args, **kwargs):
            with original(instance, *args, **kwargs) as span:
                try:
                    yield span
                finally:
                    if cls._scope_patch is not None and cls._scope_patch[2] is scope:
                        try:
                            cls._translator._policy.observe_veto(span)
                        except Exception:
                            logger.debug(
                                "Native OI scope observation failed", exc_info=True
                            )

        OITracer.start_as_current_span = scope
        cls._scope_patch = (OITracer, original, scope)

    def __init__(self, instrumentor_class: type, **kwargs: Any) -> None:
        self._instrumentor_class = instrumentor_class
        kwargs.pop("tracer_provider", None)
        self._instrumentor_kwargs = dict(kwargs)
        self._instrumentor = None
        self._is_instrumented = False
        self._is_span_processor = False
        self.name = f"openinference-{instrumentor_class.__name__}"

    @classmethod
    def _get_translator(cls) -> OpenInferenceTranslator:
        return cls._translator

    @staticmethod
    def _active_processor_state(provider: Any) -> tuple[Any, tuple[Any, ...] | None]:
        active = getattr(provider, "_active_span_processor", None)
        processors = getattr(active, "_span_processors", None) if active else None
        return active, processors

    @staticmethod
    def _delegate_is_active(delegate: Any) -> bool:
        for attribute_name in (
            "is_instrumented_by_opentelemetry",
            "_is_instrumented_by_opentelemetry",
        ):
            value = getattr(delegate, attribute_name, None)
            if value is None:
                continue
            try:
                return bool(value() if callable(value) else value)
            except Exception:
                logger.debug(
                    "Could not inspect OpenInference delegate state",
                    exc_info=True,
                )
        return False

    @classmethod
    def _existing_processor(cls, provider: Any, processor_class: type) -> Any | None:
        """Return an externally registered processor of the requested class."""
        _, processors = cls._active_processor_state(provider)
        if processors is None:
            return None
        return next(
            (
                processor
                for processor in processors
                if isinstance(processor, processor_class)
            ),
            None,
        )

    @classmethod
    def _rebuild_processor_chain(cls, provider: Any) -> None:
        active, processors = cls._active_processor_state(provider)
        if active is None or processors is None:
            return

        processor_delegates = tuple(
            registration.instrumentor
            for registration in cls._registrations.values()
            if registration.is_span_processor
        )
        other_processors = tuple(
            processor
            for processor in processors
            if processor is not cls._translator
            and processor not in processor_delegates
            and processor not in cls._active_span_processors
        )
        if cls._registrations:
            active._span_processors = (
                *processor_delegates,
                cls._translator,
                *other_processors,
            )
            cls._translator_registered = True
        else:
            # Restore adopted processors' original relative positions. Preserve
            # processors added by other owners while this wrapper was active.
            active._span_processors = tuple(
                processor
                for processor in cls._original_order
                if any(processor is item for item in other_processors)
            ) + tuple(
                processor
                for processor in other_processors
                if not any(processor is item for item in cls._original_order)
            )
            cls._translator_registered = False
        cls._active_span_processors = list(processor_delegates)

    @classmethod
    def _ensure_translator_registered(cls, provider: Any) -> None:
        active, processors = cls._active_processor_state(provider)
        if active is not None and processors is not None:
            remaining = tuple(
                processor
                for processor in processors
                if processor is not cls._translator
            )
            active._span_processors = (cls._translator, *remaining)
            cls._translator_registered = True
            return
        add_span_processor = getattr(provider, "add_span_processor", None)
        if callable(add_span_processor):
            add_span_processor(cls._translator)
            cls._translator_registered = True

    @classmethod
    def _remove_processor(cls, provider: Any, target: Any) -> None:
        active, processors = cls._active_processor_state(provider)
        if active is not None and processors is not None:
            active._span_processors = tuple(
                processor for processor in processors if processor is not target
            )

    @classmethod
    def _reset_if_idle(cls) -> None:
        if cls._registrations:
            return
        provider = cls._provider
        if provider is not None:
            cls._remove_processor(provider, cls._translator)
        cls._translator_registered = False
        cls._active_span_processors = []
        cls._provider = None
        cls._original_order = ()
        cls._translator.shutdown()
        if cls._scope_patch is not None:
            owner, original, replacement = cls._scope_patch
            cls._scope_patch = None
            if owner.start_as_current_span is replacement:
                owner.start_as_current_span = original

    def activate(self) -> None:
        """Activate or reference-count one shared OpenInference delegate."""
        if self._is_instrumented:
            return

        cls = OpenInferenceInstrumentor
        with cls._lock:
            if self._is_instrumented:
                return
            provider = trace.get_tracer_provider()
            if cls._provider is not None and cls._provider is not provider:
                raise RuntimeError(
                    "OpenInference delegates are already active on another tracer provider"
                )

            registration = cls._registrations.get(self._instrumentor_class)
            if registration is not None:
                if registration.kwargs != self._instrumentor_kwargs:
                    logger.warning(
                        "%s is already active; the first instrumentation settings win",
                        self.name,
                    )
                registration.refcount += 1
                self._instrumentor = registration.instrumentor
                self._is_span_processor = registration.is_span_processor
                self._is_instrumented = True
                return

            is_standard = callable(
                getattr(self._instrumentor_class, "instrument", None)
            )
            # Processor-style delegates receive their actual constructor
            # options; standard OTel instrumentors receive instrument kwargs.
            external = (
                None
                if is_standard
                else cls._existing_processor(provider, self._instrumentor_class)
            )
            if is_standard:
                delegate = self._instrumentor_class()
            elif external is not None:
                delegate = external
            else:
                delegate = self._instrumentor_class(**self._instrumentor_kwargs)
            is_processor = not is_standard and callable(
                getattr(provider, "add_span_processor", None)
            )
            if not is_standard and not is_processor:
                raise TypeError(
                    f"{self._instrumentor_class.__name__} is neither an "
                    "OpenInference instrumentor nor a span processor"
                )

            owned_activation = False
            _, processor_order = cls._active_processor_state(provider)
            if processor_order is None:
                raise TypeError("OpenInference requires an OTel SDK processor chain")
            if not cls._registrations:
                cls._original_order = tuple(processor_order)
            try:
                if is_standard:
                    already_active = cls._delegate_is_active(delegate)
                    if not already_active:
                        owned_activation = True
                        delegate.instrument(
                            tracer_provider=provider,
                            **self._instrumentor_kwargs,
                        )
                else:
                    external_delegate = cls._existing_processor(
                        provider, self._instrumentor_class
                    )
                    if external_delegate is not None:
                        delegate = external_delegate
                    else:
                        provider.add_span_processor(delegate)
                        owned_activation = True

                cls._provider = provider
                registration = _SharedRegistration(
                    instrumentor=delegate,
                    kwargs=dict(self._instrumentor_kwargs),
                    is_span_processor=is_processor,
                    owned_activation=owned_activation,
                )
                cls._registrations[self._instrumentor_class] = registration
                cls._observe_native_scope()
                if is_processor:
                    cls._active_span_processors.append(delegate)
                    cls._rebuild_processor_chain(provider)
                else:
                    cls._ensure_translator_registered(provider)
                    cls._rebuild_processor_chain(provider)
            except BaseException:
                cls._registrations.pop(self._instrumentor_class, None)
                if owned_activation:
                    cls._remove_processor(provider, delegate)
                if owned_activation and is_standard:
                    uninstrument = getattr(delegate, "uninstrument", None)
                    if callable(uninstrument):
                        try:
                            uninstrument()
                        except Exception:
                            logger.exception(
                                "Failed to roll back partial OpenInference activation"
                            )
                if owned_activation and is_processor:
                    shutdown = getattr(delegate, "shutdown", None)
                    if callable(shutdown):
                        try:
                            shutdown()
                        except Exception:
                            logger.exception(
                                "Failed to roll back OpenInference processor activation"
                            )
                cls._reset_if_idle()
                raise

            self._instrumentor = delegate
            self._is_span_processor = is_processor
            self._is_instrumented = True
            logger.info("%s instrumentation activated (via OpenInference)", self.name)

    def deactivate(self) -> None:
        """Release this wrapper and tear down only the last owned delegate."""
        if not self._is_instrumented:
            return

        cls = OpenInferenceInstrumentor
        with cls._lock:
            if not self._is_instrumented:
                return
            self._is_instrumented = False
            registration = cls._registrations.get(self._instrumentor_class)
            if registration is None:
                self._instrumentor = None
                self._is_span_processor = False
                cls._reset_if_idle()
                return

            registration.refcount = max(0, registration.refcount - 1)
            if registration.refcount:
                self._instrumentor = None
                self._is_span_processor = False
                return

            delegate = registration.instrumentor
            provider = cls._provider or trace.get_tracer_provider()
            cls._registrations.pop(self._instrumentor_class, None)
            try:
                if registration.is_span_processor:
                    if registration.owned_activation:
                        cls._remove_processor(provider, delegate)
                    shutdown = getattr(delegate, "shutdown", None)
                    if registration.owned_activation and callable(shutdown):
                        shutdown()
                elif registration.owned_activation:
                    uninstrument = getattr(delegate, "uninstrument", None)
                    if callable(uninstrument):
                        uninstrument()
            except Exception:
                logger.exception("Failed to deactivate %s", self.name)
            finally:
                cls._active_span_processors = [
                    processor
                    for processor in cls._active_span_processors
                    if processor is not delegate
                ]
                cls._rebuild_processor_chain(provider)
                cls._reset_if_idle()
                self._instrumentor = None
                self._is_span_processor = False
        logger.info("%s instrumentation deactivated", self.name)
