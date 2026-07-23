"""Lifecycle management for OpenLIT's native OpenTelemetry instrumentation."""

from __future__ import annotations

import importlib
import logging
import threading
from importlib.resources import files
from typing import Any

from opentelemetry import trace

from respan_instrumentation_openlit._constants import OPENLIT_INSTRUMENTATION_NAME
from respan_instrumentation_openlit._embeddings import (
    EmbeddingHook,
    install_openai_embedding_hooks,
    remove_openai_embedding_hooks,
)
from respan_instrumentation_openlit._processor import OpenLITSpanProcessor
from respan_tracing.core.tracer import RespanTracer

logger = logging.getLogger(__name__)

_LOCK = threading.RLock()
_REFCOUNT = 0
_PROCESSOR: OpenLITSpanProcessor | None = None
_PROVIDER: Any = None
_OWNED_INSTRUMENTORS: list[Any] = []
_EMBEDDING_HOOKS: list[EmbeddingHook] = []


def _is_respan_tracing_enabled() -> bool:
    tracer = getattr(RespanTracer, "_instance", None)
    if tracer is None:
        return True
    return bool(getattr(tracer, "is_enabled", True))


def _instrumentors() -> dict[str, Any]:
    try:
        registry = importlib.import_module("openlit._instrumentors")
        return dict(registry.get_all_instrumentors())
    except (ImportError, AttributeError, TypeError):
        return {}


def _is_instrumented(instrumentor: Any) -> bool:
    for name in (
        "is_instrumented_by_opentelemetry",
        "_is_instrumented_by_opentelemetry",
    ):
        value = getattr(instrumentor, name, None)
        if value is not None:
            return bool(value() if callable(value) else value)
    return False


def _active_span_processors(provider: Any) -> tuple[Any, tuple[Any, ...] | None]:
    active = getattr(provider, "_active_span_processor", None)
    processors = getattr(active, "_span_processors", None) if active else None
    return active, processors


def _register_first(provider: Any, processor: OpenLITSpanProcessor) -> None:
    active, processors = _active_span_processors(provider)
    if active is not None and processors is not None:
        remaining = tuple(item for item in processors if item is not processor)
        active._span_processors = (processor, *remaining)
    elif hasattr(provider, "add_span_processor"):
        provider.add_span_processor(processor)


def _unregister(provider: Any, processor: OpenLITSpanProcessor) -> None:
    active, processors = _active_span_processors(provider)
    if active is not None and processors is not None:
        active._span_processors = tuple(
            item for item in processors if item is not processor
        )


class OpenLITInstrumentor:
    """Enable OpenLIT once and normalize its native spans for Respan."""

    name = OPENLIT_INSTRUMENTATION_NAME

    def __init__(
        self,
        *,
        capture_content: bool = True,
        disabled_instrumentors: list[str] | None = None,
        pricing_json: str | None = None,
        disable_metrics: bool = True,
        disable_events: bool = True,
    ) -> None:
        self._capture_content = capture_content
        self._disabled_instrumentors = list(disabled_instrumentors or [])
        self._pricing_json = pricing_json
        self._disable_metrics = disable_metrics
        self._disable_events = disable_events
        self._is_instrumented = False

    def activate(self) -> None:
        """Activate OpenLIT without adding a second exporter or wrapper span."""
        global _EMBEDDING_HOOKS, _OWNED_INSTRUMENTORS, _PROCESSOR, _PROVIDER, _REFCOUNT

        if self._is_instrumented or not _is_respan_tracing_enabled():
            return
        try:
            openlit = importlib.import_module("openlit")
        except ImportError as exc:
            logger.warning("OpenLIT instrumentation unavailable: %s", exc)
            return

        with _LOCK:
            if _REFCOUNT == 0:
                before = {
                    name: _is_instrumented(value)
                    for name, value in _instrumentors().items()
                }
                pricing_json = self._pricing_json or str(
                    files("respan_instrumentation_openlit").joinpath("_pricing.json")
                )
                openlit.init(
                    capture_message_content=self._capture_content,
                    disabled_instrumentors=self._disabled_instrumentors,
                    disable_metrics=self._disable_metrics,
                    disable_events=self._disable_events,
                    pricing_json=pricing_json,
                )
                _EMBEDDING_HOOKS = install_openai_embedding_hooks(
                    capture_content=self._capture_content
                )
                _PROVIDER = trace.get_tracer_provider()
                _PROCESSOR = OpenLITSpanProcessor(capture_content=self._capture_content)
                _register_first(_PROVIDER, _PROCESSOR)
                after = _instrumentors()
                _OWNED_INSTRUMENTORS = [
                    value
                    for name, value in after.items()
                    if _is_instrumented(value) and not before.get(name, False)
                ]
            elif _PROCESSOR is not None and (
                _PROCESSOR.capture_content != self._capture_content
            ):
                logger.warning(
                    "OpenLIT is already active; the first capture_content setting wins"
                )

            _REFCOUNT += 1
            self._is_instrumented = True

    def deactivate(self) -> None:
        """Remove Respan normalization and only OpenLIT hooks owned by this adapter."""
        global _EMBEDDING_HOOKS, _OWNED_INSTRUMENTORS, _PROCESSOR, _PROVIDER, _REFCOUNT

        if not self._is_instrumented:
            return
        with _LOCK:
            self._is_instrumented = False
            _REFCOUNT = max(0, _REFCOUNT - 1)
            if _REFCOUNT:
                return
            if _PROCESSOR is not None and _PROVIDER is not None:
                _unregister(_PROVIDER, _PROCESSOR)
            remove_openai_embedding_hooks(_EMBEDDING_HOOKS)
            _EMBEDDING_HOOKS = []
            for instrumentor in reversed(_OWNED_INSTRUMENTORS):
                uninstrument = getattr(instrumentor, "uninstrument", None)
                if callable(uninstrument) and _is_instrumented(instrumentor):
                    try:
                        uninstrument()
                    except Exception:
                        logger.exception("Failed to deactivate an OpenLIT instrumentor")
            _OWNED_INSTRUMENTORS = []
            _PROCESSOR = None
            _PROVIDER = None
