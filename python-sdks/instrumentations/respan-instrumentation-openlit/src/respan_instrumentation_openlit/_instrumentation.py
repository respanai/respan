"""Lifecycle management for OpenLIT's native OpenTelemetry instrumentation."""

from __future__ import annotations

import importlib
import logging
import threading
from importlib.resources import files
from typing import Any

from opentelemetry import trace
from respan_tracing.core.tracer import RespanTracer

from respan_instrumentation_openlit._constants import OPENLIT_INSTRUMENTATION_NAME
from respan_instrumentation_openlit._embeddings import (
    EmbeddingHook,
    install_openai_embedding_hooks,
    remove_openai_embedding_hooks,
)
from respan_instrumentation_openlit._openai_hooks import (
    ChunkHook,
    FactoryHook,
    OpenAIPatch,
    RequestHook,
    capture_openai_patches,
    install_openai_request_hooks,
    install_openai_stream_factory_hooks,
    install_openai_stream_usage_hooks,
    remove_openai_request_hooks,
    remove_openai_stream_factory_hooks,
    remove_openai_stream_usage_hooks,
    restore_openai_patches,
    snapshot_openai_resource_methods,
)
from respan_instrumentation_openlit._processor import OpenLITSpanProcessor

from ._config import restore as restore_native_config
from ._config import snapshot as snapshot_native_config
from ._guard import install_context_guard, install_provider_guard, remove_provider_guard

logger = logging.getLogger(__name__)

_LOCK = threading.RLock()
_REFCOUNT = 0
_PROCESSOR: OpenLITSpanProcessor | None = None
_PROVIDER: Any = None
_OWNED_INSTRUMENTORS: list[Any] = []
_EMBEDDING_HOOKS: list[EmbeddingHook] = []
_REQUEST_HOOKS: list[RequestHook] = []
_STREAM_USAGE_HOOKS: list[ChunkHook] = []
_OPENAI_PATCHES: list[OpenAIPatch] = []
_CONFIG: tuple[Any, ...] | None = None
_TRACER_HOOK = None
_NATIVE_CONFIG = None

_DEFAULT_DISABLED_TRANSPORT_INSTRUMENTORS = (
    "httpx",
    "requests",
    "urllib",
    "urllib3",
)


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


def _uninstrument_owned(instrumentors: list[Any]) -> None:
    for instrumentor in reversed(instrumentors):
        uninstrument = getattr(instrumentor, "uninstrument", None)
        if callable(uninstrument) and _is_instrumented(instrumentor):
            try:
                uninstrument()
            except Exception:
                logger.exception("Failed to deactivate an OpenLIT instrumentor")


def _disabled_instrumentors(
    configured: list[str], *, capture_transport_spans: bool
) -> list[str]:
    result = list(dict.fromkeys(configured))
    if not capture_transport_spans:
        result.extend(
            name
            for name in _DEFAULT_DISABLED_TRANSPORT_INSTRUMENTORS
            if name not in result
        )
    return result


def _init_with_owned_instrumentors(
    openlit: Any,
    instrumentors: dict[str, Any],
    **kwargs: Any,
) -> None:
    """Make OpenLIT initialize the exact instances this adapter can unwind."""

    original = getattr(openlit, "get_all_instrumentors", None)
    if not callable(original):
        openlit.init(**kwargs)
        return

    def owned_instrumentors() -> dict[str, Any]:
        return instrumentors

    openlit.get_all_instrumentors = owned_instrumentors
    try:
        openlit.init(**kwargs)
    finally:
        if getattr(openlit, "get_all_instrumentors", None) is owned_instrumentors:
            openlit.get_all_instrumentors = original


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
        capture_transport_spans: bool = False,
        max_content_length: int = 16_000,
    ) -> None:
        if not isinstance(max_content_length, int) or not (
            128 <= max_content_length <= 16_000
        ):
            raise ValueError("max_content_length must be between 128 and 16000 bytes")
        self._capture_content = capture_content
        self._disabled_instrumentors = list(disabled_instrumentors or [])
        self._pricing_json = pricing_json
        self._disable_metrics = disable_metrics
        self._disable_events = disable_events
        self._capture_transport_spans = capture_transport_spans
        self._max_content_length = max_content_length
        self._is_instrumented = False

    def activate(self) -> None:
        """Activate OpenLIT without adding a second exporter or wrapper span."""
        global _CONFIG, _EMBEDDING_HOOKS, _OPENAI_PATCHES, _OWNED_INSTRUMENTORS
        global _PROCESSOR, _PROVIDER, _REFCOUNT, _REQUEST_HOOKS
        global _STREAM_USAGE_HOOKS, _TRACER_HOOK, _NATIVE_CONFIG

        if not _is_respan_tracing_enabled():
            return
        try:
            openlit = importlib.import_module("openlit")
        except ImportError as exc:
            logger.warning("OpenLIT instrumentation unavailable: %s", exc)
            return

        with _LOCK:
            if self._is_instrumented:
                return
            pricing_json = self._pricing_json or str(
                files("respan_instrumentation_openlit").joinpath("_pricing.json")
            )
            disabled_instrumentors = _disabled_instrumentors(
                self._disabled_instrumentors,
                capture_transport_spans=self._capture_transport_spans,
            )
            config = (
                self._capture_content,
                tuple(disabled_instrumentors),
                pricing_json,
                self._disable_metrics,
                self._disable_events,
                self._capture_transport_spans,
                self._max_content_length,
            )
            if _REFCOUNT and _CONFIG != config:
                raise RuntimeError(
                    "OpenLIT is already active with a different adapter configuration"
                )
            if _REFCOUNT and _PROVIDER is not trace.get_tracer_provider():
                raise RuntimeError(
                    "OpenLIT is already active with a different tracing provider"
                )
            if _REFCOUNT == 0:
                instrumentors = _instrumentors()
                openai_before = snapshot_openai_resource_methods()
                before = {
                    name: _is_instrumented(value)
                    for name, value in instrumentors.items()
                }
                request_hooks: list[RequestHook] = []
                factory_hooks: list[FactoryHook] = []
                openai_patches: list[OpenAIPatch] = []
                stream_usage_hooks: list[ChunkHook] = []
                embedding_hooks: list[EmbeddingHook] = []
                processor: OpenLITSpanProcessor | None = None
                provider: Any = None
                tracer_hook = None
                native_before = snapshot_native_config()
                owned_instrumentors: list[Any] = []
                try:
                    provider = trace.get_tracer_provider()
                    processor = OpenLITSpanProcessor(
                        capture_content=self._capture_content
                    )
                    _PROCESSOR = processor
                    tracer_hook = install_provider_guard(provider, processor)
                    processor._detach_hook = install_context_guard(processor)
                    _register_first(provider, processor)
                    request_hooks = install_openai_request_hooks(
                        capture_content=self._capture_content,
                        max_content_length=self._max_content_length,
                    )
                    factory_hooks = install_openai_stream_factory_hooks(
                        max_content_length=self._max_content_length
                    )
                    try:
                        _init_with_owned_instrumentors(
                            openlit,
                            instrumentors,
                            capture_message_content=self._capture_content,
                            disabled_instrumentors=disabled_instrumentors,
                            disable_metrics=self._disable_metrics,
                            disable_events=self._disable_events,
                            pricing_json=pricing_json,
                            max_content_length=self._max_content_length,
                        )
                    finally:
                        remove_openai_stream_factory_hooks(factory_hooks)
                        factory_hooks = []
                    openai_patches = capture_openai_patches(openai_before)
                    if (
                        "openai" in instrumentors
                        and "openai" not in disabled_instrumentors
                        and not _is_instrumented(instrumentors["openai"])
                    ):
                        raise RuntimeError(
                            "OpenLIT did not activate its OpenAI instrumentor"
                        )
                    after = instrumentors
                    owned_instrumentors = [
                        value
                        for name, value in after.items()
                        if _is_instrumented(value) and not before.get(name, False)
                    ]
                    stream_usage_hooks = install_openai_stream_usage_hooks()
                    embedding_hooks = install_openai_embedding_hooks(
                        capture_content=self._capture_content,
                        max_content_length=self._max_content_length,
                    )
                except Exception:  # noqa: BLE001 - transactional third-party install.
                    if not owned_instrumentors:
                        try:
                            after = instrumentors
                            owned_instrumentors = [
                                value
                                for name, value in after.items()
                                if _is_instrumented(value)
                                and not before.get(name, False)
                            ]
                        except Exception:  # noqa: BLE001 - best-effort rollback scan.
                            owned_instrumentors = []
                    if processor is not None and provider is not None:
                        _unregister(provider, processor)
                    remove_openai_embedding_hooks(embedding_hooks)
                    remove_openai_stream_usage_hooks(stream_usage_hooks)
                    remove_openai_stream_factory_hooks(factory_hooks)
                    if not openai_patches:
                        openai_patches = capture_openai_patches(openai_before)
                    _uninstrument_owned(owned_instrumentors)
                    restore_openai_patches(openai_patches)
                    remove_openai_request_hooks(request_hooks)
                    if processor is not None:
                        processor.shutdown()
                    _PROCESSOR = None
                    remove_provider_guard(tracer_hook)
                    restore_native_config(native_before, snapshot_native_config())
                    raise RuntimeError("Failed to activate OpenLIT instrumentation")

                _REQUEST_HOOKS = request_hooks
                _OPENAI_PATCHES = openai_patches
                _STREAM_USAGE_HOOKS = stream_usage_hooks
                _EMBEDDING_HOOKS = embedding_hooks
                _OWNED_INSTRUMENTORS = owned_instrumentors
                _PROVIDER = provider
                _TRACER_HOOK = tracer_hook
                _NATIVE_CONFIG = (native_before, snapshot_native_config())
                _PROCESSOR = processor
                _CONFIG = config

            _REFCOUNT += 1
            self._is_instrumented = True

    def deactivate(self) -> None:
        """Remove Respan normalization and only OpenLIT hooks owned by this adapter."""
        global _CONFIG, _EMBEDDING_HOOKS, _OPENAI_PATCHES, _OWNED_INSTRUMENTORS
        global _PROCESSOR, _PROVIDER, _REFCOUNT, _REQUEST_HOOKS
        global _STREAM_USAGE_HOOKS, _TRACER_HOOK, _NATIVE_CONFIG

        with _LOCK:
            if not self._is_instrumented:
                return
            self._is_instrumented = False
            _REFCOUNT = max(0, _REFCOUNT - 1)
            if _REFCOUNT:
                return
            if _PROCESSOR is not None and _PROVIDER is not None:
                _PROCESSOR.retire(_PROVIDER)
            remove_openai_embedding_hooks(_EMBEDDING_HOOKS)
            _EMBEDDING_HOOKS = []
            remove_openai_stream_usage_hooks(_STREAM_USAGE_HOOKS)
            _STREAM_USAGE_HOOKS = []
            _uninstrument_owned(_OWNED_INSTRUMENTORS)
            _OWNED_INSTRUMENTORS = []
            restore_openai_patches(_OPENAI_PATCHES)
            _OPENAI_PATCHES = []
            remove_openai_request_hooks(_REQUEST_HOOKS)
            _REQUEST_HOOKS = []
            if _NATIVE_CONFIG is not None:
                restore_native_config(*_NATIVE_CONFIG)
            _NATIVE_CONFIG = None
            remove_provider_guard(_TRACER_HOOK)
            _TRACER_HOOK = None
            _PROCESSOR = None
            _PROVIDER = None
            _CONFIG = None
