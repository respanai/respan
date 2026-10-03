"""Microsoft Agent Framework OTEL instrumentation plugin for Respan."""

from __future__ import annotations

import importlib
import logging
from collections.abc import Mapping
from contextvars import ContextVar
from functools import wraps
from threading import Lock
from typing import Any

from opentelemetry import trace
from respan_tracing.core.tracer import RespanTracer

from respan_instrumentation_microsoft_agent_framework import _policy
from respan_instrumentation_microsoft_agent_framework._constants import (
    AGENT_FRAMEWORK_INSTRUMENTATION_NAME,
)
from respan_instrumentation_microsoft_agent_framework._processor import (
    AgentFrameworkSpanProcessor,
)

logger = logging.getLogger(__name__)


_chat_telemetry_patch_lock = Lock()
_chat_telemetry_patch_users = 0
_chat_telemetry_layer: type[Any] | None = None
_chat_observability: Any = None
_original_chat_get_response: Any = None
_patched_chat_get_response: Any = None
_original_get_span_attributes: Any = None
_patched_get_span_attributes: Any = None
_chat_tool_definitions: ContextVar[Any | None] = ContextVar(
    "respan_agent_framework_chat_tool_definitions",
    default=None,
)
_processor_registration_lock = Lock()
_shared_processor_provider: Any = None
_shared_processor: AgentFrameworkSpanProcessor | None = None
_shared_processor_users = 0


def _patch_chat_tool_capture(observability: Any) -> bool:
    """Include request tools in Agent Framework's native chat telemetry.

    Agent Framework accepts tools through ``options`` but its telemetry layer
    currently derives span attributes from other keyword arguments. A scoped
    context value lets the attribute builder see those tools without changing
    any arguments forwarded to the provider. The Respan processor then maps
    ``gen_ai.tool.definitions`` to canonical ``llm.request.functions`` and
    strips the raw attribute.
    """
    global _chat_telemetry_layer
    global _chat_observability
    global _chat_telemetry_patch_users
    global _original_chat_get_response
    global _patched_chat_get_response
    global _original_get_span_attributes
    global _patched_get_span_attributes

    layer = getattr(observability, "ChatTelemetryLayer", None)
    if layer is None:
        logger.warning(
            "Microsoft Agent Framework chat tool capture is unavailable: "
            "ChatTelemetryLayer was not found"
        )
        return False

    with _chat_telemetry_patch_lock:
        if _chat_telemetry_patch_users:
            if layer is not _chat_telemetry_layer:
                logger.warning(
                    "Microsoft Agent Framework ChatTelemetryLayer changed while "
                    "instrumentation was active"
                )
                return False
            _chat_telemetry_patch_users += 1
            return True

        original = getattr(layer, "get_response", None)
        get_span_attributes = getattr(observability, "_get_span_attributes", None)
        if not callable(original) or not callable(get_span_attributes):
            logger.warning(
                "Microsoft Agent Framework chat tool capture is unavailable: "
                "required telemetry hooks were not found"
            )
            return False

        @wraps(original)
        def get_response_with_tool_capture(self: Any, *args: Any, **kwargs: Any) -> Any:
            if (
                not _chat_telemetry_patch_users
                or _patched_chat_get_response is not get_response_with_tool_capture
            ):
                return original(self, *args, **kwargs)
            options = kwargs.get("options")
            tools = options.get("tools") if isinstance(options, Mapping) else None
            token = _chat_tool_definitions.set(tools)
            try:
                return original(self, *args, **kwargs)
            finally:
                _chat_tool_definitions.reset(token)

        @wraps(get_span_attributes)
        def get_span_attributes_with_tool_capture(*args: Any, **kwargs: Any) -> Any:
            if (
                not _chat_telemetry_patch_users
                or _patched_get_span_attributes
                is not get_span_attributes_with_tool_capture
            ):
                return get_span_attributes(*args, **kwargs)
            tools = _chat_tool_definitions.get() if _policy.capture_content() else None
            if tools not in (None, (), []) and kwargs.get("tools") in (None, (), []):
                kwargs = {**kwargs, "tools": tools}
            return get_span_attributes(*args, **kwargs)

        _chat_telemetry_layer = layer
        _chat_observability = observability
        _original_chat_get_response = original
        _patched_chat_get_response = get_response_with_tool_capture
        _original_get_span_attributes = get_span_attributes
        _patched_get_span_attributes = get_span_attributes_with_tool_capture
        layer.get_response = get_response_with_tool_capture
        observability._get_span_attributes = get_span_attributes_with_tool_capture
        _chat_telemetry_patch_users = 1
        return True


def _unpatch_chat_tool_capture() -> None:
    global _chat_telemetry_layer
    global _chat_observability
    global _chat_telemetry_patch_users
    global _original_chat_get_response
    global _patched_chat_get_response
    global _original_get_span_attributes
    global _patched_get_span_attributes

    with _chat_telemetry_patch_lock:
        if not _chat_telemetry_patch_users:
            return
        _chat_telemetry_patch_users -= 1
        if _chat_telemetry_patch_users:
            return

        if (
            _chat_telemetry_layer is not None
            and getattr(_chat_telemetry_layer, "get_response", None)
            is _patched_chat_get_response
        ):
            _chat_telemetry_layer.get_response = _original_chat_get_response
        if (
            _chat_observability is not None
            and getattr(_chat_observability, "_get_span_attributes", None)
            is _patched_get_span_attributes
        ):
            _chat_observability._get_span_attributes = _original_get_span_attributes
        _chat_telemetry_layer = None
        _chat_observability = None
        _original_chat_get_response = None
        _patched_chat_get_response = None
        _original_get_span_attributes = None
        _patched_get_span_attributes = None


def _active_span_processors(tracer_provider: Any) -> tuple[Any, tuple[Any, ...] | None]:
    active_span_processor = getattr(tracer_provider, "_active_span_processor", None)
    processors = (
        getattr(active_span_processor, "_span_processors", None)
        if active_span_processor is not None
        else None
    )
    return active_span_processor, processors


def _register_processor(
    tracer_provider: Any,
    processor: AgentFrameworkSpanProcessor,
) -> None:
    active_span_processor, processors = _active_span_processors(tracer_provider)
    if active_span_processor is None or processors is None:
        if hasattr(tracer_provider, "add_span_processor"):
            tracer_provider.add_span_processor(processor)
            return
        raise RuntimeError(
            "Initialize an OpenTelemetry tracer provider before activating Agent Framework instrumentation"
        )

    remaining_processors = tuple(
        existing_processor
        for existing_processor in processors
        if existing_processor is not processor
    )
    active_span_processor._span_processors = (processor, *remaining_processors)


def _unregister_processor(
    tracer_provider: Any,
    processor: AgentFrameworkSpanProcessor,
) -> None:
    active_span_processor, processors = _active_span_processors(tracer_provider)
    if active_span_processor is None or processors is None:
        return
    active_span_processor._span_processors = tuple(
        existing_processor
        for existing_processor in processors
        if existing_processor is not processor
    )


def _acquire_shared_processor(tracer_provider: Any) -> AgentFrameworkSpanProcessor:
    """Register one normalizer per process-wide tracer provider."""
    global _shared_processor
    global _shared_processor_provider
    global _shared_processor_users

    with _processor_registration_lock:
        if _shared_processor_users:
            if tracer_provider is not _shared_processor_provider:
                raise RuntimeError(
                    "Microsoft Agent Framework instrumentation cannot switch tracer "
                    "providers while active"
                )
            _shared_processor_users += 1
            assert _shared_processor is not None
            return _shared_processor

        processor = AgentFrameworkSpanProcessor()
        _register_processor(tracer_provider=tracer_provider, processor=processor)
        _shared_processor_provider = tracer_provider
        _shared_processor = processor
        _shared_processor_users = 1
        return processor


def _release_shared_processor(
    tracer_provider: Any,
    processor: AgentFrameworkSpanProcessor,
) -> None:
    """Release a shared normalizer and unregister it after the final user."""
    global _shared_processor
    global _shared_processor_provider
    global _shared_processor_users

    with _processor_registration_lock:
        if (
            not _shared_processor_users
            or tracer_provider is not _shared_processor_provider
            or processor is not _shared_processor
        ):
            return

        _shared_processor_users -= 1
        if _shared_processor_users:
            return

        try:
            _unregister_processor(
                tracer_provider=tracer_provider,
                processor=processor,
            )
        finally:
            processor.shutdown()
            _shared_processor_provider = None
            _shared_processor = None


class MicrosoftAgentFrameworkInstrumentor:
    """Respan instrumentor for Microsoft Agent Framework native OTEL spans."""

    name = AGENT_FRAMEWORK_INSTRUMENTATION_NAME

    def __init__(self, *, capture_content: bool = True) -> None:
        self._capture_content = capture_content
        self._processor: AgentFrameworkSpanProcessor | None = None
        self._tracer_provider: Any = None
        self._is_instrumented = False
        self._chat_tool_capture_patched = False
        self._policy_acquired = False

    @staticmethod
    def _is_respan_tracing_enabled() -> bool:
        tracer = getattr(RespanTracer, "_instance", None)
        if tracer is None:
            return True
        return bool(getattr(tracer, "is_enabled", True))

    def _enable_agent_framework_observability(self) -> Any | None:
        try:
            observability = importlib.import_module("agent_framework.observability")
        except ImportError as exc:
            logger.warning(
                "Failed to activate Microsoft Agent Framework instrumentation - "
                "missing dependency: %s",
                exc,
            )
            return None

        settings = getattr(observability, "OBSERVABILITY_SETTINGS", None)
        if bool(getattr(settings, "is_user_disabled", False)):
            logger.info(
                "Microsoft Agent Framework instrumentation skipped because "
                "Agent Framework observability is user-disabled"
            )
            return None

        self._policy_acquired = _policy.acquire(observability, self._capture_content)
        return observability if self._policy_acquired else None

    def activate(self) -> None:
        """Activate Respan normalization for Microsoft Agent Framework spans."""
        if self._is_instrumented:
            return

        if not self._is_respan_tracing_enabled():
            logger.info(
                "Microsoft Agent Framework instrumentation skipped because "
                "Respan tracing is disabled"
            )
            return

        observability = self._enable_agent_framework_observability()
        if observability is None:
            return

        tracer_provider = trace.get_tracer_provider()
        try:
            self._chat_tool_capture_patched = _patch_chat_tool_capture(observability)
            self._processor = _acquire_shared_processor(tracer_provider)
            self._tracer_provider = tracer_provider
        except Exception:
            if self._chat_tool_capture_patched:
                _unpatch_chat_tool_capture()
                self._chat_tool_capture_patched = False
            if self._policy_acquired:
                _policy.release()
                self._policy_acquired = False
            raise
        self._is_instrumented = True
        logger.info("Microsoft Agent Framework instrumentation activated")

    def deactivate(self) -> None:
        """Deactivate Respan normalization for Microsoft Agent Framework spans."""
        if not self._is_instrumented:
            return

        try:
            if self._processor is not None and self._tracer_provider is not None:
                _release_shared_processor(
                    tracer_provider=self._tracer_provider,
                    processor=self._processor,
                )
        finally:
            try:
                if self._chat_tool_capture_patched:
                    _unpatch_chat_tool_capture()
                    self._chat_tool_capture_patched = False
            finally:
                if self._policy_acquired:
                    _policy.release()
                    self._policy_acquired = False
                self._processor = None
                self._tracer_provider = None
                self._is_instrumented = False
        logger.info("Microsoft Agent Framework instrumentation deactivated")
