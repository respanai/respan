"""LiveKit Agents instrumentation plugin for Respan."""

from __future__ import annotations

import functools
import importlib
import logging
import time
from typing import Any, Callable

from opentelemetry import trace

from respan_instrumentation_livekit._constants import (
    LIVEKIT_INSTRUMENTATION_NAME,
    LIVEKIT_RESPAN_TOOL_DEFINITIONS_ATTR,
)
from respan_instrumentation_livekit._otel_emitter import emit_livekit_tool_span
from respan_instrumentation_livekit._processor import LiveKitSpanProcessor
from respan_instrumentation_livekit._serialization import get_value, safe_json
from respan_instrumentation_livekit._translator import normalize_livekit_tools
from respan_tracing.core.tracer import RespanTracer

logger = logging.getLogger(__name__)

_ORIGINAL_UTILS_EXECUTE_FUNCTION_CALL: Callable[..., Any] | None = None
_ORIGINAL_EXPORTED_EXECUTE_FUNCTION_CALL: Callable[..., Any] | None = None
_ORIGINAL_LLM_STREAM_MAIN_TASK: Callable[..., Any] | None = None
_PATCHED_UTILS_MODULE: Any = None
_PATCHED_LLM_MODULE: Any = None
_PATCHED_LLM_STREAM_CLASS: Any = None
_ACTIVE_INSTANCES = 0


def _active_span_processors() -> tuple[Any, tuple[Any, ...] | None]:
    tracer_provider = trace.get_tracer_provider()
    active_span_processor = getattr(tracer_provider, "_active_span_processor", None)
    processors = (
        getattr(active_span_processor, "_span_processors", None)
        if active_span_processor is not None
        else None
    )
    return active_span_processor, processors


def _register_processor(processor: LiveKitSpanProcessor) -> None:
    tracer_provider = trace.get_tracer_provider()
    active_span_processor, processors = _active_span_processors()
    if active_span_processor is not None and processors is not None:
        remaining_processors = tuple(
            existing for existing in processors if existing is not processor
        )
        active_span_processor._span_processors = (processor, *remaining_processors)
        return

    if hasattr(tracer_provider, "add_span_processor"):
        tracer_provider.add_span_processor(processor)


def _unregister_processor(processor: LiveKitSpanProcessor) -> None:
    active_span_processor, processors = _active_span_processors()
    if active_span_processor is None or processors is None:
        return
    active_span_processor._span_processors = tuple(
        existing for existing in processors if existing is not processor
    )


def _patch_execute_function_call(utils_module: Any, llm_module: Any) -> None:
    global _ORIGINAL_UTILS_EXECUTE_FUNCTION_CALL
    global _ORIGINAL_EXPORTED_EXECUTE_FUNCTION_CALL
    global _PATCHED_UTILS_MODULE
    global _PATCHED_LLM_MODULE

    if _ORIGINAL_UTILS_EXECUTE_FUNCTION_CALL is not None:
        return

    original = getattr(utils_module, "execute_function_call")
    exported_original = getattr(llm_module, "execute_function_call", None)

    @functools.wraps(original)
    async def wrapped_execute_function_call(*args: Any, **kwargs: Any) -> Any:
        start_time_ns = time.time_ns()
        result = await original(*args, **kwargs)

        tool_call = args[0] if args else kwargs.get("tool_call")
        tool_name = (
            get_value(get_value(result, "fnc_call"), "name")
            or get_value(tool_call, "name")
            or "livekit.tool"
        )
        arguments = (
            get_value(get_value(result, "fnc_call"), "arguments")
            or get_value(tool_call, "arguments")
            or {}
        )
        call_id = (
            get_value(get_value(result, "fnc_call"), "call_id")
            or get_value(tool_call, "call_id")
        )
        raw_exception = get_value(result, "raw_exception")
        fnc_call_out = get_value(result, "fnc_call_out")
        output = get_value(result, "raw_output")
        if output is None and fnc_call_out is not None:
            output = get_value(fnc_call_out, "output")

        emit_livekit_tool_span(
            tool_name=str(tool_name),
            arguments=arguments,
            output=output,
            call_id=str(call_id) if call_id else None,
            start_time_ns=start_time_ns,
            error=raw_exception if isinstance(raw_exception, BaseException) else None,
        )
        return result

    _ORIGINAL_UTILS_EXECUTE_FUNCTION_CALL = original
    _ORIGINAL_EXPORTED_EXECUTE_FUNCTION_CALL = exported_original
    _PATCHED_UTILS_MODULE = utils_module
    _PATCHED_LLM_MODULE = llm_module
    setattr(utils_module, "execute_function_call", wrapped_execute_function_call)
    if exported_original is not None:
        setattr(llm_module, "execute_function_call", wrapped_execute_function_call)


def _restore_execute_function_call() -> None:
    global _ORIGINAL_UTILS_EXECUTE_FUNCTION_CALL
    global _ORIGINAL_EXPORTED_EXECUTE_FUNCTION_CALL
    global _PATCHED_UTILS_MODULE
    global _PATCHED_LLM_MODULE

    if _PATCHED_UTILS_MODULE is not None and _ORIGINAL_UTILS_EXECUTE_FUNCTION_CALL:
        setattr(
            _PATCHED_UTILS_MODULE,
            "execute_function_call",
            _ORIGINAL_UTILS_EXECUTE_FUNCTION_CALL,
        )
    if _PATCHED_LLM_MODULE is not None and _ORIGINAL_EXPORTED_EXECUTE_FUNCTION_CALL:
        setattr(
            _PATCHED_LLM_MODULE,
            "execute_function_call",
            _ORIGINAL_EXPORTED_EXECUTE_FUNCTION_CALL,
        )

    _ORIGINAL_UTILS_EXECUTE_FUNCTION_CALL = None
    _ORIGINAL_EXPORTED_EXECUTE_FUNCTION_CALL = None
    _PATCHED_UTILS_MODULE = None
    _PATCHED_LLM_MODULE = None


def _patch_llm_stream_main_task(llm_stream_class: Any) -> None:
    global _ORIGINAL_LLM_STREAM_MAIN_TASK
    global _PATCHED_LLM_STREAM_CLASS

    if _ORIGINAL_LLM_STREAM_MAIN_TASK is not None:
        return

    original = getattr(llm_stream_class, "_main_task")

    @functools.wraps(original)
    async def wrapped_main_task(self: Any, *args: Any, **kwargs: Any) -> Any:
        current_span = trace.get_current_span()
        tool_definitions = normalize_livekit_tools(getattr(self, "_tools", None))
        if tool_definitions:
            try:
                current_span.set_attribute(
                    LIVEKIT_RESPAN_TOOL_DEFINITIONS_ATTR,
                    safe_json(tool_definitions),
                )
            except Exception:
                logger.debug("Failed to attach LiveKit tool definitions", exc_info=True)
        return await original(self, *args, **kwargs)

    _ORIGINAL_LLM_STREAM_MAIN_TASK = original
    _PATCHED_LLM_STREAM_CLASS = llm_stream_class
    setattr(llm_stream_class, "_main_task", wrapped_main_task)


def _restore_llm_stream_main_task() -> None:
    global _ORIGINAL_LLM_STREAM_MAIN_TASK
    global _PATCHED_LLM_STREAM_CLASS

    if _PATCHED_LLM_STREAM_CLASS is not None and _ORIGINAL_LLM_STREAM_MAIN_TASK:
        setattr(_PATCHED_LLM_STREAM_CLASS, "_main_task", _ORIGINAL_LLM_STREAM_MAIN_TASK)
    _ORIGINAL_LLM_STREAM_MAIN_TASK = None
    _PATCHED_LLM_STREAM_CLASS = None


class LiveKitInstrumentor:
    """Respan instrumentor for LiveKit Agents."""

    name = LIVEKIT_INSTRUMENTATION_NAME

    def __init__(self) -> None:
        self._processor = LiveKitSpanProcessor()
        self._is_instrumented = False

    @staticmethod
    def _is_respan_tracing_enabled() -> bool:
        tracer = getattr(RespanTracer, "_instance", None)
        if tracer is None:
            return True
        return bool(getattr(tracer, "is_enabled", True))

    def activate(self) -> None:
        """Register LiveKit span translation and tool execution hooks."""
        global _ACTIVE_INSTANCES

        if self._is_instrumented:
            return

        if not self._is_respan_tracing_enabled():
            logger.info("LiveKit instrumentation skipped because Respan tracing is disabled")
            return

        try:
            livekit_telemetry = importlib.import_module("livekit.agents.telemetry")
            livekit_llm = importlib.import_module("livekit.agents.llm")
            livekit_llm_utils = importlib.import_module("livekit.agents.llm.utils")
        except ImportError as exc:
            logger.warning(
                "Failed to activate LiveKit instrumentation - missing dependency: %s",
                exc,
            )
            return

        set_tracer_provider = getattr(livekit_telemetry, "set_tracer_provider", None)
        if callable(set_tracer_provider):
            set_tracer_provider(trace.get_tracer_provider())

        _register_processor(self._processor)
        _patch_execute_function_call(livekit_llm_utils, livekit_llm)
        _patch_llm_stream_main_task(getattr(livekit_llm, "LLMStream"))

        _ACTIVE_INSTANCES += 1
        self._is_instrumented = True
        logger.info("LiveKit instrumentation activated")

    def deactivate(self) -> None:
        """Restore LiveKit hooks and remove the span processor."""
        global _ACTIVE_INSTANCES

        if not self._is_instrumented:
            return

        _unregister_processor(self._processor)
        _ACTIVE_INSTANCES = max(0, _ACTIVE_INSTANCES - 1)
        if _ACTIVE_INSTANCES == 0:
            _restore_execute_function_call()
            _restore_llm_stream_main_task()

        self._is_instrumented = False
        logger.info("LiveKit instrumentation deactivated")
