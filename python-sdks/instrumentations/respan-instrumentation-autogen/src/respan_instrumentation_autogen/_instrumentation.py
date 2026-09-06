"""AutoGen AgentChat instrumentation plugin for Respan."""

import importlib
import logging
from collections.abc import Mapping, Sequence
from threading import Lock
from typing import Any

from opentelemetry import context as context_api
from opentelemetry import trace

from respan_instrumentation_autogen._native_processor import (
    AutoGenNativeSpanProcessor,
)
from respan_instrumentation_openinference import OpenInferenceInstrumentor
from respan_tracing.core.tracer import RespanTracer

logger = logging.getLogger(__name__)

AUTOGEN_INSTRUMENTATION_NAME = "autogen"
OPENINFERENCE_AUTOGEN_AGENTCHAT_MODULE = (
    "openinference.instrumentation.autogen_agentchat"
)
OPENINFERENCE_AUTOGEN_AGENTCHAT_WRAPPERS_MODULE = (
    "openinference.instrumentation.autogen_agentchat._wrappers"
)
OPENINFERENCE_AUTOGEN_AGENTCHAT_CLASS_NAMES = (
    "AutogenAgentChatInstrumentor",
    "AutoGenAgentChatInstrumentor",
    "AutoGenInstrumentor",
)
TRACER_PROVIDER_KWARG = "tracer_provider"

_TOOL_EXTRACTOR_PATCH_LOCK = Lock()
_TOOL_EXTRACTOR_PATCH_REFCOUNT = 0
_ORIGINAL_TOOL_EXTRACTOR = None
_AGENT_WRAPPER_PATCHES: list[tuple[type, Any, Any]] = []

_AGENT_WRAPPER_CLASS_NAMES = (
    "_AssistantAgentOnMessagesStreamWrapper",
    "_BaseChatAgentOnMessagesStreamWrapper",
)


def _respan_get_llm_tool_attributes(tools: Any) -> dict[str, Any]:
    """Capture both AutoGen Tool objects and current ToolSchema mappings.

    OpenInference AutoGen 0.1.11 accepts ``Tool | ToolSchema`` at the model
    boundary but serializes only ``Tool`` instances. AutoGen AgentChat 0.7.5
    normally passes the mapping form returned by a workbench, so its complete
    name/description/parameter schema otherwise disappears before the span is
    started.
    """

    if not isinstance(tools, Sequence) or isinstance(tools, str | bytes):
        return {}

    from openinference.instrumentation import safe_json_dumps
    from openinference.semconv.trace import SpanAttributes, ToolAttributes

    attributes: dict[str, Any] = {}
    for tool_index, tool in enumerate(tools):
        if isinstance(tool, Mapping):
            tool_schema: Any = dict(tool)
        else:
            tool_schema = getattr(tool, "schema", None)
        if tool_schema is None:
            continue

        if not isinstance(tool_schema, str):
            tool_schema = safe_json_dumps(tool_schema)
        attributes[
            f"{SpanAttributes.LLM_TOOLS}.{tool_index}."
            f"{ToolAttributes.TOOL_JSON_SCHEMA}"
        ] = tool_schema
    return attributes


def _capture_current_agent_output(event: Any) -> None:
    """Attach an AutoGen Response while the owning agent span is current."""

    from autogen_agentchat.base import Response
    from openinference.instrumentation import get_output_attributes

    if not isinstance(event, Response):
        return
    span = trace.get_current_span()
    set_attributes = getattr(span, "set_attributes", None)
    if callable(set_attributes):
        set_attributes(dict(get_output_attributes(event)))


def _agent_wrapper_with_output_capture(original_call: Any) -> Any:
    def call(self: Any, wrapped: Any, instance: Any, args: Any, kwargs: Any) -> Any:
        generator = original_call(self, wrapped, instance, args, kwargs)
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return generator

        async def generator_with_output_capture():
            async for event in generator:
                _capture_current_agent_output(event)
                yield event

        return generator_with_output_capture()

    return call


def _patch_openinference_tool_extractor() -> None:
    global _ORIGINAL_TOOL_EXTRACTOR, _TOOL_EXTRACTOR_PATCH_REFCOUNT

    with _TOOL_EXTRACTOR_PATCH_LOCK:
        wrappers = importlib.import_module(
            OPENINFERENCE_AUTOGEN_AGENTCHAT_WRAPPERS_MODULE
        )
        if _TOOL_EXTRACTOR_PATCH_REFCOUNT == 0:
            _ORIGINAL_TOOL_EXTRACTOR = wrappers._get_llm_tool_attributes
            wrappers._get_llm_tool_attributes = _respan_get_llm_tool_attributes
            for class_name in _AGENT_WRAPPER_CLASS_NAMES:
                wrapper_class = getattr(wrappers, class_name)
                original_call = wrapper_class.__call__
                patched_call = _agent_wrapper_with_output_capture(original_call)
                wrapper_class.__call__ = patched_call
                _AGENT_WRAPPER_PATCHES.append(
                    (wrapper_class, original_call, patched_call)
                )
        _TOOL_EXTRACTOR_PATCH_REFCOUNT += 1


def _unpatch_openinference_tool_extractor() -> None:
    global _ORIGINAL_TOOL_EXTRACTOR, _TOOL_EXTRACTOR_PATCH_REFCOUNT

    with _TOOL_EXTRACTOR_PATCH_LOCK:
        if _TOOL_EXTRACTOR_PATCH_REFCOUNT == 0:
            return
        _TOOL_EXTRACTOR_PATCH_REFCOUNT -= 1
        if _TOOL_EXTRACTOR_PATCH_REFCOUNT != 0:
            return

        wrappers = importlib.import_module(
            OPENINFERENCE_AUTOGEN_AGENTCHAT_WRAPPERS_MODULE
        )
        if (
            wrappers._get_llm_tool_attributes is _respan_get_llm_tool_attributes
            and _ORIGINAL_TOOL_EXTRACTOR is not None
        ):
            wrappers._get_llm_tool_attributes = _ORIGINAL_TOOL_EXTRACTOR
        for wrapper_class, original_call, patched_call in _AGENT_WRAPPER_PATCHES:
            if wrapper_class.__call__ is patched_call:
                wrapper_class.__call__ = original_call
        _AGENT_WRAPPER_PATCHES.clear()
        _ORIGINAL_TOOL_EXTRACTOR = None


def _load_openinference_autogen_agentchat_class() -> type:
    autogen_module = importlib.import_module(OPENINFERENCE_AUTOGEN_AGENTCHAT_MODULE)
    for class_name in OPENINFERENCE_AUTOGEN_AGENTCHAT_CLASS_NAMES:
        instrumentor_class = getattr(autogen_module, class_name, None)
        if instrumentor_class is not None:
            return instrumentor_class
    expected = ", ".join(OPENINFERENCE_AUTOGEN_AGENTCHAT_CLASS_NAMES)
    raise ImportError(
        f"{OPENINFERENCE_AUTOGEN_AGENTCHAT_MODULE} does not expose any of: {expected}"
    )


class AutoGenInstrumentor:
    """Respan instrumentor for AgentChat or explicit ``api="legacy"`` autogen.

    Activates OpenInference's AutoGen AgentChat instrumentor and registers
    Respan's OpenInference translator so AutoGen spans reach the Respan OTLP
    pipeline with the expected ``traceloop.*``, ``gen_ai.*``, and
    ``respan.*`` fields. Legacy mode adapts the upstream AG2 chat, reply and
    tool wrappers; compose a provider instrumentor for actual LLM spans.
    """

    name = AUTOGEN_INSTRUMENTATION_NAME

    def __init__(self, *, api: str = "agentchat", **instrumentor_kwargs: Any) -> None:
        if api not in ("agentchat", "legacy"):
            raise ValueError("api must be 'agentchat' or 'legacy'")
        self._api = api
        instrumentor_kwargs.pop(TRACER_PROVIDER_KWARG, None)
        self._instrumentor_kwargs = instrumentor_kwargs
        self._delegate = None
        self._native_processor = AutoGenNativeSpanProcessor()
        self._tool_extractor_patched = False
        self._is_instrumented = False

    @staticmethod
    def _register_native_processor(tracer_provider, processor) -> None:
        active_span_processor = getattr(tracer_provider, "_active_span_processor", None)
        processors = (
            getattr(active_span_processor, "_span_processors", None)
            if active_span_processor is not None
            else None
        )

        if processors is not None:
            remaining_processors = tuple(
                existing_processor
                for existing_processor in processors
                if existing_processor is not processor
            )
            active_span_processor._span_processors = (
                processor,
                *remaining_processors,
            )
            return

        if hasattr(tracer_provider, "add_span_processor"):
            tracer_provider.add_span_processor(processor)

    @staticmethod
    def _unregister_native_processor(tracer_provider, processor) -> None:
        active_span_processor = getattr(tracer_provider, "_active_span_processor", None)
        processors = (
            getattr(active_span_processor, "_span_processors", None)
            if active_span_processor is not None
            else None
        )
        if processors is None:
            return
        active_span_processor._span_processors = tuple(
            existing_processor
            for existing_processor in processors
            if existing_processor is not processor
        )

    @staticmethod
    def _is_respan_tracing_enabled() -> bool:
        tracer = getattr(RespanTracer, "_instance", None)
        if tracer is None:
            return True
        return bool(getattr(tracer, "is_enabled", True))

    def activate(self) -> None:
        """Activate the selected AutoGen API and Respan's translator."""
        if self._is_instrumented:
            return

        if not self._is_respan_tracing_enabled():
            logger.info(
                "AutoGen instrumentation skipped because Respan tracing is disabled"
            )
            return

        if self._api == "legacy":
            self._activate_legacy()
            return

        try:
            autogen_instrumentor_class = _load_openinference_autogen_agentchat_class()
        except ImportError as exc:
            logger.warning(
                "Failed to activate AutoGen instrumentation - missing dependency: %s",
                exc,
            )
            return

        try:
            _patch_openinference_tool_extractor()
            self._tool_extractor_patched = True
            self._register_native_processor(
                trace.get_tracer_provider(),
                self._native_processor,
            )
            self._delegate = OpenInferenceInstrumentor(
                autogen_instrumentor_class,
                **self._instrumentor_kwargs,
            )
            self._delegate.activate()
            # OpenInference rebuilds the processor chain with its generic
            # translator first. Move the AutoGen adapter back to the front so
            # it can repair vendor-only history fields before translation.
            self._register_native_processor(
                trace.get_tracer_provider(),
                self._native_processor,
            )
            self._is_instrumented = True
            logger.info("AutoGen instrumentation activated")
        except Exception:
            if self._delegate is not None:
                try:
                    self._delegate.deactivate()
                except Exception:
                    logger.exception("Failed to clean up AutoGen instrumentation")
            self._unregister_native_processor(
                trace.get_tracer_provider(),
                self._native_processor,
            )
            if self._tool_extractor_patched:
                _unpatch_openinference_tool_extractor()
                self._tool_extractor_patched = False
            self._delegate = None
            self._is_instrumented = False
            logger.exception("Failed to activate AutoGen instrumentation")

    def _activate_legacy(self) -> None:
        from respan_instrumentation_autogen._legacy import LegacyAutoGenInstrumentor

        try:
            self._delegate = OpenInferenceInstrumentor(
                LegacyAutoGenInstrumentor, **self._instrumentor_kwargs
            )
            self._delegate.activate()
            if not LegacyAutoGenInstrumentor().is_instrumented_by_opentelemetry:
                raise RuntimeError("Install a supported legacy-pyautogen or legacy-autogen extra")
            self._is_instrumented = True
            logger.info("Legacy AutoGen instrumentation activated")
        except Exception:
            if self._delegate is not None:
                self._delegate.deactivate()
            self._delegate = None
            self._is_instrumented = False
            logger.exception("Failed to activate legacy AutoGen instrumentation")

    def deactivate(self) -> None:
        """Deactivate the instrumentation."""
        if self._is_instrumented and self._delegate is not None:
            try:
                self._delegate.deactivate()
            except Exception:
                logger.exception("Failed to deactivate AutoGen instrumentation")
        self._unregister_native_processor(
            trace.get_tracer_provider(),
            self._native_processor,
        )
        if self._tool_extractor_patched:
            _unpatch_openinference_tool_extractor()
            self._tool_extractor_patched = False
        self._delegate = None
        self._is_instrumented = False
        logger.info("AutoGen instrumentation deactivated")
