"""Mistral AI instrumentation plugin for Respan."""

from __future__ import annotations

import importlib
import json
import logging
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor
from opentelemetry.semconv_ai import SpanAttributes as TLSpanAttributes
from respan_instrumentation_openinference import OpenInferenceInstrumentor
from respan_sdk.constants.span_attributes import (
    RESPAN_SPAN_TOOL_CALLS,
    RESPAN_SPAN_TOOLS,
)
from respan_tracing.constants.tracing import SAMPLE_RATE_ATTR
from respan_tracing.core.tracer import RespanTracer

logger = logging.getLogger(__name__)

MISTRALAI_INSTRUMENTATION_NAME = "mistralai"
OPENINFERENCE_MISTRALAI_MODULE = "openinference.instrumentation.mistralai"
MISTRALAI_SDK_TRACER_NAME = "mistralai_sdk_tracer"
_OFF_CONTRACT_ALIAS_KEYS = (
    RESPAN_SPAN_TOOLS,
    RESPAN_SPAN_TOOL_CALLS,
    "tools",
    "tool_calls",
    "model",
    "prompt_tokens",
    "completion_tokens",
    "total_request_tokens",
)
_GEN_AI_MESSAGE_PREFIXES = (
    f"{TLSpanAttributes.LLM_PROMPTS}.",
    f"{TLSpanAttributes.LLM_COMPLETIONS}.",
)
_TOOL_CALLS_SUFFIX = ".tool_calls"


def _load_openinference_mistralai_class() -> type:
    mistralai_module = importlib.import_module(OPENINFERENCE_MISTRALAI_MODULE)
    return mistralai_module.MistralAIInstrumentor


def _is_mistralai_span(span: ReadableSpan) -> bool:
    scope = getattr(span, "instrumentation_scope", None)
    scope_name = getattr(scope, "name", None)
    return scope_name == OPENINFERENCE_MISTRALAI_MODULE


def _is_mistralai_sdk_span(span: ReadableSpan) -> bool:
    scope = getattr(span, "instrumentation_scope", None)
    scope_name = getattr(scope, "name", None)
    return scope_name == MISTRALAI_SDK_TRACER_NAME


def _is_gen_ai_tool_calls_attr(key: str) -> bool:
    return key.endswith(_TOOL_CALLS_SUFFIX) and key.startswith(_GEN_AI_MESSAGE_PREFIXES)


def _safe_json_str(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, default=str)


class _MistralAIOffContractAliasProcessor(SpanProcessor):
    """Clean Mistral spans before the Respan exporter sees them."""

    def on_start(self, span: Any, parent_context: Any = None) -> None:
        pass

    def on_end(self, span: ReadableSpan) -> None:
        if _is_mistralai_sdk_span(span):
            original_attrs = getattr(span, "_attributes", None)
            if original_attrs is not None:
                attrs = dict(original_attrs)
                attrs[SAMPLE_RATE_ATTR] = 0
                span._attributes = attrs
            return

        if not _is_mistralai_span(span):
            return

        original_attrs = getattr(span, "_attributes", None)
        if original_attrs is None:
            return

        attrs = dict(original_attrs)
        for key in _OFF_CONTRACT_ALIAS_KEYS:
            attrs.pop(key, None)

        for key, value in list(attrs.items()):
            if _is_gen_ai_tool_calls_attr(key):
                attrs[key] = _safe_json_str(value)

        span._attributes = attrs

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


def _active_span_processors() -> tuple[Any, Any]:
    tracer_provider = trace.get_tracer_provider()
    active_span_processor = getattr(tracer_provider, "_active_span_processor", None)
    processors = (
        getattr(active_span_processor, "_span_processors", None)
        if active_span_processor is not None
        else None
    )
    return active_span_processor, processors


class MistralAIInstrumentor:
    """Respan instrumentor for the official Mistral AI Python SDK."""

    name = MISTRALAI_INSTRUMENTATION_NAME

    def __init__(self, **instrumentor_kwargs: Any) -> None:
        self._instrumentor_kwargs = instrumentor_kwargs
        self._delegate = None
        self._cleanup_processor = None
        self._is_instrumented = False

    @staticmethod
    def _is_respan_tracing_enabled() -> bool:
        tracer = getattr(RespanTracer, "_instance", None)
        if tracer is None:
            return True
        return bool(getattr(tracer, "is_enabled", True))

    def activate(self) -> None:
        """Instrument Mistral AI via OpenInference and Respan's translator."""
        if self._is_instrumented:
            return

        if not self._is_respan_tracing_enabled():
            logger.info(
                "Mistral AI instrumentation skipped because Respan tracing is disabled"
            )
            return

        try:
            mistralai_instrumentor_class = _load_openinference_mistralai_class()
        except ImportError as exc:
            logger.warning(
                "Failed to activate Mistral AI instrumentation - missing dependency: %s",
                exc,
            )
            return

        try:
            self._delegate = OpenInferenceInstrumentor(
                mistralai_instrumentor_class,
                **self._instrumentor_kwargs,
            )
            self._delegate.activate()
            self._register_cleanup_processor()
            self._is_instrumented = True
            logger.info("Mistral AI instrumentation activated")
        except Exception:
            if self._delegate is not None:
                try:
                    self._delegate.deactivate()
                except Exception:
                    logger.exception("Failed to clean up Mistral AI instrumentation")
            self._delegate = None
            self._cleanup_processor = None
            self._is_instrumented = False
            logger.exception("Failed to activate Mistral AI instrumentation")

    def _register_cleanup_processor(self) -> None:
        translator_getter = getattr(OpenInferenceInstrumentor, "_get_translator", None)
        if translator_getter is None:
            return

        translator = translator_getter()
        active_span_processor, processors = _active_span_processors()
        if active_span_processor is None or processors is None:
            return

        cleanup_processor = _MistralAIOffContractAliasProcessor()
        rebuilt_processors = []
        inserted = False

        for processor in processors:
            if isinstance(processor, _MistralAIOffContractAliasProcessor):
                continue
            rebuilt_processors.append(processor)
            if processor is translator:
                rebuilt_processors.append(cleanup_processor)
                inserted = True

        if inserted:
            active_span_processor._span_processors = tuple(rebuilt_processors)
            self._cleanup_processor = cleanup_processor

    def _unregister_cleanup_processor(self) -> None:
        if self._cleanup_processor is None:
            return

        active_span_processor, processors = _active_span_processors()
        if active_span_processor is not None and processors is not None:
            active_span_processor._span_processors = tuple(
                processor
                for processor in processors
                if processor is not self._cleanup_processor
            )
        self._cleanup_processor = None

    def deactivate(self) -> None:
        """Deactivate the instrumentation."""
        self._unregister_cleanup_processor()
        if self._is_instrumented and self._delegate is not None:
            try:
                self._delegate.deactivate()
            except Exception:
                logger.exception("Failed to deactivate Mistral AI instrumentation")
        self._delegate = None
        self._is_instrumented = False
        logger.info("Mistral AI instrumentation deactivated")
