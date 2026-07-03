"""Hugging Face Transformers instrumentation plugin for Respan."""

from __future__ import annotations

import importlib
import logging
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanProcessor
from opentelemetry.semconv_ai import LLMRequestTypeValues
from opentelemetry.semconv_ai import SpanAttributes as TLSpanAttributes
from respan_instrumentation_huggingface._constants import (
    ASSISTANT_ROLE,
    HUGGINGFACE_GEN_AI_SYSTEM,
    HUGGINGFACE_INSTRUMENTATION_NAME,
    TRANSFORMERS_MODULE,
    TRANSFORMERS_SCOPE_NAME,
    TRANSFORMERS_TEXT_GENERATION_SPAN_NAME,
    USER_ROLE,
)
from respan_sdk.constants.llm_logging import LOG_TYPE_TEXT
from respan_sdk.constants.span_attributes import RESPAN_LOG_TYPE
from respan_tracing.core.tracer import RespanTracer

logger = logging.getLogger(__name__)


def _load_transformers_instrumentor_class() -> type:
    transformers_module = importlib.import_module(TRANSFORMERS_MODULE)
    return transformers_module.TransformersInstrumentor


def _active_span_processors(tracer_provider) -> tuple[Any, ...] | None:
    active_span_processor = getattr(tracer_provider, "_active_span_processor", None)
    processors = (
        getattr(active_span_processor, "_span_processors", None)
        if active_span_processor is not None
        else None
    )
    if processors is None:
        return None
    return tuple(processors)


def _set_active_span_processors(
    tracer_provider,
    processors: tuple[Any, ...],
) -> bool:
    active_span_processor = getattr(tracer_provider, "_active_span_processor", None)
    if active_span_processor is None:
        return False
    if getattr(active_span_processor, "_span_processors", None) is None:
        return False
    active_span_processor._span_processors = processors
    return True


def _register_processor_first(tracer_provider, processor) -> None:
    processors = _active_span_processors(tracer_provider)
    if processors is None:
        if hasattr(tracer_provider, "add_span_processor"):
            tracer_provider.add_span_processor(processor)
        return

    processors = tuple(
        existing_processor
        for existing_processor in processors
        if existing_processor is not processor
    )
    _set_active_span_processors(
        tracer_provider=tracer_provider,
        processors=(processor, *processors),
    )


def _unregister_processor(tracer_provider, processor) -> None:
    processors = _active_span_processors(tracer_provider)
    if processors is None:
        return
    _set_active_span_processors(
        tracer_provider=tracer_provider,
        processors=tuple(
            existing_processor
            for existing_processor in processors
            if existing_processor is not processor
        ),
    )


def _indexed_content_indices(attrs: dict[str, Any], prefix: str) -> set[str]:
    marker = f"{prefix}."
    indices: set[str] = set()
    for key in attrs:
        if not key.startswith(marker) or not key.endswith(".content"):
            continue
        index = key[len(marker) : -len(".content")]
        if index:
            indices.add(index)
    return indices


class HuggingFaceSpanContractProcessor(SpanProcessor):
    """Normalize upstream Transformers spans into Respan's span contract."""

    def on_start(self, span, parent_context=None) -> None:
        return None

    def on_end(self, span: ReadableSpan) -> None:
        if not _is_transformers_text_generation_span(span):
            return

        original_attrs = getattr(span, "_attributes", None)
        if original_attrs is None:
            return

        attrs = dict(original_attrs)
        attrs[TLSpanAttributes.LLM_SYSTEM] = HUGGINGFACE_GEN_AI_SYSTEM
        attrs[TLSpanAttributes.LLM_REQUEST_TYPE] = LLMRequestTypeValues.CHAT.value
        attrs[RESPAN_LOG_TYPE] = LOG_TYPE_TEXT
        attrs.setdefault(
            TLSpanAttributes.TRACELOOP_ENTITY_NAME,
            "huggingface.text_generation",
        )
        attrs.setdefault(TLSpanAttributes.TRACELOOP_ENTITY_PATH, "")

        for index in _indexed_content_indices(attrs, TLSpanAttributes.LLM_PROMPTS):
            attrs.setdefault(
                f"{TLSpanAttributes.LLM_PROMPTS}.{index}.role",
                USER_ROLE,
            )

        for index in _indexed_content_indices(attrs, TLSpanAttributes.LLM_COMPLETIONS):
            attrs.setdefault(
                f"{TLSpanAttributes.LLM_COMPLETIONS}.{index}.role",
                ASSISTANT_ROLE,
            )

        span._attributes = attrs


def _is_transformers_text_generation_span(span: ReadableSpan) -> bool:
    if span.name == TRANSFORMERS_TEXT_GENERATION_SPAN_NAME:
        return True

    scope = getattr(span, "instrumentation_scope", None)
    return getattr(scope, "name", None) == TRANSFORMERS_SCOPE_NAME


class HuggingFaceInstrumentor:
    """Respan instrumentor for Hugging Face Transformers."""

    name = HUGGINGFACE_INSTRUMENTATION_NAME

    def __init__(
        self,
        *,
        exception_logger: Any | None = None,
        use_legacy_attributes: bool = True,
        **instrumentor_kwargs: Any,
    ) -> None:
        self._constructor_kwargs = {
            "exception_logger": exception_logger,
            "use_legacy_attributes": use_legacy_attributes,
        }
        self._instrumentor_kwargs = instrumentor_kwargs
        self._delegate = None
        self._contract_processor = HuggingFaceSpanContractProcessor()
        self._is_instrumented = False

    @staticmethod
    def _is_respan_tracing_enabled() -> bool:
        tracer = getattr(RespanTracer, "_instance", None)
        if tracer is None:
            return True
        return bool(getattr(tracer, "is_enabled", True))

    def activate(self) -> None:
        """Instrument Transformers via the upstream OTEL instrumentor."""
        if self._is_instrumented:
            return

        if not self._is_respan_tracing_enabled():
            logger.info(
                "Hugging Face instrumentation skipped because Respan tracing is disabled"
            )
            return

        try:
            instrumentor_class = _load_transformers_instrumentor_class()
        except ImportError as exc:
            logger.warning(
                "Failed to activate Hugging Face instrumentation - missing dependency: %s",
                exc,
            )
            return

        tracer_provider = trace.get_tracer_provider()

        try:
            self._delegate = instrumentor_class(**self._constructor_kwargs)
            _register_processor_first(
                tracer_provider=tracer_provider,
                processor=self._contract_processor,
            )
            self._delegate.instrument(
                tracer_provider=tracer_provider,
                **self._instrumentor_kwargs,
            )
            self._is_instrumented = True
            logger.info("Hugging Face instrumentation activated")
        except Exception:
            _unregister_processor(
                tracer_provider=tracer_provider,
                processor=self._contract_processor,
            )
            if self._delegate is not None:
                try:
                    self._delegate.uninstrument()
                except Exception:
                    logger.exception("Failed to clean up Hugging Face instrumentation")
            self._delegate = None
            self._is_instrumented = False
            logger.exception("Failed to activate Hugging Face instrumentation")

    def deactivate(self) -> None:
        """Deactivate the instrumentation."""
        _unregister_processor(
            tracer_provider=trace.get_tracer_provider(),
            processor=self._contract_processor,
        )
        if self._is_instrumented and self._delegate is not None:
            try:
                self._delegate.uninstrument()
            except Exception:
                logger.exception("Failed to deactivate Hugging Face instrumentation")
        self._delegate = None
        self._is_instrumented = False
        logger.info("Hugging Face instrumentation deactivated")
