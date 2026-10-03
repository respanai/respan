"""Translate the SDK's embedding responses without inventing token usage."""

from __future__ import annotations

import logging
import math
import time
from typing import Any

from opentelemetry import context as context_api
from opentelemetry.semconv._incubating.attributes.error_attributes import ERROR_MESSAGE
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import (
    GEN_AI_USAGE_INPUT_TOKENS,
)
from opentelemetry.semconv_ai import LLMRequestTypeValues, SpanAttributes
from respan_sdk.constants.llm_logging import LOG_TYPE_EMBEDDING
from respan_sdk.constants.span_attributes import RESPAN_LOG_TYPE
from respan_tracing.utils.span_factory import build_readable_span, inject_span

from respan_instrumentation_google_genai._constants import GOOGLE_GENAI_SYSTEM_NAME
from respan_instrumentation_google_genai._otel_emitter import _current_trace_parent_ids
from respan_instrumentation_google_genai._translator import (
    _dump_value,
    _field,
    capture_content,
    safe_json,
)

logger = logging.getLogger(__name__)
_EMBEDDING_SPAN_NAME = "google_genai.embed_content"


def _input_tokens(embeddings: list[Any]) -> int | None:
    """Use complete per-input provider statistics, never character counts."""
    if not embeddings:
        return None
    total = 0
    for embedding in embeddings:
        count = _field(_field(embedding, "statistics"), "token_count")
        if (
            isinstance(count, bool)
            or not isinstance(count, (int, float))
            or not math.isfinite(count)
            or count < 0
            or int(count) != count
        ):
            return None
        total += int(count)
    return total


def build_embed_content_attrs(
    *, request_kwargs: dict[str, Any], response_or_chunks: Any = None
) -> dict[str, Any]:
    attrs: dict[str, Any] = {
        RESPAN_LOG_TYPE: LOG_TYPE_EMBEDDING,
        SpanAttributes.LLM_SYSTEM: GOOGLE_GENAI_SYSTEM_NAME,
        SpanAttributes.LLM_REQUEST_TYPE: LLMRequestTypeValues.EMBEDDING.value,
        SpanAttributes.TRACELOOP_ENTITY_NAME: _EMBEDDING_SPAN_NAME,
        SpanAttributes.TRACELOOP_ENTITY_PATH: _EMBEDDING_SPAN_NAME,
    }
    include_content = capture_content()
    if include_content:
        attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = safe_json(
            _dump_value(request_kwargs.get("contents"))
        )
    if request_kwargs.get("model"):
        attrs[SpanAttributes.LLM_REQUEST_MODEL] = request_kwargs["model"]
    workflow_name = context_api.get_value(SpanAttributes.TRACELOOP_ENTITY_NAME)
    if workflow_name:
        attrs[SpanAttributes.TRACELOOP_WORKFLOW_NAME] = workflow_name
    embeddings = _field(response_or_chunks, "embeddings")
    if embeddings is not None:
        if include_content:
            attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = safe_json(
                [_field(embedding, "values") for embedding in embeddings]
            )
        tokens = _input_tokens(embeddings)
        if tokens is not None:
            attrs[GEN_AI_USAGE_INPUT_TOKENS] = tokens
            attrs[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] = tokens
            attrs[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] = tokens
    return attrs


def emit_embed_content_span(
    *,
    request_kwargs: dict[str, Any],
    start_ns: int,
    response_or_chunks: Any = None,
    error_message: str | None = None,
    status_code: int = 200,
) -> None:
    try:
        attrs = build_embed_content_attrs(
            request_kwargs=request_kwargs, response_or_chunks=response_or_chunks
        )
        if error_message:
            attrs[ERROR_MESSAGE] = error_message
        trace_id, parent_id = _current_trace_parent_ids()
        span = build_readable_span(
            name=_EMBEDDING_SPAN_NAME,
            trace_id=trace_id,
            parent_id=parent_id,
            start_time_ns=start_ns,
            end_time_ns=time.time_ns(),
            attributes=attrs,
            status_code=status_code,
            error_message=error_message,
        )
        inject_span(span=span)
    except Exception:
        logger.debug("Failed to emit Google Gen AI embedding span", exc_info=True)
