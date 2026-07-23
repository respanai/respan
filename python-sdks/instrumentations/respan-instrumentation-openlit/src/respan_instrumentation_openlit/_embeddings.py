"""Enrich OpenLIT's native OpenAI embedding spans with full vectors."""

from __future__ import annotations

import importlib
import json
import logging
from collections.abc import Mapping, Sequence
from functools import wraps
from types import ModuleType
from typing import Any, Callable

from opentelemetry.semconv_ai import SpanAttributes

logger = logging.getLogger(__name__)

_OPENAI_EMBEDDING_MODULES = (
    "openlit.instrumentation.openai.openai",
    "openlit.instrumentation.openai.async_openai",
)

EmbeddingHook = tuple[ModuleType, Callable[..., Any], Callable[..., Any]]


def _response_data(response: Any) -> Sequence[Any]:
    data = getattr(response, "data", None)
    if data is None and isinstance(response, Mapping):
        data = response.get("data")
    if isinstance(data, Sequence) and not isinstance(data, str | bytes):
        return data
    return ()


def _embedding_vectors(response: Any) -> list[list[Any]]:
    vectors: list[list[Any]] = []
    for item in _response_data(response):
        vector = getattr(item, "embedding", None)
        if vector is None and isinstance(item, Mapping):
            vector = item.get("embedding")
        if isinstance(vector, Sequence) and not isinstance(vector, str | bytes):
            vectors.append(list(vector))
    return vectors


def _enrich_embedding_span(*, response: Any, span: Any, capture_content: bool) -> None:
    if not capture_content or span is None:
        return
    vectors = _embedding_vectors(response)
    if vectors:
        span.set_attribute(
            SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
            json.dumps(vectors, separators=(",", ":"), default=str),
        )


def install_openai_embedding_hooks(*, capture_content: bool) -> list[EmbeddingHook]:
    """Hook OpenLIT response processing without wrapping provider calls."""

    hooks: list[EmbeddingHook] = []
    for module_name in _OPENAI_EMBEDDING_MODULES:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            logger.debug("OpenLIT OpenAI module unavailable: %s", module_name)
            continue
        original = getattr(module, "process_embedding_response", None)
        if not callable(original):
            continue

        @wraps(original)
        def process_embedding_response(
            *args: Any,
            __original: Callable[..., Any] = original,
            **kwargs: Any,
        ) -> Any:
            result = __original(*args, **kwargs)
            response = kwargs.get("response", args[0] if args else result)
            span = kwargs.get("span")
            _enrich_embedding_span(
                response=response,
                span=span,
                capture_content=capture_content,
            )
            return result

        module.process_embedding_response = process_embedding_response
        hooks.append((module, original, process_embedding_response))
    return hooks


def remove_openai_embedding_hooks(hooks: list[EmbeddingHook]) -> None:
    """Restore only hook functions still owned by this adapter."""

    for module, original, replacement in reversed(hooks):
        if getattr(module, "process_embedding_response", None) is replacement:
            module.process_embedding_response = original
