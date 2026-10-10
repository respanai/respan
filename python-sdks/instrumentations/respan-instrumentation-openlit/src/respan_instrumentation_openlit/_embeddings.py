"""Enrich OpenLIT's native OpenAI embedding spans with full vectors."""

from __future__ import annotations

import importlib
import json
import logging
import math
from collections.abc import Callable, Mapping, Sequence
from functools import wraps
from types import ModuleType
from typing import Any

from opentelemetry.semconv_ai import SpanAttributes

from ._policy import allowed

logger = logging.getLogger(__name__)

_OPENAI_EMBEDDING_MODULES = (
    "openlit.instrumentation.openai.openai",
    "openlit.instrumentation.openai.async_openai",
)

EmbeddingHook = tuple[ModuleType, Callable[..., Any], Callable[..., Any]]


def _safe_get(value: Any, name: str, default: Any = None) -> Any:
    try:
        if isinstance(value, Mapping):
            return value.get(name, default)
        return getattr(value, name, default)
    except Exception:  # noqa: BLE001 - provider objects may expose descriptors.
        return default


def _response_data(response: Any) -> Sequence[Any]:
    data = _safe_get(response, "data")
    if isinstance(data, Sequence) and not isinstance(data, str | bytes):
        return data
    return ()


def _embedding_vectors(response: Any) -> list[list[Any]]:
    vectors: list[list[Any]] = []
    try:
        items = iter(_response_data(response))
        for item in items:
            vector = _safe_get(item, "embedding")
            if not isinstance(vector, Sequence) or isinstance(vector, str | bytes):
                continue
            values: list[int | float | None] = []
            try:
                for value in vector:
                    if isinstance(value, bool):
                        values.append(int(value))
                    elif isinstance(value, int) or (
                        isinstance(value, float) and math.isfinite(value)
                    ):
                        values.append(value)
                    else:
                        values.append(None)
            except Exception as exc:  # noqa: BLE001 - hostile provider iterator.
                logger.debug("OpenLIT embedding vector skipped: %s", type(exc).__name__)
                continue
            vectors.append(values)
    except Exception:  # noqa: BLE001 - fail open on hostile response collections.
        return []
    return vectors


def _enrich_embedding_span(
    *,
    response: Any,
    span: Any,
    capture_content: bool,
) -> None:
    try:
        if not allowed(span, capture_content):
            return
        vectors = _embedding_vectors(response)
        if vectors:
            span.set_attribute(
                SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
                json.dumps(
                    vectors,
                    allow_nan=False,
                    separators=(",", ":"),
                ),
            )
    except Exception as exc:  # noqa: BLE001 - tracing must not break provider calls.
        logger.debug("OpenLIT embedding enrichment skipped: %s", type(exc).__name__)


def install_openai_embedding_hooks(
    *,
    capture_content: bool,
    max_content_length: int = 16_000,
) -> list[EmbeddingHook]:
    """Hook OpenLIT response processing without wrapping provider calls."""

    del max_content_length  # Full vectors are required by the span contract.
    hooks: list[EmbeddingHook] = []
    try:
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
                if not process_embedding_response._respan_active:
                    return result
                response = kwargs.get("response", args[0] if args else result)
                span = kwargs.get("span")
                _enrich_embedding_span(
                    response=response,
                    span=span,
                    capture_content=capture_content,
                )
                return result

            process_embedding_response._respan_active = True
            module.process_embedding_response = process_embedding_response
            hooks.append((module, original, process_embedding_response))
    except Exception:
        remove_openai_embedding_hooks(hooks)
        raise
    return hooks


def remove_openai_embedding_hooks(hooks: list[EmbeddingHook]) -> None:
    """Restore only hook functions still owned by this adapter."""

    for module, original, replacement in reversed(hooks):
        replacement._respan_active = False
        if getattr(module, "process_embedding_response", None) is replacement:
            module.process_embedding_response = original
