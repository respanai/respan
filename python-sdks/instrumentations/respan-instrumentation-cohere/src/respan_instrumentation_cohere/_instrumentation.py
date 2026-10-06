"""Cohere instrumentation plugin for Respan."""

import importlib
import json
import logging
import math
import os
import threading
from collections.abc import Mapping, Sequence
from functools import lru_cache
from typing import Any

from opentelemetry import context as context_api
from opentelemetry import trace
from opentelemetry.semconv_ai import LLMRequestTypeValues, SpanAttributes
from respan_tracing.core.tracer import RespanTracer

from respan_instrumentation_cohere._processor import (
    CohereSpanProcessor,
    insert_span_processor_before_export,
    remove_span_processor,
)
from respan_instrumentation_cohere._streaming import (
    patch_stream_processors,
    restore_stream_processors,
)

logger = logging.getLogger(__name__)

COHERE_INSTRUMENTATION_NAME = "cohere"
OTEL_COHERE_MODULE = "opentelemetry.instrumentation.cohere"

_CONTENT_PATCH_LOCK = threading.RLock()
_CONTENT_PATCH_USERS = 0
_CONTENT_PATCH_ORIGINALS: dict[str, Any] = {}
_INSTRUMENTATION_USERS: dict[Any, tuple[int, bool]] = {}


def _capture_content() -> bool:
    # Match the upstream content switch, including its per-context override.
    return (os.getenv("TRACELOOP_TRACE_CONTENT") or "true").lower() == "true" or bool(
        context_api.get_value("override_enable_content_tracing")
    )


def _load_otel_cohere_module() -> Any:
    return importlib.import_module(OTEL_COHERE_MODULE)


@lru_cache(maxsize=1)
def _compatible_instrumentor_class(upstream: type) -> type:
    """Keep upstream wrappers while validating the released Cohere 5–7 APIs."""

    class CompatibleCohereInstrumentor(upstream):
        # BaseInstrumentor's singleton lookup otherwise inherits an already
        # constructed upstream instance and silently loses this override.
        _instance = None

        def instrumentation_dependencies(self):
            return ("cohere >=5.0.0, <8",)

    return CompatibleCohereInstrumentor


def _structured_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _structured_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_structured_value(item) for item in value]
    for method_name in ("model_dump", "dict"):
        method = getattr(value, method_name, None)
        if not callable(method):
            continue
        try:
            dumped = method()
        except Exception as exc:  # noqa: BLE001 - optional SDK serializers must not break calls
            logger.debug("Cohere serializer unavailable: %s", type(exc).__name__)
            continue
        if isinstance(dumped, Mapping):
            return _structured_value(dumped)
    attributes = getattr(value, "__dict__", None)
    if isinstance(attributes, Mapping):
        return _structured_value(
            {
                key: item
                for key, item in attributes.items()
                if not str(key).startswith("_")
            }
        )
    return str(value)


def _json_attribute(value: Any) -> str:
    return json.dumps(
        _structured_value(value),
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _set_reported_usage(span, response: Any) -> None:
    """Cohere returns absent billed fields as None, not zero."""
    if isinstance(response, Mapping):
        usage = response.get("usage") or response.get("meta")
    else:
        usage = getattr(response, "usage", None) or getattr(response, "meta", None)
    usage = _structured_value(usage)
    if not isinstance(usage, Mapping):
        return
    billed = usage.get("billed_units") or {}
    if not isinstance(billed, Mapping):
        return
    counts = {}
    for field, attribute in (
        ("input_tokens", SpanAttributes.LLM_USAGE_PROMPT_TOKENS),
        ("output_tokens", SpanAttributes.LLM_USAGE_COMPLETION_TOKENS),
    ):
        value = billed.get(field)
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            and value >= 0
            and int(value) == value
        ):
            counts[field] = int(value)
            span.set_attribute(attribute, int(value))
    if len(counts) == 2:
        span.set_attribute(SpanAttributes.LLM_USAGE_TOTAL_TOKENS, sum(counts.values()))


def _rerank_input(kwargs: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: _structured_value(kwargs[key])
        for key in ("model", "query", "documents", "top_n")
        if kwargs.get(key) is not None
    }


def _rerank_output(response: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    response_id = getattr(response, "id", None)
    if response_id is not None:
        result["id"] = response_id

    ranked_results: list[dict[str, Any]] = []
    for item in getattr(response, "results", None) or []:
        normalized = _structured_value(item)
        if isinstance(normalized, Mapping):
            ranked_results.append(
                {
                    key: normalized[key]
                    for key in ("index", "relevance_score", "document")
                    if normalized.get(key) is not None
                }
            )
        else:
            ranked_results.append({"value": normalized})
    result["results"] = ranked_results
    return result


def _install_content_patch(cohere_module: Any) -> bool:
    global _CONTENT_PATCH_USERS
    with _CONTENT_PATCH_LOCK:
        if _CONTENT_PATCH_USERS:
            _CONTENT_PATCH_USERS += 1
            return True

        original_input = getattr(cohere_module, "set_input_content_attributes", None)
        original_response = getattr(
            cohere_module,
            "set_response_content_attributes",
            None,
        )
        if not callable(original_input) or not callable(original_response):
            return False

        def set_input_content_attributes(span, llm_request_type, kwargs):
            original_input(span, llm_request_type, kwargs)
            if not span.is_recording() or not _capture_content():
                return
            if llm_request_type == LLMRequestTypeValues.EMBEDDING:
                payload = {
                    key: kwargs[key]
                    for key in ("texts", "images", "inputs")
                    if kwargs.get(key) is not None
                }
                span.set_attribute(
                    SpanAttributes.TRACELOOP_ENTITY_INPUT, _json_attribute(payload)
                )
                return
            if llm_request_type != LLMRequestTypeValues.RERANK:
                return
            payload = _rerank_input(kwargs)
            span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_NAME, "rerank")
            span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_PATH, "rerank")
            span.set_attribute(
                SpanAttributes.TRACELOOP_ENTITY_INPUT,
                _json_attribute(payload),
            )

        def set_response_content_attributes(span, llm_request_type, response):
            original_response(span, llm_request_type, response)
            if span.is_recording():
                _set_reported_usage(span, response)
            if not span.is_recording() or not _capture_content():
                return
            if llm_request_type == LLMRequestTypeValues.EMBEDDING:
                embeddings = getattr(response, "embeddings", None)
                if embeddings is not None:
                    span.set_attribute(
                        SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
                        _json_attribute(embeddings),
                    )
                return
            if llm_request_type != LLMRequestTypeValues.RERANK:
                return
            span.set_attribute(
                SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
                _json_attribute(_rerank_output(response)),
            )

        _CONTENT_PATCH_ORIGINALS.update(
            {
                "module": cohere_module,
                "set_input_content_attributes": original_input,
                "set_response_content_attributes": original_response,
            }
        )
        cohere_module.set_input_content_attributes = set_input_content_attributes
        cohere_module.set_response_content_attributes = set_response_content_attributes
        _CONTENT_PATCH_USERS = 1
        _CONTENT_PATCH_ORIGINALS["stream_processors"] = patch_stream_processors(
            cohere_module
        )
        return True


def _remove_content_patch() -> None:
    global _CONTENT_PATCH_USERS
    with _CONTENT_PATCH_LOCK:
        if _CONTENT_PATCH_USERS == 0:
            return
        _CONTENT_PATCH_USERS -= 1
        if _CONTENT_PATCH_USERS:
            return

        cohere_module = _CONTENT_PATCH_ORIGINALS.get("module")
        if cohere_module is not None:
            for name in (
                "set_input_content_attributes",
                "set_response_content_attributes",
            ):
                original = _CONTENT_PATCH_ORIGINALS.get(name)
                if original is not None:
                    setattr(cohere_module, name, original)
        restore_stream_processors(_CONTENT_PATCH_ORIGINALS.get("stream_processors", []))
        _CONTENT_PATCH_ORIGINALS.clear()


class CohereInstrumentor:
    """Respan instrumentor for the Cohere Python SDK."""

    name = COHERE_INSTRUMENTATION_NAME

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
        self._instrumentor_kwargs = dict(instrumentor_kwargs)
        self._instrumentor = None
        self._processor = None
        self._is_instrumented = False
        self._owns_instrumentation = False
        self._owns_content_patch = False

    @staticmethod
    def _is_respan_tracing_enabled() -> bool:
        tracer = getattr(RespanTracer, "_instance", None)
        if tracer is None:
            return True
        return bool(getattr(tracer, "is_enabled", True))

    def activate(self) -> None:
        """Instrument Cohere via OTEL and add Respan contract normalization."""
        if self._is_instrumented:
            return

        if not self._is_respan_tracing_enabled():
            logger.info(
                "Cohere instrumentation skipped because Respan tracing is disabled"
            )
            return

        try:
            cohere_module = _load_otel_cohere_module()
            cohere_instrumentor_class = _compatible_instrumentor_class(
                cohere_module.CohereInstrumentor
            )
        except ImportError as exc:
            logger.warning(
                "Failed to activate Cohere instrumentation - missing dependency: %s",
                exc,
            )
            return

        tracer_provider = trace.get_tracer_provider()
        try:
            self._owns_content_patch = _install_content_patch(cohere_module)
            self._processor = CohereSpanProcessor()
            insert_span_processor_before_export(tracer_provider, self._processor)

            upstream_instance = getattr(
                cohere_module.CohereInstrumentor, "_instance", None
            )
            if getattr(upstream_instance, "is_instrumented_by_opentelemetry", False):
                self._instrumentor = upstream_instance
            else:
                self._instrumentor = cohere_instrumentor_class(
                    **self._constructor_kwargs
                )
            already_instrumented = bool(
                getattr(
                    self._instrumentor,
                    "is_instrumented_by_opentelemetry",
                    False,
                )
            )
            if not already_instrumented:
                self._instrumentor.instrument(
                    tracer_provider=tracer_provider,
                    **self._instrumentor_kwargs,
                )
                if not bool(
                    getattr(
                        self._instrumentor,
                        "is_instrumented_by_opentelemetry",
                        False,
                    )
                ):
                    remove_span_processor(tracer_provider, self._processor)
                    self._instrumentor = None
                    self._processor = None
                    if self._owns_content_patch:
                        _remove_content_patch()
                        self._owns_content_patch = False
                    logger.warning(
                        "Cohere instrumentation skipped because the upstream "
                        "instrumentor did not activate"
                    )
                    return
                self._owns_instrumentation = True
            users, owned = _INSTRUMENTATION_USERS.get(self._instrumentor, (0, False))
            _INSTRUMENTATION_USERS[self._instrumentor] = (
                users + 1,
                owned or self._owns_instrumentation,
            )
            self._is_instrumented = True
            logger.info("Cohere instrumentation activated")
        except Exception:
            if self._instrumentor is not None and self._owns_instrumentation:
                try:
                    self._instrumentor.uninstrument()
                except Exception:
                    logger.exception("Failed to clean up Cohere instrumentation")
            if self._processor is not None:
                remove_span_processor(tracer_provider, self._processor)
            self._instrumentor = None
            self._processor = None
            self._is_instrumented = False
            self._owns_instrumentation = False
            if self._owns_content_patch:
                _remove_content_patch()
                self._owns_content_patch = False
            logger.exception("Failed to activate Cohere instrumentation")

    def deactivate(self) -> None:
        """Deactivate the instrumentation."""
        tracer_provider = trace.get_tracer_provider()
        if self._is_instrumented and self._instrumentor is not None:
            users, owned = _INSTRUMENTATION_USERS.get(
                self._instrumentor, (1, self._owns_instrumentation)
            )
            if users > 1:
                _INSTRUMENTATION_USERS[self._instrumentor] = (users - 1, owned)
            else:
                _INSTRUMENTATION_USERS.pop(self._instrumentor, None)
                if owned:
                    try:
                        self._instrumentor.uninstrument()
                    except Exception:
                        logger.exception("Failed to deactivate Cohere instrumentation")
        if self._processor is not None:
            remove_span_processor(tracer_provider, self._processor)
        if self._owns_content_patch:
            _remove_content_patch()
        self._instrumentor = None
        self._processor = None
        self._is_instrumented = False
        self._owns_instrumentation = False
        self._owns_content_patch = False
        logger.info("Cohere instrumentation deactivated")
