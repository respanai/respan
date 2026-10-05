"""Capture embedding payloads inside the framework's native telemetry span."""

import logging
from functools import wraps
from time import perf_counter, time_ns

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import SpanAttributes

from respan_instrumentation_microsoft_agent_framework import _policy
from respan_instrumentation_microsoft_agent_framework._processor import (
    _int_value,
    _json_string,
)


def _set_payload(span, key, factory):
    try:
        span.set_attribute(key, _json_string(factory()))
    except Exception:
        logging.getLogger(__name__).debug(
            "Could not serialize embedding telemetry", exc_info=True
        )


def install(observability, active):
    layer = getattr(observability, "EmbeddingTelemetryLayer", None)
    original = getattr(layer, "get_embeddings", None)
    if not callable(original):
        return None

    @wraps(original)
    async def get_embeddings(self, values, *, options=None):
        if not active() or not observability.OBSERVABILITY_SETTINGS.ENABLED:
            return await original(self, values, options=options)
        opts = options or {}
        service_url = getattr(self, "service_url", None)
        attributes = observability._get_span_attributes(
            operation_name="embeddings",
            provider_name=str(getattr(self, "otel_provider_name", "unknown")),
            model=opts.get("model") or getattr(self, "model", None) or "unknown",
            service_url=str(service_url() if callable(service_url) else "unknown"),
        )
        capture = _policy.capture_content()
        with observability._get_span(
            attributes=attributes, span_name_attribute=SpanAttributes.LLM_REQUEST_MODEL
        ) as span:
            started = perf_counter()
            if capture and span.is_recording():
                _set_payload(
                    span, SpanAttributes.TRACELOOP_ENTITY_INPUT, lambda: values
                )
            try:
                # Delegate to the same provider method as the native telemetry
                # layer; only that layer is replaced, never the provider API.
                result = await super(layer, self).get_embeddings(
                    values, options=options
                )
            except BaseException as exc:
                observability.capture_exception(
                    span=span, exception=exc, timestamp=time_ns()
                )
                capture_error = getattr(observability, "_capture_operation_error", None)
                if callable(capture_error):
                    capture_error(
                        attributes=attributes,
                        exception=exc,
                        operation_duration_histogram=self.duration_histogram,
                        duration=perf_counter() - started,
                    )
                raise
            response_attributes = dict(attributes)
            usage = result.usage or {}
            if (input_tokens := _int_value(usage.get("input_token_count"))) is not None:
                response_attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] = (
                    input_tokens
                )
            if capture and span.is_recording():
                _set_payload(
                    span,
                    SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
                    lambda: [embedding.vector for embedding in result],
                )
            observability._capture_response(
                span=span,
                attributes=response_attributes,
                token_usage_histogram=self.token_usage_histogram,
                operation_duration_histogram=self.duration_histogram,
                duration=perf_counter() - started,
            )
            return result

    layer.get_embeddings = get_embeddings
    return layer, "get_embeddings", original, get_embeddings
