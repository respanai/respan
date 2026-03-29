"""
OTLP (OpenTelemetry Protocol) JSON format constants.

Single source of truth for OTLP wire-format keys shared between the
serializer (respan-tracing exporter) and deserializer (respan-backend
traces ingest endpoint).

Reference: https://opentelemetry.io/docs/specs/otlp/
"""

# ---------------------------------------------------------------------------
# OTLP JSON value type keys
# The OTLP/HTTP JSON encoding wraps each attribute value in a typed object.
# ---------------------------------------------------------------------------
OTLP_STRING_VALUE = "stringValue"
OTLP_INT_VALUE = "intValue"
OTLP_DOUBLE_VALUE = "doubleValue"
OTLP_BOOL_VALUE = "boolValue"
OTLP_BYTES_VALUE = "bytesValue"
OTLP_ARRAY_VALUE = "arrayValue"
OTLP_KVLIST_VALUE = "kvlistValue"

# Sub-keys inside array/kvlist containers
OTLP_ARRAY_VALUES_KEY = "values"

# ---------------------------------------------------------------------------
# OTLP JSON span structure keys
# ---------------------------------------------------------------------------
OTLP_RESOURCE_SPANS_KEY = "resourceSpans"
OTLP_SCOPE_SPANS_KEY = "scopeSpans"
OTLP_RESOURCE_KEY = "resource"
OTLP_ATTRIBUTES_KEY = "attributes"
OTLP_SCOPE_KEY = "scope"
OTLP_SPANS_KEY = "spans"

OTLP_TRACE_ID_KEY = "traceId"
OTLP_SPAN_ID_KEY = "spanId"
OTLP_PARENT_SPAN_ID_KEY = "parentSpanId"
OTLP_NAME_KEY = "name"
OTLP_VERSION_KEY = "version"
OTLP_KIND_KEY = "kind"
OTLP_START_TIME_KEY = "startTimeUnixNano"
OTLP_END_TIME_KEY = "endTimeUnixNano"
OTLP_STATUS_KEY = "status"
OTLP_EVENTS_KEY = "events"
OTLP_LINKS_KEY = "links"
OTLP_FLAGS_KEY = "flags"
OTLP_TRACE_STATE_KEY = "traceState"
OTLP_DROPPED_ATTRIBUTES_COUNT_KEY = "droppedAttributesCount"

# W3C trace context flag indicating the linked span is from a remote process.
# See: https://www.w3.org/TR/trace-context/#trace-flags (bit 8 = HasRemoteParent)
OTLP_REMOTE_LINK_FLAG = 0x100

# OTLP attribute key/value pair keys
OTLP_ATTR_KEY = "key"
OTLP_ATTR_VALUE = "value"

# ---------------------------------------------------------------------------
# OTel span status codes (per OpenTelemetry spec)
# ---------------------------------------------------------------------------
OTEL_STATUS_CODE_UNSET = 0
OTEL_STATUS_CODE_OK = 1
OTEL_STATUS_CODE_ERROR = 2

OTEL_STATUS_CODE_KEY = "code"
OTEL_STATUS_MESSAGE_KEY = "message"

# ---------------------------------------------------------------------------
# OTel exception event constants
# ---------------------------------------------------------------------------
OTEL_EXCEPTION_EVENT_NAME = "exception"
OTEL_EXCEPTION_MESSAGE_KEY = "exception.message"
OTEL_EXCEPTION_TYPE_KEY = "exception.type"

# ---------------------------------------------------------------------------
# OTel instrumentation scope metadata keys (stored in passthrough metadata)
# ---------------------------------------------------------------------------
OTEL_SCOPE_NAME_KEY = "otel.scope.name"
OTEL_SCOPE_VERSION_KEY = "otel.scope.version"

# ---------------------------------------------------------------------------
# OTel Python SDK ReadableSpan internal field names
# Used by ModifiedSpan proxy to override parent / attributes at export time.
# ---------------------------------------------------------------------------
OTEL_SPAN_PARENT_FIELD = "parent"
OTEL_SPAN_PARENT_PRIVATE_FIELD = "_parent"
OTEL_SPAN_ATTRIBUTES_FIELD = "attributes"

# ---------------------------------------------------------------------------
# Error message attribute (non-standard but widely used)
# ---------------------------------------------------------------------------
ERROR_MESSAGE_ATTR = "error.message"

# ---------------------------------------------------------------------------
# Promoted attribute keys (Tier 1)
#
# Attributes that the backend extracts into typed CHLogV2 columns during
# span-to-log conversion.  After promotion they are excluded from the
# passthrough metadata to avoid duplication.
#
# This is the single source of truth — both the backend and SDK enrichment
# layers import from here.
# ---------------------------------------------------------------------------

# Respan SDK-specific extensions (values from RespanSpanAttributes enum)
RESPAN_PROMOTED_KEYS = frozenset({
    "respan.span_params.custom_identifier",
    "respan.customer_params.customer_identifier",
    "respan.customer_params.email",
    "respan.customer_params.name",
    "respan.evaluation_params.evaluation_identifier",
    "respan.threads.thread_identifier",
    "respan.sessions.session_identifier",
    "respan.trace.trace_group_identifier",
    "respan.metadata",
    "respan.entity.log_method",
    "respan.entity.log_type",
})

# Gen AI semantic conventions + OTel standard attributes
GEN_AI_PROMOTED_KEYS = frozenset({
    # Core LLM request/response attributes
    "llm.request.type",
    "gen_ai.request.model",
    "gen_ai.response.model",
    "gen_ai.request.temperature",
    "gen_ai.request.max_tokens",
    "llm.frequency_penalty",
    "llm.presence_penalty",
    "llm.is_streaming",
    # Token usage
    "gen_ai.usage.prompt_tokens",
    "gen_ai.usage.completion_tokens",
    "llm.usage.total_tokens",
    # Traceloop conventions
    "traceloop.span.kind",
    "traceloop.entity.path",
    "traceloop.entity.input",
    "traceloop.entity.output",
    "traceloop.workflow.name",
    # Non-standard variants (Instructor telemetry, OpenAI-specific)
    "gen_ai.usage.input_tokens",
    "gen_ai.usage.output_tokens",
    "gen_ai.usage.total_tokens",
    "llm.openai.chat_completions.streaming_time_to_first_token",
    "llm.embeddings.0",
    ERROR_MESSAGE_ATTR,
    # OTel v2 semantic convention keys (Pydantic AI, OpenAI instrumentor).
    # SDK enrichment remaps these to backend column names; promoted here
    # so originals don't leak into metadata passthrough.
    "gen_ai.usage.cache_read_input_tokens",
    "llm.usage.reasoning_tokens",
    "llm.request.reasoning_effort",
    # Pydantic AI agent run span attributes — consumed by SDK enrichment,
    # remapped to traceloop.entity.output / gen_ai.request.model.
    "final_result",
    "model_name",
    "agent_name",
    "logfire.msg",
    # Override keys — set by SDK enrichment for the backend override mechanism.
    # Promoted so they don't leak into passthrough metadata.
    "model",
    "prompt_tokens",
    "completion_tokens",
    "total_request_tokens",
})

# Union of all promoted keys for quick lookup
ALL_PROMOTED_KEYS = RESPAN_PROMOTED_KEYS | GEN_AI_PROMOTED_KEYS
