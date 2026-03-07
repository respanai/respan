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
# Error message attribute (non-standard but widely used)
# ---------------------------------------------------------------------------
ERROR_MESSAGE_ATTR = "error.message"
