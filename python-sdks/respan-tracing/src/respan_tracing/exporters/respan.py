import base64
import json
from collections import defaultdict
from collections.abc import Mapping
from typing import Dict, Optional, Sequence, List, Any

import requests
from opentelemetry.context import attach, detach, set_value
from opentelemetry.context import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.sdk.trace.export import SpanExportResult
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.trace import StatusCode

from respan_sdk.constants import RESPAN_DOGFOOD_HEADER
from respan_sdk.utils.data_processing.id_processing import format_trace_id, format_span_id
from respan_sdk.constants.otlp_constants import (
    OTLP_BOOL_VALUE,
    OTLP_INT_VALUE,
    OTLP_DOUBLE_VALUE,
    OTLP_STRING_VALUE,
    OTLP_BYTES_VALUE,
    OTLP_ARRAY_VALUE,
    OTLP_ARRAY_VALUES_KEY,
    OTLP_KVLIST_VALUE,
    OTLP_ATTR_KEY,
    OTLP_ATTR_VALUE,
    OTLP_TRACE_ID_KEY,
    OTLP_SPAN_ID_KEY,
    OTLP_PARENT_SPAN_ID_KEY,
    OTLP_NAME_KEY,
    OTLP_KIND_KEY,
    OTLP_START_TIME_KEY,
    OTLP_END_TIME_KEY,
    OTLP_ATTRIBUTES_KEY,
    OTLP_STATUS_KEY,
    OTLP_EVENTS_KEY,
    OTLP_LINKS_KEY,
    OTLP_FLAGS_KEY,
    OTLP_TRACE_STATE_KEY,
    OTLP_DROPPED_ATTRIBUTES_COUNT_KEY,
    OTLP_REMOTE_LINK_FLAG,
    OTLP_RESOURCE_SPANS_KEY,
    OTLP_SCOPE_SPANS_KEY,
    OTLP_RESOURCE_KEY,
    OTLP_SCOPE_KEY,
    OTLP_SPANS_KEY,
    OTLP_VERSION_KEY,
    OTEL_STATUS_CODE_UNSET,
    OTEL_STATUS_CODE_OK,
    OTEL_STATUS_CODE_ERROR,
    OTEL_STATUS_CODE_KEY,
    OTEL_STATUS_MESSAGE_KEY,
)

from ..utils.logging import get_respan_logger, build_spans_export_preview
from ..utils.preprocessing.span_processing import is_root_span_candidate
from ..constants.generic_constants import LOGGER_NAME_EXPORTER

logger = get_respan_logger(LOGGER_NAME_EXPORTER)


class ModifiedSpan:
    """A proxy wrapper that forwards the original span with optional overrides."""

    def __init__(
        self,
        original_span: ReadableSpan,
        overrides: Optional[Dict[str, Any]] = None,
    ):
        self._original_span = original_span
        self._overrides = overrides or {}

    def __getattr__(self, name):
        """Forward all attribute access to the original span unless overridden."""
        if name in self._overrides:
            return self._overrides[name]
        return getattr(self._original_span, name)


_PYDANTIC_AI_SCOPE_NAME = "pydantic-ai"
_OPENAI_CHAT_SPAN_NAMES = frozenset({"openai.chat"})
_PYDANTIC_AI_CHAT_REQUIRED_FIELDS = frozenset({
    "full_request",
    "full_response",
})

def _get_span_identity(span: ReadableSpan) -> Optional[tuple[int, int]]:
    context = span.get_span_context()
    trace_id = getattr(context, "trace_id", None)
    span_id = getattr(context, "span_id", None)
    if trace_id is None or span_id is None:
        return None
    return trace_id, span_id


def _get_parent_identity(span: ReadableSpan) -> Optional[tuple[int, int]]:
    parent = getattr(span, "parent", None)
    parent_span_id = getattr(parent, "span_id", None)
    if parent_span_id is None:
        return None

    parent_trace_id = getattr(parent, "trace_id", None)
    if parent_trace_id is None:
        context = span.get_span_context()
        parent_trace_id = getattr(context, "trace_id", None)
    if parent_trace_id is None:
        return None

    return parent_trace_id, parent_span_id


def _get_scope_name(span: ReadableSpan) -> Optional[str]:
    scope = getattr(span, "instrumentation_scope", None)
    scope_name = getattr(scope, "name", None)
    if isinstance(scope_name, str) and scope_name:
        return scope_name
    return None



def _is_pydantic_ai_chat_wrapper_span(span: ReadableSpan) -> bool:
    if _get_scope_name(span) != _PYDANTIC_AI_SCOPE_NAME:
        return False

    attributes = getattr(span, "attributes", {}) or {}
    if attributes.get("respan.entity.log_type") != "chat":
        return False

    return all(
        attributes.get(field_name) is not None
        for field_name in _PYDANTIC_AI_CHAT_REQUIRED_FIELDS
    )


def _is_openai_chat_span(span: ReadableSpan) -> bool:
    return getattr(span, "name", None) in _OPENAI_CHAT_SPAN_NAMES


def _drop_openai_chat_children(
    spans: Sequence[ReadableSpan],
) -> List[ReadableSpan]:
    """Drop `openai.chat` child spans that are children of pydantic-ai chat
    wrapper spans.  The wrapper already carries the clean, extracted attributes
    (`full_request`, `full_response`, token counts, etc.) so the raw
    `openai.chat` span with its many `gen_ai.*` custom properties is redundant.
    Any children of the dropped `openai.chat` span are reparented to the
    wrapper so the trace tree stays intact."""

    spans_list = list(spans)
    children_by_parent: dict[tuple[int, int], list[ReadableSpan]] = defaultdict(list)

    for span in spans_list:
        parent_identity = _get_parent_identity(span)
        if parent_identity is not None:
            children_by_parent[parent_identity].append(span)

    openai_identities_to_drop: set[tuple[int, int]] = set()
    reparent_map: dict[tuple[int, int], Any] = {}

    for wrapper_span in spans_list:
        if not _is_pydantic_ai_chat_wrapper_span(wrapper_span):
            continue

        wrapper_identity = _get_span_identity(wrapper_span)
        if wrapper_identity is None:
            continue

        wrapper_context = getattr(wrapper_span, "get_span_context", lambda: None)()

        for child in children_by_parent.get(wrapper_identity, []):
            if not _is_openai_chat_span(child):
                continue
            child_identity = _get_span_identity(child)
            if child_identity is None:
                continue
            openai_identities_to_drop.add(child_identity)
            reparent_map[child_identity] = wrapper_context

    if not openai_identities_to_drop:
        return spans_list

    result: List[ReadableSpan] = []
    for span in spans_list:
        span_identity = _get_span_identity(span)
        if span_identity in openai_identities_to_drop:
            continue

        parent_identity = _get_parent_identity(span)
        if parent_identity in openai_identities_to_drop and parent_identity in reparent_map:
            result.append(
                ModifiedSpan(
                    original_span=span,
                    overrides={
                        "parent": reparent_map[parent_identity],
                        "_parent": reparent_map[parent_identity],
                    },
                )
            )
        else:
            result.append(span)

    return result


def _prepare_spans_for_export(spans: Sequence[ReadableSpan]) -> List[ReadableSpan]:
    merged_spans = _drop_openai_chat_children(spans=spans)
    prepared_spans: List[ReadableSpan] = []

    for span in merged_spans:
        if is_root_span_candidate(span):
            logger.debug("Making span a root span: %s", span.name)
            prepared_spans.append(
                ModifiedSpan(
                    original_span=span,
                    overrides={
                        "parent": None,
                        "_parent": None,
                    },
                )
            )
        else:
            prepared_spans.append(span)

    return prepared_spans


def _convert_attribute_value(value: Any) -> Optional[Dict[str, Any]]:
    """Convert a Python attribute value to OTLP JSON typed wrapper."""
    if value is None:
        return None
    if isinstance(value, bool):
        return {OTLP_BOOL_VALUE: value}
    if isinstance(value, int):
        return {OTLP_INT_VALUE: str(value)}
    if isinstance(value, float):
        return {OTLP_DOUBLE_VALUE: value}
    if isinstance(value, str):
        return {OTLP_STRING_VALUE: value}
    if isinstance(value, bytes):
        return {OTLP_BYTES_VALUE: base64.b64encode(value).decode("ascii")}
    if isinstance(value, Mapping):
        converted_items = []
        for item_key, item_value in value.items():
            converted_value = _convert_attribute_value(item_value)
            if converted_value is not None:
                converted_items.append(
                    {OTLP_ATTR_KEY: str(item_key), OTLP_ATTR_VALUE: converted_value}
                )
        return {OTLP_KVLIST_VALUE: {OTLP_ARRAY_VALUES_KEY: converted_items}}
    if isinstance(value, (list, tuple)):
        converted = []
        for item in value:
            v = _convert_attribute_value(item)
            if v is not None:
                converted.append(v)
        return {OTLP_ARRAY_VALUE: {OTLP_ARRAY_VALUES_KEY: converted}}
    # Fallback: stringify
    return {OTLP_STRING_VALUE: str(value)}


# Attributes that duplicate data already captured in child spans.
# pydantic_ai.all_messages — full conversation history on the parent "agent run"
#   span; the same content is already in gen_ai.input/output.messages on each
#   "chat <model>" child span.
# logfire.json_schema — Pydantic-AI/Logfire internal metadata, not useful in Respan.
_STRIPPED_ATTRIBUTES = frozenset({
    "pydantic_ai.all_messages",
    "logfire.json_schema",
})


def _convert_attributes(attributes: Any) -> List[Dict[str, Any]]:
    """Convert a mapping of attributes to OTLP JSON key-value list."""
    if not attributes:
        return []
    result = []
    for key, value in attributes.items():
        if key in _STRIPPED_ATTRIBUTES:
            continue
        converted = _convert_attribute_value(value)
        if converted is not None:
            result.append({OTLP_ATTR_KEY: str(key), OTLP_ATTR_VALUE: converted})
    return result


def _span_to_otlp_json(span: ReadableSpan) -> Dict[str, Any]:
    """Convert a ReadableSpan (or ModifiedSpan) to OTLP JSON span dict."""
    ctx = span.get_span_context()

    trace_id = format_trace_id(ctx.trace_id) if ctx else ""
    span_id = format_span_id(ctx.span_id) if ctx else ""

    # Parent span ID
    parent_span_id = ""
    parent = getattr(span, "parent", None)
    if parent is not None:
        parent_sid = getattr(parent, "span_id", None)
        if parent_sid:
            parent_span_id = format_span_id(parent_sid)

    # Timestamps as nanosecond strings
    start_time_ns = str(span.start_time) if span.start_time else "0"
    end_time_ns = str(span.end_time) if span.end_time else "0"

    # Span kind mapping: OTel Python SpanKind enum is 0-4 (INTERNAL=0, SERVER=1, ...)
    # but OTLP wire format is 1-5 (UNSPECIFIED=0, INTERNAL=1, SERVER=2, ...)
    kind_value = 0  # SPAN_KIND_UNSPECIFIED
    if span.kind is not None:
        raw = span.kind.value if hasattr(span.kind, "value") else int(span.kind)
        kind_value = raw + 1

    # Status
    status_dict = {}
    if span.status is not None:
        code = OTEL_STATUS_CODE_UNSET
        if span.status.status_code == StatusCode.OK:
            code = OTEL_STATUS_CODE_OK
        elif span.status.status_code == StatusCode.ERROR:
            code = OTEL_STATUS_CODE_ERROR
        status_dict[OTEL_STATUS_CODE_KEY] = code
        if span.status.description:
            status_dict[OTEL_STATUS_MESSAGE_KEY] = span.status.description

    # Events
    events = []
    for event in span.events or []:
        event_dict = {
            OTLP_NAME_KEY: event.name,
            "timeUnixNano": str(event.timestamp) if event.timestamp else "0",
        }
        event_attrs = _convert_attributes(event.attributes)
        if event_attrs:
            event_dict[OTLP_ATTRIBUTES_KEY] = event_attrs
        events.append(event_dict)

    links = []
    for link in getattr(span, "links", ()) or ():
        link_ctx = getattr(link, "context", None)
        if link_ctx is None:
            continue

        link_dict = {
            OTLP_TRACE_ID_KEY: format_trace_id(link_ctx.trace_id),
            OTLP_SPAN_ID_KEY: format_span_id(link_ctx.span_id),
            OTLP_ATTRIBUTES_KEY: _convert_attributes(getattr(link, "attributes", None)),
            OTLP_FLAGS_KEY: int(link_ctx.trace_flags) | (
                OTLP_REMOTE_LINK_FLAG if getattr(link_ctx, "is_remote", False) else 0
            ),
        }

        trace_state = getattr(link_ctx, "trace_state", None)
        if trace_state:
            trace_state_header = trace_state.to_header()
            if trace_state_header:
                link_dict[OTLP_TRACE_STATE_KEY] = trace_state_header

        dropped_attributes = getattr(link, "dropped_attributes", 0) or 0
        if dropped_attributes:
            link_dict[OTLP_DROPPED_ATTRIBUTES_COUNT_KEY] = dropped_attributes

        links.append(link_dict)

    result = {
        OTLP_TRACE_ID_KEY: trace_id,
        OTLP_SPAN_ID_KEY: span_id,
        OTLP_NAME_KEY: span.name,
        OTLP_KIND_KEY: kind_value,
        OTLP_START_TIME_KEY: start_time_ns,
        OTLP_END_TIME_KEY: end_time_ns,
        OTLP_ATTRIBUTES_KEY: _convert_attributes(span.attributes),
    }

    if parent_span_id:
        result[OTLP_PARENT_SPAN_ID_KEY] = parent_span_id
    if status_dict:
        result[OTLP_STATUS_KEY] = status_dict
    if events:
        result[OTLP_EVENTS_KEY] = events
    if links:
        result[OTLP_LINKS_KEY] = links

    return result


def _get_resource_key(span: ReadableSpan) -> str:
    """Build a hashable key for grouping spans by resource."""
    resource = getattr(span, "resource", None)
    if not resource or not resource.attributes:
        return ""
    # Sort for deterministic keys
    return json.dumps(dict(sorted(resource.attributes.items())), sort_keys=True, default=str)


def _get_scope_key(span: ReadableSpan) -> str:
    """Build a hashable key for grouping spans by instrumentation scope."""
    scope = getattr(span, "instrumentation_scope", None)
    if not scope:
        return ""
    return f"{scope.name or ''}|{scope.version or ''}"


def _build_otlp_payload(spans: Sequence[ReadableSpan]) -> Dict[str, Any]:
    """
    Group spans by resource and scope, then build OTLP JSON payload.

    Structure: { resourceSpans: [ { resource, scopeSpans: [ { scope, spans } ] } ] }
    """
    # Group: resource_key -> scope_key -> list of span dicts
    resource_groups: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    resource_attrs_map: Dict[str, Any] = {}
    scope_info_map: Dict[str, Any] = {}

    for span in spans:
        r_key = _get_resource_key(span)
        s_key = _get_scope_key(span)

        if r_key not in resource_groups:
            resource_groups[r_key] = {}
            resource = getattr(span, "resource", None)
            resource_attrs_map[r_key] = resource.attributes if resource else {}

        if s_key not in resource_groups[r_key]:
            resource_groups[r_key][s_key] = []
            scope = getattr(span, "instrumentation_scope", None)
            scope_info_map[s_key] = scope

        resource_groups[r_key][s_key].append(_span_to_otlp_json(span))

    # Build OTLP JSON
    resource_spans = []
    for r_key, scope_groups in resource_groups.items():
        scope_spans = []
        for s_key, span_dicts in scope_groups.items():
            scope_entry = {OTLP_SPANS_KEY: span_dicts}
            scope = scope_info_map.get(s_key)
            if scope:
                scope_dict = {}
                if scope.name:
                    scope_dict[OTLP_NAME_KEY] = scope.name
                if scope.version:
                    scope_dict[OTLP_VERSION_KEY] = scope.version
                scope_entry[OTLP_SCOPE_KEY] = scope_dict
            scope_spans.append(scope_entry)

        rs_entry = {OTLP_SCOPE_SPANS_KEY: scope_spans}
        r_attrs = resource_attrs_map.get(r_key, {})
        if r_attrs:
            rs_entry[OTLP_RESOURCE_KEY] = {OTLP_ATTRIBUTES_KEY: _convert_attributes(r_attrs)}
        resource_spans.append(rs_entry)

    return {OTLP_RESOURCE_SPANS_KEY: resource_spans}


class RespanSpanExporter:
    """
    Custom span exporter for Respan that serializes spans as OTLP JSON
    and POSTs them to the /v2/traces endpoint.

    Anti-recursion: Uses OpenTelemetry's suppress_instrumentation context
    to prevent auto-instrumented HTTP libraries (requests, urllib3) from
    creating spans during export. This ensures no infinite trace loops
    even when the ingest endpoint is itself traced.
    """

    def __init__(
        self,
        endpoint: str,
        api_key: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: int = 30,
    ):
        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self._is_shutdown = False

        # Persistent session for TCP connection reuse across export() calls.
        # At 1% prod sampling with 3-5 traces per request, connection overhead matters.
        self._session = requests.Session()
        self._session.headers.update({
            "Content-Type": "application/json",
            # Anti-recursion marker: tells the server "don't emit new traces
            # while processing this request" — but still ingest the payload.
            # Prevents infinite loops when the ingest endpoint is itself observed.
            RESPAN_DOGFOOD_HEADER: "1",
        })
        if headers:
            self._session.headers.update(headers)
        if api_key:
            self._session.headers["Authorization"] = f"Bearer {api_key}"

        self._traces_url = f"{self.endpoint}/v2/traces"
        logger.debug("OTLP JSON traces endpoint: %s", self._traces_url)

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        """Export spans as OTLP JSON to /v2/traces."""
        if self._is_shutdown:
            return SpanExportResult.FAILURE

        modified_spans = _prepare_spans_for_export(spans=spans)

        # Debug preview
        try:
            if logger.isEnabledFor(10):  # logging.DEBUG
                preview = build_spans_export_preview(modified_spans)
                logger.debug("Export preview (sanitized): %s", preview)
        except Exception:
            pass

        # Build OTLP JSON payload
        payload = _build_otlp_payload(modified_spans)

        # Suppress OTel instrumentation during export to prevent recursion.
        # Without this, auto-instrumented `requests` would create spans for
        # the export POST, which would be exported, creating more spans, etc.
        token = attach(set_value(_SUPPRESS_INSTRUMENTATION_KEY, True))
        try:
            response = self._session.post(
                url=self._traces_url,
                data=json.dumps(payload, default=str),
                timeout=self.timeout,
            )
            if response.status_code < 400:
                logger.debug(
                    "Exported %d spans successfully (HTTP %d)",
                    len(modified_spans),
                    response.status_code,
                )
                return SpanExportResult.SUCCESS
            else:
                logger.warning(
                    "Failed to export spans: HTTP %d — %s",
                    response.status_code,
                    response.text[:500],
                )
                return SpanExportResult.FAILURE
        except Exception as e:
            logger.warning("Failed to export spans: %s", e)
            return SpanExportResult.FAILURE
        finally:
            detach(token)

    def shutdown(self):
        """Shutdown the exporter and close the HTTP session."""
        self._is_shutdown = True
        self._session.close()

    def force_flush(self, timeout_millis: int = 30000):
        """Force flush — no-op for HTTP JSON exporter (each export is synchronous)."""
        return True
