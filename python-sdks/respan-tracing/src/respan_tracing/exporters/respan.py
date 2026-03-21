import base64
import json
from typing import Dict, Optional, Sequence, List, Any

import requests
from opentelemetry.context import attach, detach, set_value
from opentelemetry.context import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.sdk.trace.export import SpanExportResult
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.trace import StatusCode

from respan_sdk.constants import RESPAN_DOGFOOD_HEADER
from respan_sdk.constants.otlp_constants import (
    OTLP_BOOL_VALUE,
    OTLP_INT_VALUE,
    OTLP_DOUBLE_VALUE,
    OTLP_STRING_VALUE,
    OTLP_BYTES_VALUE,
    OTLP_ARRAY_VALUE,
    OTLP_ARRAY_VALUES_KEY,
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
from ..utils.preprocessing.span_processing import is_root_span_candidate, _is_adk_span
from ..constants.generic_constants import LOGGER_NAME_EXPORTER

logger = get_respan_logger(LOGGER_NAME_EXPORTER)


class ModifiedSpan:
    """A proxy wrapper that forwards all attributes to the original span except parent_span_id"""

    def __init__(self, original_span: ReadableSpan):
        self._original_span = original_span

    def __getattr__(self, name):
        """Forward all attribute access to the original span"""
        if name in ("parent_span_id", "parent", "_parent"):
            return None  # Override parent to None for root-promoted spans
        return getattr(self._original_span, name)


class EnrichedSpan:
    """Proxy wrapper that overrides span name and/or injects extra attributes.

    Used to enrich ADK spans with OpenLLMetry-compatible attributes so the
    backend can extract trace-level fields (name, input, output, tokens).
    """

    def __init__(
        self,
        original_span: ReadableSpan,
        name: Optional[str] = None,
        extra_attributes: Optional[Dict[str, Any]] = None,
        stripped_keys: Optional[set] = None,
    ):
        self._original_span = original_span
        self._name_override = name
        self._extra_attributes = extra_attributes or {}
        self._stripped_keys = stripped_keys or set()

    @property
    def name(self):
        return self._name_override if self._name_override is not None else self._original_span.name

    @property
    def attributes(self):
        attrs = dict(self._original_span.attributes or {})
        for key in self._stripped_keys:
            attrs.pop(key, None)
        attrs.update(self._extra_attributes)
        return attrs

    def __getattr__(self, name):
        return getattr(self._original_span, name)


# Internal/redundant ADK attributes that should NOT appear as custom properties.
_ADK_STRIP_ATTRS = {
    # Internal tracking IDs
    "gcp.vertex.agent.event_id",
    "gcp.vertex.agent.invocation_id",
    # Content moved to input/output fields
    "gcp.vertex.agent.llm_request",
    "gcp.vertex.agent.llm_response",
    # Session ID moved to thread_identifier
    "gcp.vertex.agent.session_id",
    # Tool I/O moved to span input/output
    "gcp.vertex.agent.tool_call_args",
    "gcp.vertex.agent.tool_response",
    # Infrastructure attributes
    "otel.scope.name",
    "otel.scope.version",
    "service.name",
    # Empty/redundant gen_ai attributes
    "gen_ai.agent.description",
    "gen_ai.agent.version",
    "gen_ai.operation.name",
    "gen_ai.response.finish_reasons",
}


def _extract_adk_input(llm_request_json: str) -> Optional[str]:
    """Parse ADK llm_request JSON into formatted input messages JSON string."""
    try:
        req = json.loads(llm_request_json)
        contents = req.get("contents", [])
        messages = []
        for content in contents:
            role = content.get("role", "user")
            parts = content.get("parts", [])
            text_parts = [p.get("text", "") for p in parts if "text" in p]
            messages.append({"role": role, "content": "\n".join(text_parts)})
        return json.dumps(messages) if messages else None
    except (json.JSONDecodeError, AttributeError, TypeError):
        return None


def _extract_adk_output(llm_response_json: str) -> Optional[str]:
    """Parse ADK llm_response JSON into formatted output message JSON string."""
    try:
        resp = json.loads(llm_response_json)
        content = resp.get("content", {})
        parts = content.get("parts", [])
        text_parts = [p.get("text", "") for p in parts if "text" in p]
        role = content.get("role", "model")
        if role == "model":
            role = "assistant"
        return json.dumps({"role": role, "content": "\n".join(text_parts)})
    except (json.JSONDecodeError, AttributeError, TypeError):
        return None


def _enrich_adk_spans(spans: Sequence[ReadableSpan]) -> List[ReadableSpan]:
    """Enrich ADK spans with OpenLLMetry-compatible attributes.

    Maps ADK-specific attributes to the conventions the backend expects:
    - Root "invocation" span: rename to agent name, add input/output, mark as workflow
    - LLM spans: add llm.request.type, map token attribute names
    - All child spans: add traceloop.entity.path to prevent root promotion
    """
    adk_spans = []
    other_spans = []
    for span in spans:
        if _is_adk_span(span):
            adk_spans.append(span)
        else:
            other_spans.append(span)

    if not adk_spans:
        return list(spans)

    # Group ADK spans by trace ID for cross-span enrichment
    trace_groups: Dict[str, List[ReadableSpan]] = {}
    for span in adk_spans:
        ctx = span.get_span_context()
        trace_id = format(ctx.trace_id, "032x") if ctx else "unknown"
        trace_groups.setdefault(trace_id, []).append(span)

    enriched: List[ReadableSpan] = list(other_spans)
    for trace_spans in trace_groups.values():
        enriched.extend(_enrich_adk_trace_group(trace_spans))
    return enriched


def _enrich_adk_trace_group(spans: List[ReadableSpan]) -> List[ReadableSpan]:
    """Enrich a group of ADK spans from the same trace."""
    # --- First pass: collect cross-span info ---
    agent_name = None
    first_input = None
    last_output = None
    session_id = None

    for span in spans:
        prefix = (span.name or "").split(" ")[0]
        attrs = span.attributes or {}

        if prefix == "invoke_agent" and not agent_name:
            agent_name = attrs.get("gen_ai.agent.name")

        # Collect input/output from LLM spans (prefer gen_ai.*, fall back to ADK attrs)
        if prefix in ("call_llm", "generate_content"):
            if first_input is None:
                first_input = (
                    attrs.get("gen_ai.input.messages")
                    or attrs.get("gen_ai.prompt")
                    or _extract_adk_input(attrs.get("gcp.vertex.agent.llm_request", ""))
                )
            output = (
                attrs.get("gen_ai.output.messages")
                or attrs.get("gen_ai.completion")
                or _extract_adk_output(attrs.get("gcp.vertex.agent.llm_response", ""))
            )
            if output:
                last_output = output

        # Also check invoke_agent spans (ADK sometimes puts llm_request/response here)
        elif prefix == "invoke_agent":
            llm_req = attrs.get("gcp.vertex.agent.llm_request", "")
            llm_resp = attrs.get("gcp.vertex.agent.llm_response", "")
            if first_input is None and llm_req and llm_req != "{}":
                first_input = _extract_adk_input(llm_req)
            if llm_resp and llm_resp != "{}":
                parsed_output = _extract_adk_output(llm_resp)
                if parsed_output:
                    last_output = parsed_output

        # Collect session ID for thread_identifier
        if not session_id:
            session_id = attrs.get("gen_ai.conversation.id")

    # Read resource-level identifiers (set by _core.py from instrumentor config)
    resource = getattr(spans[0], "resource", None) if spans else None
    resource_attrs = resource.attributes if resource else {}
    customer_id = resource_attrs.get("respan.customer_params.customer_identifier")
    thread_id = resource_attrs.get("respan.threads.thread_identifier")

    # --- Second pass: enrich each span ---
    enriched = []
    for span in spans:
        prefix = (span.name or "").split(" ")[0]
        attrs = span.attributes or {}
        extra: Dict[str, Any] = {}
        new_name = None

        if prefix == "invocation":
            # Root span: rename to agent name, add workflow kind, propagate I/O
            if agent_name:
                new_name = agent_name
            extra["traceloop.span.kind"] = "workflow"
            if first_input:
                extra["traceloop.entity.input"] = first_input
            if last_output:
                extra["traceloop.entity.output"] = last_output
            # Inject identifiers from resource attributes
            if customer_id:
                extra["respan.customer_params.customer_identifier"] = customer_id
            # Use session_id as thread if not explicitly set
            effective_thread = thread_id or session_id
            if effective_thread:
                extra["respan.threads.thread_identifier"] = effective_thread
        else:
            # Child spans: add entity path to prevent root promotion
            extra["traceloop.entity.path"] = f"adk.{prefix}"

            if prefix in ("call_llm", "generate_content"):
                # LLM spans: add request type + map token attributes
                extra["llm.request.type"] = "chat"
                input_tokens = attrs.get("gen_ai.usage.input_tokens")
                output_tokens = attrs.get("gen_ai.usage.output_tokens")
                if input_tokens is not None:
                    extra["gen_ai.usage.prompt_tokens"] = input_tokens
                if output_tokens is not None:
                    extra["gen_ai.usage.completion_tokens"] = output_tokens
                # Map input/output to traceloop entity attributes (with ADK fallback)
                messages_in = (
                    attrs.get("gen_ai.input.messages")
                    or attrs.get("gen_ai.prompt")
                    or _extract_adk_input(attrs.get("gcp.vertex.agent.llm_request", ""))
                )
                if messages_in:
                    extra["traceloop.entity.input"] = messages_in
                messages_out = (
                    attrs.get("gen_ai.output.messages")
                    or attrs.get("gen_ai.completion")
                    or _extract_adk_output(attrs.get("gcp.vertex.agent.llm_response", ""))
                )
                if messages_out:
                    extra["traceloop.entity.output"] = messages_out

            elif prefix == "execute_tool":
                tool_args = attrs.get("gcp.vertex.agent.tool_call_args")
                if tool_args:
                    extra["traceloop.entity.input"] = tool_args
                tool_resp = attrs.get("gcp.vertex.agent.tool_response")
                if tool_resp:
                    extra["traceloop.entity.output"] = tool_resp

        if new_name or extra:
            enriched.append(EnrichedSpan(span, name=new_name, extra_attributes=extra, stripped_keys=_ADK_STRIP_ATTRS))
        else:
            enriched.append(span)

    return enriched


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
    if isinstance(value, (list, tuple)):
        converted = []
        for item in value:
            v = _convert_attribute_value(item)
            if v is not None:
                converted.append(v)
        return {OTLP_ARRAY_VALUE: {OTLP_ARRAY_VALUES_KEY: converted}}
    # Fallback: stringify
    return {OTLP_STRING_VALUE: str(value)}


def _convert_attributes(attributes: Any) -> List[Dict[str, Any]]:
    """Convert a mapping of attributes to OTLP JSON key-value list."""
    if not attributes:
        return []
    result = []
    for key, value in attributes.items():
        converted = _convert_attribute_value(value)
        if converted is not None:
            result.append({OTLP_ATTR_KEY: str(key), OTLP_ATTR_VALUE: converted})
    return result


def _span_to_otlp_json(span: ReadableSpan) -> Dict[str, Any]:
    """Convert a ReadableSpan (or ModifiedSpan) to OTLP JSON span dict."""
    ctx = span.get_span_context()

    trace_id = format(ctx.trace_id, "032x") if ctx else ""
    span_id = format(ctx.span_id, "016x") if ctx else ""

    # Parent span ID
    parent_span_id = ""
    parent = getattr(span, "parent", None)
    if parent is not None:
        parent_sid = getattr(parent, "span_id", None)
        if parent_sid:
            parent_span_id = format(parent_sid, "016x")

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

        # Enrich ADK spans with OpenLLMetry-compatible attributes before
        # root promotion so the enriched traceloop.span.kind is visible.
        enriched_spans = _enrich_adk_spans(spans)

        # Apply root-span promotion logic
        modified_spans: List[ReadableSpan] = []
        for span in enriched_spans:
            if is_root_span_candidate(span):
                logger.debug("Making span a root span: %s", span.name)
                modified_spans.append(ModifiedSpan(span))
            else:
                modified_spans.append(span)

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
