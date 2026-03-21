"""ADK span enrichment for OpenLLMetry-compatible export.

Extracted from respan.py to keep the core OTLP exporter free of
ADK-specific logic.  Registered as an enricher on RespanSpanExporter
by _core.py when a Google ADK instrumentor is present.
"""

import json
from typing import Dict, Optional, Sequence, List, Any

from opentelemetry.sdk.trace import ReadableSpan

from respan_sdk.constants.adk_constants import is_adk_span
from .respan import EnrichedSpan


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
        if is_adk_span(span):
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
