"""ADK span enrichment for OpenLLMetry-compatible export.

Converts raw ADK OTel spans into enriched spans with traceloop.* and
gen_ai.* attributes that the Respan backend expects.  Uses EnrichedSpan
from respan_tracing to wrap originals — preserving all metadata (resource,
events, links, status) without rebuilding from scratch.
"""

import json
import logging
from typing import Dict, List, Optional, Sequence, Any, Set

from opentelemetry.sdk.trace import ReadableSpan

from respan_tracing.exporters.respan import EnrichedSpan

from .adk_detection import is_adk_span

logger = logging.getLogger(__name__)


# Internal/redundant ADK attributes that should NOT appear in enriched spans.
_ADK_STRIP_ATTRS: Set[str] = {
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


def enrich_adk_batch(spans: Sequence[ReadableSpan]) -> List[ReadableSpan]:
    """Enrich ADK spans with OpenLLMetry-compatible attributes.

    Separates ADK spans from other spans, groups ADK spans by trace ID,
    enriches each group, and returns the combined result.
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
    """Enrich a group of ADK spans from the same trace.

    Two-pass algorithm:
    1. Collect cross-span info (agent name, first input, last output, session ID).
    2. Wrap each span with EnrichedSpan — adds traceloop attributes, strips
       internal ADK attributes, and optionally renames/clears parent.

    Customer identifier and thread identifier are already on the original spans
    (bridged by RespanSpanProcessor.on_start from the _PROPAGATED_ATTRIBUTES
    ContextVar) and are preserved automatically by EnrichedSpan.
    """
    # --- First pass: collect cross-span info ---
    agent_name = None
    first_input = None
    last_output = None
    session_id = None

    for span in spans:
        prefix = (span.name or "").split(" ")[0]
        attrs = span.attributes or {}

        if not agent_name:
            agent_name = attrs.get("gen_ai.agent.name")

        # Collect input/output from LLM spans
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

        # Also check invoke_agent spans
        elif prefix == "invoke_agent":
            llm_req = attrs.get("gcp.vertex.agent.llm_request", "")
            llm_resp = attrs.get("gcp.vertex.agent.llm_response", "")
            if first_input is None and llm_req and llm_req != "{}":
                first_input = _extract_adk_input(llm_req)
            if llm_resp and llm_resp != "{}":
                parsed_output = _extract_adk_output(llm_resp)
                if parsed_output:
                    last_output = parsed_output

        # Collect session ID for thread_identifier fallback
        if not session_id:
            session_id = attrs.get("gen_ai.conversation.id")

    # --- Second pass: enrich each span ---
    enriched = []
    for span in spans:
        prefix = (span.name or "").split(" ")[0]
        attrs = span.attributes or {}
        extra: Dict[str, Any] = {}
        name_override = None
        is_root = False

        if prefix == "invocation":
            # Root span: rename to agent name, add workflow kind, propagate I/O
            if agent_name:
                name_override = agent_name
            extra["traceloop.span.kind"] = "workflow"
            if first_input:
                extra["traceloop.entity.input"] = first_input
            if last_output:
                extra["traceloop.entity.output"] = last_output
            # Fall back to ADK session_id when no thread_identifier was propagated
            if session_id and not attrs.get("respan.threads.thread_identifier"):
                extra["respan.threads.thread_identifier"] = session_id
            is_root = True
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
                # Map input/output
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

        enriched.append(EnrichedSpan(
            span,
            extra_attrs=extra,
            clear_parent=is_root,
            name_override=name_override,
            strip_attrs=_ADK_STRIP_ATTRS,
        ))

    return enriched
