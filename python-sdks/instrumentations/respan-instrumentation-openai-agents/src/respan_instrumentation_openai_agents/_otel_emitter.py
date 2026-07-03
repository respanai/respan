"""Emit OpenAI Agents SDK spans as OTEL ReadableSpan objects.

Each per-type emitter converts an OpenAI Agents SDK Trace/Span into a
``ReadableSpan`` with ``traceloop.*`` and ``gen_ai.*`` attributes, then
injects it into the OTEL pipeline via ``inject_span()``.

Attribute mapping follows the same conventions as the decorator-based
spans (``traceloop.span.kind``, ``traceloop.entity.*``) and auto-
instrumented LLM spans (``llm.request.type``, ``gen_ai.*``).

**Critical:** ALL child spans must have a non-empty
``traceloop.entity.path`` to prevent accidental root-span promotion
by ``is_root_span_candidate()``.
"""

import json
import logging
from typing import Any, Dict, Union

from agents.tracing.span_data import (
    AgentSpanData,
    CustomSpanData,
    FunctionSpanData,
    GenerationSpanData,
    GuardrailSpanData,
    HandoffSpanData,
    ResponseSpanData,
    TaskSpanData,
    TurnSpanData,
)
from agents.tracing.spans import Span, SpanImpl
from agents.tracing.traces import Trace

from opentelemetry.semconv_ai import SpanAttributes, LLMRequestTypeValues

from respan_sdk.constants.llm_logging import (
    LOG_TYPE_AGENT,
    LOG_TYPE_CUSTOM,
    LOG_TYPE_GENERATION,
    LOG_TYPE_GUARDRAIL,
    LOG_TYPE_HANDOFF,
    LOG_TYPE_RESPONSE,
    LOG_TYPE_TASK,
    LOG_TYPE_TOOL,
    LOG_TYPE_WORKFLOW,
)
from respan_sdk.constants.span_attributes import (
    GEN_AI_SYSTEM,
    LLM_REQUEST_MODEL,
    LLM_REQUEST_TYPE,
    LLM_USAGE_COMPLETION_TOKENS,
    LLM_USAGE_PROMPT_TOKENS,
    RESPAN_LOG_TYPE,
    RESPAN_METADATA_AGENT_NAME,
    RESPAN_METADATA_FROM_AGENT,
    RESPAN_METADATA_GUARDRAIL_NAME,
    RESPAN_METADATA_TO_AGENT,
    RESPAN_METADATA_TRIGGERED,
    RESPAN_SPAN_HANDOFFS,
    RESPAN_SPAN_TOOL_CALLS,
    RESPAN_SPAN_TOOLS,
)
from respan_sdk.utils.serialization import serialize_value
from respan_tracing.utils.span_factory import build_readable_span, inject_span

from respan_instrumentation_openai_agents._utils import (
    _format_input_messages,
    _format_output,
    _parse_ts,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_timestamps(item: SpanImpl):
    """Extract start/end nanoseconds from an SDK span's ISO timestamps."""
    start_ns = end_ns = None
    if item.started_at:
        try:
            start_ns = int(_parse_ts(item.started_at).timestamp() * 1e9)
        except Exception:
            pass
    if item.ended_at:
        try:
            end_ns = int(_parse_ts(item.ended_at).timestamp() * 1e9)
        except Exception:
            pass
    return start_ns, end_ns


def _base_attrs(
    span_kind: str,
    entity_name: str,
    entity_path: str,
    log_type: str,
) -> Dict[str, Any]:
    """Build the common attribute dict shared by all emitters."""
    return {
        SpanAttributes.TRACELOOP_SPAN_KIND: span_kind,
        SpanAttributes.TRACELOOP_ENTITY_NAME: entity_name,
        SpanAttributes.TRACELOOP_ENTITY_PATH: entity_path,
        RESPAN_LOG_TYPE: log_type,
    }


def _llm_base_attrs(
    entity_name: str,
    entity_path: str,
    log_type: str,
) -> Dict[str, Any]:
    """Common attrs for an LLM span — deliberately WITHOUT a span kind.

    Stamping ``traceloop.span.kind="task"`` on the LLM span makes the
    backend file it as a generic task, so ``total_cost``/``total_tokens``
    never roll up ($0). Omitting the kind (the span stays processable via
    its ``traceloop.entity.path``) lets the backend classify it as a
    ``chat`` LLM call off ``llm.request.type``/``gen_ai.system``.
    """
    return {
        SpanAttributes.TRACELOOP_ENTITY_NAME: entity_name,
        SpanAttributes.TRACELOOP_ENTITY_PATH: entity_path,
        RESPAN_LOG_TYPE: log_type,
    }


def _safe_json(obj: Any) -> str:
    """JSON-encode *obj*, falling back to str() on failure."""
    try:
        return json.dumps(obj, default=str)
    except Exception:
        return str(obj)


def _set_json_structured_attr(attrs: Dict[str, Any], key: str, value: Any) -> None:
    """Store structured values as JSON strings for OTEL attribute safety."""
    if value:
        attrs[key] = _safe_json(value)


def _extract_tools(tools: list) -> list:
    """Convert Response API tool definitions to Chat Completions format."""
    result = []
    for tool in tools:
        tool_dict = serialize_value(tool)
        if not isinstance(tool_dict, dict):
            continue
        tool_type = tool_dict.get("type", "")
        if tool_type == "function":
            func = {
                "name": tool_dict.get("name", ""),
            }
            desc = tool_dict.get("description")
            if desc:
                func["description"] = desc
            params = tool_dict.get("parameters")
            if params:
                func["parameters"] = params
            result.append({"type": "function", "function": func})
        else:
            result.append(tool_dict)
    return result


def _extract_tool_calls(output: list) -> list:
    """Extract function tool calls from Response API output items."""
    result = []
    for item in output:
        item_dict = serialize_value(item)
        if isinstance(item_dict, dict) and item_dict.get("type") == "function_call":
            result.append({
                "id": item_dict.get("call_id", ""),
                "type": "function",
                "function": {
                    "name": item_dict.get("name", ""),
                    "arguments": item_dict.get("arguments", ""),
                },
            })
    return result


# ---------------------------------------------------------------------------
# Per-type emitters
# ---------------------------------------------------------------------------


def emit_trace(trace_obj: Trace) -> None:
    """Emit a Trace (root workflow span)."""
    attrs = _base_attrs(
        span_kind="workflow",
        entity_name=trace_obj.name or "trace",
        entity_path="",  # root — no parent path
        log_type=LOG_TYPE_WORKFLOW,
    )
    attrs[SpanAttributes.TRACELOOP_WORKFLOW_NAME] = trace_obj.name or "trace"

    span = build_readable_span(
        name=f"{trace_obj.name}.workflow",
        trace_id=trace_obj.trace_id,
        span_id=trace_obj.trace_id,  # root span uses trace_id as span_id
        attributes=attrs,
    )
    inject_span(span)


def emit_agent(item: SpanImpl, span_data: AgentSpanData) -> None:
    """Emit an AgentSpanData span."""
    start_ns, end_ns = _resolve_timestamps(item)
    name = span_data.name or "agent"
    attrs = _base_attrs(
        span_kind="agent",
        entity_name=name,
        entity_path=name,
        log_type=LOG_TYPE_AGENT,
    )
    attrs[SpanAttributes.TRACELOOP_WORKFLOW_NAME] = name
    attrs[RESPAN_METADATA_AGENT_NAME] = name
    if span_data.tools:
        attrs[RESPAN_SPAN_TOOLS] = _safe_json(span_data.tools)
    if span_data.handoffs:
        attrs[RESPAN_SPAN_HANDOFFS] = _safe_json(span_data.handoffs)

    span = build_readable_span(
        name=f"{name}.agent",
        trace_id=item.trace_id,
        span_id=item.span_id,
        parent_id=item.parent_id or item.trace_id,
        start_time_ns=start_ns,
        end_time_ns=end_ns,
        attributes=attrs,
        status_code=400 if item.error else 200,
        error_message=str(item.error) if item.error else None,
    )
    inject_span(span)


def emit_response(item: SpanImpl, span_data: ResponseSpanData) -> None:
    """Emit a ResponseSpanData span (the actual LLM call)."""
    start_ns, end_ns = _resolve_timestamps(item)
    attrs = _llm_base_attrs(
        entity_name="response",
        entity_path="response",
        log_type=LOG_TYPE_RESPONSE,
    )
    attrs[LLM_REQUEST_TYPE] = LLMRequestTypeValues.CHAT.value
    attrs[GEN_AI_SYSTEM] = "openai"

    # Input
    input_msgs = _format_input_messages(span_data.input)
    if input_msgs:
        attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = _safe_json(input_msgs)

    # Response data
    resp = span_data.response
    if resp:
        model = getattr(resp, "model", None) or ""
        if model:
            attrs[LLM_REQUEST_MODEL] = model

        if hasattr(resp, "output") and resp.output:
            output = _format_output(resp.output)
            attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = output

            tool_calls = _extract_tool_calls(resp.output)
            _set_json_structured_attr(attrs, RESPAN_SPAN_TOOL_CALLS, tool_calls)

        if hasattr(resp, "tools") and resp.tools:
            tools_list = _extract_tools(resp.tools)
            _set_json_structured_attr(attrs, RESPAN_SPAN_TOOLS, tools_list)

        usage = getattr(resp, "usage", None)
        if usage:
            attrs[LLM_USAGE_PROMPT_TOKENS] = getattr(usage, "input_tokens", 0) or 0
            attrs[LLM_USAGE_COMPLETION_TOKENS] = getattr(usage, "output_tokens", 0) or 0

    span = build_readable_span(
        name="openai.chat",
        trace_id=item.trace_id,
        span_id=item.span_id,
        parent_id=item.parent_id or item.trace_id,
        start_time_ns=start_ns,
        end_time_ns=end_ns,
        attributes=attrs,
        status_code=400 if item.error else 200,
        error_message=str(item.error) if item.error else None,
    )
    inject_span(span)


def emit_function(item: SpanImpl, span_data: FunctionSpanData) -> None:
    """Emit a FunctionSpanData span (tool call)."""
    start_ns, end_ns = _resolve_timestamps(item)
    name = span_data.name or "function"
    attrs = _base_attrs(
        span_kind="tool",
        entity_name=name,
        entity_path=name,
        log_type=LOG_TYPE_TOOL,
    )

    input_str = serialize_value(span_data.input) or ""
    if not isinstance(input_str, str):
        input_str = json.dumps(input_str, default=str)
    attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = _safe_json([{"role": "tool", "content": input_str}])

    output_str = serialize_value(span_data.output) or ""
    if not isinstance(output_str, str):
        output_str = json.dumps(output_str, default=str)
    attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = _safe_json({"role": "tool", "content": output_str})

    span = build_readable_span(
        name=f"{name}.tool",
        trace_id=item.trace_id,
        span_id=item.span_id,
        parent_id=item.parent_id or item.trace_id,
        start_time_ns=start_ns,
        end_time_ns=end_ns,
        attributes=attrs,
        status_code=400 if item.error else 200,
        error_message=str(item.error) if item.error else None,
    )
    inject_span(span)


def emit_generation(item: SpanImpl, span_data: GenerationSpanData) -> None:
    """Emit a GenerationSpanData span."""
    start_ns, end_ns = _resolve_timestamps(item)
    attrs = _llm_base_attrs(
        entity_name="generation",
        entity_path="generation",
        log_type=LOG_TYPE_GENERATION,
    )
    attrs[LLM_REQUEST_TYPE] = LLMRequestTypeValues.CHAT.value
    attrs[GEN_AI_SYSTEM] = "openai"

    if span_data.model:
        attrs[LLM_REQUEST_MODEL] = span_data.model

    input_msgs = _format_input_messages(span_data.input)
    if input_msgs:
        attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = _safe_json(input_msgs)

    output = _format_output(span_data.output)
    attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = output
    _set_json_structured_attr(
        attrs,
        RESPAN_SPAN_TOOL_CALLS,
        _extract_tool_calls(span_data.output or []),
    )

    if span_data.usage:
        u = span_data.usage
        attrs[LLM_USAGE_PROMPT_TOKENS] = u.get("prompt_tokens") or u.get("input_tokens") or 0
        attrs[LLM_USAGE_COMPLETION_TOKENS] = u.get("completion_tokens") or u.get("output_tokens") or 0

    span = build_readable_span(
        name="openai.chat",
        trace_id=item.trace_id,
        span_id=item.span_id,
        parent_id=item.parent_id or item.trace_id,
        start_time_ns=start_ns,
        end_time_ns=end_ns,
        attributes=attrs,
        status_code=400 if item.error else 200,
        error_message=str(item.error) if item.error else None,
    )
    inject_span(span)


def emit_handoff(item: SpanImpl, span_data: HandoffSpanData) -> None:
    """Emit a HandoffSpanData span."""
    start_ns, end_ns = _resolve_timestamps(item)
    from_agent = span_data.from_agent or ""
    to_agent = span_data.to_agent or ""
    attrs = _base_attrs(
        span_kind="task",
        entity_name="handoff",
        entity_path="handoff",
        log_type=LOG_TYPE_HANDOFF,
    )
    attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = _safe_json(from_agent)
    attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = _safe_json(to_agent)
    attrs[RESPAN_METADATA_FROM_AGENT] = from_agent
    attrs[RESPAN_METADATA_TO_AGENT] = to_agent

    span = build_readable_span(
        name="handoff.task",
        trace_id=item.trace_id,
        span_id=item.span_id,
        parent_id=item.parent_id or item.trace_id,
        start_time_ns=start_ns,
        end_time_ns=end_ns,
        attributes=attrs,
        status_code=400 if item.error else 200,
        error_message=str(item.error) if item.error else None,
    )
    inject_span(span)


def emit_guardrail(item: SpanImpl, span_data: GuardrailSpanData) -> None:
    """Emit a GuardrailSpanData span."""
    start_ns, end_ns = _resolve_timestamps(item)
    name = f"guardrail:{span_data.name}"
    attrs = _base_attrs(
        span_kind="task",
        entity_name=name,
        entity_path=name,
        log_type=LOG_TYPE_GUARDRAIL,
    )
    attrs[RESPAN_METADATA_GUARDRAIL_NAME] = span_data.name
    attrs[RESPAN_METADATA_TRIGGERED] = str(span_data.triggered)

    span = build_readable_span(
        name=f"{name}.task",
        trace_id=item.trace_id,
        span_id=item.span_id,
        parent_id=item.parent_id or item.trace_id,
        start_time_ns=start_ns,
        end_time_ns=end_ns,
        attributes=attrs,
        status_code=400 if item.error else 200,
        error_message=str(item.error) if item.error else None,
    )
    inject_span(span)


def emit_custom(item: SpanImpl, span_data: CustomSpanData) -> None:
    """Emit a CustomSpanData span."""
    start_ns, end_ns = _resolve_timestamps(item)
    name = span_data.name or "custom"
    attrs = _base_attrs(
        span_kind="task",
        entity_name=name,
        entity_path=name,
        log_type=LOG_TYPE_CUSTOM,
    )
    data = span_data.data or {}
    for k, v in data.items():
        if k in ("model",):
            attrs[LLM_REQUEST_MODEL] = v
        elif k == "prompt_tokens":
            attrs[LLM_USAGE_PROMPT_TOKENS] = v
        elif k == "completion_tokens":
            attrs[LLM_USAGE_COMPLETION_TOKENS] = v
        elif k == "input":
            attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = _safe_json(v)
        elif k == "output":
            attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = _safe_json(v)
        else:
            attrs[f"respan.metadata.{k}"] = str(v)

    span = build_readable_span(
        name=f"{name}.task",
        trace_id=item.trace_id,
        span_id=item.span_id,
        parent_id=item.parent_id or item.trace_id,
        start_time_ns=start_ns,
        end_time_ns=end_ns,
        attributes=attrs,
        status_code=400 if item.error else 200,
        error_message=str(item.error) if item.error else None,
    )
    inject_span(span)


def _emit_structural(
    item: SpanImpl,
    span_data: Any,
    name: str,
    log_type: str,
    agent_name: str = None,
) -> None:
    """Emit a structural parent span (Task/Turn).

    These are the SDK's tree scaffolding — one ``TaskSpanData`` per
    ``Runner.run`` and one ``TurnSpanData`` per agent-loop turn. Without
    them the agent/LLM/tool children orphan onto the trace root and the
    backend renders the trace **flat**. Emitting them with the SDK's real
    ``span_id``/``parent_id`` re-nests the tree.
    """
    start_ns, end_ns = _resolve_timestamps(item)
    attrs = _base_attrs(
        span_kind="task",
        entity_name=name,
        entity_path=name,
        log_type=log_type,
    )
    if agent_name:
        attrs[RESPAN_METADATA_AGENT_NAME] = agent_name

    usage = getattr(span_data, "usage", None) or {}
    if usage:
        attrs[LLM_USAGE_PROMPT_TOKENS] = (
            usage.get("prompt_tokens") or usage.get("input_tokens") or 0
        )
        attrs[LLM_USAGE_COMPLETION_TOKENS] = (
            usage.get("completion_tokens") or usage.get("output_tokens") or 0
        )

    span = build_readable_span(
        name=f"{name}.task",
        trace_id=item.trace_id,
        span_id=item.span_id,
        parent_id=item.parent_id or item.trace_id,
        start_time_ns=start_ns,
        end_time_ns=end_ns,
        attributes=attrs,
        status_code=400 if item.error else 200,
        error_message=str(item.error) if item.error else None,
    )
    inject_span(span)


def emit_task(item: SpanImpl, span_data: TaskSpanData) -> None:
    """Emit a TaskSpanData span (one top-level Runner run)."""
    _emit_structural(
        item, span_data, span_data.name or "agent-run", LOG_TYPE_WORKFLOW
    )


def emit_turn(item: SpanImpl, span_data: TurnSpanData) -> None:
    """Emit a TurnSpanData span (one agent-loop turn)."""
    _emit_structural(
        item,
        span_data,
        f"turn-{span_data.turn}",
        LOG_TYPE_TASK,
        span_data.agent_name,
    )


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_EMITTERS = {
    ResponseSpanData: emit_response,
    FunctionSpanData: emit_function,
    GenerationSpanData: emit_generation,
    AgentSpanData: emit_agent,
    HandoffSpanData: emit_handoff,
    GuardrailSpanData: emit_guardrail,
    CustomSpanData: emit_custom,
    TaskSpanData: emit_task,
    TurnSpanData: emit_turn,
}


def emit_sdk_item(item: Union[Trace, Span[Any]]) -> None:
    """Convert an OpenAI Agents SDK Trace or Span and inject into OTEL pipeline."""
    if isinstance(item, Trace):
        emit_trace(item)
        return

    if isinstance(item, SpanImpl):
        emitter = _EMITTERS.get(type(item.span_data))
        if emitter is None:
            logger.warning("Unknown span data type: %s", type(item.span_data).__name__)
            return
        try:
            emitter(item, item.span_data)
        except Exception:
            logger.exception("Error emitting %s", type(item.span_data).__name__)
        return

    logger.debug("Skipping unsupported item type: %s", type(item).__name__)
