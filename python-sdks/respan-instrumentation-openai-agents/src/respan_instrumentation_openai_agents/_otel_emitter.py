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
from typing import Any, Dict, Optional, Union

from agents.tracing.span_data import (
    AgentSpanData,
    CustomSpanData,
    FunctionSpanData,
    GenerationSpanData,
    GuardrailSpanData,
    HandoffSpanData,
    ResponseSpanData,
)
from agents.tracing.spans import Span, SpanImpl
from agents.tracing.traces import Trace

from respan_tracing.utils.span_factory import build_readable_span, inject_span

from ._converter import _format_input_messages, _format_output, _parse_ts, _serialize

logger = logging.getLogger(__name__)

# Attribute keys (matching opentelemetry-semconv-ai)
_SPAN_KIND = "traceloop.span.kind"
_ENTITY_NAME = "traceloop.entity.name"
_ENTITY_PATH = "traceloop.entity.path"
_ENTITY_INPUT = "traceloop.entity.input"
_ENTITY_OUTPUT = "traceloop.entity.output"
_WORKFLOW_NAME = "traceloop.workflow.name"
_LLM_REQUEST_TYPE = "llm.request.type"
_GEN_AI_SYSTEM = "gen_ai.system"
_GEN_AI_MODEL = "gen_ai.request.model"
_GEN_AI_PROMPT_TOKENS = "gen_ai.usage.prompt_tokens"
_GEN_AI_COMPLETION_TOKENS = "gen_ai.usage.completion_tokens"
_LOG_TYPE = "respan.entity.log_type"


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
        _SPAN_KIND: span_kind,
        _ENTITY_NAME: entity_name,
        _ENTITY_PATH: entity_path,
        _LOG_TYPE: log_type,
    }


def _safe_json(obj: Any) -> str:
    """JSON-encode *obj*, falling back to str() on failure."""
    try:
        return json.dumps(obj, default=str)
    except Exception:
        return str(obj)


# ---------------------------------------------------------------------------
# Per-type emitters
# ---------------------------------------------------------------------------


def emit_trace(trace_obj: Trace) -> None:
    """Emit a Trace (root workflow span)."""
    attrs = _base_attrs(
        span_kind="workflow",
        entity_name=trace_obj.name or "trace",
        entity_path="",  # root — no parent path
        log_type="workflow",
    )
    attrs[_WORKFLOW_NAME] = trace_obj.name or "trace"

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
        log_type="agent",
    )
    attrs[_WORKFLOW_NAME] = name
    attrs["respan.metadata.agent_name"] = name
    if span_data.tools:
        attrs["respan.span.tools"] = _safe_json(span_data.tools)
    if span_data.handoffs:
        attrs["respan.span.handoffs"] = _safe_json(span_data.handoffs)

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
    attrs = _base_attrs(
        span_kind="task",
        entity_name="response",
        entity_path="response",
        log_type="response",
    )
    attrs[_LLM_REQUEST_TYPE] = "chat"
    attrs[_GEN_AI_SYSTEM] = "openai"

    # Input
    input_msgs = _format_input_messages(span_data.input)
    if input_msgs:
        attrs[_ENTITY_INPUT] = _safe_json(input_msgs)

    # Response data
    resp = span_data.response
    if resp:
        model = getattr(resp, "model", None) or ""
        if model:
            attrs[_GEN_AI_MODEL] = model

        if hasattr(resp, "output") and resp.output:
            output = _format_output(resp.output)
            attrs[_ENTITY_OUTPUT] = _safe_json(output)

        usage = getattr(resp, "usage", None)
        if usage:
            attrs[_GEN_AI_PROMPT_TOKENS] = getattr(usage, "input_tokens", 0) or 0
            attrs[_GEN_AI_COMPLETION_TOKENS] = getattr(usage, "output_tokens", 0) or 0

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
        log_type="tool",
    )

    input_str = _serialize(span_data.input) or ""
    if not isinstance(input_str, str):
        input_str = json.dumps(input_str, default=str)
    attrs[_ENTITY_INPUT] = _safe_json([{"role": "tool", "content": input_str}])

    output_str = _serialize(span_data.output) or ""
    if not isinstance(output_str, str):
        output_str = json.dumps(output_str, default=str)
    attrs[_ENTITY_OUTPUT] = _safe_json({"role": "tool", "content": output_str})

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
    attrs = _base_attrs(
        span_kind="task",
        entity_name="generation",
        entity_path="generation",
        log_type="generation",
    )
    attrs[_LLM_REQUEST_TYPE] = "chat"

    if span_data.model:
        attrs[_GEN_AI_MODEL] = span_data.model

    input_msgs = _format_input_messages(span_data.input)
    if input_msgs:
        attrs[_ENTITY_INPUT] = _safe_json(input_msgs)

    output = _format_output(span_data.output)
    attrs[_ENTITY_OUTPUT] = _safe_json(output)

    if span_data.usage:
        u = span_data.usage
        attrs[_GEN_AI_PROMPT_TOKENS] = u.get("prompt_tokens") or u.get("input_tokens") or 0
        attrs[_GEN_AI_COMPLETION_TOKENS] = u.get("completion_tokens") or u.get("output_tokens") or 0

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
        log_type="handoff",
    )
    attrs[_ENTITY_INPUT] = _safe_json(from_agent)
    attrs[_ENTITY_OUTPUT] = _safe_json(to_agent)
    attrs["respan.metadata.from_agent"] = from_agent
    attrs["respan.metadata.to_agent"] = to_agent

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
        log_type="guardrail",
    )
    attrs["respan.metadata.guardrail_name"] = span_data.name
    attrs["respan.metadata.triggered"] = str(span_data.triggered)

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
        log_type="custom",
    )
    data = span_data.data or {}
    for k, v in data.items():
        if k in ("model",):
            attrs[_GEN_AI_MODEL] = v
        elif k == "prompt_tokens":
            attrs[_GEN_AI_PROMPT_TOKENS] = v
        elif k == "completion_tokens":
            attrs[_GEN_AI_COMPLETION_TOKENS] = v
        elif k == "input":
            attrs[_ENTITY_INPUT] = _safe_json(v)
        elif k == "output":
            attrs[_ENTITY_OUTPUT] = _safe_json(v)
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
