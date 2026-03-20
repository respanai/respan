"""Convert OpenAI Agents SDK Trace/Span objects to Respan log dicts.

Produces flat dicts matching the backend ``/v1/traces/ingest`` contract:

- ``input``  → list of messages or content
- ``output`` → completion message dict
- ``prompt_tokens`` / ``completion_tokens`` at top level
- ``model``, ``log_type``, ``span_name``, tracing IDs, etc.

Uses ``RespanTextLogParams`` from ``respan-sdk`` for schema validation.
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

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
from respan_sdk.constants.llm_logging import (
    LOG_TYPE_AGENT,
    LOG_TYPE_CUSTOM,
    LOG_TYPE_GENERATION,
    LOG_TYPE_GUARDRAIL,
    LOG_TYPE_HANDOFF,
    LOG_TYPE_RESPONSE,
    LOG_TYPE_TOOL,
)
from respan_sdk.respan_types.param_types import RespanTextLogParams

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _serialize(obj: Any) -> Any:
    """Recursively convert *obj* to plain JSON-serializable Python types."""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize(item) for item in obj]
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump(mode="json")
        except Exception:
            return {
                k: _serialize(v)
                for k, v in vars(obj).items()
                if not k.startswith("_")
            }
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    return str(obj)


def _responses_api_item_to_message(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Convert a single Responses API input/output item to a chat message dict.

    Handles item types produced by OpenAI's Responses API:
    - ``message`` → ``{"role": ..., "content": ...}``
    - ``function_call`` → ``{"role": "assistant", "content": "", "tool_calls": [...]}``
    - ``function_call_output`` → ``{"role": "tool", "content": ..., "tool_call_id": ...}``
    - ``item_reference`` → skipped (internal bookkeeping)
    """
    item_type = item.get("type", "")

    if item_type == "message":
        # ResponseInputMessageItem or ResponseOutputMessage
        role = item.get("role", "user")
        content_blocks = item.get("content", [])
        if isinstance(content_blocks, str):
            return {"role": role, "content": content_blocks}
        # Extract text from content blocks
        text_parts = []
        for block in content_blocks:
            if isinstance(block, dict):
                bt = block.get("type", "")
                if bt in ("input_text", "output_text", "text"):
                    text_parts.append(block.get("text", ""))
                elif bt == "input_image":
                    text_parts.append("[image]")
                elif bt == "input_file":
                    text_parts.append("[file]")
                else:
                    text_parts.append(block.get("text", str(block)))
            elif isinstance(block, str):
                text_parts.append(block)
        return {"role": role, "content": "\n".join(text_parts)}

    if item_type == "function_call":
        return {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": item.get("call_id", ""),
                "type": "function",
                "function": {
                    "name": item.get("name", ""),
                    "arguments": item.get("arguments", ""),
                },
            }],
        }

    if item_type == "function_call_output":
        return {
            "role": "tool",
            "content": item.get("output", ""),
            "tool_call_id": item.get("call_id", ""),
        }

    # item_reference, reasoning, etc. — skip
    return None


def _format_input_messages(raw_input: Any) -> List[Dict[str, Any]]:
    """Wrap raw input into proper ``[{"role": ..., "content": ...}]`` format."""
    serialized = _serialize(raw_input)
    if serialized is None:
        return []
    # Already a list of message dicts
    if isinstance(serialized, list):
        # Detect Responses API format: any item with a "type" field like
        # "function_call", "function_call_output", "message", etc.
        # The first item may be a plain {role, content} dict (message items
        # lose their "type" during model_dump), so check ALL items.
        has_responses_api_items = any(
            isinstance(item, dict) and "type" in item
            for item in serialized
        )
        if has_responses_api_items:
            messages = []
            for item in serialized:
                if not isinstance(item, dict):
                    continue
                if "type" in item:
                    msg = _responses_api_item_to_message(item)
                    if msg is not None:
                        messages.append(msg)
                elif "role" in item:
                    # Plain message dict (already converted or standard)
                    messages.append(item)
            return messages if messages else serialized
        # Standard chat messages (role/content only, no "type")
        if serialized and isinstance(serialized[0], dict) and "role" in serialized[0]:
            return serialized
        return serialized
    # Plain string → wrap as user message
    if isinstance(serialized, str):
        return [{"role": "user", "content": serialized}]
    # Dict (e.g. function args) → wrap as user message with JSON content
    if isinstance(serialized, dict):
        return [{"role": "user", "content": json.dumps(serialized, default=str)}]
    return [{"role": "user", "content": str(serialized)}]


def _format_output(resp_output: Any) -> Dict[str, Any]:
    """Extract a clean ``{"role": "assistant", "content": ...}`` from Response output.

    The Responses API returns output as a list of items. We extract text content
    and tool calls into a standard completion message format.
    """
    serialized = _serialize(resp_output)
    if not serialized:
        return {"role": "assistant", "content": "", "_is_placeholder": True}

    if isinstance(serialized, str):
        return {"role": "assistant", "content": serialized}

    if isinstance(serialized, dict):
        # Already a message dict
        if "role" in serialized:
            return serialized
        return {"role": "assistant", "content": str(serialized)}

    if isinstance(serialized, list):
        # Responses API format: list of output items
        text_parts = []
        tool_calls = []
        for item in serialized:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type", "")
            if item_type == "message":
                # ResponseOutputMessage — extract text from content blocks
                for block in item.get("content", []):
                    if isinstance(block, dict) and block.get("type") == "output_text":
                        text_parts.append(block.get("text", ""))
            elif item_type == "function_call":
                tool_calls.append({
                    "id": item.get("call_id", ""),
                    "type": "function",
                    "function": {
                        "name": item.get("name", ""),
                        "arguments": item.get("arguments", ""),
                    },
                })
            elif item_type == "output_text":
                # Direct content block
                text_parts.append(item.get("text", ""))

        msg: Dict[str, Any] = {"role": "assistant", "content": "\n".join(text_parts)}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        return msg

    return {"role": "assistant", "content": str(serialized)}


def _parse_ts(ts: str) -> datetime:
    """Parse an ISO-8601 timestamp string to datetime."""
    ts = ts.replace("Z", "+00:00")
    return datetime.fromisoformat(ts)


def _build_base_params(item: SpanImpl) -> RespanTextLogParams:
    """Build a ``RespanTextLogParams`` with the common fields every span shares."""
    started = item.started_at  # ISO-8601 string or None
    ended = item.ended_at

    latency = 0.0
    if started and ended:
        try:
            latency = (_parse_ts(ended) - _parse_ts(started)).total_seconds()
        except Exception:
            pass

    return RespanTextLogParams(
        trace_unique_id=item.trace_id,
        span_unique_id=item.span_id,
        span_parent_id=item.parent_id or item.trace_id,
        start_time=started,
        timestamp=ended,
        latency=latency,
        status_code=400 if item.error else 200,
        status="error" if item.error else "success",
        error_message=str(item.error) if item.error else None,
    )


# ---------------------------------------------------------------------------
# Span-type converters  (mutate *params* in-place)
# ---------------------------------------------------------------------------


def _convert_response(params: RespanTextLogParams, span_data: ResponseSpanData) -> None:
    """ResponseSpanData → has the actual LLM call with model + tokens."""
    params.span_name = "response"
    params.log_type = LOG_TYPE_RESPONSE
    params.input = _format_input_messages(span_data.input)

    resp = span_data.response
    if resp is None:
        return

    params.model = getattr(resp, "model", None) or ""

    # Format output as proper {"role": "assistant", "content": "..."} message
    if hasattr(resp, "output") and resp.output:
        params.output = _format_output(resp.output)

    # token usage — ResponseUsage uses input_tokens/output_tokens
    usage = getattr(resp, "usage", None)
    if usage:
        params.prompt_tokens = getattr(usage, "input_tokens", 0) or 0
        params.completion_tokens = getattr(usage, "output_tokens", 0) or 0


def _convert_function(params: RespanTextLogParams, span_data: FunctionSpanData) -> None:
    params.span_name = span_data.name
    params.log_type = LOG_TYPE_TOOL
    # Tool input = function arguments
    input_str = _serialize(span_data.input) or ""
    if not isinstance(input_str, str):
        input_str = json.dumps(input_str, default=str)
    params.input = [{"role": "tool", "content": input_str}]
    # Tool output = function result
    output_str = _serialize(span_data.output) or ""
    if not isinstance(output_str, str):
        output_str = json.dumps(output_str, default=str)
    params.output = {"role": "tool", "content": output_str}
    params.span_tools = [span_data.name]


def _convert_generation(params: RespanTextLogParams, span_data: GenerationSpanData) -> None:
    params.span_name = "generation"
    params.log_type = LOG_TYPE_GENERATION
    params.model = span_data.model or ""
    params.input = _format_input_messages(span_data.input)
    params.output = _format_output(span_data.output)
    if span_data.usage:
        u = span_data.usage
        params.prompt_tokens = u.get("prompt_tokens") or u.get("input_tokens") or 0
        params.completion_tokens = u.get("completion_tokens") or u.get("output_tokens") or 0


def _convert_agent(params: RespanTextLogParams, span_data: AgentSpanData) -> None:
    params.span_name = span_data.name
    params.log_type = LOG_TYPE_AGENT
    params.span_workflow_name = span_data.name
    meta: Dict[str, str] = {"agent_name": span_data.name}
    if span_data.output_type:
        meta["output_type"] = span_data.output_type
    params.metadata = meta
    if span_data.tools:
        params.span_tools = span_data.tools
    if span_data.handoffs:
        params.span_handoffs = span_data.handoffs


def _convert_handoff(params: RespanTextLogParams, span_data: HandoffSpanData) -> None:
    from_agent = span_data.from_agent or ""
    to_agent = span_data.to_agent or ""
    params.span_name = "handoff"
    params.log_type = LOG_TYPE_HANDOFF
    params.span_handoffs = [f"{from_agent} -> {to_agent}"]
    params.input = from_agent
    params.output = to_agent
    params.metadata = {
        "from_agent": from_agent,
        "to_agent": to_agent,
    }


def _convert_guardrail(params: RespanTextLogParams, span_data: GuardrailSpanData) -> None:
    params.span_name = f"guardrail:{span_data.name}"
    params.log_type = LOG_TYPE_GUARDRAIL
    params.metadata = {
        "guardrail_name": span_data.name,
        "triggered": str(span_data.triggered),
    }


def _convert_custom(params: RespanTextLogParams, span_data: CustomSpanData) -> None:
    params.span_name = span_data.name
    params.log_type = LOG_TYPE_CUSTOM
    params.metadata = {k: str(v) for k, v in (span_data.data or {}).items()}
    # Promote well-known keys
    for key in ("model", "prompt_tokens", "completion_tokens"):
        if key in span_data.data:
            setattr(params, key, span_data.data[key])
    if "input" in span_data.data:
        params.input = span_data.data["input"]
    if "output" in span_data.data:
        params.output = span_data.data["output"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_CONVERTERS = {
    ResponseSpanData: _convert_response,
    FunctionSpanData: _convert_function,
    GenerationSpanData: _convert_generation,
    AgentSpanData: _convert_agent,
    HandoffSpanData: _convert_handoff,
    GuardrailSpanData: _convert_guardrail,
    CustomSpanData: _convert_custom,
}


def convert_to_respan_log(
    item: Union[Trace, Span[Any]],
) -> Optional[Dict[str, Any]]:
    """Convert an OpenAI Agents SDK Trace or Span to a flat Respan log dict.

    The returned dict matches the ``/v1/traces/ingest`` contract:
    ``input``, ``output``, ``prompt_tokens``,
    ``completion_tokens``, ``model``, etc.

    Uses ``RespanTextLogParams`` for schema validation, then serializes
    via ``model_dump(mode="json", exclude_none=True)``.
    """
    if isinstance(item, Trace):
        params = RespanTextLogParams(
            trace_unique_id=item.trace_id,
            span_unique_id=item.trace_id,
            span_name=item.name,
            log_type=LOG_TYPE_AGENT,
        )
        return params.model_dump(mode="json", exclude_none=True)

    if isinstance(item, SpanImpl):
        params = _build_base_params(item)
        converter = _CONVERTERS.get(type(item.span_data))
        if converter is None:
            logger.warning("Unknown span data type: %s", type(item.span_data).__name__)
            return None
        try:
            converter(params, item.span_data)
            return params.model_dump(mode="json", exclude_none=True)
        except Exception:
            logger.exception("Error converting %s", type(item.span_data).__name__)
            return None

    return None
