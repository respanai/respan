"""Convert OpenAI Agents SDK Trace/Span objects to Respan log dicts.

Produces flat dicts matching the backend ``/v1/traces/ingest`` contract:

- ``input``  → list of messages or content
- ``output`` → completion message dict
- ``prompt_tokens`` / ``completion_tokens`` at top level
- ``model``, ``log_type``, ``span_name``, tracing IDs, etc.
"""

import logging
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
        import json
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


def _base_span_dict(item: SpanImpl) -> Dict[str, Any]:
    """Build the common fields every span shares."""
    started = item.started_at  # ISO-8601 string or None
    ended = item.ended_at

    latency = 0.0
    if started and ended:
        from datetime import datetime, timezone

        def _parse_ts(ts: str) -> datetime:
            # Handle both 'Z' suffix and '+00:00'
            ts = ts.replace("Z", "+00:00")
            return datetime.fromisoformat(ts)

        try:
            latency = (_parse_ts(ended) - _parse_ts(started)).total_seconds()
        except Exception:
            pass

    d: Dict[str, Any] = {
        "trace_unique_id": item.trace_id,
        "span_unique_id": item.span_id,
        "span_parent_id": item.parent_id or item.trace_id,
        "start_time": started,
        "timestamp": ended,
        "latency": latency,
        "status_code": 400 if item.error else 200,
        "status": "error" if item.error else "success",
    }
    if item.error:
        d["error_message"] = str(item.error)
    return d


# ---------------------------------------------------------------------------
# Span-type converters  (mutate *d* in-place)
# ---------------------------------------------------------------------------


def _convert_response(d: Dict[str, Any], span_data: ResponseSpanData) -> None:
    """ResponseSpanData → has the actual LLM call with model + tokens."""
    d["span_name"] = "response"
    d["log_type"] = "response"
    d["input"] = _format_input_messages(span_data.input)

    resp = span_data.response
    if resp is None:
        return

    d["model"] = getattr(resp, "model", None) or ""

    # Format output as proper {"role": "assistant", "content": "..."} message
    if hasattr(resp, "output") and resp.output:
        d["output"] = _format_output(resp.output)

    # token usage — ResponseUsage uses input_tokens/output_tokens
    usage = getattr(resp, "usage", None)
    if usage:
        d["prompt_tokens"] = getattr(usage, "input_tokens", 0) or 0
        d["completion_tokens"] = getattr(usage, "output_tokens", 0) or 0


def _convert_function(d: Dict[str, Any], span_data: FunctionSpanData) -> None:
    d["span_name"] = span_data.name
    d["log_type"] = "tool"
    # Tool input = function arguments
    input_str = _serialize(span_data.input) or ""
    if not isinstance(input_str, str):
        import json
        input_str = json.dumps(input_str, default=str)
    d["input"] = [{"role": "tool", "content": input_str}]
    # Tool output = function result
    output_str = _serialize(span_data.output) or ""
    if not isinstance(output_str, str):
        import json
        output_str = json.dumps(output_str, default=str)
    d["output"] = {"role": "tool", "content": output_str}
    d["span_tools"] = [span_data.name]


def _convert_generation(d: Dict[str, Any], span_data: GenerationSpanData) -> None:
    d["span_name"] = "generation"
    d["log_type"] = "generation"
    d["model"] = span_data.model or ""
    d["input"] = _format_input_messages(span_data.input)
    d["output"] = _format_output(span_data.output)
    if span_data.usage:
        u = span_data.usage
        d["prompt_tokens"] = u.get("prompt_tokens") or u.get("input_tokens") or 0
        d["completion_tokens"] = u.get("completion_tokens") or u.get("output_tokens") or 0


def _convert_agent(d: Dict[str, Any], span_data: AgentSpanData) -> None:
    d["span_name"] = span_data.name
    d["log_type"] = "agent"
    d["span_workflow_name"] = span_data.name
    meta: Dict[str, str] = {"agent_name": span_data.name}
    if span_data.output_type:
        meta["output_type"] = span_data.output_type
    d["metadata"] = meta
    if span_data.tools:
        d["span_tools"] = span_data.tools
    if span_data.handoffs:
        d["span_handoffs"] = span_data.handoffs


def _convert_handoff(d: Dict[str, Any], span_data: HandoffSpanData) -> None:
    from_agent = span_data.from_agent or ""
    to_agent = span_data.to_agent or ""
    d["span_name"] = "handoff"
    d["log_type"] = "handoff"
    d["span_handoffs"] = [f"{from_agent} -> {to_agent}"]
    d["input"] = from_agent
    d["output"] = to_agent
    d["metadata"] = {
        "from_agent": from_agent,
        "to_agent": to_agent,
    }


def _convert_guardrail(d: Dict[str, Any], span_data: GuardrailSpanData) -> None:
    d["span_name"] = f"guardrail:{span_data.name}"
    d["log_type"] = "guardrail"
    d["metadata"] = {
        "guardrail_name": span_data.name,
        "triggered": str(span_data.triggered),
    }


def _convert_custom(d: Dict[str, Any], span_data: CustomSpanData) -> None:
    d["span_name"] = span_data.name
    d["log_type"] = "custom"
    d["metadata"] = {k: str(v) for k, v in (span_data.data or {}).items()}
    # Promote well-known keys
    for key in ("model", "prompt_tokens", "completion_tokens"):
        if key in span_data.data:
            d[key] = span_data.data[key]
    if "input" in span_data.data:
        d["input"] = span_data.data["input"]
    if "output" in span_data.data:
        d["output"] = span_data.data["output"]


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
    """
    if isinstance(item, Trace):
        return {
            "trace_unique_id": item.trace_id,
            "span_unique_id": item.trace_id,
            "span_name": item.name,
            "log_type": "agent",
        }

    if isinstance(item, SpanImpl):
        d = _base_span_dict(item)
        converter = _CONVERTERS.get(type(item.span_data))
        if converter is None:
            logger.warning("Unknown span data type: %s", type(item.span_data).__name__)
            return None
        try:
            converter(d, item.span_data)
            return d
        except Exception:
            logger.exception("Error converting %s", type(item.span_data).__name__)
            return None

    return None
