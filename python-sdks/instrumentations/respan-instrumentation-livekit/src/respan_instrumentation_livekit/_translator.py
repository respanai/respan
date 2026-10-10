"""Translate both released LiveKit event and GenAI attribute shapes."""

from __future__ import annotations

from livekit.agents.telemetry import trace_types
from opentelemetry.semconv._incubating.attributes import gen_ai_attributes as GenAI
from opentelemetry.semconv_ai import SpanAttributes as AI
from respan_sdk.constants.llm_logging import LogMethodChoices
from respan_sdk.constants.span_attributes import RESPAN_LOG_METHOD, RESPAN_LOG_TYPE

from ._constants import (
    LIVEKIT_RESPAN_PROVIDER_NAME_ATTR,
    LIVEKIT_RESPAN_TOOL_DEFINITIONS_ATTR,
)
from ._serialization import data, parse_jsonish, safe_json, text


def _calls(values):
    result = []
    for raw in values if isinstance(values, (list, tuple)) else []:
        call = parse_jsonish(raw)
        if not isinstance(call, dict):
            continue
        fn = call.get("function", {})
        if call.get("type") in {"tool_call", "function_call"}:
            fn = {"name": call.get("name"), "arguments": call.get("arguments")}
        if not isinstance(fn, dict) or not isinstance(fn.get("name"), str):
            continue
        args = fn.get("arguments")
        args = text(args) if isinstance(args, str) else safe_json(args)
        item = {
            "type": "function",
            "function": {"name": text(fn["name"]), "arguments": args},
        }
        cid = call.get("id", call.get("call_id"))
        if isinstance(cid, str):
            item["id"] = text(cid)
        result.append(item)
    return result


def _message(raw):
    if not isinstance(raw, dict):
        return None
    result = {"role": text(raw.get("role")) or "user"}
    content = raw.get("content")
    parts = raw.get("parts")
    calls = _calls(raw.get("tool_calls"))
    if isinstance(parts, list):
        values = []
        raw_calls = []
        for p in parts:
            if not isinstance(p, dict):
                continue
            if p.get("type") in {"tool_call", "function_call"}:
                raw_calls.append(p)
            elif p.get("type") == "tool_call_response":
                result["role"] = "tool"
                result["tool_call_id"] = text(p.get("id"))
                values.append(p.get("response"))
            elif p.get("type") == "text":
                values.append(p.get("content", p.get("text")))
            else:
                values.append(p)
        if values:
            content = values[0] if len(values) == 1 else values
        calls.extend(_calls(raw_calls))
    if content is not None:
        result["content"] = data(content)
    if calls:
        result["tool_calls"] = calls
    if isinstance(raw.get("tool_call_id"), str):
        result["tool_call_id"] = text(raw["tool_call_id"])
    return result


def _publish(attrs, messages, prefix):
    for i, m in enumerate(messages):
        base = f"{prefix}.{i}"
        attrs[base + ".role"] = m["role"]
        if "content" in m:
            attrs[base + ".content"] = (
                m["content"]
                if isinstance(m["content"], str)
                else safe_json(m["content"])
            )
        if m.get("tool_calls"):
            attrs[base + ".tool_calls"] = safe_json(m["tool_calls"])
        if m.get("tool_call_id"):
            attrs[base + ".tool_call_id"] = m["tool_call_id"]


def _events(events):
    roles = {
        trace_types.EVENT_GEN_AI_SYSTEM_MESSAGE: "system",
        trace_types.EVENT_GEN_AI_USER_MESSAGE: "user",
        trace_types.EVENT_GEN_AI_ASSISTANT_MESSAGE: "assistant",
        trace_types.EVENT_GEN_AI_TOOL_MESSAGE: "tool",
    }
    prompt = []
    completion = []
    for event in events:
        a = dict(getattr(event, "attributes", {}) or {})
        name = getattr(event, "name", None)
        if name in roles:
            a.setdefault("role", roles[name])
            prompt.append(a)
        elif name == trace_types.EVENT_GEN_AI_CHOICE:
            a.setdefault("role", "assistant")
            completion.append(a)
    return prompt, completion


def is_livekit_llm_span(span_name, attrs):
    return attrs.get(GenAI.GEN_AI_OPERATION_NAME) == "chat" and span_name in {
        "llm_request",
        "llm_node",
    }


def _count(value):
    return value if type(value) is int and value >= 0 else None


def usage_attributes(source):
    attrs = {}
    source = source if isinstance(source, dict) else {}
    keys = {
        "input": GenAI.GEN_AI_USAGE_INPUT_TOKENS,
        "output": GenAI.GEN_AI_USAGE_OUTPUT_TOKENS,
        "total": AI.LLM_USAGE_TOTAL_TOKENS,
        "cache_read": AI.LLM_USAGE_CACHE_READ_INPUT_TOKENS,
        "cache_creation": AI.LLM_USAGE_CACHE_CREATION_INPUT_TOKENS,
        "reasoning": AI.LLM_USAGE_REASONING_TOKENS,
    }
    for k, v in keys.items():
        count = _count(source.get(k))
        if count is not None:
            attrs[v] = count
    if GenAI.GEN_AI_USAGE_INPUT_TOKENS in attrs:
        attrs[AI.LLM_USAGE_PROMPT_TOKENS] = attrs[GenAI.GEN_AI_USAGE_INPUT_TOKENS]
    if GenAI.GEN_AI_USAGE_OUTPUT_TOKENS in attrs:
        attrs[AI.LLM_USAGE_COMPLETION_TOKENS] = attrs[GenAI.GEN_AI_USAGE_OUTPUT_TOKENS]
    if (
        "total" not in source
        and "input" in source
        and "output" in source
        and _count(source["input"]) is not None
        and _count(source["output"]) is not None
    ):
        attrs[AI.LLM_USAGE_TOTAL_TOKENS] = source["input"] + source["output"]
    return attrs


def build_livekit_llm_attrs(
    *, span_name, attrs, events, capture=True, source_usage=None
):
    provider = attrs.get(LIVEKIT_RESPAN_PROVIDER_NAME_ATTR) or attrs.get(
        GenAI.GEN_AI_PROVIDER_NAME
    )
    result = {
        RESPAN_LOG_METHOD: LogMethodChoices.TRACING_INTEGRATION.value,
        RESPAN_LOG_TYPE: "chat",
        AI.LLM_REQUEST_TYPE: "chat",
        AI.TRACELOOP_ENTITY_NAME: span_name,
        AI.TRACELOOP_ENTITY_PATH: "",
    }
    if isinstance(provider, str):
        result[AI.LLM_SYSTEM] = text(provider)
        result[GenAI.GEN_AI_PROVIDER_NAME] = text(provider)
    if isinstance(attrs.get(GenAI.GEN_AI_REQUEST_MODEL), str):
        result[AI.LLM_REQUEST_MODEL] = text(attrs[GenAI.GEN_AI_REQUEST_MODEL])
    result[AI.LLM_IS_STREAMING] = True
    result.update(usage_attributes(source_usage))
    if not capture:
        return result
    old_in, old_out = _events(events)
    prompts = parse_jsonish(attrs.get(GenAI.GEN_AI_INPUT_MESSAGES))
    outputs = parse_jsonish(attrs.get(GenAI.GEN_AI_OUTPUT_MESSAGES))
    prompts = prompts if isinstance(prompts, list) else old_in
    outputs = outputs if isinstance(outputs, list) else old_out
    instructions = parse_jsonish(attrs.get(GenAI.GEN_AI_SYSTEM_INSTRUCTIONS))
    if isinstance(instructions, list):
        prompts = [{"role": "system", "parts": instructions}, *prompts]
    prompts = [m for r in prompts if (m := _message(r)) is not None]
    outputs = [m for r in outputs if (m := _message(r)) is not None]
    if prompts:
        result[AI.TRACELOOP_ENTITY_INPUT] = safe_json(prompts)
        _publish(result, prompts, AI.LLM_PROMPTS)
    if outputs:
        result[AI.TRACELOOP_ENTITY_OUTPUT] = safe_json(outputs)
        _publish(result, outputs, AI.LLM_COMPLETIONS)
    tools = parse_jsonish(
        attrs.get(
            LIVEKIT_RESPAN_TOOL_DEFINITIONS_ATTR,
            attrs.get(GenAI.GEN_AI_TOOL_DEFINITIONS),
        )
    )
    if isinstance(tools, list) and tools:
        result[AI.LLM_REQUEST_FUNCTIONS] = safe_json(tools, schema=True)
    return result


def normalize_livekit_tools(tools):
    from livekit.agents.llm import ToolContext

    result = ToolContext(tools).parse_function_tools("openai")
    return data(result, schema=True) if isinstance(result, list) else []


def build_native_tool_attrs(attrs, *, capture):
    """Promote the SDK's existing Session function_tool span, without another span."""
    return build_tool_span_attrs(
        tool_name=attrs.get(
            trace_types.ATTR_FUNCTION_TOOL_NAME, attrs.get(GenAI.GEN_AI_TOOL_NAME)
        ),
        call_id=attrs.get(
            trace_types.ATTR_FUNCTION_TOOL_ID, attrs.get(GenAI.GEN_AI_TOOL_CALL_ID)
        ),
        arguments=attrs.get(trace_types.ATTR_FUNCTION_TOOL_ARGS),
        output=parse_jsonish(attrs.get(trace_types.ATTR_FUNCTION_TOOL_OUTPUT)),
        capture=capture,
        capture_output=attrs.get(trace_types.ATTR_FUNCTION_TOOL_IS_ERROR) is not True
        and trace_types.ATTR_FUNCTION_TOOL_OUTPUT in attrs,
    )


def build_tool_span_attrs(
    *, tool_name, arguments, output, call_id=None, capture=True, capture_output=True
):
    result = {
        RESPAN_LOG_TYPE: "tool",
        RESPAN_LOG_METHOD: LogMethodChoices.TRACING_INTEGRATION.value,
        AI.TRACELOOP_ENTITY_NAME: text(tool_name),
        AI.TRACELOOP_ENTITY_PATH: "",
    }
    if isinstance(call_id, str):
        result[GenAI.GEN_AI_TOOL_CALL_ID] = text(call_id)
    if capture:
        result[AI.TRACELOOP_ENTITY_INPUT] = safe_json(
            {"name": tool_name, "arguments": parse_jsonish(arguments)}
        )
        if capture_output:
            result[AI.TRACELOOP_ENTITY_OUTPUT] = safe_json(output)
    return result
