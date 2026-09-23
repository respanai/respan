"""Translate current OpenAI Agents SDK trace items into canonical OTEL spans."""

from __future__ import annotations

import logging
import math
import os
from collections.abc import Mapping
from typing import Any

from agents.tracing.span_data import (
    AgentSpanData,
    CustomSpanData,
    FunctionSpanData,
    GenerationSpanData,
    GuardrailSpanData,
    HandoffSpanData,
    MCPListToolsSpanData,
    ResponseSpanData,
    SpeechGroupSpanData,
    SpeechSpanData,
    TaskSpanData,
    TranscriptionSpanData,
    TurnSpanData,
)
from agents.tracing.spans import Span, SpanImpl
from agents.tracing.traces import Trace
from opentelemetry import context as context_api
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import (
    GEN_AI_USAGE_INPUT_TOKENS,
    GEN_AI_USAGE_OUTPUT_TOKENS,
)
from opentelemetry.semconv_ai import LLMRequestTypeValues, SpanAttributes
from respan_sdk.constants.llm_logging import (
    LOG_TYPE_AGENT,
    LOG_TYPE_CHAT,
    LOG_TYPE_CUSTOM,
    LOG_TYPE_GUARDRAIL,
    LOG_TYPE_HANDOFF,
    LOG_TYPE_SPEECH,
    LOG_TYPE_TASK,
    LOG_TYPE_TOOL,
    LOG_TYPE_TRANSCRIPTION,
    LOG_TYPE_WORKFLOW,
)
from respan_sdk.constants.span_attributes import (
    RESPAN_INTERNAL_SPAN_NAME_DETAIL,
    RESPAN_INTERNAL_SPAN_NAME_KIND,
    RESPAN_LOG_TYPE,
    RESPAN_METADATA,
    RESPAN_METADATA_AGENT_NAME,
    RESPAN_METADATA_FROM_AGENT,
    RESPAN_METADATA_GUARDRAIL_NAME,
    RESPAN_METADATA_TO_AGENT,
    RESPAN_METADATA_TRIGGERED,
    RESPAN_TRACE_GROUP_ID,
)
from respan_tracing.constants.context_constants import ENABLE_CONTENT_TRACING_KEY
from respan_tracing.utils.span_factory import build_readable_span, inject_span

from respan_instrumentation_openai_agents._serialization import (
    error_status_code,
    flatten_metadata_attributes,
    json_string,
    json_value,
    parse_json_string,
    safe_error_message,
    safe_text,
)
from respan_instrumentation_openai_agents._utils import (
    _format_input_messages,
    _format_output,
    _parse_ts,
)

logger = logging.getLogger(__name__)


def _resolve_timestamps(item: SpanImpl) -> tuple[int | None, int | None]:
    start_ns = end_ns = None
    if item.started_at:
        try:
            start_ns = int(_parse_ts(item.started_at).timestamp() * 1e9)
        except (TypeError, ValueError, OverflowError):
            pass
    if item.ended_at:
        try:
            end_ns = int(_parse_ts(item.ended_at).timestamp() * 1e9)
        except (TypeError, ValueError, OverflowError):
            pass
    return start_ns, end_ns


def _base_attrs(
    *,
    entity_name: str,
    entity_path: str,
    log_type: str,
    metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    normalized_name = safe_text(entity_name, default="unknown")
    normalized_path = safe_text(entity_path)
    attrs: dict[str, Any] = {
        SpanAttributes.TRACELOOP_ENTITY_NAME: normalized_name,
        SpanAttributes.TRACELOOP_ENTITY_PATH: normalized_path,
        RESPAN_LOG_TYPE: log_type,
    }
    if metadata:
        attrs[RESPAN_METADATA] = json_string(metadata)
        for key, value in flatten_metadata_attributes(metadata).items():
            attrs[f"{RESPAN_METADATA}.{key}"] = value
    return attrs


def _item_metadata(
    item: Any, extra_metadata: Mapping[str, Any] | None
) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    trace_metadata = getattr(item, "trace_metadata", None)
    if isinstance(trace_metadata, Mapping):
        metadata.update(trace_metadata)
    span_metadata = getattr(getattr(item, "span_data", None), "metadata", None)
    if isinstance(span_metadata, Mapping):
        metadata.update(span_metadata)
    if extra_metadata:
        metadata.update(extra_metadata)
    return metadata


def _set_message_attrs(
    attrs: dict[str, Any], prefix: str, messages: list[dict[str, Any]]
) -> None:
    for index, message in enumerate(messages[:50]):
        message_prefix = f"{prefix}.{index}"
        role = message.get("role")
        content = message.get("content")
        tool_calls = message.get("tool_calls")
        if role is not None:
            attrs[f"{message_prefix}.role"] = safe_text(
                role, default="unknown", limit=64
            )
        if content is not None:
            attrs[f"{message_prefix}.content"] = (
                safe_text(content, limit=4_000)
                if isinstance(content, str)
                else json_string(content)
            )
        if tool_calls:
            attrs[f"{message_prefix}.tool_calls"] = json_string(tool_calls)


def _set_completion_attrs(
    attrs: dict[str, Any], *, content: str, tool_calls: list[dict[str, Any]]
) -> None:
    prefix = f"{SpanAttributes.LLM_COMPLETIONS}.0"
    attrs[f"{prefix}.role"] = "assistant"
    attrs[f"{prefix}.content"] = safe_text(content, limit=4_000)
    if tool_calls:
        attrs[f"{prefix}.tool_calls"] = json_string(tool_calls)


def _get_field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    try:
        return getattr(value, name, default)
    except Exception:  # noqa: BLE001 - SDK objects may implement hostile descriptors.
        return default


def _tool_definition(tool: Any) -> dict[str, Any] | None:
    dumped = json_value(tool)
    if isinstance(dumped, str):
        return {"type": "function", "function": {"name": dumped}}
    if not isinstance(dumped, Mapping):
        return None
    tool_type = safe_text(dumped.get("type"), default="function", limit=64)
    function = dumped.get("function")
    if isinstance(function, Mapping):
        name = function.get("name")
        if not name:
            return None
        normalized = {"name": safe_text(name, default="unknown")}
        for key in ("description", "parameters"):
            if function.get(key) is not None:
                normalized[key] = function[key]
        return {"type": tool_type, "function": normalized}

    name = dumped.get("name")
    if not name:
        return dict(dumped)
    normalized = {"name": safe_text(name, default="unknown")}
    for key in ("description", "parameters"):
        if dumped.get(key) is not None:
            normalized[key] = dumped[key]
    return {"type": tool_type, "function": normalized}


def _extract_tools(tools: Any) -> list[dict[str, Any]]:
    if not tools:
        return []
    if isinstance(tools, str) or not isinstance(tools, (list, tuple)):
        tools = [tools]
    result = []
    for tool in tools[:50]:
        definition = _tool_definition(tool)
        if definition is not None:
            result.append(definition)
    return result


def _normalize_tool_call(value: Any) -> dict[str, Any] | None:
    dumped = json_value(value)
    if not isinstance(dumped, Mapping):
        return None
    function = dumped.get("function")
    if isinstance(function, Mapping):
        name = function.get("name")
        arguments = function.get("arguments", {})
    else:
        item_type = dumped.get("type", "")
        name = dumped.get("name") or (
            item_type if item_type.endswith("_call") else None
        )
        hosted_arguments = {
            key: value
            for key, value in dumped.items()
            if key
            not in {
                "id",
                "call_id",
                "type",
                "name",
                "status",
                "result",
                "output",
                "outputs",
            }
        }
        arguments = dumped.get(
            "arguments", dumped.get("action", dumped.get("operation", hosted_arguments))
        )
    if not name:
        return None
    normalized: dict[str, Any] = {
        "type": "function",
        "function": {
            "name": safe_text(name, default="unknown"),
            "arguments": (
                arguments if isinstance(arguments, str) else json_string(arguments)
            ),
        },
    }
    for output_key in ("result", "output", "outputs"):
        if output_key in dumped:
            normalized["output"] = dumped[output_key]
            break
    call_id = dumped.get("call_id") or dumped.get("id")
    if call_id:
        normalized["id"] = safe_text(call_id, limit=256)
    return normalized


def _extract_tool_calls(output: Any) -> list[dict[str, Any]]:
    dumped = json_value(output)
    items = dumped if isinstance(dumped, list) else [dumped]
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in items[:50]:
        if not isinstance(item, Mapping):
            continue
        candidates: list[Any] = []
        if str(item.get("type", "")).endswith("_call"):
            candidates.append(item)
        nested = item.get("tool_calls")
        if isinstance(nested, list):
            candidates.extend(nested)
        for candidate in candidates:
            normalized = _normalize_tool_call(candidate)
            if normalized is None:
                continue
            signature = json_string(normalized)
            if signature in seen:
                continue
            seen.add(signature)
            result.append(normalized)
    return result


def _usage_value(usage: Any, *names: str) -> int | None:
    for name in names:
        value = _get_field(usage, name)
        if isinstance(value, bool):
            continue
        if isinstance(value, int) and value >= 0:
            return int(value)
        if isinstance(value, float) and math.isfinite(value) and value >= 0:
            return int(value)
    return None


def _set_usage(attrs: dict[str, Any], usage: Any) -> None:
    if not usage:
        return
    input_tokens = _usage_value(usage, "input_tokens", "prompt_tokens")
    output_tokens = _usage_value(usage, "output_tokens", "completion_tokens")
    total_tokens = _usage_value(usage, "total_tokens")
    if input_tokens is not None:
        attrs[GEN_AI_USAGE_INPUT_TOKENS] = input_tokens
        attrs[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] = input_tokens
    if output_tokens is not None:
        attrs[GEN_AI_USAGE_OUTPUT_TOKENS] = output_tokens
        attrs[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS] = output_tokens
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens
    if total_tokens is not None:
        attrs[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] = total_tokens


def _agent_tools(agent_context: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    return _extract_tools(agent_context.get("tools")) if agent_context else []


def _is_streaming(
    span_data: Any, explicit: bool, agent_context: Mapping[str, Any] | None
) -> bool:
    if explicit:
        return True
    model_config = _get_field(span_data, "model_config")
    if isinstance(model_config, Mapping) and model_config.get("stream") is True:
        return True
    return bool(agent_context and agent_context.get("is_streaming"))


def _apply_content_privacy(attributes: dict[str, Any]) -> None:
    enabled = os.getenv("TRACELOOP_TRACE_CONTENT", "true").strip().lower()
    if (
        enabled not in {"0", "false", "no", "off"}
        and context_api.get_value(ENABLE_CONTENT_TRACING_KEY) is not False
    ):
        return
    for key in tuple(attributes):
        if key in {
            SpanAttributes.TRACELOOP_ENTITY_INPUT,
            SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
            SpanAttributes.LLM_REQUEST_FUNCTIONS,
        } or key.startswith(
            (f"{SpanAttributes.LLM_PROMPTS}.", f"{SpanAttributes.LLM_COMPLETIONS}.")
        ):
            attributes.pop(key, None)


def _build_item_span(
    item: SpanImpl,
    *,
    name: str,
    attributes: dict[str, Any],
) -> Any:
    start_ns, end_ns = _resolve_timestamps(item)
    error = item.error
    message = safe_error_message(error)
    status_code = error_status_code(error) if error else 200
    if message:
        attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = json_string(
            {"error": {"message": message, "status_code": status_code}}
        )
    _apply_content_privacy(attributes)
    return build_readable_span(
        name=safe_text(name, default="span"),
        trace_id=item.trace_id,
        span_id=item.span_id,
        parent_id=item.parent_id or item.trace_id,
        start_time_ns=start_ns,
        end_time_ns=end_ns,
        attributes=attributes,
        status_code=status_code,
        error_message=message,
    )


def emit_trace(
    trace_obj: Trace,
    *,
    extra_metadata: Mapping[str, Any] | None = None,
    start_ns: int | None = None,
    end_ns: int | None = None,
) -> None:
    name = safe_text(trace_obj.name, default="trace")
    metadata: dict[str, Any] = {}
    trace_metadata = getattr(trace_obj, "metadata", None)
    if isinstance(trace_metadata, Mapping):
        metadata.update(trace_metadata)
    if extra_metadata:
        metadata.update(extra_metadata)
    attrs = _base_attrs(
        entity_name=name, entity_path="", log_type=LOG_TYPE_WORKFLOW, metadata=metadata
    )
    attrs[SpanAttributes.TRACELOOP_WORKFLOW_NAME] = name
    attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = json_string({"workflow_name": name})
    group_id = getattr(trace_obj, "group_id", None)
    if group_id:
        attrs[RESPAN_TRACE_GROUP_ID] = safe_text(group_id, limit=512)
    _apply_content_privacy(attrs)
    inject_span(
        build_readable_span(
            name=f"{name}.workflow",
            trace_id=trace_obj.trace_id,
            span_id=trace_obj.trace_id,
            start_time_ns=start_ns,
            end_time_ns=end_ns,
            attributes=attrs,
        )
    )


def emit_agent(
    item: SpanImpl,
    span_data: AgentSpanData,
    *,
    extra_metadata: Mapping[str, Any] | None = None,
    **_: Any,
) -> None:
    name = safe_text(span_data.name, default="agent")
    attrs = _base_attrs(
        entity_name=name,
        entity_path=name,
        log_type=LOG_TYPE_AGENT,
        metadata=_item_metadata(item, extra_metadata),
    )
    attrs[SpanAttributes.TRACELOOP_WORKFLOW_NAME] = name
    attrs[RESPAN_METADATA_AGENT_NAME] = name
    attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = json_string(
        {
            "handoffs": span_data.handoffs or [],
            "name": name,
            "output_type": span_data.output_type,
            "tools": span_data.tools or [],
        }
    )
    inject_span(_build_item_span(item, name=f"{name}.agent", attributes=attrs))


def _llm_attrs(
    item: SpanImpl,
    *,
    entity_name: str,
    extra_metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    attrs = _base_attrs(
        entity_name=entity_name,
        entity_path=entity_name,
        log_type=LOG_TYPE_CHAT,
        metadata=_item_metadata(item, extra_metadata),
    )
    attrs[SpanAttributes.LLM_REQUEST_TYPE] = LLMRequestTypeValues.CHAT.value
    attrs[SpanAttributes.LLM_SYSTEM] = "openai"
    return attrs


def _set_llm_content(
    attrs: dict[str, Any], *, input_value: Any, output_value: Any
) -> None:
    messages = _format_input_messages(input_value)
    if messages:
        attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = json_string(messages)
        _set_message_attrs(attrs, SpanAttributes.LLM_PROMPTS, messages)
    tool_calls = _extract_tool_calls(output_value)
    output = _format_output(output_value)
    if output_value is not None:
        attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = json_string(
            {"content": output, "tool_calls": tool_calls}
        )
        _set_completion_attrs(attrs, content=output, tool_calls=tool_calls)


def emit_response(
    item: SpanImpl,
    span_data: ResponseSpanData,
    *,
    extra_metadata: Mapping[str, Any] | None = None,
    agent_context: Mapping[str, Any] | None = None,
    is_streaming: bool = False,
) -> None:
    attrs = _llm_attrs(item, entity_name="response", extra_metadata=extra_metadata)
    response = span_data.response
    output = _get_field(response, "output") if response is not None else None
    _set_llm_content(attrs, input_value=span_data.input, output_value=output)
    model = _get_field(response, "model")
    if model:
        attrs[SpanAttributes.LLM_REQUEST_MODEL] = safe_text(model)
        attrs[SpanAttributes.LLM_RESPONSE_MODEL] = safe_text(model)
    tools = _extract_tools(_get_field(response, "tools")) or _agent_tools(agent_context)
    if tools:
        attrs[SpanAttributes.LLM_REQUEST_FUNCTIONS] = json_string(tools)
    _set_usage(attrs, _get_field(span_data, "usage") or _get_field(response, "usage"))
    attrs[SpanAttributes.LLM_IS_STREAMING] = _is_streaming(
        span_data, is_streaming, agent_context
    )
    inject_span(_build_item_span(item, name="openai.chat", attributes=attrs))


def emit_generation(
    item: SpanImpl,
    span_data: GenerationSpanData,
    *,
    extra_metadata: Mapping[str, Any] | None = None,
    agent_context: Mapping[str, Any] | None = None,
    is_streaming: bool = False,
) -> None:
    attrs = _llm_attrs(item, entity_name="generation", extra_metadata=extra_metadata)
    if span_data.model:
        attrs[SpanAttributes.LLM_REQUEST_MODEL] = safe_text(span_data.model)
        attrs[SpanAttributes.LLM_RESPONSE_MODEL] = safe_text(span_data.model)
    # Streaming Chat Completions reports a complete Responses object inside
    # GenerationSpanData.output. Unwrap it before bounded serialization, or
    # tool calls disappear and the assistant answer becomes response JSON.
    output = span_data.output
    if isinstance(output, (list, tuple)):
        output = [
            child
            for entry in output
            for child in (
                _get_field(entry, "output")
                if isinstance(_get_field(entry, "output"), (list, tuple))
                else [entry]
            )
        ]
    _set_llm_content(attrs, input_value=span_data.input, output_value=output)
    tools = _agent_tools(agent_context)
    if tools:
        attrs[SpanAttributes.LLM_REQUEST_FUNCTIONS] = json_string(tools)
    _set_usage(attrs, span_data.usage)
    attrs[SpanAttributes.LLM_IS_STREAMING] = _is_streaming(
        span_data, is_streaming, agent_context
    )
    inject_span(_build_item_span(item, name="openai.chat", attributes=attrs))


def emit_function(
    item: SpanImpl,
    span_data: FunctionSpanData,
    *,
    extra_metadata: Mapping[str, Any] | None = None,
    **_: Any,
) -> None:
    name = safe_text(span_data.name, default="function")
    attrs = _base_attrs(
        entity_name=name,
        entity_path=name,
        log_type=LOG_TYPE_TOOL,
        metadata=_item_metadata(item, extra_metadata),
    )
    attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = json_string(
        {"arguments": parse_json_string(span_data.input), "name": name}
    )
    attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = json_string(span_data.output)
    inject_span(_build_item_span(item, name=f"{name}.tool", attributes=attrs))


def emit_handoff(
    item: SpanImpl,
    span_data: HandoffSpanData,
    *,
    extra_metadata: Mapping[str, Any] | None = None,
    **_: Any,
) -> None:
    from_agent = safe_text(span_data.from_agent, default="unknown")
    to_agent = safe_text(span_data.to_agent, default="unknown")
    attrs = _base_attrs(
        entity_name="handoff",
        entity_path="handoff",
        log_type=LOG_TYPE_HANDOFF,
        metadata=_item_metadata(item, extra_metadata),
    )
    attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = json_string(
        {"from_agent": from_agent}
    )
    attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = json_string({"to_agent": to_agent})
    attrs[RESPAN_METADATA_FROM_AGENT] = from_agent
    attrs[RESPAN_METADATA_TO_AGENT] = to_agent
    attrs[RESPAN_INTERNAL_SPAN_NAME_KIND] = "handoff"
    attrs[RESPAN_INTERNAL_SPAN_NAME_DETAIL] = f"{from_agent}_to_{to_agent}"
    inject_span(_build_item_span(item, name="handoff.task", attributes=attrs))


def emit_guardrail(
    item: SpanImpl,
    span_data: GuardrailSpanData,
    *,
    extra_metadata: Mapping[str, Any] | None = None,
    **_: Any,
) -> None:
    name = safe_text(span_data.name, default="guardrail")
    attrs = _base_attrs(
        entity_name=name,
        entity_path=name,
        log_type=LOG_TYPE_GUARDRAIL,
        metadata=_item_metadata(item, extra_metadata),
    )
    attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = json_string({"name": name})
    attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = json_string(
        {"triggered": bool(span_data.triggered)}
    )
    attrs[RESPAN_METADATA_GUARDRAIL_NAME] = name
    attrs[RESPAN_METADATA_TRIGGERED] = bool(span_data.triggered)
    inject_span(_build_item_span(item, name=f"{name}.task", attributes=attrs))


def emit_custom(
    item: SpanImpl,
    span_data: CustomSpanData,
    *,
    extra_metadata: Mapping[str, Any] | None = None,
    **_: Any,
) -> None:
    name = safe_text(span_data.name, default="custom")
    metadata = _item_metadata(item, extra_metadata)
    attrs = _base_attrs(
        entity_name=name,
        entity_path=name,
        log_type=LOG_TYPE_CUSTOM,
        metadata=metadata,
    )
    data = span_data.data or {}
    if data.get("model"):
        attrs[SpanAttributes.LLM_REQUEST_MODEL] = safe_text(data["model"])
    _set_usage(attrs, data)
    if "input" in data:
        attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = json_string(data["input"])
    if "output" in data:
        attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = json_string(data["output"])
    excluded = {
        "completion_tokens",
        "input",
        "input_tokens",
        "model",
        "output",
        "output_tokens",
        "prompt_tokens",
        "total_tokens",
    }
    remaining = {key: value for key, value in data.items() if key not in excluded}
    if remaining:
        attrs[RESPAN_METADATA] = json_string({**metadata, **remaining})
    inject_span(_build_item_span(item, name=f"{name}.task", attributes=attrs))


def _emit_structural(
    item: SpanImpl,
    span_data: Any,
    *,
    name: str,
    log_type: str,
    input_value: Mapping[str, Any],
    extra_metadata: Mapping[str, Any] | None,
) -> None:
    attrs = _base_attrs(
        entity_name=name,
        entity_path=name,
        log_type=log_type,
        metadata=_item_metadata(item, extra_metadata),
    )
    attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = json_string(input_value)
    agent_name = getattr(span_data, "agent_name", None)
    if agent_name:
        attrs[RESPAN_METADATA_AGENT_NAME] = safe_text(agent_name)
    usage = getattr(span_data, "usage", None)
    if usage is not None:
        # Task/turn usage is a rollup of child model calls, not another charge.
        metadata = _item_metadata(item, extra_metadata)
        metadata["openai_agents.usage"] = usage
        attrs[RESPAN_METADATA] = json_string(metadata)
    inject_span(_build_item_span(item, name=f"{name}.task", attributes=attrs))


def emit_task(
    item: SpanImpl,
    span_data: TaskSpanData,
    *,
    extra_metadata: Mapping[str, Any] | None = None,
    **_: Any,
) -> None:
    name = safe_text(span_data.name, default="agent-run")
    _emit_structural(
        item,
        span_data,
        name=name,
        log_type=LOG_TYPE_WORKFLOW,
        input_value={"name": name},
        extra_metadata=extra_metadata,
    )


def emit_turn(
    item: SpanImpl,
    span_data: TurnSpanData,
    *,
    extra_metadata: Mapping[str, Any] | None = None,
    **_: Any,
) -> None:
    turn = safe_text(span_data.turn, default="unknown", limit=64)
    name = f"turn-{turn}"
    _emit_structural(
        item,
        span_data,
        name=name,
        log_type=LOG_TYPE_TASK,
        input_value={"agent_name": span_data.agent_name, "turn": span_data.turn},
        extra_metadata=extra_metadata,
    )


def emit_mcp_tools(
    item: SpanImpl,
    span_data: MCPListToolsSpanData,
    *,
    extra_metadata: Mapping[str, Any] | None = None,
    **_: Any,
) -> None:
    server = safe_text(span_data.server, default="mcp")
    name = f"{server}.list_tools"
    attrs = _base_attrs(
        entity_name=name,
        entity_path=name,
        log_type=LOG_TYPE_TOOL,
        metadata=_item_metadata(item, extra_metadata),
    )
    attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = json_string(
        {"name": name, "arguments": {"server": server}}
    )
    attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = json_string(span_data.result or [])
    inject_span(_build_item_span(item, name=f"{name}.tool", attributes=attrs))


def emit_audio(
    item: SpanImpl,
    span_data: Any,
    *,
    extra_metadata: Mapping[str, Any] | None = None,
    **_: Any,
) -> None:
    transcription = isinstance(span_data, TranscriptionSpanData)
    group = isinstance(span_data, SpeechGroupSpanData)
    name = "speech_group" if group else "transcription" if transcription else "speech"
    log_type = (
        LOG_TYPE_TASK
        if group
        else LOG_TYPE_TRANSCRIPTION
        if transcription
        else LOG_TYPE_SPEECH
    )
    attrs = _base_attrs(
        entity_name=name,
        entity_path=name,
        log_type=log_type,
        metadata=_item_metadata(item, extra_metadata),
    )
    input_value = span_data.input
    if transcription:
        input_value = {"data": input_value, "format": span_data.input_format}
    if input_value is not None:
        attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = json_string(input_value)
    output = getattr(span_data, "output", None)
    if output is not None:
        if isinstance(span_data, SpeechSpanData):
            output = {"data": output, "format": span_data.output_format}
        attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = json_string(output)
    model = getattr(span_data, "model", None)
    if model:
        attrs[SpanAttributes.LLM_REQUEST_MODEL] = safe_text(model)
    if not group:
        attrs[RESPAN_INTERNAL_SPAN_NAME_KIND] = (
            "transcribe" if transcription else "speech"
        )
    inject_span(_build_item_span(item, name=name, attributes=attrs))


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
    MCPListToolsSpanData: emit_mcp_tools,
    TranscriptionSpanData: emit_audio,
    SpeechSpanData: emit_audio,
    SpeechGroupSpanData: emit_audio,
}


def emit_sdk_item(
    item: Trace | Span[Any],
    *,
    extra_metadata: Mapping[str, Any] | None = None,
    agent_context: Mapping[str, Any] | None = None,
    is_streaming: bool = False,
    trace_start_ns: int | None = None,
    trace_end_ns: int | None = None,
) -> bool:
    """Convert one completed SDK item and inject it into the OTEL pipeline."""
    if isinstance(item, Trace):
        emit_trace(
            item,
            extra_metadata=extra_metadata,
            start_ns=trace_start_ns,
            end_ns=trace_end_ns,
        )
        return True
    if not isinstance(item, SpanImpl):
        logger.debug("Skipping unsupported item type: %s", type(item).__name__)
        return False
    emitter = _EMITTERS.get(type(item.span_data))
    if emitter is None:
        logger.warning("Unknown span data type: %s", type(item.span_data).__name__)
        return False
    try:
        emitter(
            item,
            item.span_data,
            extra_metadata=extra_metadata,
            agent_context=agent_context,
            is_streaming=is_streaming,
        )
    except Exception:
        logger.exception("Error emitting %s", type(item.span_data).__name__)
        return False
    return True
