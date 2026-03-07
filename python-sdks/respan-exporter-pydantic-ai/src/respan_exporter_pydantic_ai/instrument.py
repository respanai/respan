import json
import logging
from collections.abc import Iterable, Mapping
from typing import Any, Optional

from opentelemetry.semconv_ai import SpanAttributes
from pydantic_ai.agent import Agent
from pydantic_ai.models.instrumented import InstrumentationSettings
from respan_tracing.core.tracer import RespanTracer

logger = logging.getLogger(__name__)

_REDUNDANT_PYDANTIC_AI_SPAN_ATTRIBUTES = frozenset({
    "gen_ai.input.messages",
    "gen_ai.output.messages",
    "gen_ai.system_instructions",
    "gen_ai.tool.definitions",
    "gen_ai.response.finish_reasons",
    "model_request_parameters",
    "logfire.json_schema",
})


def _serialize_json(value: Any) -> str:
    return json.dumps(value, default=str, separators=(",", ":"))


def _coerce_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _serialize_message_content(parts: list[dict[str, Any]]) -> Optional[str]:
    if not parts:
        return None

    text_parts: list[str] = []
    for part in parts:
        if part.get("type") != "text" or not isinstance(part.get("content"), str):
            return _serialize_json(value=parts)
        text_parts.append(part["content"])

    return "\n".join(text_parts)


def _serialize_tool_result(value: Any) -> str:
    if isinstance(value, str):
        return value
    return _serialize_json(value=value)


def _serialize_tool_arguments(value: Any) -> str:
    if isinstance(value, str):
        return value
    return _serialize_json(value=value)


def _build_openai_tool_calls(parts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tool_calls: list[dict[str, Any]] = []

    for part in parts:
        if part.get("type") != "tool_call":
            continue

        tool_calls.append({
            "id": part.get("id"),
            "type": "function",
            "function": {
                "name": part.get("name"),
                "arguments": _serialize_tool_arguments(
                    value=part.get("arguments", {})
                ),
            },
        })

    return tool_calls


def _build_legacy_message(
    message: dict[str, Any],
) -> list[dict[str, Any]]:
    role = message.get("role")
    parts = [
        _coerce_mapping(value=part)
        for part in message.get("parts", [])
        if isinstance(part, Mapping)
    ]

    if not role:
        return []

    tool_response_parts = [
        part for part in parts if part.get("type") == "tool_call_response"
    ]
    remaining_parts = [
        part for part in parts if part.get("type") != "tool_call_response"
    ]
    legacy_messages: list[dict[str, Any]] = []

    if role == "assistant":
        assistant_message: dict[str, Any] = {"role": "assistant"}
        assistant_content_parts = [
            part for part in remaining_parts if part.get("type") != "tool_call"
        ]
        assistant_content = _serialize_message_content(parts=assistant_content_parts)
        assistant_tool_calls = _build_openai_tool_calls(parts=remaining_parts)

        if assistant_content is not None:
            assistant_message["content"] = assistant_content
        if assistant_tool_calls:
            assistant_message["tool_calls"] = assistant_tool_calls

        finish_reason = message.get("finish_reason")
        if finish_reason is not None:
            assistant_message["finish_reason"] = finish_reason

        if len(assistant_message) > 1:
            legacy_messages.append(assistant_message)
    elif remaining_parts:
        message_content = _serialize_message_content(parts=remaining_parts)
        legacy_message: dict[str, Any] = {"role": role}
        if message_content is not None:
            legacy_message["content"] = message_content
        legacy_messages.append(legacy_message)

    for tool_response_part in tool_response_parts:
        legacy_messages.append({
            "role": "tool",
            "content": _serialize_tool_result(value=tool_response_part.get("result")),
            "tool_call_id": tool_response_part.get("id"),
            "name": tool_response_part.get("name"),
        })

    return legacy_messages


def _build_legacy_messages(messages: list[Any]) -> list[dict[str, Any]]:
    legacy_messages: list[dict[str, Any]] = []

    for message in messages:
        legacy_messages.extend(_build_legacy_message(message=_coerce_mapping(value=message)))

    return legacy_messages


def _build_openai_tools(parameters: Any) -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = []
    function_tools = list(getattr(parameters, "function_tools", []) or [])
    output_tools = list(getattr(parameters, "output_tools", []) or [])

    for tool in [*function_tools, *output_tools]:
        name = getattr(tool, "name", None)
        if not name:
            continue

        function: dict[str, Any] = {"name": name}
        description = getattr(tool, "description", None)
        parameters_json_schema = getattr(tool, "parameters_json_schema", None)
        strict = getattr(tool, "strict", None)

        if description:
            function["description"] = description
        if parameters_json_schema:
            function["parameters"] = parameters_json_schema
        if strict is not None:
            function["strict"] = strict

        tools.append({
            "type": "function",
            "function": function,
        })

    return tools


def _build_legacy_indexed_attributes(
    prefix: str,
    messages: list[dict[str, Any]],
) -> dict[str, Any]:
    attributes: dict[str, Any] = {}

    for message_index, message in enumerate(messages):
        role = message.get("role")
        if role is not None:
            attributes[f"{prefix}.{message_index}.role"] = role

        content = message.get("content")
        if content is not None:
            if not isinstance(content, str):
                content = _serialize_json(value=content)
            attributes[f"{prefix}.{message_index}.content"] = content

        tool_call_id = message.get("tool_call_id")
        if tool_call_id is not None:
            attributes[f"{prefix}.{message_index}.tool_call_id"] = tool_call_id

        name = message.get("name")
        if name is not None:
            attributes[f"{prefix}.{message_index}.name"] = name

        finish_reason = message.get("finish_reason")
        if finish_reason is not None:
            attributes[f"{prefix}.{message_index}.finish_reason"] = finish_reason

    return attributes


def _get_raw_span_attributes(span: Any) -> Optional[dict[str, Any]]:
    bounded_attributes = getattr(span, "_attributes", None)
    raw_attributes = getattr(bounded_attributes, "_dict", None)
    return raw_attributes if isinstance(raw_attributes, dict) else None


def _set_structured_span_attribute(span: Any, key: str, value: Any) -> None:
    if value is None:
        return

    # OTel span attributes reject dict/list-of-dict values, but Respan's OTLP
    # ingest can preserve them when they are present on the span.
    raw_attributes = _get_raw_span_attributes(span=span)
    if raw_attributes is not None:
        raw_attributes[key] = value


def _remove_span_attributes(span: Any, keys: Iterable[str] = ()) -> None:
    raw_attributes = _get_raw_span_attributes(span=span)
    if raw_attributes is None:
        return

    for key in keys:
        raw_attributes.pop(key, None)


def _build_response_format(parameters: Any) -> Optional[dict[str, Any]]:
    output_mode = getattr(parameters, "output_mode", None)
    output_object = getattr(parameters, "output_object", None)

    if output_object is not None and getattr(output_object, "json_schema", None):
        json_schema_payload: dict[str, Any] = {
            "schema": output_object.json_schema,
        }
        if getattr(output_object, "name", None):
            json_schema_payload["name"] = output_object.name
        if getattr(output_object, "description", None):
            json_schema_payload["description"] = output_object.description
        if getattr(output_object, "strict", None) is not None:
            json_schema_payload["strict"] = output_object.strict
        return {
            "type": "json_schema",
            "json_schema": json_schema_payload,
        }

    if output_mode in {"text", "image"}:
        return {"type": output_mode}

    return None


def _extract_response_finish_reason(span: Any) -> Optional[str]:
    raw_attributes = _get_raw_span_attributes(span=span)
    if raw_attributes is None:
        return None

    finish_reasons = raw_attributes.get("gen_ai.response.finish_reasons")
    if isinstance(finish_reasons, (list, tuple)):
        for finish_reason in finish_reasons:
            if isinstance(finish_reason, str) and finish_reason:
                return finish_reason
        return None

    if isinstance(finish_reasons, str) and finish_reasons:
        return finish_reasons

    return None


def _set_legacy_indexed_tool_call_attributes(
    span: Any,
    prefix: str,
    messages: list[dict[str, Any]],
) -> None:
    for message_index, message in enumerate(messages):
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list) or not tool_calls:
            continue

        normalized_tool_calls: list[dict[str, Any]] = []
        for tool_call in tool_calls:
            if not isinstance(tool_call, Mapping):
                continue

            function = tool_call.get("function")
            if not isinstance(function, Mapping):
                continue

            normalized_tool_call: dict[str, Any] = {
                "type": str(tool_call.get("type") or "function"),
                "function": {
                    "name": function.get("name"),
                },
            }

            if tool_call.get("id") is not None:
                normalized_tool_call["id"] = tool_call.get("id")

            arguments = function.get("arguments")
            if arguments is not None:
                normalized_tool_call["function"]["arguments"] = (
                    arguments
                    if isinstance(arguments, str)
                    else _serialize_json(value=arguments)
                )

            normalized_tool_calls.append(normalized_tool_call)

        if normalized_tool_calls:
            _set_structured_span_attribute(
                span=span,
                key=f"{prefix}.{message_index}.tool_calls",
                value=normalized_tool_calls,
            )


def _normalize_running_tools_span_attributes(span: Any) -> None:
    raw_attributes = _get_raw_span_attributes(span=span)
    if not isinstance(raw_attributes, dict) or span.name != "running tools":
        return

    tools = raw_attributes.get("tools")
    if isinstance(tools, (list, tuple)) and all(
        isinstance(tool_name, str) for tool_name in tools
    ):
        raw_attributes["tools"] = [
            {
                "type": "function",
                "function": {"name": tool_name},
            }
            for tool_name in tools
        ]


def _iter_respan_span_processors(tracer_provider: Any):
    active_span_processor = getattr(tracer_provider, "_active_span_processor", None)
    span_processors = getattr(active_span_processor, "_span_processors", ())

    for buffering_processor in span_processors:
        filtering_processor = getattr(buffering_processor, "original_processor", None)
        respan_processor = getattr(filtering_processor, "processor", None)
        if respan_processor is not None:
            yield respan_processor


def _install_span_normalizer(tracer: RespanTracer) -> None:
    normalizer = getattr(
        tracer,
        "_respan_pydantic_ai_span_normalizer",
        _normalize_running_tools_span_attributes,
    )

    if not hasattr(tracer, "_respan_pydantic_ai_span_normalizer"):
        existing_callback = getattr(tracer, "span_postprocess_callback", None)

        if existing_callback is None:
            combined_callback = normalizer
        else:
            def combined_callback(span: Any) -> None:
                existing_callback(span)
                normalizer(span)

        tracer._respan_pydantic_ai_span_normalizer = normalizer
        tracer._respan_pydantic_ai_combined_callback = combined_callback
        tracer.span_postprocess_callback = combined_callback

    combined_callback = tracer._respan_pydantic_ai_combined_callback
    for respan_processor in _iter_respan_span_processors(tracer.tracer_provider):
        respan_processor.span_postprocess_callback = combined_callback

        if getattr(respan_processor, "original_on_end", None) is None:
            respan_processor.original_on_end = respan_processor.processor.on_end
            respan_processor.processor.on_end = respan_processor._wrapped_on_end


class RespanPydanticAIInstrumentationSettings(InstrumentationSettings):
    """Add Respan-compatible attributes on top of Pydantic AI's OTel spans."""

    def handle_messages(  # type: ignore[override]
        self,
        input_messages: list[Any],
        response: Any,
        system: str,
        span: Any,
        parameters: Any = None,
    ) -> None:
        super().handle_messages(
            input_messages=input_messages,
            response=response,
            system=system,
            span=span,
            parameters=parameters,
        )

        if not span.is_recording():
            return

        legacy_prompt_messages = _build_legacy_messages(
            messages=self.messages_to_otel_messages(messages=input_messages)
        )
        legacy_completion_messages = _build_legacy_messages(
            messages=self.messages_to_otel_messages(messages=[response])
        )

        compatibility_attributes: dict[str, Any] = {
            SpanAttributes.LLM_REQUEST_TYPE: "chat",
            **_build_legacy_indexed_attributes(
                prefix=SpanAttributes.LLM_PROMPTS,
                messages=legacy_prompt_messages,
            ),
            **_build_legacy_indexed_attributes(
                prefix=SpanAttributes.LLM_COMPLETIONS,
                messages=legacy_completion_messages,
            ),
        }
        finish_reason = _extract_response_finish_reason(span=span)
        if finish_reason:
            compatibility_attributes[
                f"{SpanAttributes.LLM_COMPLETIONS}.0.finish_reason"
            ] = finish_reason
        span.set_attributes(attributes=compatibility_attributes)
        _set_legacy_indexed_tool_call_attributes(
            span=span,
            prefix=SpanAttributes.LLM_PROMPTS,
            messages=legacy_prompt_messages,
        )
        _set_legacy_indexed_tool_call_attributes(
            span=span,
            prefix=SpanAttributes.LLM_COMPLETIONS,
            messages=legacy_completion_messages,
        )

        response_format = _build_response_format(parameters=parameters)
        if response_format:
            _set_structured_span_attribute(
                span=span,
                key="response_format",
                value=response_format,
            )

        tools = _build_openai_tools(parameters=parameters)
        if tools:
            _set_structured_span_attribute(span=span, key="tools", value=tools)

        for completion_message in legacy_completion_messages:
            tool_calls = completion_message.get("tool_calls")
            if isinstance(tool_calls, list) and tool_calls:
                _set_structured_span_attribute(
                    span=span,
                    key="tool_calls",
                    value=tool_calls,
                )
                break

        _remove_span_attributes(
            span=span,
            keys=_REDUNDANT_PYDANTIC_AI_SPAN_ATTRIBUTES,
        )


def instrument_pydantic_ai(
    agent: Optional[Agent] = None,
    include_content: bool = True,
    include_binary_content: bool = True,
) -> None:
    """
    Instruments Pydantic AI with Respan telemetry via OpenTelemetry.
    
    If an agent is provided, instruments only that agent.
    Otherwise, instruments all Pydantic AI agents globally.
    
    Args:
        agent: Optional Agent to instrument. If None, instruments globally.
        include_content: Whether to include message content in telemetry.
        include_binary_content: Whether to include binary content in telemetry.
    """
    if not RespanTracer.is_initialized():
        logger.warning(
            "Respan telemetry is not initialized. "
            "Please initialize RespanTelemetry before calling instrument_pydantic_ai()."
        )
        return
    
    tracer = RespanTracer()
    
    if not tracer.is_enabled:
        logger.warning("Respan telemetry is disabled.")
        return

    _install_span_normalizer(tracer=tracer)
    
    # tracer_provider is guaranteed to exist here: is_initialized() and is_enabled
    # guards above ensure _setup_tracer_provider() has run. Pydantic AI also accepts
    # None (falls back to global provider), but we always have the explicit one.
    settings = RespanPydanticAIInstrumentationSettings(
        tracer_provider=tracer.tracer_provider,
        include_content=include_content,
        include_binary_content=include_binary_content,
        # We use version 2 by default to support standard OTel semantic conventions
        version=2,
    )
    
    if agent is not None:
        agent.instrument = settings
    else:
        Agent.instrument_all(instrument=settings)
