import json
import logging
import os
from typing import Any, Optional

from pydantic_ai.agent import Agent
from pydantic_ai.models.instrumented import InstrumentationSettings
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.semconv_ai import SpanAttributes, TraceloopSpanKindValues
from respan_sdk.constants.llm_logging import (
    LOG_TYPE_AGENT,
    LOG_TYPE_CHAT,
    LOG_TYPE_EMBEDDING,
    LOG_TYPE_RESPONSE,
    LOG_TYPE_SPEECH,
    LOG_TYPE_TASK,
    LOG_TYPE_TOOL,
    LOG_TYPE_TRANSCRIPTION,
    LogMethodChoices,
)
from respan_sdk.respan_types._internal_types import Function, FunctionTool, TextModelResponseFormat
from respan_sdk.respan_types.param_types import RespanTextLogParams
from respan_sdk.respan_types.span_types import RespanSpanAttributes
from respan_sdk.utils.data_processing.id_processing import format_trace_id, format_span_id
from respan_tracing.core.tracer import RespanTracer

from respan_exporter_pydantic_ai.constants import (
    DEFAULT_RESPAN_GATEWAY_BASE_URL,
    ENRICHMENT_STRIP_ATTRS,
    MODEL_NAME_ATTR,
    RESPAN_RESPONSE_FORMAT_ATTR,
    RESPAN_TOOLS_ATTR,
    PYDANTIC_AI_ADD_PROCESSOR_PATCH_MARKER,
    PYDANTIC_AI_AGENT_NAME_ATTR,
    PYDANTIC_AI_ENRICHMENT_MARKER,
    PYDANTIC_AI_INPUT_MESSAGES_ATTR,
    PYDANTIC_AI_LEGACY_AGENT_NAME_ATTR,
    PYDANTIC_AI_LEGACY_TOOL_ARGUMENTS_ATTR,
    PYDANTIC_AI_LEGACY_TOOL_RESULT_ATTR,
    PYDANTIC_AI_OPENAI_HANDLE_REQUEST_PATCH_MARKER,
    PYDANTIC_AI_OPERATION_NAME_ATTR,
    PYDANTIC_AI_OUTPUT_MESSAGES_ATTR,
    PYDANTIC_AI_REQUEST_PARAMETERS_ATTR,
    PYDANTIC_AI_RUNNING_TOOLS_SPAN_NAME,
    PYDANTIC_AI_SYSTEM_ATTR,
    PYDANTIC_AI_TOOL_ARGUMENTS_ATTR,
    PYDANTIC_AI_TOOL_DEFINITIONS_ATTR,
    PYDANTIC_AI_TOOL_NAME_ATTR,
    PYDANTIC_AI_TOOL_RESULT_ATTR,
    PYDANTIC_AI_USAGE_INPUT_TOKENS_ATTR,
    PYDANTIC_AI_USAGE_OUTPUT_TOKENS_ATTR,
)

logger = logging.getLogger(__name__)

_RESPAN_TEXT_LOG_FIELDS = frozenset(RespanTextLogParams.model_fields.keys())
_PYDANTIC_AI_OPERATION_TO_LOG_TYPE = {
    "chat": LOG_TYPE_CHAT,
    "embedding": LOG_TYPE_EMBEDDING,
    "response": LOG_TYPE_RESPONSE,
    "speech": LOG_TYPE_SPEECH,
    "transcription": LOG_TYPE_TRANSCRIPTION,
}


def _safe_json_loads(value: Any) -> Any:
    if not isinstance(value, str):
        return value

    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def _normalize_base_url(base_url: Any) -> str:
    if base_url is None:
        return ""
    return str(base_url).strip().rstrip("/").lower()


def _is_respan_gateway_base_url(base_url: Any) -> bool:
    normalized_base_url = _normalize_base_url(base_url=base_url)
    if not normalized_base_url:
        return False

    respan_base_url = _normalize_base_url(
        base_url=os.getenv("RESPAN_BASE_URL") or DEFAULT_RESPAN_GATEWAY_BASE_URL
    )
    return (
        normalized_base_url == respan_base_url
        or normalized_base_url.startswith(f"{respan_base_url}/")
    )


def _extract_openai_instance_base_url(instance: Any) -> Any:
    client = getattr(instance, "_client", None)
    if client is None:
        client = getattr(instance, "client", None)
    return getattr(client, "base_url", None)


def _get_span_attributes(span: Any) -> dict[str, Any]:
    attributes = getattr(span, "attributes", None)
    if attributes:
        return dict(attributes)

    private_attributes = getattr(span, "_attributes", None)
    if private_attributes:
        return dict(private_attributes)

    return {}


def _extract_span_workflow_name(span: Any) -> Optional[str]:
    attributes = _get_span_attributes(span=span)
    workflow_name = attributes.get(SpanAttributes.TRACELOOP_WORKFLOW_NAME)
    if isinstance(workflow_name, str) and workflow_name:
        return workflow_name

    existing_workflow_name = attributes.get("span_workflow_name")
    if isinstance(existing_workflow_name, str) and existing_workflow_name:
        return existing_workflow_name

    return None


def _build_gateway_trace_extra_body(span: Any) -> dict[str, Any]:
    span_context = span.get_span_context()
    if span_context is None:
        return {}

    extra_body = {
        "trace_unique_id": format_trace_id(span_context.trace_id),
        "span_unique_id": format_span_id(span_context.span_id),
        "span_name": getattr(span, "name", None) or "openai.chat",
        "span_workflow_name": _extract_span_workflow_name(span=span),
    }

    parent = getattr(span, "parent", None)
    parent_span_id = getattr(parent, "span_id", None)
    if parent_span_id:
        extra_body["span_parent_id"] = format_span_id(parent_span_id)

    return {
        key: value
        for key, value in extra_body.items()
        if value is not None and value != ""
    }


def _inject_gateway_trace_extra_body(
    span: Any,
    kwargs: dict[str, Any],
    instance: Any,
) -> None:
    if not _is_respan_gateway_base_url(
        base_url=_extract_openai_instance_base_url(instance=instance)
    ):
        return

    extra_body = kwargs.get("extra_body")
    if extra_body is None:
        patched_extra_body: dict[str, Any] = {}
    elif isinstance(extra_body, dict):
        patched_extra_body = dict(extra_body)
    else:
        return

    for key, value in _build_gateway_trace_extra_body(span=span).items():
        patched_extra_body.setdefault(key, value)

    kwargs["extra_body"] = patched_extra_body


def _extract_request_parameters(attributes: dict[str, Any]) -> Optional[dict[str, Any]]:
    request_parameters = _safe_json_loads(
        value=attributes.get(PYDANTIC_AI_REQUEST_PARAMETERS_ATTR)
    )
    if isinstance(request_parameters, dict):
        return request_parameters
    return None


def _extract_messages(
    attributes: dict[str, Any], attr_name: str
) -> Optional[list[dict[str, Any]]]:
    messages = _safe_json_loads(value=attributes.get(attr_name))
    if isinstance(messages, list):
        return messages
    return None


def _extract_tool_span_value(attributes: dict[str, Any], *attr_names: str) -> Any:
    for attr_name in attr_names:
        value = attributes.get(attr_name)
        if value is not None:
            return value
    return None


def _extract_tool_name_sequence(attributes: dict[str, Any]) -> Optional[list[str]]:
    raw_tools = attributes.get(RESPAN_TOOLS_ATTR)
    if not isinstance(raw_tools, (list, tuple)):
        return None
    if not all(isinstance(tool_name, str) for tool_name in raw_tools):
        return None
    return list(raw_tools)


def _extract_respan_model(attributes: dict[str, Any]) -> Optional[str]:
    for attr_name in (
        SpanAttributes.LLM_RESPONSE_MODEL,
        SpanAttributes.LLM_REQUEST_MODEL,
        MODEL_NAME_ATTR,
    ):
        value = attributes.get(attr_name)
        if isinstance(value, str) and value:
            return value
    return None


def _extract_respan_usage(attributes: dict[str, Any]) -> dict[str, int]:
    usage = {}

    prompt_tokens = attributes.get(PYDANTIC_AI_USAGE_INPUT_TOKENS_ATTR)
    if isinstance(prompt_tokens, int):
        usage["prompt_tokens"] = prompt_tokens

    completion_tokens = attributes.get(PYDANTIC_AI_USAGE_OUTPUT_TOKENS_ATTR)
    if isinstance(completion_tokens, int):
        usage["completion_tokens"] = completion_tokens

    if usage:
        usage["total_request_tokens"] = sum(usage.values())

    return usage


def _normalize_tool_definition(
    tool_definition: dict[str, Any],
) -> Optional[FunctionTool]:
    function_payload = tool_definition.get("function")
    if isinstance(function_payload, dict):
        return FunctionTool.model_validate(tool_definition)

    tool_name = tool_definition.get("name")
    if not tool_name:
        return None

    parameters_schema = tool_definition.get("parameters") or tool_definition.get(
        "parameters_json_schema"
    )
    return FunctionTool(
        type=str(tool_definition.get("type", "function")),
        function=Function(
            name=tool_name,
            description=tool_definition.get("description"),
            parameters=parameters_schema if isinstance(parameters_schema, dict) else None,
            strict=tool_definition.get("strict"),
        ),
    )


def _extract_tools(attributes: dict[str, Any]) -> Optional[list[dict[str, Any]]]:
    tool_definitions = attributes.get(RESPAN_TOOLS_ATTR)
    if not isinstance(tool_definitions, list):
        tool_definitions = _safe_json_loads(value=tool_definitions)

    if not isinstance(tool_definitions, list):
        tool_definitions = _safe_json_loads(
            value=attributes.get(PYDANTIC_AI_TOOL_DEFINITIONS_ATTR)
        )

    if not isinstance(tool_definitions, list):
        request_parameters = _extract_request_parameters(attributes=attributes)
        if not request_parameters:
            return None

        tool_definitions = [
            *(request_parameters.get("function_tools") or []),
            *(request_parameters.get("output_tools") or []),
        ]

    normalized_tools = []
    for tool_definition in tool_definitions:
        if not isinstance(tool_definition, dict):
            continue

        normalized_tool = _normalize_tool_definition(tool_definition=tool_definition)
        if normalized_tool is not None:
            normalized_tools.append(normalized_tool.model_dump(exclude_none=True))

    if normalized_tools:
        return normalized_tools
    return None


def _build_json_schema_response_format(
    output_object: dict[str, Any],
) -> dict[str, Any]:
    response_format = TextModelResponseFormat(type="json_schema")

    output_schema = output_object.get("json_schema")
    if not isinstance(output_schema, dict):
        return response_format.model_dump()

    json_schema_payload = {"schema": output_schema}

    output_name = output_object.get("name")
    if output_name:
        json_schema_payload["name"] = output_name

    output_description = output_object.get("description")
    if output_description:
        json_schema_payload["description"] = output_description

    strict = output_object.get("strict")
    if strict is not None:
        json_schema_payload["strict"] = strict

    response_format.json_schema = json_schema_payload
    return response_format.model_dump()


def _extract_response_format(
    attributes: dict[str, Any],
) -> Optional[dict[str, Any]]:
    existing_response_format = attributes.get(RESPAN_RESPONSE_FORMAT_ATTR)
    if isinstance(existing_response_format, dict):
        return TextModelResponseFormat.model_validate(
            existing_response_format
        ).model_dump()

    parsed_existing_response_format = _safe_json_loads(value=existing_response_format)
    if isinstance(parsed_existing_response_format, dict):
        return TextModelResponseFormat.model_validate(
            parsed_existing_response_format
        ).model_dump()

    request_parameters = _extract_request_parameters(attributes=attributes)
    if not request_parameters:
        return None

    output_mode = request_parameters.get("output_mode")
    if not output_mode:
        return None

    if output_mode == "text":
        return TextModelResponseFormat(type="text").model_dump()

    if output_mode == "image":
        return TextModelResponseFormat(type="image").model_dump()

    if output_mode in {"native", "prompted"}:
        output_object = request_parameters.get("output_object") or {}
        if isinstance(output_object, dict):
            return _build_json_schema_response_format(output_object=output_object)
        return TextModelResponseFormat(type="json_schema").model_dump()

    logger.debug("Unknown Pydantic AI output_mode %r, skipping response_format extraction", output_mode)
    return None


def _extract_log_type(span: ReadableSpan, attributes: dict[str, Any]) -> Optional[str]:
    if isinstance(attributes.get(PYDANTIC_AI_TOOL_NAME_ATTR), str):
        return LOG_TYPE_TOOL

    if isinstance(attributes.get(PYDANTIC_AI_AGENT_NAME_ATTR), str) or isinstance(
        attributes.get(PYDANTIC_AI_LEGACY_AGENT_NAME_ATTR), str
    ):
        return LOG_TYPE_AGENT

    operation_name = attributes.get(PYDANTIC_AI_OPERATION_NAME_ATTR)
    if isinstance(operation_name, str):
        return _PYDANTIC_AI_OPERATION_TO_LOG_TYPE.get(operation_name)

    running_tool_names = _extract_tool_name_sequence(attributes=attributes)
    if span.name == PYDANTIC_AI_RUNNING_TOOLS_SPAN_NAME and running_tool_names:
        return LOG_TYPE_TASK

    return None


def _is_pydantic_ai_span(span: ReadableSpan, attributes: dict[str, Any]) -> bool:
    return (
        bool(attributes.get(PYDANTIC_AI_SYSTEM_ATTR))
        or PYDANTIC_AI_REQUEST_PARAMETERS_ATTR in attributes
        or PYDANTIC_AI_TOOL_DEFINITIONS_ATTR in attributes
        or bool(attributes.get(PYDANTIC_AI_TOOL_NAME_ATTR))
        or bool(attributes.get(PYDANTIC_AI_AGENT_NAME_ATTR))
        or bool(attributes.get(PYDANTIC_AI_LEGACY_AGENT_NAME_ATTR))
        or bool(attributes.get(PYDANTIC_AI_TOOL_ARGUMENTS_ATTR))
        or bool(attributes.get(PYDANTIC_AI_TOOL_RESULT_ATTR))
        or bool(attributes.get(PYDANTIC_AI_LEGACY_TOOL_ARGUMENTS_ATTR))
        or bool(attributes.get(PYDANTIC_AI_LEGACY_TOOL_RESULT_ATTR))
        or (
            span.name == PYDANTIC_AI_RUNNING_TOOLS_SPAN_NAME
            and _extract_tool_name_sequence(attributes=attributes) is not None
        )
    )


def _set_respan_log_field(
    attributes: dict[str, Any], field_name: str, value: Any
) -> None:
    if value is None or field_name not in _RESPAN_TEXT_LOG_FIELDS:
        return
    attributes.setdefault(field_name, value)


def _set_span_field(
    attributes: dict[str, Any], field_name: str, value: Any
) -> None:
    if value is None:
        return
    attributes.setdefault(field_name, value)


def _apply_traceloop_field_mapping(
    span: ReadableSpan,
    attributes: dict[str, Any],
    enriched_attributes: dict[str, Any],
) -> None:
    log_type = _extract_log_type(span=span, attributes=attributes)
    tool_name = attributes.get(PYDANTIC_AI_TOOL_NAME_ATTR)
    tool_input = _extract_tool_span_value(
        attributes,
        PYDANTIC_AI_TOOL_ARGUMENTS_ATTR,
        PYDANTIC_AI_LEGACY_TOOL_ARGUMENTS_ATTR,
    )
    tool_output = _extract_tool_span_value(
        attributes,
        PYDANTIC_AI_TOOL_RESULT_ATTR,
        PYDANTIC_AI_LEGACY_TOOL_RESULT_ATTR,
    )
    agent_name = _extract_tool_span_value(
        attributes,
        PYDANTIC_AI_AGENT_NAME_ATTR,
        PYDANTIC_AI_LEGACY_AGENT_NAME_ATTR,
    )
    running_tool_names = _extract_tool_name_sequence(attributes=attributes)

    if log_type == LOG_TYPE_TOOL and isinstance(tool_name, str):
        _set_span_field(
            attributes=enriched_attributes,
            field_name=SpanAttributes.TRACELOOP_SPAN_KIND,
            value=TraceloopSpanKindValues.TOOL.value,
        )
        _set_span_field(
            attributes=enriched_attributes,
            field_name=SpanAttributes.TRACELOOP_ENTITY_NAME,
            value=tool_name,
        )
        _set_span_field(
            attributes=enriched_attributes,
            field_name=SpanAttributes.TRACELOOP_ENTITY_PATH,
            value=enriched_attributes.get(SpanAttributes.TRACELOOP_ENTITY_PATH) or tool_name,
        )
        _set_span_field(
            attributes=enriched_attributes,
            field_name=SpanAttributes.TRACELOOP_ENTITY_INPUT,
            value=tool_input,
        )
        _set_span_field(
            attributes=enriched_attributes,
            field_name=SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
            value=tool_output,
        )
        return

    if log_type == LOG_TYPE_AGENT and isinstance(agent_name, str):
        _set_span_field(
            attributes=enriched_attributes,
            field_name=SpanAttributes.TRACELOOP_SPAN_KIND,
            value=TraceloopSpanKindValues.AGENT.value,
        )
        _set_span_field(
            attributes=enriched_attributes,
            field_name=SpanAttributes.TRACELOOP_ENTITY_NAME,
            value=agent_name,
        )
        _set_span_field(
            attributes=enriched_attributes,
            field_name=SpanAttributes.TRACELOOP_ENTITY_PATH,
            value=enriched_attributes.get(SpanAttributes.TRACELOOP_ENTITY_PATH) or agent_name,
        )
        return

    if log_type == LOG_TYPE_TASK and span.name == PYDANTIC_AI_RUNNING_TOOLS_SPAN_NAME and running_tool_names:
        task_name = "running_tools"
        _set_span_field(
            attributes=enriched_attributes,
            field_name=SpanAttributes.TRACELOOP_SPAN_KIND,
            value=TraceloopSpanKindValues.TASK.value,
        )
        _set_span_field(
            attributes=enriched_attributes,
            field_name=SpanAttributes.TRACELOOP_ENTITY_NAME,
            value=task_name,
        )
        _set_span_field(
            attributes=enriched_attributes,
            field_name=SpanAttributes.TRACELOOP_ENTITY_PATH,
            value=enriched_attributes.get(SpanAttributes.TRACELOOP_ENTITY_PATH) or task_name,
        )


def _apply_respan_field_mapping(
    span: ReadableSpan,
    attributes: dict[str, Any],
    enriched_attributes: dict[str, Any],
) -> None:
    log_type = _extract_log_type(span=span, attributes=attributes)
    _set_span_field(
        attributes=enriched_attributes,
        field_name=RespanSpanAttributes.LOG_METHOD.value,
        value=LogMethodChoices.TRACING_INTEGRATION.value,
    )
    _set_span_field(
        attributes=enriched_attributes,
        field_name=RespanSpanAttributes.LOG_TYPE.value,
        value=log_type,
    )

    _set_respan_log_field(
        attributes=enriched_attributes,
        field_name="model",
        value=_extract_respan_model(attributes=attributes),
    )

    usage_values = _extract_respan_usage(attributes=attributes)
    for field_name, value in usage_values.items():
        _set_respan_log_field(
            attributes=enriched_attributes,
            field_name=field_name,
            value=value,
        )

    _set_respan_log_field(
        attributes=enriched_attributes,
        field_name="full_request",
        value=_extract_messages(
            attributes=attributes,
            attr_name=PYDANTIC_AI_INPUT_MESSAGES_ATTR,
        ),
    )
    _set_respan_log_field(
        attributes=enriched_attributes,
        field_name="full_response",
        value=_extract_messages(
            attributes=attributes,
            attr_name=PYDANTIC_AI_OUTPUT_MESSAGES_ATTR,
        ),
    )

    tool_name = attributes.get(PYDANTIC_AI_TOOL_NAME_ATTR)
    if isinstance(tool_name, str):
        _set_respan_log_field(
            attributes=enriched_attributes,
            field_name="span_tools",
            value=[tool_name],
        )

    running_tool_names = _extract_tool_name_sequence(attributes=attributes)
    if span.name == PYDANTIC_AI_RUNNING_TOOLS_SPAN_NAME and running_tool_names:
        _set_respan_log_field(
            attributes=enriched_attributes,
            field_name="span_tools",
            value=running_tool_names,
        )

    _set_respan_log_field(
        attributes=enriched_attributes,
        field_name="input",
        value=_extract_tool_span_value(
            attributes,
            PYDANTIC_AI_TOOL_ARGUMENTS_ATTR,
            PYDANTIC_AI_LEGACY_TOOL_ARGUMENTS_ATTR,
        ),
    )
    _set_respan_log_field(
        attributes=enriched_attributes,
        field_name="output",
        value=_extract_tool_span_value(
            attributes,
            PYDANTIC_AI_TOOL_RESULT_ATTR,
            PYDANTIC_AI_LEGACY_TOOL_RESULT_ATTR,
        ),
    )

    agent_name = _extract_tool_span_value(
        attributes,
        PYDANTIC_AI_AGENT_NAME_ATTR,
        PYDANTIC_AI_LEGACY_AGENT_NAME_ATTR,
    )
    if log_type == LOG_TYPE_AGENT and isinstance(agent_name, str):
        _set_respan_log_field(
            attributes=enriched_attributes,
            field_name="span_workflow_name",
            value=agent_name,
        )


def _enrich_pydantic_ai_span(span: ReadableSpan) -> None:
    try:
        attributes = dict(getattr(span, "attributes", {}) or {})
        if not _is_pydantic_ai_span(span=span, attributes=attributes):
            return

        tools = _extract_tools(attributes=attributes)
        response_format = _extract_response_format(attributes=attributes)
        log_type = _extract_log_type(span=span, attributes=attributes)
        if tools is None and response_format is None and log_type is None:
            return

        enriched_attributes = {
            k: v for k, v in attributes.items()
            if k not in ENRICHMENT_STRIP_ATTRS
        }
        _apply_respan_field_mapping(
            span=span,
            attributes=attributes,
            enriched_attributes=enriched_attributes,
        )
        _apply_traceloop_field_mapping(
            span=span,
            attributes=attributes,
            enriched_attributes=enriched_attributes,
        )

        span._attributes = enriched_attributes
    except Exception:
        logger.exception("Failed to enrich Pydantic AI span attributes.")


def _wrap_span_processor(span_processor: Any) -> None:
    if getattr(span_processor, PYDANTIC_AI_ENRICHMENT_MARKER, False):
        return

    original_on_end = span_processor.on_end

    def _wrapped_on_end(span: ReadableSpan) -> None:
        _enrich_pydantic_ai_span(span=span)
        original_on_end(span)

    span_processor.on_end = _wrapped_on_end
    setattr(span_processor, PYDANTIC_AI_ENRICHMENT_MARKER, True)


def _install_respan_gateway_trace_correlation() -> None:
    # Deferred import: opentelemetry-instrumentation-openai is an optional
    # dependency that is only present when the user instruments OpenAI calls.
    try:
        from opentelemetry.instrumentation.openai.shared import chat_wrappers
    except ImportError:
        logger.debug(
            "opentelemetry-instrumentation-openai not installed, "
            "skipping gateway trace correlation"
        )
        return

    if getattr(
        chat_wrappers,
        PYDANTIC_AI_OPENAI_HANDLE_REQUEST_PATCH_MARKER,
        False,
    ):
        return

    original_handle_request = chat_wrappers._handle_request

    async def _wrapped_handle_request(span: Any, kwargs: dict[str, Any], instance: Any) -> Any:
        try:
            _inject_gateway_trace_extra_body(
                span=span,
                kwargs=kwargs,
                instance=instance,
            )
        except Exception:
            logger.exception("Failed to correlate Respan gateway log with active chat span.")

        return await original_handle_request(span, kwargs, instance)

    chat_wrappers._handle_request = _wrapped_handle_request
    setattr(
        chat_wrappers,
        PYDANTIC_AI_OPENAI_HANDLE_REQUEST_PATCH_MARKER,
        True,
    )


def _install_pydantic_ai_span_enrichment(tracer: RespanTracer) -> None:
    tracer_provider = getattr(tracer, "tracer_provider", None)
    if tracer_provider is None:
        return

    if not getattr(tracer_provider, PYDANTIC_AI_ADD_PROCESSOR_PATCH_MARKER, False):
        original_add_span_processor = tracer_provider.add_span_processor

        def _wrapped_add_span_processor(span_processor: Any) -> None:
            _wrap_span_processor(span_processor=span_processor)
            original_add_span_processor(span_processor)

        tracer_provider.add_span_processor = _wrapped_add_span_processor
        setattr(tracer_provider, PYDANTIC_AI_ADD_PROCESSOR_PATCH_MARKER, True)

    active_span_processor = getattr(tracer_provider, "_active_span_processor", None)
    span_processors = getattr(active_span_processor, "_span_processors", ())
    for buffering_processor in span_processors:
        _wrap_span_processor(span_processor=buffering_processor)


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
    
    # tracer_provider is guaranteed to exist here: is_initialized() and is_enabled
    # guards above ensure _setup_tracer_provider() has run. Pydantic AI also accepts
    # None (falls back to global provider), but we always have the explicit one.
    _install_pydantic_ai_span_enrichment(tracer=tracer)
    _install_respan_gateway_trace_correlation()

    settings = InstrumentationSettings(
        tracer_provider=tracer.tracer_provider,
        include_content=include_content,
        include_binary_content=include_binary_content,
        # Version 4 uses the current GenAI semantic conventions, including
        # execute_tool spans with gen_ai.tool.call.* attributes.
        version=4,
    )
    
    if agent is not None:
        agent.instrument = settings
    else:
        Agent.instrument_all(instrument=settings)
