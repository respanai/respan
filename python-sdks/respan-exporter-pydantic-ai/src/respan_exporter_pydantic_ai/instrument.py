import json
import logging
from typing import Any, Optional

from pydantic_ai.agent import Agent
from pydantic_ai.models.instrumented import InstrumentationSettings
from opentelemetry.sdk.trace import ReadableSpan
from respan_sdk.respan_types._internal_types import Function, FunctionTool, TextModelResponseFormat
from respan_tracing.core.tracer import RespanTracer

logger = logging.getLogger(__name__)

PYDANTIC_AI_REQUEST_PARAMETERS_ATTR = "model_request_parameters"
PYDANTIC_AI_TOOL_DEFINITIONS_ATTR = "gen_ai.tool.definitions"
_PYDANTIC_AI_ENRICHMENT_MARKER = "_respan_pydantic_ai_enrichment_installed"
_PYDANTIC_AI_ADD_PROCESSOR_PATCH_MARKER = (
    "_respan_pydantic_ai_add_span_processor_patched"
)


def _safe_json_loads(value: Any) -> Any:
    if not isinstance(value, str):
        return value

    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def _extract_request_parameters(attributes: dict[str, Any]) -> Optional[dict[str, Any]]:
    request_parameters = _safe_json_loads(
        value=attributes.get(PYDANTIC_AI_REQUEST_PARAMETERS_ATTR)
    )
    if isinstance(request_parameters, dict):
        return request_parameters
    return None


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
    tool_definitions = attributes.get("tools")
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
    existing_response_format = attributes.get("response_format")
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

    return TextModelResponseFormat(type=str(output_mode)).model_dump()


def _is_pydantic_ai_span(attributes: dict[str, Any]) -> bool:
    return bool(attributes.get("gen_ai.system")) and (
        PYDANTIC_AI_REQUEST_PARAMETERS_ATTR in attributes
        or PYDANTIC_AI_TOOL_DEFINITIONS_ATTR in attributes
    )


def _enrich_pydantic_ai_span(span: ReadableSpan) -> None:
    try:
        attributes = dict(getattr(span, "attributes", {}) or {})
        if not _is_pydantic_ai_span(attributes=attributes):
            return

        tools = _extract_tools(attributes=attributes)
        response_format = _extract_response_format(attributes=attributes)
        if tools is None and response_format is None:
            return

        enriched_attributes = dict(attributes)
        if tools is not None:
            enriched_attributes["tools"] = tools
        if response_format is not None:
            enriched_attributes["response_format"] = response_format

        span._attributes = enriched_attributes
    except Exception:
        logger.exception("Failed to enrich Pydantic AI span attributes.")


def _wrap_span_processor(span_processor: Any) -> None:
    if getattr(span_processor, _PYDANTIC_AI_ENRICHMENT_MARKER, False):
        return

    original_on_end = span_processor.on_end

    def _wrapped_on_end(span: ReadableSpan) -> None:
        _enrich_pydantic_ai_span(span=span)
        original_on_end(span)

    span_processor.on_end = _wrapped_on_end
    setattr(span_processor, _PYDANTIC_AI_ENRICHMENT_MARKER, True)


def _install_pydantic_ai_span_enrichment(tracer: RespanTracer) -> None:
    tracer_provider = getattr(tracer, "tracer_provider", None)
    if tracer_provider is None:
        return

    if not getattr(tracer_provider, _PYDANTIC_AI_ADD_PROCESSOR_PATCH_MARKER, False):
        original_add_span_processor = tracer_provider.add_span_processor

        def _wrapped_add_span_processor(span_processor: Any) -> None:
            _wrap_span_processor(span_processor=span_processor)
            original_add_span_processor(span_processor)

        tracer_provider.add_span_processor = _wrapped_add_span_processor
        setattr(tracer_provider, _PYDANTIC_AI_ADD_PROCESSOR_PATCH_MARKER, True)

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

    settings = InstrumentationSettings(
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
