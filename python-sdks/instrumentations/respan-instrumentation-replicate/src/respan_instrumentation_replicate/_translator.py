"""Translate Replicate SDK calls into canonical Respan span attributes."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from opentelemetry.semconv_ai import LLMRequestTypeValues, SpanAttributes

from respan_instrumentation_replicate._constants import (
    ASSISTANT_ROLE,
    DEPLOYMENT_KEY,
    ERROR_KEY,
    ID_KEY,
    INPUT_KEY,
    LOGS_KEY,
    MAX_TEXT_LENGTH,
    METRICS_KEY,
    MODEL_KEY,
    OUTPUT_KEY,
    PROMPT_KEY,
    PREDICTION_RESPAN_MODEL_ATTR,
    REF_KEY,
    REPLICATE_SYSTEM_NAME,
    RESPAN_PARAMS_KEY,
    RESPAN_PARAMS_MODEL_KEY,
    STATUS_KEY,
    USER_ROLE,
    VERSION_KEY,
)
from respan_sdk.constants.llm_logging import (
    LOG_TYPE_TASK,
    LOG_TYPE_TEXT,
    LogMethodChoices,
)
from respan_sdk.constants.span_attributes import (
    RESPAN_LOG_METHOD,
    RESPAN_LOG_TYPE,
    RESPAN_METADATA,
    RESPAN_SPAN_ATTRIBUTES_MAP,
    RESPAN_TRACE_GROUP_ID,
)
from respan_sdk.utils.serialization import serialize_value


def safe_json(value: Any) -> str:
    """Serialize arbitrary Replicate values into an OTEL-safe JSON string."""
    try:
        return json.dumps(
            serialize_value(value=value), default=str, separators=(",", ":")
        )
    except Exception:
        return json.dumps(str(value), separators=(",", ":"))


def _truncate_text(value: str) -> str:
    if len(value) <= MAX_TEXT_LENGTH:
        return value
    return f"{value[:MAX_TEXT_LENGTH]}...<truncated>"


def _to_mapping(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value

    for method_name in ("model_dump", "dict"):
        method = getattr(value, method_name, None)
        if callable(method):
            try:
                converted = method()
            except Exception:
                continue
            if isinstance(converted, Mapping):
                return converted

    value_dict = getattr(value, "__dict__", None)
    if isinstance(value_dict, Mapping):
        return value_dict
    return None


def _file_output_text(value: Any) -> str | None:
    if value.__class__.__name__ != "FileOutput":
        return None
    url = getattr(value, "url", None)
    return str(url) if url else str(value)


def output_to_text(value: Any) -> str:
    """Convert Replicate output or stream chunks to readable completion text."""
    if value is None:
        return ""
    if isinstance(value, str):
        return _truncate_text(value)
    if isinstance(value, bytes):
        return f"<bytes length={len(value)}>"
    if isinstance(value, bytearray):
        return f"<bytearray length={len(value)}>"

    file_output = _file_output_text(value)
    if file_output is not None:
        return _truncate_text(file_output)

    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        if all(isinstance(item, str) for item in value):
            return _truncate_text("".join(value))
        text_parts = [output_to_text(item) for item in value]
        if any(text_parts):
            return _truncate_text("".join(text_parts))

    mapping = _to_mapping(value)
    if mapping is not None:
        for key in (OUTPUT_KEY, "data", "text", "content"):
            if mapping.get(key) is not None:
                return output_to_text(mapping[key])
        return _truncate_text(safe_json(mapping))

    return _truncate_text(str(value))


def _prediction_mapping(prediction: Any) -> Mapping[str, Any]:
    return _to_mapping(prediction) or {}


def model_from_ref_or_prediction(
    *,
    ref: Any = None,
    kwargs: Mapping[str, Any] | None = None,
    prediction: Any = None,
) -> str | None:
    kwargs = kwargs or {}
    respan_params = _to_mapping(kwargs.get(RESPAN_PARAMS_KEY))
    if respan_params is not None:
        model_override = respan_params.get(RESPAN_PARAMS_MODEL_KEY)
        if model_override:
            return str(model_override)

    if prediction is not None:
        prediction_model_override = getattr(
            prediction, PREDICTION_RESPAN_MODEL_ATTR, None
        )
        if prediction_model_override:
            return str(prediction_model_override)

    for value in (
        kwargs.get(MODEL_KEY),
        kwargs.get(VERSION_KEY),
        kwargs.get(DEPLOYMENT_KEY),
        ref,
    ):
        if value:
            return str(value)

    prediction_map = _prediction_mapping(prediction)
    model = prediction_map.get(MODEL_KEY)
    version = prediction_map.get(VERSION_KEY)
    if model and version:
        model_text = str(model)
        version_text = str(version)
        if version_text.startswith(f"{model_text}:"):
            return version_text
        return f"{model_text}:{version_text}"
    if model:
        return str(model)
    if version:
        return str(version)
    return None


def _prompt_content(input_value: Any) -> str:
    mapping = _to_mapping(input_value)
    if mapping is not None:
        for key in (PROMPT_KEY, "text", "query", "input"):
            if mapping.get(key) is not None:
                return output_to_text(mapping[key])
        return safe_json(mapping)
    return output_to_text(input_value)


def _base_llm_attrs(*, span_name: str, model: str | None, stream: bool) -> dict[str, Any]:
    attrs: dict[str, Any] = {
        RESPAN_LOG_METHOD: LogMethodChoices.TRACING_INTEGRATION.value,
        RESPAN_LOG_TYPE: LOG_TYPE_TEXT,
        SpanAttributes.LLM_SYSTEM: REPLICATE_SYSTEM_NAME,
        SpanAttributes.LLM_REQUEST_TYPE: LLMRequestTypeValues.COMPLETION.value,
        SpanAttributes.TRACELOOP_ENTITY_NAME: span_name,
        SpanAttributes.TRACELOOP_ENTITY_PATH: span_name,
    }
    if model:
        attrs[SpanAttributes.LLM_REQUEST_MODEL] = model
    if stream:
        attrs[SpanAttributes.LLM_IS_STREAMING] = True
    return attrs


def _base_task_attrs(*, span_name: str) -> dict[str, Any]:
    return {
        RESPAN_LOG_METHOD: LogMethodChoices.TRACING_INTEGRATION.value,
        RESPAN_LOG_TYPE: LOG_TYPE_TASK,
        SpanAttributes.TRACELOOP_ENTITY_NAME: span_name,
        SpanAttributes.TRACELOOP_ENTITY_PATH: span_name,
    }


def _apply_respan_params(attributes: dict[str, Any], params: Any) -> str | None:
    params_mapping = _to_mapping(params)
    if params_mapping is None:
        return None

    span_name = params_mapping.get("span_name")
    workflow_name = params_mapping.get("workflow_name")
    if workflow_name and "trace_group_identifier" not in params_mapping:
        attributes.setdefault(RESPAN_TRACE_GROUP_ID, str(workflow_name))

    for key, value in params_mapping.items():
        if key in {
            "parent_span_id",
            "span_id",
            "span_name",
            "trace_id",
            "trace_name",
            "workflow_name",
        }:
            continue
        attr_key = RESPAN_SPAN_ATTRIBUTES_MAP.get(str(key))
        if attr_key is None:
            continue
        if attr_key == RESPAN_METADATA and isinstance(value, Mapping):
            for metadata_key, metadata_value in value.items():
                attributes[f"{RESPAN_METADATA}.{metadata_key}"] = (
                    metadata_value
                    if isinstance(metadata_value, str)
                    else str(metadata_value)
                )
        else:
            attributes[attr_key] = value
    return str(span_name) if span_name else None


def _set_request_attrs(
    *,
    attrs: dict[str, Any],
    ref: Any,
    input_value: Any,
    kwargs: Mapping[str, Any],
) -> None:
    entity_input = {REF_KEY: ref, INPUT_KEY: input_value}
    params = {key: value for key, value in kwargs.items() if key != RESPAN_PARAMS_KEY}
    if params:
        entity_input["params"] = params
    attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = safe_json(entity_input)

    prompt_text = _prompt_content(input_value)
    if prompt_text:
        prompt_prefix = f"{SpanAttributes.LLM_PROMPTS}.0"
        attrs[f"{prompt_prefix}.role"] = USER_ROLE
        attrs[f"{prompt_prefix}.content"] = prompt_text


def _set_prediction_metadata(attrs: dict[str, Any], prediction: Any) -> None:
    prediction_map = _prediction_mapping(prediction)
    for key in (ID_KEY, STATUS_KEY, ERROR_KEY, LOGS_KEY):
        value = prediction_map.get(key)
        if value:
            attrs[f"{RESPAN_METADATA}.replicate_{key}"] = output_to_text(value)

    metrics = prediction_map.get(METRICS_KEY)
    if metrics:
        attrs[f"{RESPAN_METADATA}.replicate_metrics"] = safe_json(metrics)


def _set_output_attrs(
    *,
    attrs: dict[str, Any],
    output: Any,
    prediction: Any,
    error: Exception | None,
) -> None:
    if error is not None:
        completion_text = str(error)
    elif output is not None:
        completion_text = output_to_text(output)
    else:
        prediction_map = _prediction_mapping(prediction)
        completion_text = output_to_text(prediction_map.get(OUTPUT_KEY))

    if completion_text:
        attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = completion_text
        completion_prefix = f"{SpanAttributes.LLM_COMPLETIONS}.0"
        attrs[f"{completion_prefix}.role"] = ASSISTANT_ROLE
        attrs[f"{completion_prefix}.content"] = completion_text


def build_model_call_span_data(
    *,
    span_name: str,
    ref: Any = None,
    input_value: Any = None,
    kwargs: Mapping[str, Any] | None = None,
    output: Any = None,
    prediction: Any = None,
    error: Exception | None = None,
    stream: bool = False,
) -> tuple[str, dict[str, Any]]:
    """Build canonical text-completion span data for Replicate model calls."""
    kwargs = kwargs or {}
    model = model_from_ref_or_prediction(ref=ref, kwargs=kwargs, prediction=prediction)
    attrs = _base_llm_attrs(span_name=span_name, model=model, stream=stream)
    resolved_span_name = _apply_respan_params(attrs, kwargs.get(RESPAN_PARAMS_KEY))
    if resolved_span_name:
        span_name = resolved_span_name
        attrs[SpanAttributes.TRACELOOP_ENTITY_NAME] = span_name
        attrs[SpanAttributes.TRACELOOP_ENTITY_PATH] = span_name

    _set_request_attrs(attrs=attrs, ref=ref, input_value=input_value, kwargs=kwargs)
    if prediction is not None:
        _set_prediction_metadata(attrs, prediction)
    _set_output_attrs(attrs=attrs, output=output, prediction=prediction, error=error)
    return span_name, attrs


def build_operation_span_data(
    *,
    span_name: str,
    input_value: Any = None,
    output: Any = None,
    error: Exception | None = None,
) -> tuple[str, dict[str, Any]]:
    """Build a non-LLM task span for Replicate SDK management operations."""
    attrs = _base_task_attrs(span_name=span_name)
    if input_value is not None:
        attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = safe_json(input_value)
    if output is not None:
        attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = safe_json(output)
    if error is not None:
        attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = str(error)
    return span_name, attrs
