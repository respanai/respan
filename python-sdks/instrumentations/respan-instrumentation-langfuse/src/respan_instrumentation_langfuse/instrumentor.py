"""Translate Langfuse OTLP spans into the canonical Respan span contract."""

from __future__ import annotations

import inspect
import json
import logging
from collections.abc import Collection, Mapping, Sequence
from http import HTTPStatus
from typing import Any

import wrapt
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.sdk.trace.export import SpanExportResult
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv._incubating.attributes.error_attributes import ERROR_MESSAGE
from opentelemetry.semconv_ai import LLMRequestTypeValues, SpanAttributes
from opentelemetry.trace.status import StatusCode
from respan_sdk.constants.llm_logging import (
    LOG_TYPE_AGENT,
    LOG_TYPE_CHAT,
    LOG_TYPE_EMBEDDING,
    LOG_TYPE_GUARDRAIL,
    LOG_TYPE_TASK,
    LOG_TYPE_TOOL,
    LOG_TYPE_WORKFLOW,
    LogMethodChoices,
)
from respan_sdk.constants.span_attributes import (
    RESPAN_CUSTOMER_PARAMS_ID,
    RESPAN_LOG_METHOD,
    RESPAN_LOG_TYPE,
    RESPAN_METADATA,
    RESPAN_SESSION_ID,
    RESPAN_TRACE_GROUP_ID,
)
from respan_tracing.utils.span_factory import build_readable_span, inject_span

logger = logging.getLogger(__name__)

_instruments = ("langfuse >= 3.12.0, <5.0.0",)

_OBSERVATION_TYPE = "langfuse.observation.type"
_OBSERVATION_INPUT = "langfuse.observation.input"
_OBSERVATION_OUTPUT = "langfuse.observation.output"
_OBSERVATION_MODEL = "langfuse.observation.model.name"
_LEGACY_OBSERVATION_MODEL = "langfuse.observation.model"
_OBSERVATION_USAGE = "langfuse.observation.usage_details"
_OBSERVATION_MODEL_PARAMETERS = "langfuse.observation.model.parameters"
_OBSERVATION_LEVEL = "langfuse.observation.level"
_OBSERVATION_STATUS_MESSAGE = "langfuse.observation.status_message"
_TRACE_NAME = "langfuse.trace.name"
_TRACE_INPUT = "langfuse.trace.input"
_TRACE_OUTPUT = "langfuse.trace.output"
_TRACE_USER_ID = "user.id"
_TRACE_SESSION_ID = "session.id"
_METADATA_PREFIXES = (
    "langfuse.trace.metadata.",
    "langfuse.observation.metadata.",
)


def _json_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return value


def _json_string(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return json.dumps(value, default=str, separators=(",", ":"))


def _content(value: Any) -> str:
    parsed = _json_value(value)
    if isinstance(parsed, str):
        return parsed
    return json.dumps(parsed, default=str, separators=(",", ":"))


def _messages(value: Any, *, default_role: str) -> list[dict[str, Any]]:
    parsed = _json_value(value)
    if parsed is None:
        return []
    if isinstance(parsed, Mapping) and isinstance(parsed.get("messages"), list):
        parsed = parsed["messages"]
    elif isinstance(parsed, Mapping) and any(
        key in parsed for key in ("role", "content", "tool_calls")
    ):
        parsed = [parsed]
    if isinstance(parsed, Sequence) and not isinstance(parsed, (str, bytes)):
        messages = []
        for item in parsed:
            if not isinstance(item, Mapping):
                continue
            role = item.get("role") or item.get("type") or default_role
            content = item.get(
                "content", "" if "role" in item or "tool_calls" in item else item
            )
            if content is None:
                content = ""
            message = {"role": str(role), "content": _content(content)}
            if isinstance(item.get("tool_calls"), list):
                message["tool_calls"] = item["tool_calls"]
            messages.append(message)
        if messages:
            return messages
    return [{"role": default_role, "content": _content(parsed)}]


def _usage(attributes: Mapping[str, Any]) -> dict[str, int]:
    current = _json_value(attributes.get(_OBSERVATION_USAGE))
    current = current if isinstance(current, Mapping) else {}
    candidates = {
        "prompt": (
            current.get("prompt_tokens"),
            current.get("input_tokens"),
            current.get("input"),
            attributes.get("langfuse.usage.input"),
        ),
        "completion": (
            current.get("completion_tokens"),
            current.get("output_tokens"),
            current.get("output"),
            attributes.get("langfuse.usage.output"),
        ),
        "total": (
            current.get("total_tokens"),
            current.get("total"),
            attributes.get("langfuse.usage.total"),
        ),
    }
    resolved: dict[str, int] = {}
    for key, values in candidates.items():
        for value in values:
            if isinstance(value, int) and not isinstance(value, bool):
                resolved[key] = value
                break
    if "total" not in resolved and ("prompt" in resolved or "completion" in resolved):
        resolved["total"] = resolved.get("prompt", 0) + resolved.get("completion", 0)
    return resolved


def _log_type(observation_type: str, *, is_root: bool) -> str:
    if observation_type == "generation":
        return LOG_TYPE_CHAT
    if observation_type == "embedding":
        return LOG_TYPE_EMBEDDING
    if observation_type == "agent":
        return LOG_TYPE_AGENT
    if observation_type == "tool":
        return LOG_TYPE_TOOL
    if observation_type == "guardrail":
        return LOG_TYPE_GUARDRAIL
    return LOG_TYPE_WORKFLOW if is_root else LOG_TYPE_TASK


def _translate_span(source_span: Any) -> tuple[dict[str, Any], int, str | None]:
    source = dict(source_span.attributes or {})
    observation_type = str(source.get(_OBSERVATION_TYPE, "span")).lower()
    log_type = _log_type(observation_type, is_root=source_span.parent is None)
    input_value = source.get(_OBSERVATION_INPUT, source.get(_TRACE_INPUT))
    output_value = source.get(_OBSERVATION_OUTPUT, source.get(_TRACE_OUTPUT))

    attributes: dict[str, Any] = {
        RESPAN_LOG_METHOD: LogMethodChoices.TRACING_INTEGRATION.value,
        RESPAN_LOG_TYPE: log_type,
        SpanAttributes.TRACELOOP_ENTITY_NAME: source_span.name,
        SpanAttributes.TRACELOOP_ENTITY_PATH: (
            "" if source_span.parent is None else source_span.name
        ),
    }
    if input_value is not None:
        attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT] = _json_string(input_value)
    if output_value is not None:
        attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = _json_string(output_value)

    trace_name = source.get(_TRACE_NAME)
    if trace_name:
        attributes[RESPAN_TRACE_GROUP_ID] = str(trace_name)
    if source.get(_TRACE_USER_ID):
        attributes[RESPAN_CUSTOMER_PARAMS_ID] = str(source[_TRACE_USER_ID])
    if source.get(_TRACE_SESSION_ID):
        attributes[RESPAN_SESSION_ID] = str(source[_TRACE_SESSION_ID])
    for key, value in source.items():
        for prefix in _METADATA_PREFIXES:
            if key.startswith(prefix):
                attributes[f"{RESPAN_METADATA}.{key.removeprefix(prefix)}"] = value
                break

    if log_type in (LOG_TYPE_CHAT, LOG_TYPE_EMBEDDING):
        attributes[SpanAttributes.LLM_REQUEST_TYPE] = (
            LLMRequestTypeValues.EMBEDDING.value
            if log_type == LOG_TYPE_EMBEDDING
            else LLMRequestTypeValues.CHAT.value
        )
        model = source.get(_OBSERVATION_MODEL) or source.get(_LEGACY_OBSERVATION_MODEL)
        if model:
            attributes[SpanAttributes.LLM_REQUEST_MODEL] = str(model)
    if log_type == LOG_TYPE_CHAT:
        parameters = _json_value(source.get(_OBSERVATION_MODEL_PARAMETERS))
        if isinstance(parameters, Mapping) and isinstance(
            parameters.get("tools"), list
        ):
            attributes[SpanAttributes.LLM_REQUEST_FUNCTIONS] = _json_string(
                parameters["tools"]
            )
        for index, message in enumerate(_messages(input_value, default_role="user")):
            prefix = f"{SpanAttributes.LLM_PROMPTS}.{index}"
            attributes[f"{prefix}.role"] = message["role"]
            attributes[f"{prefix}.content"] = message["content"]
            if message.get("tool_calls"):
                attributes[f"{prefix}.tool_calls"] = _json_string(message["tool_calls"])
        for index, message in enumerate(
            _messages(output_value, default_role="assistant")
        ):
            prefix = f"{SpanAttributes.LLM_COMPLETIONS}.{index}"
            attributes[f"{prefix}.role"] = message["role"]
            attributes[f"{prefix}.content"] = message["content"]
            if message.get("tool_calls"):
                attributes[f"{prefix}.tool_calls"] = _json_string(message["tool_calls"])

    if log_type in (LOG_TYPE_CHAT, LOG_TYPE_EMBEDDING):
        usage = _usage(source)
        if "prompt" in usage:
            attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] = usage["prompt"]
            attributes[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] = usage["prompt"]
        if "completion" in usage:
            attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] = usage["completion"]
            attributes[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS] = usage["completion"]
        if "total" in usage:
            attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] = usage["total"]

    source_status = getattr(source_span, "status", None)
    status_code = getattr(source_status, "status_code", StatusCode.UNSET)
    level = str(source.get(_OBSERVATION_LEVEL, "")).upper()
    is_error = status_code is StatusCode.ERROR or level == "ERROR"
    error_message = None
    if is_error:
        error_message = getattr(source_status, "description", None) or source.get(
            _OBSERVATION_STATUS_MESSAGE
        )
    if error_message:
        attributes[ERROR_MESSAGE] = str(error_message)
    return (
        attributes,
        HTTPStatus.INTERNAL_SERVER_ERROR if is_error else HTTPStatus.OK,
        str(error_message) if error_message else None,
    )


def _inject_langfuse_span(source_span: Any) -> bool:
    attributes, status_code, error_message = _translate_span(source_span)
    parent_id = (
        format(source_span.parent.span_id, "016x") if source_span.parent else None
    )
    span = build_readable_span(
        name=source_span.name,
        trace_id=format(source_span.context.trace_id, "032x"),
        span_id=format(source_span.context.span_id, "016x"),
        parent_id=parent_id,
        start_time_ns=source_span.start_time,
        end_time_ns=source_span.end_time,
        attributes=attributes,
        status_code=status_code,
        error_message=error_message,
        kind=source_span.kind,
        merge_propagated=False,
    )
    return inject_span(span)


class LangfuseInstrumentor(BaseInstrumentor):
    """Intercept Langfuse's OTLP exporter and inject canonical Respan spans."""

    _exporter_class: type | None = None
    _original_export: Any = None
    _installed_export: Any = None
    _export_enabled = False
    _exported_span_count = 0

    @property
    def exported_span_count(self) -> int:
        """Number of Langfuse spans injected during the active session."""
        return self._exported_span_count

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        del kwargs
        self._exported_span_count = 0
        self._patch_otlp_exporter()
        logger.info("Langfuse instrumentation enabled for Respan")

    def _uninstrument(self, **kwargs: Any) -> None:
        del kwargs
        self._export_enabled = False
        if (
            self._exporter_class is not None
            and self._original_export is not None
            and inspect.getattr_static(self._exporter_class, "export")
            is self._installed_export
        ):
            self._exporter_class.export = self._original_export
        self._exporter_class = None
        self._original_export = None
        self._installed_export = None
        logger.info("Langfuse instrumentation disabled")

    @staticmethod
    def _is_langfuse_exporter(instance: Any) -> bool:
        endpoint = str(getattr(instance, "_endpoint", "")).lower()
        return "langfuse" in endpoint or "/api/public/otel" in endpoint

    def _export_spans(self, spans: Sequence[Any]) -> SpanExportResult:
        try:
            source_spans = [
                span
                for span in spans
                if _OBSERVATION_TYPE in dict(getattr(span, "attributes", {}) or {})
            ]
            if not source_spans:
                return SpanExportResult.SUCCESS
            result = (
                SpanExportResult.SUCCESS
                if all(_inject_langfuse_span(span) for span in source_spans)
                else SpanExportResult.FAILURE
            )
            if result is SpanExportResult.SUCCESS:
                self._exported_span_count += len(source_spans)
            return result
        except Exception:
            logger.exception("Failed to translate Langfuse spans")
            return SpanExportResult.FAILURE

    def _patch_otlp_exporter(self) -> None:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )

        self._exporter_class = OTLPSpanExporter
        self._original_export = inspect.getattr_static(OTLPSpanExporter, "export")

        def export_wrapper(wrapped: Any, instance: Any, args: Any, kwargs: Any) -> Any:
            if not self._export_enabled or not self._is_langfuse_exporter(instance):
                return wrapped(*args, **kwargs)
            spans = args[0] if args else kwargs.get("spans", ())
            return self._export_spans(spans)

        wrapt.wrap_function_wrapper(
            "opentelemetry.exporter.otlp.proto.http.trace_exporter",
            "OTLPSpanExporter.export",
            export_wrapper,
        )
        self._installed_export = inspect.getattr_static(OTLPSpanExporter, "export")
        self._export_enabled = True


__all__ = ["LangfuseInstrumentor"]
