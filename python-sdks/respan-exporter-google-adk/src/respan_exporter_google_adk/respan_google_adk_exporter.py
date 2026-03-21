"""Google ADK span → Respan log exporter.

Converts OpenTelemetry ``ReadableSpan`` objects emitted by Google ADK into
Respan-compatible log dicts and ships them over HTTP.
"""

from __future__ import annotations

import logging
import os
import random
import time
from typing import Any, Dict, List, Optional, Sequence

import httpx
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.trace import StatusCode

from respan_sdk.constants.llm_logging import (
    LOG_TYPE_AGENT,
    LOG_TYPE_CUSTOM,
    LOG_TYPE_GENERATION,
    LOG_TYPE_TOOL,
)
from respan_sdk.respan_types.param_types import RespanTextLogParams

from .utils import (
    build_metadata,
    coerce_int,
    extract_span_type,
    gemini_request_to_input_text,
    gemini_request_to_prompt_messages,
    gemini_response_to_completion_message,
    get_attr,
    message_to_text,
    ns_to_datetime,
    ns_to_seconds,
    safe_json_parse,
    serialize,
    span_id_hex,
    trace_id_hex,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-type converter functions
# ---------------------------------------------------------------------------

def _invocation_to_log(data: RespanTextLogParams, span: ReadableSpan) -> None:
    data.log_type = LOG_TYPE_AGENT
    data.span_name = "invocation"


def _invoke_agent_to_log(data: RespanTextLogParams, span: ReadableSpan) -> None:
    data.log_type = LOG_TYPE_AGENT
    data.span_name = span.name

    agent_name = get_attr(span, "gen_ai.agent.name")
    if agent_name:
        data.span_workflow_name = agent_name

    meta = build_metadata(span)
    if agent_name:
        meta["agent_name"] = agent_name
    agent_desc = get_attr(span, "gen_ai.agent.description")
    if agent_desc:
        meta["agent_description"] = agent_desc
    agent_ver = get_attr(span, "gen_ai.agent.version")
    if agent_ver:
        meta["agent_version"] = agent_ver
    if meta:
        data.metadata = meta


def _generation_content_to_log(data: RespanTextLogParams, span: ReadableSpan) -> None:
    """Shared logic for generate_content and call_llm spans."""
    data.log_type = LOG_TYPE_GENERATION
    data.span_name = span.name

    # Model
    model = get_attr(span, "gen_ai.request.model")
    if model:
        data.model = model

    # Token counts
    prompt_tokens = coerce_int(get_attr(span, "gen_ai.usage.input_tokens"))
    completion_tokens = coerce_int(get_attr(span, "gen_ai.usage.output_tokens"))
    if prompt_tokens is not None:
        data.prompt_tokens = prompt_tokens
    if completion_tokens is not None:
        data.completion_tokens = completion_tokens
    if prompt_tokens is not None and completion_tokens is not None:
        data.total_request_tokens = prompt_tokens + completion_tokens

    # Max tokens
    max_tokens = coerce_int(get_attr(span, "gen_ai.request.max_tokens"))
    if max_tokens is not None:
        data.max_tokens = max_tokens

    # LLM request → prompt_messages + input
    llm_request_raw = get_attr(span, "gcp.vertex.agent.llm_request")
    llm_request = safe_json_parse(llm_request_raw)
    if isinstance(llm_request, dict):
        data.full_request = serialize(llm_request)
        prompt_msgs = gemini_request_to_prompt_messages(llm_request)
        if prompt_msgs:
            data.prompt_messages = prompt_msgs
        input_text = gemini_request_to_input_text(llm_request)
        if input_text:
            data.input = input_text
    elif llm_request is not None:
        data.input = serialize(llm_request)

    # LLM response → completion_message + output + tool_calls
    llm_response_raw = get_attr(span, "gcp.vertex.agent.llm_response")
    llm_response = safe_json_parse(llm_response_raw)
    if isinstance(llm_response, dict):
        data.full_response = serialize(llm_response)
        comp_msg = gemini_response_to_completion_message(llm_response)
        if comp_msg:
            data.completion_message = comp_msg
            completion_text = message_to_text(comp_msg)
            if completion_text:
                data.output = completion_text
            # Extract tool_calls from completion message
            if comp_msg.get("tool_calls"):
                data.tool_calls = comp_msg["tool_calls"]
    elif llm_response is not None:
        data.output = serialize(llm_response)

    # Metadata
    meta = build_metadata(span)
    gen_ai_system = get_attr(span, "gen_ai.system")
    if gen_ai_system:
        meta["gen_ai_system"] = gen_ai_system
    finish_reasons = get_attr(span, "gen_ai.response.finish_reasons")
    if finish_reasons:
        meta["finish_reasons"] = finish_reasons
    reasoning_tokens = coerce_int(get_attr(span, "gen_ai.usage.experimental.reasoning_tokens"))
    if reasoning_tokens is not None:
        meta["reasoning_tokens"] = reasoning_tokens
    reasoning_limit = coerce_int(get_attr(span, "gen_ai.usage.experimental.reasoning_tokens_limit"))
    if reasoning_limit is not None:
        meta["reasoning_tokens_limit"] = reasoning_limit
    sys_instr_tokens = coerce_int(get_attr(span, "gen_ai.usage.experimental.system_instruction_tokens"))
    if sys_instr_tokens is not None:
        meta["system_instruction_tokens"] = sys_instr_tokens
    top_p = get_attr(span, "gen_ai.request.top_p")
    if top_p is not None:
        meta["top_p"] = top_p
    agent_ver = get_attr(span, "gen_ai.agent.version")
    if agent_ver:
        meta["agent_version"] = agent_ver
    if meta:
        data.metadata = meta


def _generate_content_to_log(data: RespanTextLogParams, span: ReadableSpan) -> None:
    _generation_content_to_log(data, span)


def _call_llm_to_log(data: RespanTextLogParams, span: ReadableSpan) -> None:
    _generation_content_to_log(data, span)


def _execute_tool_to_log(data: RespanTextLogParams, span: ReadableSpan) -> None:
    data.log_type = LOG_TYPE_TOOL
    data.span_name = span.name

    tool_name = get_attr(span, "gen_ai.tool.name")
    if tool_name:
        data.span_tools = [tool_name]

    # Input / output
    tool_args_raw = get_attr(span, "gcp.vertex.agent.tool_call_args")
    if tool_args_raw is not None:
        data.input = serialize(safe_json_parse(tool_args_raw))
    tool_resp_raw = get_attr(span, "gcp.vertex.agent.tool_response")
    if tool_resp_raw is not None:
        data.output = serialize(safe_json_parse(tool_resp_raw))

    # Metadata
    meta = build_metadata(span)
    if tool_name:
        meta["tool_name"] = tool_name
    tool_desc = get_attr(span, "gen_ai.tool.description")
    if tool_desc:
        meta["tool_description"] = tool_desc
    tool_type = get_attr(span, "gen_ai.tool.type")
    if tool_type:
        meta["tool_type"] = tool_type
    tool_call_id = get_attr(span, "gen_ai.tool.call_id")
    if tool_call_id:
        meta["tool_call_id"] = tool_call_id
    if meta:
        data.metadata = meta


def _send_data_to_log(data: RespanTextLogParams, span: ReadableSpan) -> None:
    data.log_type = LOG_TYPE_CUSTOM
    data.span_name = "send_data"

    meta = build_metadata(span)
    raw_data = get_attr(span, "gcp.vertex.agent.data")
    if raw_data is not None:
        meta["data"] = safe_json_parse(raw_data)
    if meta:
        data.metadata = meta


def _context_caching_to_log(data: RespanTextLogParams, span: ReadableSpan) -> None:
    data.log_type = LOG_TYPE_CUSTOM
    data.span_name = "handle_context_caching"


def _create_cache_to_log(data: RespanTextLogParams, span: ReadableSpan) -> None:
    data.log_type = LOG_TYPE_CUSTOM
    data.span_name = "create_cache"

    meta = build_metadata(span)
    for attr_key in ("cache_contents_count", "model", "ttl_seconds", "cache_name"):
        val = get_attr(span, attr_key)
        if val is not None:
            meta[attr_key] = val
    if meta:
        data.metadata = meta


def _unknown_to_log(data: RespanTextLogParams, span: ReadableSpan) -> None:
    data.log_type = LOG_TYPE_CUSTOM
    data.span_name = span.name
    meta = build_metadata(span)
    if meta:
        data.metadata = meta


# Routing table: span_type → converter
_CONVERTERS = {
    "invocation": _invocation_to_log,
    "invoke_agent": _invoke_agent_to_log,
    "generate_content": _generate_content_to_log,
    "call_llm": _call_llm_to_log,
    "execute_tool": _execute_tool_to_log,
    "send_data": _send_data_to_log,
    "handle_context_caching": _context_caching_to_log,
    "create_cache": _create_cache_to_log,
}


# ---------------------------------------------------------------------------
# Public conversion function
# ---------------------------------------------------------------------------

def convert_span_to_respan_log(span: ReadableSpan) -> Optional[Dict[str, Any]]:
    """Convert an OTel ``ReadableSpan`` from Google ADK to a Respan log dict.

    Returns ``None`` if conversion fails or the span is unrecognised.
    """
    try:
        ctx = span.context
        if ctx is None:
            return None

        tid = trace_id_hex(ctx.trace_id)
        sid = span_id_hex(ctx.span_id)

        # Parent ID
        parent = getattr(span, "parent", None)
        if parent is not None:
            pid = span_id_hex(parent.span_id)
        else:
            pid = tid

        # Timestamps
        start_ns = span.start_time or 0
        end_ns = span.end_time or 0

        # Error detection
        has_error = False
        error_message = None

        status = getattr(span, "status", None)
        if status and status.status_code == StatusCode.ERROR:
            has_error = True
            error_message = status.description

        error_type = get_attr(span, "error.type")
        if error_type:
            has_error = True
            if not error_message:
                error_message = str(error_type)

        data = RespanTextLogParams(
            trace_unique_id=tid,
            span_unique_id=sid,
            span_parent_id=pid,
            start_time=ns_to_datetime(start_ns) if start_ns else None,
            timestamp=ns_to_datetime(end_ns) if end_ns else None,
            latency=ns_to_seconds(start_ns, end_ns) if (start_ns and end_ns) else None,
            error_bit=1 if has_error else 0,
            status_code=400 if has_error else 200,
            error_message=error_message,
            customer_identifier=get_attr(span, "user.id"),
            session_identifier=get_attr(span, "gcp.vertex.agent.session_id"),
        )

        # Route to per-type converter
        span_type = extract_span_type(span)
        converter = _CONVERTERS.get(span_type, _unknown_to_log)
        converter(data, span)

        return data.model_dump(mode="json")

    except Exception:
        logger.exception("Failed to convert ADK span to Respan log")
        return None


# ---------------------------------------------------------------------------
# Exporter class
# ---------------------------------------------------------------------------

class RespanGoogleADKExporter(SpanExporter):
    """OTel ``SpanExporter`` that ships Google ADK spans to Respan."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint: str = "https://api.respan.ai/api/v1/traces/ingest",
        customer_identifier: Optional[str] = None,
        customer_name: Optional[str] = None,
        customer_email: Optional[str] = None,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
    ):
        self.api_key = api_key or os.environ.get("RESPAN_API_KEY")
        self.endpoint = endpoint
        self.customer_identifier = customer_identifier
        self.customer_name = customer_name
        self.customer_email = customer_email
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self._client = httpx.Client(timeout=30.0)

    # -- SpanExporter interface ---------------------------------------------

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        if not spans:
            return SpanExportResult.SUCCESS

        if not self.api_key:
            logger.warning("RESPAN_API_KEY not set — skipping export")
            return SpanExportResult.FAILURE

        converted: List[Dict[str, Any]] = []
        for span in spans:
            log = convert_span_to_respan_log(span)
            if log is None:
                continue
            # Apply exporter-level defaults
            if self.customer_identifier and not log.get("customer_identifier"):
                log["customer_identifier"] = self.customer_identifier
            if self.customer_name and not log.get("customer_name"):
                log["customer_name"] = self.customer_name
            if self.customer_email and not log.get("customer_email"):
                log["customer_email"] = self.customer_email
            converted.append(log)

        if not converted:
            return SpanExportResult.SUCCESS

        payload = {"data": converted}
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        attempt = 0
        delay = self.base_delay
        while True:
            attempt += 1
            try:
                response = self._client.post(
                    url=self.endpoint, headers=headers, json=payload,
                )
                if response.status_code < 300:
                    logger.debug("Exported %d spans to Respan", len(converted))
                    return SpanExportResult.SUCCESS
                if 400 <= response.status_code < 500:
                    logger.error(
                        "Respan client error %d: %s",
                        response.status_code,
                        response.text,
                    )
                    return SpanExportResult.FAILURE
                logger.warning("Server error %d, retrying…", response.status_code)
            except httpx.RequestError as exc:
                logger.warning("Request failed: %s", exc)

            if attempt >= self.max_retries:
                logger.error("Max retries reached, giving up on this batch.")
                return SpanExportResult.FAILURE

            sleep_time = delay + random.uniform(0, 0.1 * delay)
            time.sleep(sleep_time)
            delay = min(delay * 2, self.max_delay)

    def shutdown(self) -> None:
        self._client.close()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True
