"""Respan Google ADK Exporter - Export Google ADK traces to Respan tracing endpoint."""
import logging
import os
import random
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import requests

from respan_exporter_google_adk.types import TraceContext
from respan_exporter_google_adk.utils import (
    as_dict,
    clean_payload,
    coerce_datetime,
    coerce_token_count,
    extract_genai_messages,
    find_root_span,
    format_rfc3339,
    get_attr,
    infer_trace_start_time,
    is_blank_value,
    messages_to_text,
    normalize_span_id,
    normalize_trace_id,
    serialize_value,
    to_completion_message,
    to_prompt_messages,
)
from respan_sdk.constants.llm_logging import LOG_TYPE_MAP
from respan_sdk.respan_types.log_types import RespanFullLogParams

logger = logging.getLogger(__name__)

DEFAULT_ENDPOINT = "https://api.respan.ai/api/v1/traces/ingest"

# Mapping from ADK span names to Respan log types
_ADK_LOG_TYPE_MAP: Dict[str, str] = {
    "invocation": "workflow",
    "agent_run": "agent",
    "call_llm": "generation",
    "execute_tool": "tool",
}


class RespanGoogleAdkExporter:
    """
    Export Google ADK traces/spans to Respan tracing endpoint.

    The exporter accepts:
    - a trace object with a .spans collection
    - a dict with a "spans" field
    - a list of span objects/dicts
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        base_url: Optional[str] = None,
        environment: Optional[str] = None,
        customer_identifier: Optional[Union[str, int]] = None,
        timeout: int = 10,
    ) -> None:
        self.api_key = api_key or os.getenv("RESPAN_API_KEY")
        if base_url is None:
            base_url = (
                os.getenv("RESPAN_BASE_URL")
                or "https://api.respan.ai/api"
            )
        self.endpoint = endpoint or self._build_endpoint(base_url=base_url)
        self.environment = (
            environment
            or os.getenv("RESPAN_ENVIRONMENT")
            or "production"
        )
        self.customer_identifier = (
            customer_identifier
            or os.getenv("RESPAN_CUSTOMER_IDENTIFIER")
        )
        self.timeout = timeout

    def _build_endpoint(self, base_url: Optional[str]) -> str:
        """Build the ingest endpoint URL from base URL."""
        if not base_url:
            return DEFAULT_ENDPOINT
        base = base_url.rstrip("/")
        if base.endswith("/v1/traces/ingest"):
            return base
        if base.endswith("/v1/traces"):
            return f"{base}/ingest"
        if base.endswith("/api"):
            return f"{base}/v1/traces/ingest"
        return f"{base}/api/v1/traces/ingest"

    def export(self, trace_or_spans: Any) -> List[Dict[str, Any]]:
        """Export trace or spans to Respan."""
        payloads = self.build_payload(trace_or_spans=trace_or_spans)
        if not payloads:
            return payloads
        if not self.api_key:
            logger.warning(
                "Respan API key is not set; skipping export to %s",
                self.endpoint,
            )
            return payloads
        self._send(payloads=payloads)
        return payloads

    def export_trace(self, trace_or_spans: Any) -> List[Dict[str, Any]]:
        """Alias for export method."""
        return self.export(trace_or_spans=trace_or_spans)

    def build_payload(self, trace_or_spans: Any) -> List[Dict[str, Any]]:
        """Build payload from trace or spans."""
        trace_obj, spans = self._normalize_trace(trace_or_spans=trace_or_spans)
        if not spans:
            return []
        trace_context = self._extract_trace_context(trace_obj=trace_obj, spans=spans)
        span_id_map: Dict[str, str] = {}
        for span in spans:
            raw_span_id = get_attr(span, "span_id", "id", "uid")
            if raw_span_id is None:
                continue
            raw_span_id = str(raw_span_id)
            span_id_map[raw_span_id] = normalize_span_id(
                span_id=raw_span_id,
                trace_id=trace_context.trace_id,
            )
        payloads: List[Dict[str, Any]] = []
        for span in spans:
            payload = self._span_to_respan(
                span=span,
                trace_context=trace_context,
                span_id_map=span_id_map,
            )
            if payload:
                payloads.append(payload)
        if payloads:
            self._propagate_trace_output(payloads=payloads)
        return payloads

    def _propagate_trace_output(self, payloads: List[Dict[str, Any]]) -> None:
        """Propagate output from generation spans to workflow/agent/task spans."""
        trace_output: Optional[str] = None
        for payload in payloads:
            output_value = payload.get("output")
            if is_blank_value(value=output_value):
                continue
            log_type = payload.get("log_type")
            if log_type == "generation":
                trace_output = output_value
                break
            if trace_output is None:
                trace_output = output_value
        if trace_output is not None:
            for payload in payloads:
                if is_blank_value(value=payload.get("output")) and payload.get(
                    "log_type"
                ) in (
                    "workflow",
                    "agent",
                    "task",
                ):
                    payload["output"] = trace_output

    def _normalize_trace(self, trace_or_spans: Any) -> Tuple[Optional[Any], List[Any]]:
        """Normalize trace input to (trace_obj, spans) tuple."""
        if trace_or_spans is None:
            return None, []
        if isinstance(trace_or_spans, (list, tuple, set)):
            return None, list(trace_or_spans)
        if isinstance(trace_or_spans, dict):
            spans = trace_or_spans.get("spans")
            if spans is not None:
                return trace_or_spans, list(spans)
            return None, [trace_or_spans]
        spans = get_attr(trace_or_spans, "spans", "span_events")
        if spans is not None:
            return trace_or_spans, list(spans)
        return None, [trace_or_spans]

    def _extract_trace_context(
        self,
        trace_obj: Optional[Any],
        spans: Sequence[Any],
    ) -> TraceContext:
        """Extract trace context from trace object and spans."""
        trace_id = get_attr(trace_obj, "trace_id", "id", "uid")
        trace_name = get_attr(trace_obj, "name", "trace_name", "title")
        workflow_name = get_attr(trace_obj, "workflow_name", "workflow")
        session_identifier = get_attr(trace_obj, "session_identifier", "session_id")
        conversation_id: Optional[str] = None
        trace_group_identifier = get_attr(
            trace_obj,
            "trace_group_identifier",
            "group_identifier",
            "group_id",
        )

        trace_metadata: Dict[str, Any] = {}

        customer_identifier = get_attr(
            trace_obj,
            "customer_identifier",
            "customer_id",
            "user_id",
            "user_identifier",
            "user",
        )
        if not customer_identifier:
            customer_identifier = self.customer_identifier

        trace_start_time = coerce_datetime(
            value=get_attr(trace_obj, "start_time", "started_at", "start", "start_timestamp")
        )

        root_span = find_root_span(spans=spans)
        root_attrs: Dict[str, Any] = {}
        if root_span is not None:
            root_attrs = as_dict(
                value=get_attr(root_span, "attributes", "metadata", "tags", "data")
            ) or {}

        # Extract trace info from root span GenAI attributes
        if not trace_name:
            trace_name = (
                root_attrs.get("gen_ai.agent.name")
                or root_attrs.get("gen_ai.request.model")
            )
        if not trace_name:
            root_name = get_attr(root_span, "name")
            if root_name:
                trace_name = str(root_name)

        if not trace_id:
            for span in spans:
                trace_id = get_attr(span, "trace_id", "traceId")
                if trace_id:
                    break
        if not trace_id:
            trace_id = str(uuid.uuid4())

        if not trace_name and trace_id:
            trace_name = str(trace_id)

        if not workflow_name:
            workflow_name = trace_name

        # Extract session/conversation from GenAI attributes
        if not session_identifier:
            conversation_id = root_attrs.get("gen_ai.conversation.id")
            if conversation_id:
                session_identifier = conversation_id

        if not trace_start_time:
            trace_start_time = infer_trace_start_time(spans=spans)

        return TraceContext(
            trace_id=str(trace_id),
            trace_name=str(trace_name) if trace_name else None,
            workflow_name=str(workflow_name) if workflow_name else None,
            metadata=trace_metadata,
            session_identifier=session_identifier,
            trace_group_identifier=trace_group_identifier,
            start_time=trace_start_time,
            customer_identifier=customer_identifier,
            conversation_id=conversation_id,
        )

    def _span_to_respan(
        self,
        span: Any,
        trace_context: TraceContext,
        span_id_map: Optional[Dict[str, str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Convert span to Respan payload format."""
        span_id = get_attr(span, "span_id", "id", "uid")
        parent_id = get_attr(span, "parent_id", "parent_span_id", "parentId")
        span_name = get_attr(span, "name", "span_name", "operation_name")

        span_attrs = as_dict(
            value=get_attr(span, "attributes", "metadata", "tags", "data")
        ) or {}

        # -- Extract GenAI semantic convention attributes --

        # Model
        model = (
            span_attrs.get("gen_ai.response.model")
            or span_attrs.get("gen_ai.request.model")
        )

        # Token usage
        prompt_tokens_raw = span_attrs.get("gen_ai.usage.input_tokens")
        completion_tokens_raw = span_attrs.get("gen_ai.usage.output_tokens")

        # Messages
        input_messages = extract_genai_messages(
            attributes=span_attrs, key="gen_ai.input.messages"
        )
        output_messages = extract_genai_messages(
            attributes=span_attrs, key="gen_ai.output.messages"
        )

        # Agent info
        agent_name = span_attrs.get("gen_ai.agent.name")
        agent_id = span_attrs.get("gen_ai.agent.id")

        # Input/Output
        span_input = get_attr(span, "input", "input_data", "request", "prompt")
        if span_input is None and input_messages:
            span_input = input_messages

        span_output = get_attr(span, "output", "output_data", "response")
        if span_output is None and output_messages:
            output_text = messages_to_text(messages=output_messages)
            if output_text:
                span_output = output_text

        # Build prompt_messages / completion_message
        prompt_messages = (
            to_prompt_messages(value=span_input) if span_input is not None else None
        )
        completion_message = (
            to_completion_message(value=span_output) if span_output is not None else None
        )
        if prompt_messages is None and input_messages:
            prompt_messages = input_messages
        if completion_message is None and output_messages:
            completion_message = output_messages[0] if output_messages else None

        if is_blank_value(value=span_output) and output_messages:
            output_text = messages_to_text(messages=output_messages)
            if output_text:
                span_output = output_text
                if completion_message is None:
                    completion_message = {"role": "assistant", "content": output_text}

        if span_output is None and isinstance(completion_message, dict):
            completion_content = completion_message.get("content")
            if completion_content:
                span_output = completion_content
        if completion_message is None and isinstance(span_output, str) and span_output.strip():
            completion_message = {"role": "assistant", "content": span_output.strip()}
        if prompt_messages is None and isinstance(span_input, str) and span_input.strip():
            prompt_messages = [{"role": "user", "content": span_input.strip()}]

        # Error
        error = get_attr(span, "error", "exception", "err")
        if error is None:
            error = span_attrs.get("error")

        # Timestamps
        start_time = coerce_datetime(
            value=get_attr(span, "start_time", "started_at", "start", "start_timestamp"),
            reference=trace_context.start_time,
        )
        end_time = coerce_datetime(
            value=get_attr(span, "end_time", "ended_at", "end", "end_timestamp", "timestamp"),
            reference=trace_context.start_time,
        )

        now = datetime.now(timezone.utc)
        if start_time is None and end_time is None:
            start_time = now
            end_time = now
        elif start_time is None:
            start_time = end_time
        elif end_time is None:
            end_time = start_time
        if start_time and end_time and end_time < start_time:
            end_time = start_time

        latency = get_attr(span, "latency", "duration")
        if latency is None and start_time and end_time:
            latency = (end_time - start_time).total_seconds()
        if latency is not None and latency < 0:
            latency = 0.0

        if not span_id:
            span_id = str(uuid.uuid4())
        span_id_str = str(span_id)
        if not span_name:
            span_name = span_id

        log_type = self._map_log_type(
            span_name=str(span_name) if span_name else None,
            parent_id=parent_id,
            model=model,
        )

        # Build metadata with LLM config and agent info
        span_metadata: Dict[str, Any] = dict(trace_context.metadata)
        if agent_name:
            span_metadata["adk_agent_name"] = agent_name
        if agent_id:
            span_metadata["adk_agent_id"] = agent_id

        # LLM config
        temperature = span_attrs.get("gen_ai.request.temperature")
        top_k = span_attrs.get("gen_ai.request.top_k")
        top_p = span_attrs.get("gen_ai.request.top_p")
        max_tokens = span_attrs.get("gen_ai.request.max_output_tokens")
        llm_config: Dict[str, Any] = {}
        if temperature is not None:
            llm_config["temperature"] = temperature
        if top_k is not None:
            llm_config["top_k"] = top_k
        if top_p is not None:
            llm_config["top_p"] = top_p
        if max_tokens is not None:
            llm_config["max_tokens"] = max_tokens
        if llm_config:
            span_metadata["llm_config"] = llm_config

        # Conversation ID
        conv_id = span_attrs.get("gen_ai.conversation.id") or trace_context.conversation_id
        if conv_id:
            span_metadata["conversation_id"] = conv_id

        trace_hex_id = normalize_trace_id(trace_id=trace_context.trace_id)
        if span_id_map and span_id_str in span_id_map:
            span_hex_id = span_id_map[span_id_str]
        else:
            span_hex_id = normalize_span_id(
                span_id=span_id_str,
                trace_id=trace_context.trace_id,
            )
        parent_hex_id = (
            span_id_map.get(str(parent_id))
            if span_id_map and parent_id is not None
            else normalize_span_id(span_id=str(parent_id), trace_id=trace_context.trace_id)
            if parent_id
            else None
        )

        if "adk_trace_id" not in span_metadata:
            span_metadata["adk_trace_id"] = trace_context.trace_id
        if "adk_span_id" not in span_metadata:
            span_metadata["adk_span_id"] = str(span_id)
        if parent_id and "adk_parent_id" not in span_metadata:
            span_metadata["adk_parent_id"] = str(parent_id)

        input_value = serialize_value(value=span_input) if span_input is not None else None
        output_value = serialize_value(value=span_output) if span_output is not None else None

        payload = {
            "trace_unique_id": trace_hex_id,
            "trace_name": trace_context.trace_name,
            "span_unique_id": span_hex_id,
            "span_parent_id": parent_hex_id,
            "span_name": str(span_name) if span_name else None,
            "span_workflow_name": trace_context.workflow_name,
            "trace_id": trace_hex_id,
            "span_id": span_hex_id,
            "parent_id": parent_hex_id,
            "environment": self.environment,
            "customer_identifier": trace_context.customer_identifier,
            "log_type": log_type,
            "start_time": format_rfc3339(value=start_time),
            "timestamp": format_rfc3339(value=end_time),
            "latency": latency,
            "input": input_value,
            "output": output_value,
            "model": model,
            "metadata": span_metadata or None,
            "session_identifier": trace_context.session_identifier,
            "trace_group_identifier": trace_context.trace_group_identifier,
        }
        if prompt_messages is not None:
            payload["prompt_messages"] = prompt_messages
        if completion_message is not None:
            payload["completion_message"] = completion_message
        payload["respan_params"] = {
            "environment": self.environment,
            "has_webhook": False,
        }
        payload["disable_log"] = False

        # Token usage
        prompt_tokens_value = coerce_token_count(value=prompt_tokens_raw)
        completion_tokens_value = coerce_token_count(value=completion_tokens_raw)
        if prompt_tokens_value is not None or completion_tokens_value is not None:
            payload["prompt_tokens"] = prompt_tokens_value
            payload["completion_tokens"] = completion_tokens_value
            total = (prompt_tokens_value or 0) + (completion_tokens_value or 0)
            payload["total_request_tokens"] = total if total > 0 else None

        if error:
            payload["error_message"] = str(error)
            payload["status_code"] = 500
        else:
            payload["status_code"] = get_attr(span, "status_code") or 200

        # Tool info
        tool_name = (
            get_attr(span, "tool_name", "tool")
            or span_attrs.get("gen_ai.tool.name")
            or span_attrs.get("tool.name")
            or span_attrs.get("tool_name")
        )
        if tool_name:
            payload["span_tools"] = [str(tool_name)]

        if not payload.get("span_unique_id") and payload.get("trace_unique_id"):
            payload["span_unique_id"] = payload["trace_unique_id"]

        cleaned_payload = clean_payload(payload=payload)

        try:
            RespanFullLogParams(**cleaned_payload)
        except Exception as exc:
            logger.warning(
                "ADK span payload failed RespanFullLogParams validation: %s",
                exc,
            )

        return cleaned_payload

    def _map_log_type(
        self,
        span_name: Optional[str],
        parent_id: Optional[str],
        model: Optional[str],
    ) -> str:
        """Map ADK span name to Respan log type."""
        if span_name:
            # Check ADK-specific mapping first
            adk_type = _ADK_LOG_TYPE_MAP.get(span_name)
            if adk_type:
                return adk_type
            # Fall back to generic LOG_TYPE_MAP
            name_lower = span_name.lower()
            for key, value in LOG_TYPE_MAP.items():
                if key in name_lower:
                    return value
        if model:
            return "generation"
        if parent_id is None:
            return "workflow"
        return "task"

    def _send(self, payloads: List[Dict[str, Any]]) -> None:
        """Send payloads to Respan endpoint with retry on transient errors."""
        max_retries = 3
        base_delay = 1.0
        max_delay = 10.0
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        for attempt in range(max_retries):
            try:
                response = requests.post(
                    url=self.endpoint,
                    json=payloads,
                    headers=headers,
                    timeout=self.timeout,
                )
                if response.status_code in (200, 201):
                    return
                if response.status_code < 500:
                    logger.warning(
                        "Respan export failed with status %s: %s",
                        response.status_code,
                        response.text,
                    )
                    return
                logger.warning(
                    "Respan export failed with status %s (attempt %d/%d)",
                    response.status_code,
                    attempt + 1,
                    max_retries,
                )
            except Exception as exc:
                logger.warning(
                    "Respan export request failed (attempt %d/%d): %s",
                    attempt + 1,
                    max_retries,
                    exc,
                )

            if attempt < max_retries - 1:
                delay = min(base_delay * (2 ** attempt), max_delay)
                delay += random.uniform(0, delay * 0.1)
                time.sleep(delay)
