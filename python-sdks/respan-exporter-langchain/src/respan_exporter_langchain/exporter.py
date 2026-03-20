"""Respan LangChain Exporter - Export LangChain traces to Respan tracing endpoint."""
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

import requests

from respan_exporter_langchain.types import TraceContext
from respan_exporter_langchain.utils import (
    clean_payload,
    format_rfc3339,
    is_blank_value,
    normalize_span_id,
    normalize_trace_id,
    serialize_value,
    to_completion_message,
    to_prompt_messages,
)
from respan_sdk.constants.llm_logging import LOG_TYPE_MAP
from respan_sdk.respan_types.log_types import RespanFullLogParams
from respan_sdk.utils import RetryHandler

logger = logging.getLogger(__name__)

DEFAULT_ENDPOINT = "https://api.respan.ai/api/v1/traces/ingest"


class RespanLangchainExporter:
    """
    Export LangChain traces/spans to Respan tracing endpoint.

    This exporter is used internally by RespanCallbackHandler. It accepts a
    list of span dicts (built from callback events) and converts them to
    Respan ingest payloads.
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
        self._retry_handler = RetryHandler(
            max_retries=3,
            retry_delay=1.0,
            backoff_multiplier=2.0,
            max_delay=10.0,
        )

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

    def export(
        self,
        spans: List[Dict[str, Any]],
        trace_name: Optional[str] = None,
        session_identifier: Optional[Union[str, int]] = None,
        trace_group_identifier: Optional[Union[str, int]] = None,
    ) -> List[Dict[str, Any]]:
        """Export spans to Respan."""
        payloads = self.build_payload(
            spans=spans,
            trace_name=trace_name,
            session_identifier=session_identifier,
            trace_group_identifier=trace_group_identifier,
        )
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

    def build_payload(
        self,
        spans: List[Dict[str, Any]],
        trace_name: Optional[str] = None,
        session_identifier: Optional[Union[str, int]] = None,
        trace_group_identifier: Optional[Union[str, int]] = None,
    ) -> List[Dict[str, Any]]:
        """Build Respan payloads from span dicts."""
        if not spans:
            return []

        trace_context = self._extract_trace_context(
            spans=spans,
            trace_name=trace_name,
            session_identifier=session_identifier,
            trace_group_identifier=trace_group_identifier,
        )

        # Build span ID normalization map
        span_id_map: Dict[str, str] = {}
        for span in spans:
            raw_span_id = span.get("span_id")
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

    def _extract_trace_context(
        self,
        spans: List[Dict[str, Any]],
        trace_name: Optional[str] = None,
        session_identifier: Optional[Union[str, int]] = None,
        trace_group_identifier: Optional[Union[str, int]] = None,
    ) -> TraceContext:
        """Extract trace context from spans."""
        # Find root span (no parent)
        root_span = None
        for span in spans:
            if span.get("parent_id") is None:
                root_span = span
                break
        if root_span is None and spans:
            root_span = spans[0]

        # Use root span's run_id as trace_id
        trace_id = str(root_span.get("span_id", uuid.uuid4())) if root_span else str(uuid.uuid4())

        if not trace_name and root_span:
            trace_name = root_span.get("name")
        if not trace_name:
            trace_name = trace_id

        workflow_name = trace_name

        # Extract metadata from root span
        trace_metadata = dict(root_span.get("metadata") or {}) if root_span else {}

        # Try to extract customer_identifier from metadata
        customer_identifier = None
        for key in ("customer_identifier", "customer_id", "user_id", "user"):
            if key in trace_metadata:
                customer_identifier = trace_metadata.get(key)
                break
        if not customer_identifier:
            customer_identifier = self.customer_identifier

        # Session identifier from metadata
        if not session_identifier:
            for key in ("session_id", "session_identifier", "session"):
                if key in trace_metadata:
                    session_identifier = trace_metadata.get(key)
                    break

        # Find earliest start time
        start_time = None
        for span in spans:
            st = span.get("start_time")
            if isinstance(st, datetime):
                if start_time is None or st < start_time:
                    start_time = st

        return TraceContext(
            trace_id=trace_id,
            trace_name=trace_name,
            workflow_name=workflow_name,
            metadata=trace_metadata,
            session_identifier=session_identifier,
            trace_group_identifier=trace_group_identifier,
            start_time=start_time,
            customer_identifier=customer_identifier,
        )

    def _span_to_respan(
        self,
        span: Dict[str, Any],
        trace_context: TraceContext,
        span_id_map: Dict[str, str],
    ) -> Optional[Dict[str, Any]]:
        """Convert a span dict to Respan payload format."""
        span_id = span.get("span_id")
        parent_id = span.get("parent_id")
        span_name = span.get("name")
        span_type = span.get("span_type", "task")
        model = span.get("model")
        error = span.get("error")

        span_input = span.get("input")
        span_output = span.get("output")

        prompt_messages = span.get("prompt_messages")
        completion_message = span.get("completion_message")

        # Build prompt_messages and completion_message if not provided
        if prompt_messages is None and span_input is not None:
            prompt_messages = to_prompt_messages(value=span_input)
        if completion_message is None and span_output is not None:
            completion_message = to_completion_message(value=span_output)

        # Fallback: create simple message wrappers
        if prompt_messages is None and isinstance(span_input, str) and span_input.strip():
            prompt_messages = [{"role": "user", "content": span_input.strip()}]
        if completion_message is None and isinstance(span_output, str) and span_output.strip():
            completion_message = {"role": "assistant", "content": span_output.strip()}

        start_time = span.get("start_time")
        end_time = span.get("end_time")

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

        latency = None
        if start_time and end_time:
            latency = (end_time - start_time).total_seconds()
        if latency is not None and latency < 0:
            latency = 0.0

        if not span_id:
            span_id = str(uuid.uuid4())
        span_id_str = str(span_id)
        if not span_name:
            span_name = span_id_str

        log_type = self._map_log_type(
            span_type=span_type,
            parent_id=parent_id,
            model=model,
        )

        span_metadata = dict(span.get("metadata") or {})
        tags = span.get("tags", [])
        if tags:
            span_metadata["langchain_tags"] = tags
        merged_metadata = {**trace_context.metadata, **span_metadata}

        trace_hex_id = normalize_trace_id(trace_id=trace_context.trace_id)
        span_hex_id = span_id_map.get(span_id_str) or normalize_span_id(
            span_id=span_id_str,
            trace_id=trace_context.trace_id,
        )
        parent_hex_id = (
            span_id_map.get(str(parent_id))
            if parent_id is not None
            else None
        )
        if parent_hex_id is None and parent_id is not None:
            parent_hex_id = normalize_span_id(
                span_id=str(parent_id),
                trace_id=trace_context.trace_id,
            )

        # Store original LangChain IDs in metadata
        if "langchain_run_id" not in merged_metadata:
            merged_metadata["langchain_run_id"] = span_id_str
        if parent_id and "langchain_parent_run_id" not in merged_metadata:
            merged_metadata["langchain_parent_run_id"] = str(parent_id)

        input_value = serialize_value(value=span_input) if span_input is not None else None
        output_value = serialize_value(value=span_output) if span_output is not None else None

        payload: Dict[str, Any] = {
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
            "metadata": merged_metadata or None,
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

        # Populate span_tools for tool spans (matches Agno exporter pattern)
        if log_type == "tool" and span_name:
            payload["span_tools"] = [str(span_name)]

        # Token usage
        prompt_tokens = span.get("prompt_tokens")
        completion_tokens = span.get("completion_tokens")
        total_tokens = span.get("total_tokens")
        if any(v is not None for v in (prompt_tokens, completion_tokens, total_tokens)):
            payload["prompt_tokens"] = prompt_tokens
            payload["completion_tokens"] = completion_tokens
            payload["total_request_tokens"] = total_tokens
            # Calculate total if missing
            if total_tokens is None or total_tokens == 0:
                pt = int(prompt_tokens or 0)
                ct = int(completion_tokens or 0)
                if pt or ct:
                    payload["total_request_tokens"] = pt + ct

        if error:
            payload["error_message"] = str(error)
            payload["status_code"] = 500
        else:
            payload["status_code"] = 200

        if not payload.get("span_unique_id") and payload.get("trace_unique_id"):
            payload["span_unique_id"] = payload["trace_unique_id"]

        cleaned_payload = clean_payload(payload=payload)

        try:
            RespanFullLogParams(**cleaned_payload)
        except Exception as exc:
            logger.warning(
                "LangChain span payload failed RespanFullLogParams validation: %s",
                exc,
            )

        return cleaned_payload

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

    def _map_log_type(
        self,
        span_type: str,
        parent_id: Optional[str],
        model: Optional[str],
    ) -> str:
        """Map span type to Respan log type."""
        if span_type:
            type_str = str(span_type).lower()
            for key, value in LOG_TYPE_MAP.items():
                if key in type_str:
                    return value
        if model:
            return "generation"
        if parent_id is None:
            return "workflow"
        return "task"

    def _send(self, payloads: List[Dict[str, Any]]) -> None:
        """Send payloads to Respan endpoint with retry on transient errors."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        def _do_request() -> None:
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
            raise RuntimeError(
                "Respan export server error %s: %s"
                % (response.status_code, response.text[:200])
            )

        try:
            self._retry_handler.execute(_do_request, context="Respan LangChain export")
        except Exception as exc:
            logger.error("Respan LangChain export failed after retries: %s", exc)
