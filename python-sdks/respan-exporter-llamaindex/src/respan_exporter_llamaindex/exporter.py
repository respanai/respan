"""Respan LlamaIndex Exporter - Export LlamaIndex traces to Respan."""
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

import requests

logger = logging.getLogger(__name__)

DEFAULT_ENDPOINT = "https://api.respan.ai/api/v1/traces/ingest"

# LlamaIndex class names that indicate LLM calls
_LLM_CLASS_NAMES = frozenset({
    "openai", "anthropic", "cohere", "gemini", "groq", "mistral",
    "azure_openai", "bedrock", "huggingface", "ollama", "replicate",
    "together", "litellm", "llm",
})

# LlamaIndex class names that indicate agent spans
_AGENT_CLASS_NAMES = frozenset({
    "agent", "reagentworker", "openaiagent", "functioncallingagent",
})

# Span name patterns that indicate LLM calls
_LLM_SPAN_PATTERNS = frozenset({
    "llm.chat", "llm.complete", "llm.predict", "llm.stream_chat",
    "llm.stream_complete",
})


def _is_llm_span(instance_class: Optional[str], span_name: Optional[str]) -> bool:
    """Check if span represents an LLM call."""
    if instance_class:
        lower = instance_class.lower()
        if any(name in lower for name in _LLM_CLASS_NAMES):
            return True
    if span_name:
        lower = span_name.lower()
        if any(pattern in lower for pattern in _LLM_SPAN_PATTERNS):
            return True
    return False


def _is_agent_span(instance_class: Optional[str], span_name: Optional[str]) -> bool:
    """Check if span represents an agent."""
    if instance_class:
        lower = instance_class.lower()
        if any(name in lower for name in _AGENT_CLASS_NAMES):
            return True
    if span_name and "agent" in span_name.lower():
        return True
    return False


def _coerce_int(value: Any) -> Optional[int]:
    """Coerce value to int if possible."""
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value.strip()))
        except (ValueError, TypeError):
            return None
    return None


def _serialize(value: Any) -> Optional[str]:
    """Serialize value to JSON string."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, default=str)
    except Exception:
        return str(value)


class RespanLlamaIndexExporter:
    """Export LlamaIndex span records to Respan."""

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
            base_url = os.getenv("RESPAN_BASE_URL") or "https://api.respan.ai/api"
        self.endpoint = endpoint or self._build_endpoint(base_url)
        self.environment = environment or os.getenv("RESPAN_ENVIRONMENT") or "production"
        self.customer_identifier = customer_identifier or os.getenv("RESPAN_CUSTOMER_IDENTIFIER")
        self.timeout = timeout

    def _build_endpoint(self, base_url: Optional[str]) -> str:
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
        trace_id: str,
        trace_name: Optional[str] = None,
        session_identifier: Optional[Union[str, int]] = None,
    ) -> None:
        """Build payloads and send to Respan."""
        payloads = self.build_payloads(
            spans=spans, trace_id=trace_id, trace_name=trace_name,
            session_identifier=session_identifier,
        )
        if not payloads:
            return
        if not self.api_key:
            logger.warning("Respan API key is not set; skipping export")
            return
        self._send(payloads)

    def build_payloads(
        self,
        spans: List[Dict[str, Any]],
        trace_id: str,
        trace_name: Optional[str] = None,
        session_identifier: Optional[Union[str, int]] = None,
    ) -> List[Dict[str, Any]]:
        """Convert span records to Respan payloads."""
        if not spans:
            return []

        trace_hex_id = uuid.uuid5(uuid.NAMESPACE_DNS, trace_id).hex
        if not trace_name:
            trace_name = trace_id

        # Build span ID map for hex normalization
        span_id_map: Dict[str, str] = {}
        for span in spans:
            sid = span.get("span_id", "")
            if sid:
                span_id_map[sid] = uuid.uuid5(uuid.NAMESPACE_DNS, f"{trace_id}:{sid}").hex[:16]

        payloads: List[Dict[str, Any]] = []
        for span in spans:
            payload = self._span_to_payload(
                span=span,
                trace_hex_id=trace_hex_id,
                trace_name=trace_name,
                span_id_map=span_id_map,
                session_identifier=session_identifier,
            )
            if payload:
                payloads.append(payload)
        return payloads

    def _span_to_payload(
        self,
        span: Dict[str, Any],
        trace_hex_id: str,
        trace_name: str,
        span_id_map: Dict[str, str],
        session_identifier: Optional[Union[str, int]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Convert a single span record to Respan payload."""
        span_id = span.get("span_id", "")
        parent_id = span.get("parent_id")
        span_name = span.get("span_name", "")
        instance_class = span.get("instance_class")
        input_data = span.get("input")
        result = span.get("result")
        error = span.get("error")
        metadata = span.get("metadata", {}) or {}
        start_time = span.get("start_time")
        end_time = span.get("end_time")

        # Determine log type
        log_type = self._map_log_type(
            instance_class=instance_class,
            span_name=span_name,
            parent_id=parent_id,
            metadata=metadata,
        )

        # Calculate latency
        latency = None
        if start_time and end_time:
            latency = (end_time - start_time).total_seconds()

        # Resolve hex IDs
        span_hex_id = span_id_map.get(span_id, uuid.uuid4().hex[:16])
        parent_hex_id = span_id_map.get(parent_id) if parent_id else None

        # Extract model from metadata
        model = metadata.get("model") or metadata.get("llm.model_name")

        # Format times
        start_str = self._format_time(start_time) if start_time else None
        end_str = self._format_time(end_time) if end_time else None

        payload: Dict[str, Any] = {
            "trace_unique_id": trace_hex_id,
            "trace_name": trace_name,
            "span_unique_id": span_hex_id,
            "span_parent_id": parent_hex_id,
            "span_name": span_name or instance_class or span_id,
            "span_workflow_name": trace_name,
            "trace_id": trace_hex_id,
            "span_id": span_hex_id,
            "parent_id": parent_hex_id,
            "environment": self.environment,
            "customer_identifier": self.customer_identifier,
            "session_identifier": session_identifier,
            "log_type": log_type,
            "start_time": start_str,
            "timestamp": end_str,
            "latency": latency,
            "input": _serialize(input_data),
            "output": _serialize(result),
            "model": model,
            "metadata": {
                "llamaindex_span_id": span_id,
                "llamaindex_instance_class": instance_class,
                **(metadata or {}),
            },
            "respan_params": {
                "environment": self.environment,
                "has_webhook": False,
            },
            "disable_log": False,
        }

        # Token usage from metadata (coerce to int)
        prompt_tokens = _coerce_int(metadata.get("prompt_tokens") or metadata.get("llm.token_count.prompt"))
        completion_tokens = _coerce_int(metadata.get("completion_tokens") or metadata.get("llm.token_count.completion"))
        total_tokens = _coerce_int(metadata.get("total_tokens") or metadata.get("llm.token_count.total"))
        if any(v is not None for v in (prompt_tokens, completion_tokens, total_tokens)):
            payload["prompt_tokens"] = prompt_tokens
            payload["completion_tokens"] = completion_tokens
            if total_tokens:
                payload["total_request_tokens"] = total_tokens
            elif prompt_tokens or completion_tokens:
                payload["total_request_tokens"] = (prompt_tokens or 0) + (completion_tokens or 0)

        if error:
            payload["error_message"] = str(error)
            payload["status_code"] = 500
        else:
            payload["status_code"] = 200

        # Clean None values
        return {k: v for k, v in payload.items() if v is not None}

    def _map_log_type(
        self,
        instance_class: Optional[str],
        span_name: Optional[str],
        parent_id: Optional[str],
        metadata: Dict[str, Any],
    ) -> str:
        """Map span to Respan log type."""
        if _is_llm_span(instance_class, span_name):
            return "generation"
        if _is_agent_span(instance_class, span_name):
            return "agent"
        if metadata.get("model") or metadata.get("llm.model_name"):
            return "generation"
        if parent_id is None:
            return "workflow"
        return "task"

    @staticmethod
    def _format_time(dt: Optional[datetime]) -> Optional[str]:
        if dt is None:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

    def _send(self, payloads: List[Dict[str, Any]]) -> None:
        """Send payloads to Respan endpoint."""
        try:
            response = requests.post(
                url=self.endpoint,
                json=payloads,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=self.timeout,
            )
            if response.status_code not in (200, 201):
                logger.warning(
                    "Respan export failed with status %s: %s",
                    response.status_code,
                    response.text,
                )
        except Exception as exc:
            logger.warning("Respan export request failed: %s", exc)
