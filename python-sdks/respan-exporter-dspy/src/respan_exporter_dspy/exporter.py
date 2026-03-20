"""Respan dspy Exporter - Export dspy traces to Respan."""
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import requests

from respan_exporter_dspy.types import TraceContext
from respan_exporter_dspy.utils import (
    as_dict,
    clean_payload,
    coerce_datetime,
    coerce_token_count,
    extract_openinference_messages,
    extract_span_metadata,
    find_root_span,
    format_rfc3339,
    get_attr,
    infer_trace_start_time,
    is_blank_value,
    merge_openinference_metadata,
    normalize_span_id,
    normalize_trace_id,
    pick_metadata_value,
    serialize_value,
)
from respan_sdk.constants.llm_logging import LOG_TYPE_MAP
from respan_sdk.respan_types.log_types import RespanFullLogParams

logger = logging.getLogger(__name__)

DEFAULT_ENDPOINT = "https://api.respan.ai/api/v1/traces/ingest"


class RespanDSPyExporter:
    """Export dspy traces/spans to Respan."""

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
        self.endpoint = endpoint or self._build_endpoint(base_url=base_url)
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

    def export(self, trace_or_spans: Any) -> List[Dict[str, Any]]:
        """Export trace or spans to Respan."""
        payloads = self.build_payload(trace_or_spans=trace_or_spans)
        if not payloads:
            return payloads
        if not self.api_key:
            logger.warning("Respan API key is not set; skipping export")
            return payloads
        self._send(payloads=payloads)
        return payloads

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
            span_id_map[raw_span_id] = normalize_span_id(span_id=raw_span_id, trace_id=trace_context.trace_id)
        payloads: List[Dict[str, Any]] = []
        for span in spans:
            payload = self._span_to_respan(span=span, trace_context=trace_context, span_id_map=span_id_map)
            if payload:
                payloads.append(payload)
        if payloads:
            self._propagate_trace_output(payloads=payloads)
        return payloads

    def _propagate_trace_output(self, payloads: List[Dict[str, Any]]) -> None:
        """Propagate output from generation spans to workflow/agent/task spans."""
        trace_output: Optional[str] = None
        for p in payloads:
            output = p.get("output")
            if is_blank_value(value=output):
                continue
            if p.get("log_type") == "generation":
                trace_output = output
                break
            if trace_output is None:
                trace_output = output
        if trace_output is not None:
            for p in payloads:
                if is_blank_value(value=p.get("output")) and p.get("log_type") in ("workflow", "agent", "task"):
                    p["output"] = trace_output

    def _normalize_trace(self, trace_or_spans: Any) -> Tuple[Optional[Any], List[Any]]:
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

    def _extract_trace_context(self, trace_obj: Optional[Any], spans: Sequence[Any]) -> TraceContext:
        trace_id = get_attr(trace_obj, "trace_id", "id", "uid")
        trace_name = get_attr(trace_obj, "name", "trace_name", "title")
        workflow_name = get_attr(trace_obj, "workflow_name", "workflow")
        session_identifier = get_attr(trace_obj, "session_identifier", "session_id")
        trace_group_identifier = get_attr(trace_obj, "trace_group_identifier", "group_identifier", "group_id")

        trace_metadata = as_dict(value=get_attr(trace_obj, "metadata", "attributes", "tags")) or {}
        trace_metadata = merge_openinference_metadata(metadata=trace_metadata)

        customer_identifier = get_attr(trace_obj, "customer_identifier", "customer_id", "user_id", "user")
        if not customer_identifier:
            customer_identifier = self.customer_identifier

        trace_start_time = coerce_datetime(value=get_attr(trace_obj, "start_time", "started_at", "start"))

        root_span = find_root_span(spans=spans)
        root_metadata = extract_span_metadata(span=root_span) if root_span else {}

        if not trace_id:
            for span in spans:
                trace_id = get_attr(span, "trace_id", "traceId")
                if trace_id:
                    break
        if not trace_id:
            trace_id = str(uuid.uuid4())

        if not trace_name:
            trace_name = pick_metadata_value(root_metadata, "dspy.agent.type", "agent.name")
        if not trace_name:
            trace_name = str(trace_id)

        if not workflow_name:
            workflow_name = trace_name

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
        )

    def _span_to_respan(self, span: Any, trace_context: TraceContext, span_id_map: Optional[Dict[str, str]] = None) -> Optional[Dict[str, Any]]:
        span_id = get_attr(span, "span_id", "id", "uid")
        parent_id = get_attr(span, "parent_id", "parent_span_id", "parentId")
        span_name = get_attr(span, "name", "span_name", "operation_name")
        span_kind = get_attr(span, "type", "span_type", "kind")

        span_path = get_attr(span, "span_path", "path")

        span_metadata = as_dict(value=get_attr(span, "metadata", "attributes", "tags", "data")) or {}
        span_metadata = merge_openinference_metadata(metadata=span_metadata)

        if not span_path:
            span_path = pick_metadata_value(span_metadata, "graph.node.id")

        if span_kind is None:
            span_kind = pick_metadata_value(span_metadata, "openinference.span.kind", "span.kind")

        span_input = get_attr(span, "input", "input_data", "request", "prompt")
        if span_input is None:
            for key in ("input.value", "input_value", "traceloop.entity.input"):
                if key in span_metadata:
                    span_input = span_metadata.get(key)
                    break

        span_output = get_attr(span, "output", "output_data", "response")
        if span_output is None:
            for key in ("output.value", "output_value", "traceloop.entity.output"):
                if key in span_metadata:
                    span_output = span_metadata.get(key)
                    break

        input_messages = extract_openinference_messages(metadata=span_metadata, prefix="llm.input_messages")
        output_messages = extract_openinference_messages(metadata=span_metadata, prefix="llm.output_messages")
        if span_input is None and input_messages:
            span_input = input_messages
        if span_output is None and output_messages:
            span_output = output_messages

        model = get_attr(span, "model", "model_name")
        if model is None:
            model = pick_metadata_value(span_metadata, "llm.model_name", "llm.model", "model")

        usage = get_attr(span, "usage", "token_usage")
        usage = as_dict(value=usage)
        if usage is None:
            prompt_tokens = span_metadata.get("llm.token_count.prompt")
            completion_tokens = span_metadata.get("llm.token_count.completion")
            total_tokens = span_metadata.get("llm.token_count.total")
            if any(v is not None for v in (prompt_tokens, completion_tokens, total_tokens)):
                usage = {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": total_tokens}

        error = get_attr(span, "error", "exception", "err")

        start_time = coerce_datetime(value=get_attr(span, "start_time", "started_at", "start"), reference=trace_context.start_time)
        end_time = coerce_datetime(value=get_attr(span, "end_time", "ended_at", "end", "timestamp"), reference=trace_context.start_time)

        now = datetime.now(timezone.utc)
        if start_time is None and end_time is None:
            start_time = end_time = now
        elif start_time is None:
            start_time = end_time
        elif end_time is None:
            end_time = start_time
        if start_time and end_time and end_time < start_time:
            end_time = start_time

        latency = get_attr(span, "latency", "duration")
        if latency is None and start_time and end_time:
            latency = (end_time - start_time).total_seconds()

        if not span_id:
            span_id = str(uuid.uuid4())
        span_id_str = str(span_id)
        if not span_name:
            span_name = span_id

        log_type = self._map_log_type(span_kind=span_kind, parent_id=parent_id, model=model)

        merged_metadata = {**trace_context.metadata, **(span_metadata or {})}
        trace_hex_id = normalize_trace_id(trace_id=trace_context.trace_id)
        span_hex_id = span_id_map.get(span_id_str) if span_id_map else normalize_span_id(span_id=span_id_str, trace_id=trace_context.trace_id)
        if not span_hex_id:
            span_hex_id = normalize_span_id(span_id=span_id_str, trace_id=trace_context.trace_id)
        if parent_id:
            parent_hex_id = (
                (span_id_map.get(str(parent_id)) if span_id_map else None)
                or normalize_span_id(span_id=str(parent_id), trace_id=trace_context.trace_id)
            )
        else:
            parent_hex_id = None

        if "dspy_trace_id" not in merged_metadata:
            merged_metadata["dspy_trace_id"] = trace_context.trace_id
        if "dspy_span_id" not in merged_metadata:
            merged_metadata["dspy_span_id"] = str(span_id)
        if parent_id and "dspy_parent_id" not in merged_metadata:
            merged_metadata["dspy_parent_id"] = str(parent_id)

        input_value = serialize_value(value=span_input) if span_input is not None else None
        output_value = serialize_value(value=span_output) if span_output is not None else None

        payload = {
            "trace_unique_id": trace_hex_id,
            "trace_name": trace_context.trace_name,
            "span_unique_id": span_hex_id,
            "span_parent_id": parent_hex_id,
            "span_name": str(span_name) if span_name else None,
            "span_path": span_path,
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
            "respan_params": {"environment": self.environment, "has_webhook": False},
            "disable_log": False,
        }

        if usage:
            pt = usage.get("prompt_tokens") or usage.get("input_tokens")
            ct = usage.get("completion_tokens") or usage.get("output_tokens")
            tt = usage.get("total_tokens") or usage.get("total")
            payload["prompt_tokens"] = pt
            payload["completion_tokens"] = ct
            payload["total_request_tokens"] = tt
            coerced_total = coerce_token_count(value=tt)
            if coerced_total is None or coerced_total == 0:
                cp = coerce_token_count(value=pt) or 0
                cc = coerce_token_count(value=ct) or 0
                if cp or cc:
                    payload["total_request_tokens"] = cp + cc

        if error:
            payload["error_message"] = str(error)
            payload["status_code"] = 500
        else:
            payload["status_code"] = get_attr(span, "status_code") or 200

        tool_name = get_attr(span, "tool_name", "tool") or merged_metadata.get("tool_name") or merged_metadata.get("tool.name")
        if tool_name:
            payload["span_tools"] = [str(tool_name)]

        if not payload.get("span_unique_id") and payload.get("trace_unique_id"):
            payload["span_unique_id"] = payload["trace_unique_id"]

        cleaned = clean_payload(payload=payload)

        try:
            RespanFullLogParams(**cleaned)
        except Exception as exc:
            logger.warning("dspy span payload failed validation: %s", exc)

        return cleaned

    def _map_log_type(self, span_kind: Any, parent_id: Optional[str], model: Optional[str]) -> str:
        if span_kind:
            kind_str = str(span_kind).lower()
            for key, value in LOG_TYPE_MAP.items():
                if key in kind_str:
                    return value
        if model:
            return "generation"
        if parent_id is None:
            return "workflow"
        return "task"

    def _send(self, payloads: List[Dict[str, Any]]) -> None:
        try:
            response = requests.post(
                url=self.endpoint, json=payloads,
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                timeout=self.timeout,
            )
            if response.status_code not in (200, 201):
                logger.warning("Respan export failed with status %s: %s", response.status_code, response.text)
        except Exception as exc:
            logger.warning("Respan export request failed: %s", exc)
