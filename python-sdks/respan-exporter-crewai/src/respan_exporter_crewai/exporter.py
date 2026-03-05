"""Respan CrewAI Exporter - Export CrewAI traces to Respan tracing endpoint."""
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import requests

from respan_sdk.constants import RESPAN_DOGFOOD_HEADER
from respan_sdk.constants.llm_logging import LogMethodChoices

from respan_exporter_crewai.types import TraceContext
from respan_exporter_crewai.utils import (
    as_dict,
    build_traces_ingest_url,
    clean_payload,
    coerce_datetime,
    coerce_token_count,
    extract_metadata_payload,
    extract_openinference_choice_texts,
    extract_openinference_messages,
    extract_span_metadata,
    find_root_span,
    format_rfc3339,
    get_attr,
    infer_trace_start_time,
    is_blank_value,
    merge_openinference_metadata,
    messages_to_text,
    normalize_span_id,
    normalize_trace_id,
    pick_metadata_value,
    serialize_value,
    to_completion_message,
    to_prompt_messages,
)
from respan_sdk.constants.llm_logging import LOG_TYPE_MAP
from respan_sdk.respan_types.log_types import RespanFullLogParams

logger = logging.getLogger(__name__)

DEFAULT_ENDPOINT = "https://api.respan.ai/api/v1/traces/ingest"


class RespanCrewAIExporter:
    """
    Export CrewAI traces/spans to Respan tracing endpoint.

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
            if customer_identifier is not None
            else os.getenv("RESPAN_CUSTOMER_IDENTIFIER")
        )
        self.timeout = timeout

    def _build_endpoint(self, base_url: Optional[str]) -> str:
        """Build the ingest endpoint URL from base URL."""
        return build_traces_ingest_url(base_url=base_url, default_endpoint=DEFAULT_ENDPOINT)

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
        self.send(payloads=payloads)
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
        """Propagate output from generation spans to workflow/agent/task spans.

        Intentional: only the first generation span's output is propagated to
        workflow/agent/task spans that lack output. Later generations are not
        merged so the trace reflects the first completion as the trace-level result.
        """
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
        trace_group_identifier = get_attr(
            trace_obj,
            "trace_group_identifier",
            "group_identifier",
            "group_id",
        )

        trace_metadata = as_dict(
            value=get_attr(trace_obj, "metadata", "attributes", "tags")
        ) or {}
        trace_metadata = merge_openinference_metadata(metadata=trace_metadata)

        customer_identifier = get_attr(
            trace_obj,
            "customer_identifier",
            "customer_id",
            "user_id",
            "user_identifier",
            "user",
        )
        if customer_identifier is None and isinstance(trace_metadata, dict):
            for key in ("customer_identifier", "customer_id", "user_id", "user"):
                if key in trace_metadata:
                    customer_identifier = trace_metadata.get(key)
                    break
        if customer_identifier is None:
            customer_identifier = self.customer_identifier

        trace_start_time = coerce_datetime(
            value=get_attr(trace_obj, "start_time", "started_at", "start", "start_timestamp")
        )

        root_span = find_root_span(spans=spans)
        root_metadata = extract_span_metadata(span=root_span) if root_span is not None else {}
        if root_metadata:
            root_user_metadata = extract_metadata_payload(metadata=root_metadata)
            if root_user_metadata:
                trace_metadata = {**root_user_metadata, **trace_metadata}

        if not trace_id:
            for span in spans:
                trace_id = get_attr(span, "trace_id", "traceId")
                if trace_id:
                    break
        if not trace_id:
            trace_id = str(uuid.uuid4())

        if not trace_name:
            trace_name = pick_metadata_value(
                root_metadata,
                "graph.node.name",
                "agent.name",
                "crewai.workflow.name",
                "workflow.name",
            )
        if not trace_name and trace_id:
            trace_name = str(trace_id)

        if not workflow_name:
            workflow_name = trace_name

        if session_identifier is None:
            session_identifier = pick_metadata_value(
                root_metadata,
                "session.id",
                "session_id",
                "session",
            )

        if trace_group_identifier is None:
            trace_group_identifier = pick_metadata_value(
                root_metadata,
                "trace_group_identifier",
                "group_identifier",
                "group_id",
            )

        if customer_identifier is None:
            customer_identifier = (
                pick_metadata_value(
                    root_metadata,
                    "user.id",
                    "user_id",
                    "customer_identifier",
                    "customer_id",
                    "user",
                )
                or customer_identifier
            )

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
        span_kind = get_attr(span, "type", "span_type", "kind")
        span_path = get_attr(span, "span_path", "path")

        span_metadata = as_dict(
            value=get_attr(span, "metadata", "attributes", "tags", "data")
        ) or {}
        span_metadata = merge_openinference_metadata(metadata=span_metadata)

        if span_kind is None and isinstance(span_metadata, dict):
            span_kind = pick_metadata_value(
                span_metadata,
                "openinference.span.kind",
                "span.kind",
            )

        if not span_name and isinstance(span_metadata, dict):
            span_name = pick_metadata_value(
                span_metadata,
                "graph.node.name",
                "agent.name",
            )

        if not span_path and isinstance(span_metadata, dict):
            span_path = pick_metadata_value(span_metadata, "graph.node.id")

        span_input = get_attr(span, "input", "input_data", "request", "prompt")
        if span_input is None and isinstance(span_metadata, dict) and "input" in span_metadata:
            span_input = span_metadata.pop("input")
        if span_input is None and isinstance(span_metadata, dict):
            for key in ("input.value", "input_value", "traceloop.entity.input"):
                if key in span_metadata:
                    span_input = span_metadata.get(key)
                    break

        span_output = get_attr(span, "output", "output_data", "response")
        if span_output is None and isinstance(span_metadata, dict) and "output" in span_metadata:
            span_output = span_metadata.pop("output")
        if span_output is None and isinstance(span_metadata, dict):
            for key in ("output.value", "output_value", "traceloop.entity.output"):
                if key in span_metadata:
                    span_output = span_metadata.get(key)
                    break

        input_messages = None
        output_messages = None
        if isinstance(span_metadata, dict):
            input_messages = extract_openinference_messages(
                metadata=span_metadata,
                prefix="llm.input_messages",
            )
            output_messages = extract_openinference_messages(
                metadata=span_metadata,
                prefix="llm.output_messages",
            )
            if span_input is None and input_messages:
                span_input = input_messages
            if span_output is None and output_messages:
                span_output = output_messages

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

        if is_blank_value(value=span_output):
            choice_texts = extract_openinference_choice_texts(metadata=span_metadata)
            if choice_texts:
                span_output = "\n".join(choice_texts)
                if completion_message is None:
                    completion_message = {"role": "assistant", "content": span_output}
            elif output_messages:
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

        model = get_attr(span, "model", "model_name")
        if model is None and isinstance(span_metadata, dict) and "model" in span_metadata:
            model = span_metadata.pop("model")
        if model is None and isinstance(span_metadata, dict):
            model = pick_metadata_value(
                span_metadata,
                "llm.model_name",
                "llm.model",
                "embedding.model_name",
            )

        usage = get_attr(span, "usage", "token_usage")
        if usage is None and isinstance(span_metadata, dict) and "usage" in span_metadata:
            usage = span_metadata.pop("usage")
        usage = as_dict(value=usage)
        if usage is None and isinstance(span_metadata, dict):
            prompt_tokens = span_metadata.get("llm.token_count.prompt")
            completion_tokens = span_metadata.get("llm.token_count.completion")
            total_tokens = span_metadata.get("llm.token_count.total")
            if any(
                value is not None
                for value in (prompt_tokens, completion_tokens, total_tokens)
            ):
                usage = {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                }
                if not (prompt_tokens is not None and completion_tokens is not None):
                    usage["total_tokens"] = total_tokens

        error = get_attr(span, "error", "exception", "err")
        if error is None and isinstance(span_metadata, dict) and "error" in span_metadata:
            error = span_metadata.pop("error")

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
            span_kind=span_kind,
            parent_id=parent_id,
            model=model,
        )

        merged_metadata = {**trace_context.metadata, **(span_metadata or {})}
        trace_hex_id = normalize_trace_id(trace_id=trace_context.trace_id)
        if span_id_map and span_id_str in span_id_map:
            span_hex_id = span_id_map[span_id_str]
        else:
            span_hex_id = normalize_span_id(
                span_id=span_id_str,
                trace_id=trace_context.trace_id,
            )
        if parent_id is not None:
            parent_hex_id = span_id_map.get(str(parent_id)) if span_id_map else None
            if parent_hex_id is None:
                parent_hex_id = normalize_span_id(
                    span_id=str(parent_id),
                    trace_id=trace_context.trace_id,
                )
        else:
            parent_hex_id = None

        if "crewai_trace_id" not in merged_metadata:
            merged_metadata["crewai_trace_id"] = trace_context.trace_id
        if "crewai_span_id" not in merged_metadata:
            merged_metadata["crewai_span_id"] = str(span_id)
        if parent_id and "crewai_parent_id" not in merged_metadata:
            merged_metadata["crewai_parent_id"] = str(parent_id)

        input_value = serialize_value(value=span_input) if span_input is not None else None
        output_value = serialize_value(value=span_output) if span_output is not None else None

        payload = {
            "log_method": LogMethodChoices.TRACING_INTEGRATION.value,
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

        if usage:
            prompt_tokens = usage.get("prompt_tokens")
            if prompt_tokens is None:
                prompt_tokens = usage.get("input_tokens")
            completion_tokens = usage.get("completion_tokens")
            if completion_tokens is None:
                completion_tokens = usage.get("output_tokens")
            total_tokens = usage.get("total_tokens")
            if total_tokens is None:
                total_tokens = usage.get("total")

            payload["prompt_tokens"] = prompt_tokens
            payload["completion_tokens"] = completion_tokens
            # Do not set total_request_tokens when prompt_tokens and completion_tokens
            # are present; Respan backend calculates total from them.
            prompt_tokens_value = coerce_token_count(value=prompt_tokens)
            completion_tokens_value = coerce_token_count(value=completion_tokens)
            has_both = (prompt_tokens_value is not None) and (completion_tokens_value is not None)
            if not has_both and coerce_token_count(value=total_tokens) is not None:
                payload["total_request_tokens"] = total_tokens

        if error:
            payload["error_message"] = str(error)
            payload["status_code"] = 500
        else:
            payload["status_code"] = get_attr(span, "status_code") or 200

        tool_name = (
            get_attr(span, "tool_name", "tool")
            or merged_metadata.get("tool_name")
            or merged_metadata.get("tool.name")
        )
        if tool_name:
            payload["span_tools"] = [str(tool_name)]

        if not payload.get("span_unique_id") and payload.get("trace_unique_id"):
            payload["span_unique_id"] = payload["trace_unique_id"]

        cleaned_payload = clean_payload(payload=payload)

        try:
            validated = RespanFullLogParams(**cleaned_payload)
            return validated.model_dump(mode="json", exclude_none=True)
        except Exception as exc:
            logger.warning(
                "CrewAI span payload failed RespanFullLogParams validation: %s; skipping span",
                exc,
            )
            return None

    def _map_log_type(
        self,
        span_kind: Any,
        parent_id: Optional[str],
        model: Optional[str],
    ) -> str:
        """Map span kind to Respan log type."""
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

    def send(self, payloads: List[Dict[str, Any]]) -> None:
        """Send payloads to Respan endpoint with retries and anti-recursion header.

        Uses inline retry (same policy as respan_sdk.utils.RetryHandler). Prefer
        RetryHandler when respan-sdk exports it for consistency with other exporters.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            RESPAN_DOGFOOD_HEADER: "1",
        }
        max_retries = 3
        retry_delay = 1.0
        backoff_multiplier = 2.0
        max_delay = 30.0
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    url=self.endpoint,
                    json=payloads,
                    headers=headers,
                    timeout=self.timeout,
                )
                if response.status_code == 429:
                    raise RuntimeError("Respan rate limited (429)")
                if response.status_code >= 500:
                    raise RuntimeError(
                        "Respan ingest server error status_code=%s"
                        % (response.status_code,)
                    )
                if 200 <= response.status_code < 300:
                    return
                else:
                    logger.warning(
                        "Respan export failed with status %s: %s",
                        response.status_code,
                        response.text,
                    )
                    return
            except Exception as exc:
                if attempt == max_retries - 1:
                delay = min(
                    retry_delay * (backoff_multiplier ** attempt),
                    max_delay,
                )
                logger.warning(
                    "Respan export retry %s/%s in %.1fs: %s",
                    attempt + 1,
                    max_retries,
                    delay,
                    exc,
                )
                time.sleep(delay)
