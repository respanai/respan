"""Respan Google ADK Converter - Convert Google ADK traces to Respan payload format."""
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

from respan_instrumentation_google_adk.types import TraceContext
from respan_instrumentation_google_adk.utils import (
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
from respan_sdk.constants.llm_logging import (
    LOG_TYPE_AGENT,
    LOG_TYPE_GENERATION,
    LOG_TYPE_MAP,
    LOG_TYPE_TASK,
    LOG_TYPE_TOOL,
    LOG_TYPE_WORKFLOW,
)
from respan_sdk.respan_types.log_types import RespanFullLogParams

logger = logging.getLogger(__name__)

# Mapping from ADK span names to Respan log types
_ADK_LOG_TYPE_MAP: Dict[str, str] = {
    "invocation": LOG_TYPE_WORKFLOW,
    "agent_run": LOG_TYPE_AGENT,
    "invoke_agent": LOG_TYPE_AGENT,
    "call_llm": LOG_TYPE_GENERATION,
    "generate_content": LOG_TYPE_GENERATION,
    "execute_tool": LOG_TYPE_TOOL,
}

# ADK root run label — replaced with `adk_agent_name` when available (see
# `_resolve_root_workflow_span_name`).
_GENERIC_ADK_ROOT_SPAN_NAMES: frozenset[str] = frozenset({"invocation"})


class AdkSpanConverter:
    """Convert Google ADK traces/spans to Respan payload format.

    The converter accepts:
    - a trace object with a .spans collection
    - a dict with a "spans" field
    - a list of span objects/dicts
    """

    def __init__(
        self,
        environment: Optional[str] = None,
        customer_identifier: Optional[str] = None,
    ) -> None:
        self.environment = environment or "production"
        self.customer_identifier = customer_identifier

    def convert(self, trace_or_spans: Any) -> List[Dict[str, Any]]:
        """Convert trace or spans to Respan payload format."""
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
            self._propagate_trace_input(payloads=payloads)
            self._resolve_root_workflow_span_name(payloads=payloads)
        return payloads

    def _propagate_trace_output(self, payloads: List[Dict[str, Any]]) -> None:
        """Propagate output from generation spans to workflow/agent/task spans."""
        trace_output: Optional[str] = None
        for payload in payloads:
            output_value = payload.get("output")
            if is_blank_value(value=output_value):
                continue
            log_type = payload.get("log_type")
            if log_type == LOG_TYPE_GENERATION:
                trace_output = output_value
                break
            if trace_output is None:
                trace_output = output_value
        if trace_output is not None:
            for payload in payloads:
                if is_blank_value(value=payload.get("output")) and payload.get(
                    "log_type"
                ) in (
                    LOG_TYPE_WORKFLOW,
                    LOG_TYPE_AGENT,
                    LOG_TYPE_TASK,
                ):
                    payload["output"] = trace_output

    def _propagate_trace_input(self, payloads: List[Dict[str, Any]]) -> None:
        """Propagate input from generation spans to workflow/agent spans."""
        trace_input: Optional[str] = None
        for payload in payloads:
            input_value = payload.get("input")
            if is_blank_value(value=input_value):
                continue
            log_type = payload.get("log_type")
            if log_type == LOG_TYPE_GENERATION:
                trace_input = input_value
                break
            if trace_input is None:
                trace_input = input_value
        if trace_input is not None:
            for payload in payloads:
                if is_blank_value(value=payload.get("input")) and payload.get(
                    "log_type"
                ) in (
                    LOG_TYPE_WORKFLOW,
                    LOG_TYPE_AGENT,
                    LOG_TYPE_TASK,
                ):
                    payload["input"] = trace_input

    def _resolve_root_workflow_span_name(self, payloads: List[Dict[str, Any]]) -> None:
        """Replace generic ADK root labels (e.g. ``invocation``) with a real agent name.

        Prefer ``metadata.adk_agent_name`` on the root workflow payload (from
        ``gen_ai.agent.name`` on that span), otherwise the first agent child's
        name. ``_map_log_type`` already ran on the raw OTel name; only the
        emitted ``span_name`` changes here.
        """
        root_payload: Optional[Dict[str, Any]] = None
        for payload in payloads:
            if (
                payload.get("log_type") == LOG_TYPE_WORKFLOW
                and payload.get("parent_id") is None
                and payload.get("span_parent_id") is None
            ):
                root_payload = payload
                break

        if root_payload is None:
            return

        current = root_payload.get("span_name")
        if str(current) not in _GENERIC_ADK_ROOT_SPAN_NAMES:
            return

        root_meta = root_payload.get("metadata") or {}
        agent_name: Optional[str] = root_meta.get("adk_agent_name")
        if not agent_name:
            for payload in payloads:
                if payload.get("log_type") != LOG_TYPE_AGENT:
                    continue
                meta = payload.get("metadata") or {}
                agent_name = meta.get("adk_agent_name")
                if agent_name:
                    break

        if agent_name:
            root_payload["span_name"] = agent_name

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
        # If root span has no agent name (e.g. bare 'invocation' span),
        # look at child spans for the first agent name
        if not trace_name:
            for span in spans:
                child_attrs = as_dict(
                    value=get_attr(span, "attributes", "metadata", "tags", "data")
                ) or {}
                child_agent_name = child_attrs.get("gen_ai.agent.name")
                if child_agent_name:
                    trace_name = str(child_agent_name)
                    break
        if not trace_name:
            root_name = get_attr(root_span, "name")
            if root_name and str(root_name) != "invocation":
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
        conversation_id = root_attrs.get("gen_ai.conversation.id")
        if not session_identifier and conversation_id:
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

        # Token usage (OpenLLMetry names first, then OTel GenAI semconv fallbacks)
        prompt_tokens_raw = (
            span_attrs.get("gen_ai.usage.prompt_tokens")
            or span_attrs.get("gen_ai.usage.input_tokens")
        )
        completion_tokens_raw = (
            span_attrs.get("gen_ai.usage.completion_tokens")
            or span_attrs.get("gen_ai.usage.output_tokens")
        )

        # Messages (try OpenLLMetry names first, then OTel GenAI semconv,
        # then ADK-specific attributes)
        input_messages = extract_genai_messages(
            attributes=span_attrs, key="gen_ai.prompt"
        ) or extract_genai_messages(
            attributes=span_attrs, key="gen_ai.input.messages"
        )
        output_messages = extract_genai_messages(
            attributes=span_attrs, key="gen_ai.completion"
        ) or extract_genai_messages(
            attributes=span_attrs, key="gen_ai.output.messages"
        )

        # ADK stores full request/response in gcp.vertex.agent.llm_request/response
        if not input_messages or not output_messages:
            input_msgs, output_msgs = self._extract_adk_messages(span_attrs)
            if not input_messages and input_msgs:
                input_messages = input_msgs
            if not output_messages and output_msgs:
                output_messages = output_msgs

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
        max_tokens = (
            span_attrs.get("gen_ai.request.max_tokens")
            or span_attrs.get("gen_ai.request.max_output_tokens")
        )
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

        trace_hex_id = normalize_trace_id(trace_id=trace_context.trace_id)
        if span_id_map and span_id_str in span_id_map:
            span_hex_id = span_id_map[span_id_str]
        else:
            span_hex_id = normalize_span_id(
                span_id=span_id_str,
                trace_id=trace_context.trace_id,
            )
        parent_hex_id = None
        if parent_id:
            if span_id_map:
                parent_hex_id = span_id_map.get(str(parent_id))
            if parent_hex_id is None:
                parent_hex_id = normalize_span_id(
                    span_id=str(parent_id),
                    trace_id=trace_context.trace_id,
                )

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
            payload["usage"] = {
                "prompt_tokens": prompt_tokens_value or 0,
                "completion_tokens": completion_tokens_value or 0,
            }

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

    @staticmethod
    def _extract_adk_messages(
        span_attrs: Dict[str, Any],
    ) -> Tuple[Optional[List[Dict[str, Any]]], Optional[List[Dict[str, Any]]]]:
        """Extract input/output messages from ADK-specific attributes.

        Google ADK stores full LLM request/response JSON in
        ``gcp.vertex.agent.llm_request`` and ``gcp.vertex.agent.llm_response``.
        """
        import json as _json

        input_messages: Optional[List[Dict[str, Any]]] = None
        output_messages: Optional[List[Dict[str, Any]]] = None

        # Parse llm_request for input messages
        llm_request_raw = span_attrs.get("gcp.vertex.agent.llm_request")
        if llm_request_raw:
            try:
                req = _json.loads(llm_request_raw) if isinstance(llm_request_raw, str) else llm_request_raw
                if isinstance(req, dict):
                    # ADK request has contents array with role/parts
                    contents = req.get("contents") or []
                    if contents and isinstance(contents, list):
                        msgs = []
                        for c in contents:
                            if isinstance(c, dict):
                                role = c.get("role", "user")
                                # Normalize Gemini "model" role to OpenAI "assistant"
                                if role == "model":
                                    role = "assistant"
                                parts = c.get("parts") or []
                                text_parts = []
                                for p in parts:
                                    if isinstance(p, dict) and p.get("text"):
                                        text_parts.append(p["text"])
                                if text_parts:
                                    msgs.append({"role": role, "content": "\n".join(text_parts)})
                        if msgs:
                            input_messages = msgs
            except Exception:
                pass

        # Parse llm_response for output messages
        llm_response_raw = span_attrs.get("gcp.vertex.agent.llm_response")
        if llm_response_raw:
            try:
                resp = _json.loads(llm_response_raw) if isinstance(llm_response_raw, str) else llm_response_raw
                if isinstance(resp, dict):
                    # ADK response has content.parts with text
                    content = resp.get("content") or {}
                    if isinstance(content, dict):
                        role = content.get("role", "model")
                        # Normalize "model" role to "assistant" for OpenAI format
                        if role == "model":
                            role = "assistant"
                        parts = content.get("parts") or []
                        text_parts = []
                        for p in parts:
                            if isinstance(p, dict) and p.get("text"):
                                text_parts.append(p["text"])
                        if text_parts:
                            output_messages = [{"role": role, "content": "\n".join(text_parts)}]
            except Exception:
                pass

        return input_messages, output_messages

    def _map_log_type(
        self,
        span_name: Optional[str],
        parent_id: Optional[str],
        model: Optional[str],
    ) -> str:
        """Map ADK span name to Respan log type."""
        if span_name:
            # Check ADK-specific mapping first (exact match)
            adk_type = _ADK_LOG_TYPE_MAP.get(span_name)
            if adk_type:
                return adk_type
            # ADK span names can include suffixes (e.g. "invoke_agent greeting_agent",
            # "generate_content gemini-2.5-flash") — check the prefix
            span_prefix = span_name.split(" ")[0]
            if span_prefix != span_name:
                adk_type = _ADK_LOG_TYPE_MAP.get(span_prefix)
                if adk_type:
                    return adk_type
            # Fall back to generic LOG_TYPE_MAP
            name_lower = span_name.lower()
            for key, value in LOG_TYPE_MAP.items():
                if key in name_lower:
                    return value
        if model:
            return LOG_TYPE_GENERATION
        if parent_id is None:
            return LOG_TYPE_WORKFLOW
        return LOG_TYPE_TASK
