import logging
import random
import threading
import time
import warnings
from typing import Any, Dict, List, Optional, Union

import httpx
from agents.tracing.processor_interface import TracingProcessor
from agents.tracing.processors import BatchTraceProcessor, BackendSpanExporter
from agents.tracing.span_data import (
    AgentSpanData,
    CustomSpanData,
    FunctionSpanData,
    GenerationSpanData,
    GuardrailSpanData,
    HandoffSpanData,
    ResponseSpanData,
)
from agents.tracing.spans import Span, SpanImpl
from agents.tracing.traces import Trace
from respan_sdk.constants.llm_logging import (
    LOG_TYPE_AGENT,
    LOG_TYPE_CUSTOM,
    LOG_TYPE_GENERATION,
    LOG_TYPE_GUARDRAIL,
    LOG_TYPE_HANDOFF,
    LOG_TYPE_RESPONSE,
    LOG_TYPE_TOOL,
)
from respan_sdk.respan_types.param_types import RespanTextLogParams
from respan_sdk.utils.serialization import safe_attr, safe_serialize

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — Responses API item types and Chat Completions roles
# ---------------------------------------------------------------------------

ITEM_TYPE_MESSAGE = "message"
ITEM_TYPE_FUNCTION_CALL = "function_call"
ITEM_TYPE_FUNCTION_CALL_OUTPUT = "function_call_output"

CONTENT_TYPE_OUTPUT_TEXT = "output_text"
CONTENT_TYPE_INPUT_TEXT = "input_text"
CONTENT_TYPE_TEXT = "text"

ROLE_SYSTEM = "system"
ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"
ROLE_TOOL = "tool"

TOOL_CALL_TYPE_FUNCTION = "function"

METADATA_KEY_FROM_AGENT = "from_agent"
METADATA_KEY_TO_AGENT = "to_agent"
METADATA_KEY_OUTPUT_TYPE = "output_type"
METADATA_KEY_AGENT_NAME = "agent_name"

GUARDRAIL_TRIGGERED_MSG = "guardrail triggered"

# Responses API item field names (used with safe_attr() to extract values)
FIELD_NAME = "name"
FIELD_ARGUMENTS = "arguments"
FIELD_CALL_ID = "call_id"
FIELD_OUTPUT = "output"

# Usage dict keys — Responses API uses input_tokens/output_tokens,
# Chat Completions API uses prompt_tokens/completion_tokens
USAGE_KEY_INPUT_TOKENS = "input_tokens"
USAGE_KEY_OUTPUT_TOKENS = "output_tokens"
USAGE_KEY_PROMPT_TOKENS = "prompt_tokens"
USAGE_KEY_COMPLETION_TOKENS = "completion_tokens"
USAGE_KEY_INPUT_DETAILS = "input_tokens_details"
USAGE_KEY_CACHED_TOKENS = "cached_tokens"

_CUSTOM_SPAN_PASSTHROUGH_KEYS = ("input", "output", "model", "prompt_tokens", "completion_tokens")


# ---------------------------------------------------------------------------
# Responses API → Chat Completions format converters
#
# The backend renders input/output using Chat Completions message format
# (role + content). These helpers convert Responses API objects into that
# format so the trace UI shows clean System/User/Assistant/Tool messages.
# ---------------------------------------------------------------------------

def _extract_text_from_content(content) -> str:
    """Extract plain text from Responses API content items.

    Content items have type 'output_text' or 'input_text' with a 'text' field.
    Falls back to str() for unknown shapes.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            t = safe_attr(item, "type")
            if t in (CONTENT_TYPE_OUTPUT_TEXT, CONTENT_TYPE_INPUT_TEXT, CONTENT_TYPE_TEXT):
                parts.append(safe_attr(item, "text", ""))
            elif isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                parts.append(item["text"])
        return "\n".join(parts) if parts else ""
    return str(content) if content else ""


def _input_to_prompt_messages(input_items, instructions=None):
    """Convert Responses API input items to Chat Completions prompt_messages.

    Returns (messages_list, user_text_summary).
    """
    messages = []
    user_texts = []

    if instructions:
        messages.append({"role": ROLE_SYSTEM, "content": str(instructions)})

    if isinstance(input_items, str):
        messages.append({"role": ROLE_USER, "content": input_items})
        return messages, input_items

    if not isinstance(input_items, list):
        return messages, str(input_items) if input_items else ""

    for item in input_items:
        item_type = safe_attr(item, "type")

        if item_type == ITEM_TYPE_MESSAGE:
            role = safe_attr(item, "role", ROLE_USER)
            content = safe_attr(item, "content")
            text = _extract_text_from_content(content)
            messages.append({"role": role, "content": text})
            if role == ROLE_USER:
                user_texts.append(text)

        elif item_type == ITEM_TYPE_FUNCTION_CALL:
            name = safe_attr(item, FIELD_NAME)
            arguments = safe_attr(item, FIELD_ARGUMENTS, "")
            call_id = safe_attr(item, FIELD_CALL_ID)
            tc = {"type": TOOL_CALL_TYPE_FUNCTION, "function": {"name": name, "arguments": arguments}}
            if call_id:
                tc["id"] = call_id
            messages.append({"role": ROLE_ASSISTANT, "tool_calls": [tc]})

        elif item_type == ITEM_TYPE_FUNCTION_CALL_OUTPUT:
            call_id = safe_attr(item, FIELD_CALL_ID)
            output = safe_attr(item, FIELD_OUTPUT, "")
            msg = {"role": ROLE_TOOL, "content": str(output)}
            if call_id:
                msg["tool_call_id"] = call_id
            messages.append(msg)

        elif isinstance(item, dict) and "role" in item:
            role = item.get("role", ROLE_USER)
            content = item.get("content", "")
            text = _extract_text_from_content(content) if content else ""
            msg = {"role": role, "content": text}
            messages.append(msg)
            if role == ROLE_USER:
                user_texts.append(text)

        elif isinstance(item, str):
            messages.append({"role": ROLE_USER, "content": item})
            user_texts.append(item)

    return messages, "\n".join(user_texts) if user_texts else ""


def _output_to_completion(output_items):
    """Convert Responses API output items to Chat Completions format.

    Returns (completion_message, tool_calls_list, tool_names, assistant_text).
    """
    tool_calls = []
    tool_names = []
    text_parts = []

    if not output_items:
        return None, tool_calls, tool_names, ""

    items = output_items if isinstance(output_items, list) else [output_items]
    for item in items:
        item_type = safe_attr(item, "type")

        if item_type in (CONTENT_TYPE_OUTPUT_TEXT, CONTENT_TYPE_TEXT):
            text_parts.append(safe_attr(item, "text", ""))

        elif item_type == ITEM_TYPE_MESSAGE:
            content = safe_attr(item, "content")
            text_parts.append(_extract_text_from_content(content))

        elif item_type == ITEM_TYPE_FUNCTION_CALL:
            name = safe_attr(item, FIELD_NAME)
            arguments = safe_attr(item, FIELD_ARGUMENTS, "")
            call_id = safe_attr(item, FIELD_CALL_ID)
            tc = {"type": TOOL_CALL_TYPE_FUNCTION, "function": {"name": name, "arguments": arguments}}
            if call_id:
                tc["id"] = call_id
            tool_calls.append(tc)
            if name:
                tool_names.append(name)

    assistant_text = "\n".join(text_parts)

    completion = {"role": ROLE_ASSISTANT}
    if tool_calls and not assistant_text:
        completion["tool_calls"] = tool_calls
    elif tool_calls:
        completion["content"] = assistant_text
        completion["tool_calls"] = tool_calls
    else:
        completion["content"] = assistant_text

    return completion, tool_calls, tool_names, assistant_text


def _extract_token_count(primary, fallback_dict, primary_key, fallback_key):
    """Extract a token count from an SDK usage object, falling back to a raw dict.

    Uses identity checks (``is not None``) to correctly handle ``0`` as a
    valid token count.
    """
    val = safe_attr(primary, primary_key)
    if val is not None:
        return int(val)
    if isinstance(fallback_dict, dict):
        val = fallback_dict.get(fallback_key)
        if val is not None:
            return int(val)
    return None


def _response_data_to_respan_log(
    data: RespanTextLogParams, span_data: ResponseSpanData
) -> None:
    """Convert ResponseSpanData to Respan log format.

    Converts Responses API objects into Chat Completions message format so
    the trace UI renders clean System/User/Assistant/Tool messages with
    proper tool call display.
    """
    data.span_name = span_data.type
    data.log_type = LOG_TYPE_RESPONSE

    instructions = None
    if span_data.response and hasattr(span_data.response, "instructions"):
        instructions = span_data.response.instructions

    if span_data.input:
        prompt_messages, _user_text = _input_to_prompt_messages(
            input_items=span_data.input, instructions=instructions,
        )
        data.prompt_messages = prompt_messages
        data.input = prompt_messages
    elif instructions:
        prompt_messages = [{"role": ROLE_SYSTEM, "content": str(instructions)}]
        data.prompt_messages = prompt_messages
        data.input = prompt_messages

    if span_data.response:
        if hasattr(span_data.response, "model"):
            data.model = span_data.response.model

        if hasattr(span_data.response, "output") and span_data.response.output:
            completion, tool_calls, tool_names, _text = _output_to_completion(
                span_data.response.output
            )
            if completion:
                data.completion_message = completion
                data.output = completion

            if tool_calls:
                data.tool_calls = tool_calls
                data.has_tool_calls = True
                data.span_tools = tool_names

        if hasattr(span_data.response, "tools") and span_data.response.tools:
            data.tools = safe_serialize(span_data.response.tools)

        if hasattr(span_data.response, "usage") and span_data.response.usage:
            usage = span_data.response.usage
            raw = safe_serialize(usage)
            data.usage = raw

            pt = _extract_token_count(
                primary=usage, fallback_dict=raw,
                primary_key=USAGE_KEY_INPUT_TOKENS, fallback_key=USAGE_KEY_INPUT_TOKENS,
            )
            ct = _extract_token_count(
                primary=usage, fallback_dict=raw,
                primary_key=USAGE_KEY_OUTPUT_TOKENS, fallback_key=USAGE_KEY_OUTPUT_TOKENS,
            )
            if pt is not None:
                data.prompt_tokens = pt
            if ct is not None:
                data.completion_tokens = ct

            details = safe_attr(usage, USAGE_KEY_INPUT_DETAILS)
            if details is None and isinstance(raw, dict):
                details = raw.get(USAGE_KEY_INPUT_DETAILS)
            if details:
                cached = safe_attr(details, USAGE_KEY_CACHED_TOKENS)
                if cached is None and isinstance(details, dict):
                    cached = details.get(USAGE_KEY_CACHED_TOKENS)
                if cached is not None:
                    data.prompt_cache_hit_tokens = int(cached)


def _function_data_to_respan_log(
    data: RespanTextLogParams, span_data: FunctionSpanData
) -> None:
    """Convert FunctionSpanData to Respan log format."""
    data.span_name = span_data.name
    data.log_type = LOG_TYPE_TOOL
    data.input = safe_serialize(span_data.input)
    data.output = safe_serialize(span_data.output)
    data.span_tools = [span_data.name]


def _generation_data_to_respan_log(
    data: RespanTextLogParams, span_data: GenerationSpanData
) -> None:
    """Convert GenerationSpanData to Respan log format.

    Extracts prompt_tokens/completion_tokens from the usage dict so the
    backend can calculate cost, regardless of whether the usage dict uses
    Chat Completions keys or Responses API keys.
    """
    data.span_name = span_data.type
    data.log_type = LOG_TYPE_GENERATION
    data.model = span_data.model
    data.input = safe_serialize(span_data.input)
    data.output = safe_serialize(span_data.output)
    if span_data.usage:
        raw = safe_serialize(span_data.usage) if not isinstance(span_data.usage, dict) else span_data.usage
        data.usage = raw
        if isinstance(raw, dict):
            pt = raw.get(USAGE_KEY_PROMPT_TOKENS)
            if pt is None:
                pt = raw.get(USAGE_KEY_INPUT_TOKENS)
            ct = raw.get(USAGE_KEY_COMPLETION_TOKENS)
            if ct is None:
                ct = raw.get(USAGE_KEY_OUTPUT_TOKENS)
            if pt is not None:
                data.prompt_tokens = int(pt)
            if ct is not None:
                data.completion_tokens = int(ct)


def _handoff_data_to_respan_log(
    data: RespanTextLogParams, span_data: HandoffSpanData
) -> None:
    """Convert HandoffSpanData to Respan log format."""
    data.span_name = span_data.type
    data.log_type = LOG_TYPE_HANDOFF
    data.span_handoffs = [f"{span_data.from_agent} -> {span_data.to_agent}"]
    data.metadata = {
        METADATA_KEY_FROM_AGENT: span_data.from_agent,
        METADATA_KEY_TO_AGENT: span_data.to_agent,
    }


def _custom_data_to_respan_log(
    data: RespanTextLogParams, span_data: CustomSpanData
) -> None:
    """Convert CustomSpanData to Respan log format."""
    data.span_name = span_data.name
    data.log_type = LOG_TYPE_CUSTOM
    data.metadata = span_data.data

    for key in _CUSTOM_SPAN_PASSTHROUGH_KEYS:
        if key in span_data.data:
            setattr(data, key, span_data.data[key])


def _agent_data_to_respan_log(
    data: RespanTextLogParams, span_data: AgentSpanData
) -> None:
    """Convert AgentSpanData to Respan log format."""
    data.span_name = span_data.name
    data.log_type = LOG_TYPE_AGENT
    data.span_workflow_name = span_data.name

    if span_data.tools:
        data.span_tools = span_data.tools
    if span_data.handoffs:
        data.span_handoffs = span_data.handoffs

    data.metadata = {
        METADATA_KEY_OUTPUT_TYPE: span_data.output_type,
        METADATA_KEY_AGENT_NAME: span_data.name,
    }


def _guardrail_data_to_respan_log(
    data: RespanTextLogParams, span_data: GuardrailSpanData
) -> None:
    """Convert GuardrailSpanData to Respan log format."""
    data.span_name = f"guardrail:{span_data.name}"
    data.log_type = LOG_TYPE_GUARDRAIL
    data.has_warnings = span_data.triggered
    if span_data.triggered:
        data.warnings_dict = {
            f"guardrail:{span_data.name}": GUARDRAIL_TRIGGERED_MSG
        }


# ---------------------------------------------------------------------------
# Public conversion function — used by both RespanSpanExporter and
# LocalSpanCollector so conversion logic is defined once.
# ---------------------------------------------------------------------------

def convert_to_respan_log(
    item: Union[Trace, Span[Any]],
    default_model: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Convert an OpenAI Agents SDK Trace or Span to a Respan log dict.

    Handles all 7 span data types (response, function, generation, handoff,
    custom, agent, guardrail) plus root Trace objects.

    Args:
        item: A Trace or Span object from the OpenAI Agents SDK.
        default_model: Fallback model name for spans that don't carry their
            own model (agent, tool, handoff, custom, guardrail, root trace).

    Returns:
        A JSON-serializable dict matching ``RespanTextLogParams``, or ``None``
        if the item type is unrecognised.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Pydantic serializer warnings",
            category=UserWarning,
        )

        if isinstance(item, Trace):
            return RespanTextLogParams(
                trace_unique_id=item.trace_id,
                span_unique_id=item.trace_id,
                span_name=item.name,
                log_type=LOG_TYPE_AGENT,
                model=default_model,
            ).model_dump(mode="json")

        if isinstance(item, SpanImpl):
            parent_id = item.parent_id or item.trace_id
            data = RespanTextLogParams(
                trace_unique_id=item.trace_id,
                span_unique_id=item.span_id,
                span_parent_id=parent_id,
                start_time=item.started_at,
                timestamp=item.ended_at,
                error_bit=1 if item.error else 0,
                status_code=400 if item.error else 200,
                error_message=str(item.error) if item.error else None,
                model=default_model,
            )
            data.latency = (data.timestamp - data.start_time).total_seconds()
            try:
                if isinstance(item.span_data, ResponseSpanData):
                    _response_data_to_respan_log(data, item.span_data)
                elif isinstance(item.span_data, FunctionSpanData):
                    _function_data_to_respan_log(data, item.span_data)
                elif isinstance(item.span_data, GenerationSpanData):
                    _generation_data_to_respan_log(data, item.span_data)
                elif isinstance(item.span_data, HandoffSpanData):
                    _handoff_data_to_respan_log(data, item.span_data)
                elif isinstance(item.span_data, CustomSpanData):
                    _custom_data_to_respan_log(data, item.span_data)
                elif isinstance(item.span_data, AgentSpanData):
                    _agent_data_to_respan_log(data, item.span_data)
                elif isinstance(item.span_data, GuardrailSpanData):
                    _guardrail_data_to_respan_log(data, item.span_data)
                else:
                    logger.warning(f"Unknown span data type: {item.span_data}")
                    return None
                return data.model_dump(mode="json")
            except Exception as e:
                logger.error(
                    f"Error converting span data of {item.span_data} to Respan log: {e}"
                )
                return None

    return None


# ---------------------------------------------------------------------------
# LocalSpanCollector — in-process span collection for self-hosted use
# ---------------------------------------------------------------------------

class LocalSpanCollector(TracingProcessor):
    """Thread-safe, in-process span collector for self-hosted deployments.

    Instead of sending spans over HTTP, this processor converts them using
    the same ``convert_to_respan_log`` logic and stores them in memory keyed
    by ``trace_id``.  After an agent run completes, call
    ``pop_trace(trace_id)`` to retrieve (and remove) the spans for that run.

    Register globally once at application startup::

        from agents import set_trace_processors
        collector = LocalSpanCollector(default_model="gpt-4o")
        set_trace_processors([collector])

    Then after each ``Runner.run_streamed()``::

        spans = collector.pop_trace(trace_id)
        for span_data in spans:
            log_request(...)
    """

    def __init__(self, default_model: Optional[str] = None) -> None:
        self._traces: Dict[str, List[Dict[str, Any]]] = {}
        self._lock = threading.Lock()
        self._default_model = default_model

    # -- TracingProcessor interface -----------------------------------------

    def on_trace_start(self, trace: Trace) -> None:
        pass

    def on_trace_end(self, trace: Trace) -> None:
        data = convert_to_respan_log(
            item=trace, default_model=self._default_model,
        )
        if data:
            with self._lock:
                self._traces.setdefault(trace.trace_id, []).insert(0, data)

    def on_span_start(self, span: Span[Any]) -> None:
        pass

    def on_span_end(self, span: Span[Any]) -> None:
        data = convert_to_respan_log(
            item=span, default_model=self._default_model,
        )
        if data:
            trace_id = span.trace_id if hasattr(span, "trace_id") else None
            if trace_id:
                with self._lock:
                    self._traces.setdefault(trace_id, []).append(data)

    def shutdown(self) -> None:
        with self._lock:
            self._traces.clear()

    def force_flush(self) -> None:
        pass  # All conversion is synchronous — nothing to flush.

    # -- Public API ---------------------------------------------------------

    def pop_trace(self, trace_id: str) -> List[Dict[str, Any]]:
        """Retrieve and remove all collected spans for a trace.

        Returns an empty list if no spans were collected for *trace_id*.
        Thread-safe — safe to call from any request thread.
        """
        with self._lock:
            return self._traces.pop(trace_id, [])


class RespanSpanExporter(BackendSpanExporter):
    """Custom exporter for Respan that handles all span types."""

    def __init__(
        self,
        api_key: Union[str, None] = None,
        organization: Union[str, None] = None,
        project: Union[str, None] = None,
        endpoint: str = "https://api.respan.ai/api/v1/traces/ingest",
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
        default_model: Optional[str] = None,
    ):
        """Initialize the Respan exporter.

        Args:
            api_key: The API key for authentication.
            organization: The organization ID.
            project: The project ID.
            endpoint: The HTTP endpoint to which traces/spans are posted.
            max_retries: Maximum number of retries upon failures.
            base_delay: Base delay (in seconds) for the first backoff.
            max_delay: Maximum delay (in seconds) for backoff growth.
            default_model: Fallback model name for spans that don't carry
                their own (agent, tool, handoff, etc.).
        """
        super().__init__(
            api_key=api_key,
            organization=organization,
            project=project,
            endpoint=endpoint,
            max_retries=max_retries,
            base_delay=base_delay,
            max_delay=max_delay,
        )
        self._default_model = default_model

    def set_endpoint(self, endpoint: str) -> None:
        """Dynamically change the endpoint URL."""
        self.endpoint = endpoint
        logger.info(f"Respan exporter endpoint changed to: {endpoint}")

    def _respan_export(
        self, item: Union[Trace, Span[Any]]
    ) -> Optional[Dict[str, Any]]:
        """Delegates to the module-level ``convert_to_respan_log`` function."""
        return convert_to_respan_log(
            item=item, default_model=self._default_model,
        )

    def export(self, items: list[Union[Trace, Span[Any]]]) -> None:
        """Export traces and spans to the Respan backend."""
        if not items:
            return

        if not self.api_key:
            logger.warning("API key is not set, skipping trace export")
            return

        data = [self._respan_export(item) for item in items]
        data = [item for item in data if item]

        if not data:
            return

        payload = {"data": data}

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "OpenAI-Beta": "traces=v1",
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
                    logger.debug(f"Exported {len(data)} items to Respan")
                    return

                if 400 <= response.status_code < 500:
                    logger.error(
                        f"Respan client error {response.status_code}: {response.text}"
                    )
                    return

                logger.warning(f"Server error {response.status_code}, retrying.")
            except httpx.RequestError as exc:
                logger.warning(f"Request failed: {exc}")

            if attempt >= self.max_retries:
                logger.error("Max retries reached, giving up on this batch.")
                return

            sleep_time = delay + random.uniform(0, 0.1 * delay)
            time.sleep(sleep_time)
            delay = min(delay * 2, self.max_delay)


class RespanTraceProcessor(BatchTraceProcessor):
    """A processor that uses RespanSpanExporter to send traces and spans to Respan."""

    def __init__(
        self,
        api_key: Union[str, None] = None,
        organization: Union[str, None] = None,
        project: Union[str, None] = None,
        endpoint: str = "https://api.respan.ai/api/openai/v1/traces/ingest",
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
        max_queue_size: int = 8192,
        max_batch_size: int = 128,
        schedule_delay: float = 5.0,
        export_trigger_ratio: float = 0.7,
        default_model: Optional[str] = None,
    ):
        """Initialize the Respan processor.

        Args:
            api_key: The API key for authentication.
            organization: The organization ID.
            project: The project ID.
            endpoint: The HTTP endpoint to which traces/spans are posted.
            max_retries: Maximum number of retries upon failures.
            base_delay: Base delay (in seconds) for the first backoff.
            max_delay: Maximum delay (in seconds) for backoff growth.
            max_queue_size: The maximum number of spans to store in the queue.
            max_batch_size: The maximum number of spans to export in a single batch.
            schedule_delay: The delay between checks for new spans to export.
            export_trigger_ratio: The ratio of the queue size at which we will trigger an export.
            default_model: Fallback model name for spans that don't carry
                their own (agent, tool, handoff, etc.).
        """
        exporter = RespanSpanExporter(
            api_key=api_key,
            organization=organization,
            project=project,
            endpoint=endpoint,
            max_retries=max_retries,
            base_delay=base_delay,
            max_delay=max_delay,
            default_model=default_model,
        )

        super().__init__(
            exporter=exporter,
            max_queue_size=max_queue_size,
            max_batch_size=max_batch_size,
            schedule_delay=schedule_delay,
            export_trigger_ratio=export_trigger_ratio,
        )

        self._respan_exporter = exporter

    def set_endpoint(self, endpoint: str) -> None:
        """Dynamically change the endpoint URL."""
        self._respan_exporter.set_endpoint(endpoint)
