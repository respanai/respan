import logging
import random
import threading
import time
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

logger = logging.getLogger(__name__)


def _serialize(obj):
    """Recursively convert *obj* to plain JSON-serializable Python types.

    Pydantic v2 defers serializer construction for models with forward
    references (``MockValSer``).  The deferred rebuild uses
    ``sys._getframe(5)`` which fails in shallow call stacks (Celery
    workers, asyncio callbacks).  By never storing foreign Pydantic
    model instances on ``RespanTextLogParams``, we sidestep the issue
    entirely — ``data.model_dump()`` only ever sees plain types.
    """
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize(item) for item in obj]
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump(mode="json")
        except Exception:
            # MockValSer or other serializer failure — extract public attrs
            return {
                k: _serialize(v)
                for k, v in vars(obj).items()
                if not k.startswith("_")
            }
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    return str(obj)


# Internal helper functions for converting span data to Respan log format
def _response_data_to_respan_log(
    data: RespanTextLogParams, span_data: ResponseSpanData
) -> None:
    """Convert ResponseSpanData — pass raw usage through, BE parses."""
    data.span_name = span_data.type
    data.log_type = LOG_TYPE_RESPONSE
    data.input = _serialize(span_data.input)

    if span_data.response:
        if hasattr(span_data.response, "model"):
            data.model = span_data.response.model
        if hasattr(span_data.response, "usage") and span_data.response.usage:
            data.usage = _serialize(span_data.response.usage)
        if hasattr(span_data.response, "output"):
            data.output = _serialize(span_data.response.output)


def _function_data_to_respan_log(
    data: RespanTextLogParams, span_data: FunctionSpanData
) -> None:
    """Convert FunctionSpanData to Respan log format."""
    data.span_name = span_data.name
    data.log_type = LOG_TYPE_TOOL
    data.input = _serialize(span_data.input)
    data.output = _serialize(span_data.output)
    data.span_tools = [span_data.name]


def _generation_data_to_respan_log(
    data: RespanTextLogParams, span_data: GenerationSpanData
) -> None:
    """Convert GenerationSpanData — pass raw usage through, BE parses."""
    data.span_name = span_data.type
    data.log_type = LOG_TYPE_GENERATION
    data.model = span_data.model
    data.input = _serialize(span_data.input)
    data.output = _serialize(span_data.output)
    if span_data.usage:
        data.usage = span_data.usage


def _handoff_data_to_respan_log(
    data: RespanTextLogParams, span_data: HandoffSpanData
) -> None:
    """Convert HandoffSpanData to Respan log format."""
    data.span_name = span_data.type
    data.log_type = LOG_TYPE_HANDOFF
    data.span_handoffs = [f"{span_data.from_agent} -> {span_data.to_agent}"]
    data.metadata = {
        "from_agent": span_data.from_agent,
        "to_agent": span_data.to_agent,
    }


def _custom_data_to_respan_log(
    data: RespanTextLogParams, span_data: CustomSpanData
) -> None:
    """Convert CustomSpanData to Respan log format."""
    data.span_name = span_data.name
    data.log_type = LOG_TYPE_CUSTOM
    data.metadata = span_data.data

    for key in ["input", "output", "model", "prompt_tokens", "completion_tokens"]:
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
        "output_type": span_data.output_type,
        "agent_name": span_data.name,
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
            f"guardrail:{span_data.name}": "guardrail triggered"
        }


# ---------------------------------------------------------------------------
# Public conversion function — used by both RespanSpanExporter and
# LocalSpanCollector so conversion logic is defined once.
# ---------------------------------------------------------------------------

def convert_to_respan_log(
    item: Union[Trace, Span[Any]],
) -> Optional[Dict[str, Any]]:
    """Convert an OpenAI Agents SDK Trace or Span to a Respan log dict.

    Handles all 7 span data types (response, function, generation, handoff,
    custom, agent, guardrail) plus root Trace objects.

    Args:
        item: A Trace or Span object from the OpenAI Agents SDK.

    Returns:
        A JSON-serializable dict matching ``RespanTextLogParams``, or ``None``
        if the item type is unrecognised.
    """
    if isinstance(item, Trace):
        return RespanTextLogParams(
            trace_unique_id=item.trace_id,
            span_unique_id=item.trace_id,
            span_name=item.name,
            log_type=LOG_TYPE_AGENT,
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
        collector = LocalSpanCollector()
        set_trace_processors([collector])

    Then after each ``Runner.run_streamed()``::

        spans = collector.pop_trace(trace_id)
        for span_data in spans:
            log_request(...)
    """

    def __init__(self) -> None:
        self._traces: Dict[str, List[Dict[str, Any]]] = {}
        self._lock = threading.Lock()

    # -- TracingProcessor interface -----------------------------------------

    def on_trace_start(self, trace: Trace) -> None:
        pass

    def on_trace_end(self, trace: Trace) -> None:
        data = convert_to_respan_log(trace)
        if data:
            with self._lock:
                self._traces.setdefault(trace.trace_id, []).insert(0, data)

    def on_span_start(self, span: Span[Any]) -> None:
        pass

    def on_span_end(self, span: Span[Any]) -> None:
        data = convert_to_respan_log(span)
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
    """
    Custom exporter for Respan that handles all span types and allows for dynamic endpoint configuration.
    """

    def __init__(
        self,
        api_key: Union[str, None] = None,
        organization: Union[str, None] = None,
        project: Union[str, None] = None,
        endpoint: str = "https://api.respan.ai/api/v1/traces/ingest",
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
    ):
        """
        Initialize the Respan exporter.

        Args:
            api_key: The API key for authentication. Defaults to os.environ["OPENAI_API_KEY"] if not provided.
            organization: The organization ID. Defaults to os.environ["OPENAI_ORG_ID"] if not provided.
            project: The project ID. Defaults to os.environ["OPENAI_PROJECT_ID"] if not provided.
            endpoint: The HTTP endpoint to which traces/spans are posted.
            max_retries: Maximum number of retries upon failures.
            base_delay: Base delay (in seconds) for the first backoff.
            max_delay: Maximum delay (in seconds) for backoff growth.
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

    def set_endpoint(self, endpoint: str) -> None:
        """
        Dynamically change the endpoint URL.

        Args:
            endpoint: The new endpoint URL to use for exporting spans.
        """
        self.endpoint = endpoint
        logger.info(f"Respan exporter endpoint changed to: {endpoint}")

    def _respan_export(
        self, item: Union[Trace, Span[Any]]
    ) -> Optional[Dict[str, Any]]:
        """Process different span types and extract all JSON serializable attributes.

        Delegates to the module-level ``convert_to_respan_log`` function.
        """
        return convert_to_respan_log(item)

    def export(self, items: list[Union[Trace, Span[Any]]]) -> None:
        """
        Export traces and spans to the Respan backend.

        Args:
            items: List of Trace or Span objects to export.
        """
        if not items:
            return

        if not self.api_key:
            logger.warning("API key is not set, skipping trace export")
            return

        # Process each item with our custom exporter
        data = [self._respan_export(item) for item in items]
        # Filter out None values
        data = [item for item in data if item]

        if not data:
            return

        payload = {"data": data}

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "OpenAI-Beta": "traces=v1",
        }

        # Exponential backoff loop
        attempt = 0
        delay = self.base_delay
        while True:
            attempt += 1
            try:
                response = self._client.post(
                    url=self.endpoint, headers=headers, json=payload
                )

                # If the response is successful, break out of the loop
                if response.status_code < 300:
                    logger.debug(f"Exported {len(data)} items to Respan")
                    return

                # If the response is a client error (4xx), we won't retry
                if 400 <= response.status_code < 500:
                    logger.error(
                        f"Respan client error {response.status_code}: {response.text}"
                    )
                    return

                # For 5xx or other unexpected codes, treat it as transient and retry
                logger.warning(f"Server error {response.status_code}, retrying.")
            except httpx.RequestError as exc:
                # Network or other I/O error, we'll retry
                logger.warning(f"Request failed: {exc}")

            # If we reach here, we need to retry or give up
            if attempt >= self.max_retries:
                logger.error("Max retries reached, giving up on this batch.")
                return

            # Exponential backoff + jitter
            sleep_time = delay + random.uniform(0, 0.1 * delay)  # 10% jitter
            time.sleep(sleep_time)
            delay = min(delay * 2, self.max_delay)


class RespanTraceProcessor(BatchTraceProcessor):
    """
    A processor that uses RespanSpanExporter to send traces and spans to Respan.
    """

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
    ):
        """
        Initialize the Respan processor.

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
        """

        # Create the exporter
        exporter = RespanSpanExporter(
            api_key=api_key,
            organization=organization,
            project=project,
            endpoint=endpoint,
            max_retries=max_retries,
            base_delay=base_delay,
            max_delay=max_delay,
        )

        # Initialize the BatchTraceProcessor with our exporter
        super().__init__(
            exporter=exporter,
            max_queue_size=max_queue_size,
            max_batch_size=max_batch_size,
            schedule_delay=schedule_delay,
            export_trigger_ratio=export_trigger_ratio,
        )

        # Store the exporter for easy access
        self._respan_exporter = exporter

    def set_endpoint(self, endpoint: str) -> None:
        """
        Dynamically change the endpoint URL.

        Args:
            endpoint: The new endpoint URL to use for exporting spans.
        """
        self._respan_exporter.set_endpoint(endpoint)
