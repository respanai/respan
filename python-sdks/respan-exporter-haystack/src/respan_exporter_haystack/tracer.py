"""Respan Tracer implementation for Haystack content tracing."""

import time
import uuid
from typing import Any, Dict, List, Optional

from haystack import logging
from haystack.tracing import Span, Tracer

from respan_exporter_haystack.logger import RespanLogger
from respan_exporter_haystack.utils.config_utils import resolve_platform_logs_url
from respan_exporter_haystack.utils.tracing_utils import format_span_for_api

logger = logging.getLogger(__name__)


import threading

class RespanTracer(Tracer):
    """
    Custom tracer implementation for Respan that integrates with Haystack's tracing system.
    
    This tracer captures all pipeline operations and sends them to Respan for monitoring.
    It implements the Haystack Tracer protocol to seamlessly integrate with Haystack pipelines.
    """

    def __init__(
        self,
        name: str,
        api_key: str,
        base_url: str,
        metadata: Optional[Dict[str, Any]] = None,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
        platform_url: Optional[str] = None,
        timeout: float = 10.0,
    ):
        """
        Initialize the Respan tracer.
        
        Args:
            name: Name of the trace/pipeline
            api_key: Respan API key
            base_url: Respan API base URL
            metadata: Additional metadata to attach to traces
            max_retries: Maximum number of attempts for sending traces
            base_delay: Base delay in seconds between retries
            max_delay: Maximum delay in seconds between retries
            platform_url: Optional URL for the logs UI (defaults derived from base_url)
            timeout: Timeout in seconds for HTTP requests
        """
        self.name = name
        self.api_key = api_key
        self.base_url = base_url
        self.metadata = metadata or {}
        self.platform_logs_base = resolve_platform_logs_url(base_url=base_url, platform_url=platform_url)
        
        # Initialize the logger for sending data
        self.kw_logger = RespanLogger(
            api_key=api_key,
            base_url=base_url,
            max_retries=max_retries,
            base_delay=base_delay,
            max_delay=max_delay,
            timeout=timeout,
        )
        
        # Trace state
        self.trace_id = str(uuid.uuid4())
        self.spans: Dict[str, Dict[str, Any]] = {}
        self.span_objects: Dict[str, "RespanSpan"] = {}  # Registry of active span objects
        self.completed_spans: List[Dict[str, Any]] = []  # Collect spans for batch submission
        self.trace_url: Optional[str] = None
        self.start_time = None
        self.pipeline_finished = False

    def trace(self, operation_name: str, tags: Optional[Dict[str, Any]] = None, parent_span: Optional["RespanSpan"] = None) -> "RespanSpan":
        """
        Start a new span for tracing an operation.
        
        Args:
            operation_name: Name of the operation being traced
            tags: Additional tags/metadata for the span
            parent_span: Optional parent span for nested operations
            
        Returns:
            A new RespanSpan instance
        """
        if self.start_time is None:
            self.start_time = time.time()
            
        span_id = str(uuid.uuid4())
        parent_id = parent_span.span_id if parent_span else None
        
        # Keep track of component name if available, often added via tag later but initializing it 
        span = RespanSpan(
            tracer=self,
            operation_name=operation_name,
            span_id=span_id,
            trace_id=self.trace_id,
            tags=tags or {},
            parent_id=parent_id,
        )
        
        self.spans[span_id] = {
            "operation_name": operation_name,
            "span_id": span_id,
            "trace_id": self.trace_id,
            "parent_id": parent_id,
            "tags": tags or {},
            "start_time": time.time(),
            "data": {},
        }
        
        self.span_objects[span_id] = span
        
        return span

    def current_span(self) -> Optional["RespanSpan"]:
        """Get the current active span."""
        # Return reference to existing span object if available
        if self.span_objects:
            span_id = list(self.span_objects.keys())[-1]
            return self.span_objects[span_id]
        return None

    def finalize_span(
        self,
        span_id: str,
        output: Any = None,
        error: Optional[Exception] = None,
    ):
        """
        Finalize a span and collect it for batch submission.
        
        Args:
            span_id: ID of the span to finalize
            output: Output data from the operation
            error: Exception if operation failed
        """
        if span_id not in self.spans:
            return
            
        span_data = self.spans[span_id]
        end_time = time.time()
        span_data["end_time"] = end_time
        span_data["latency"] = end_time - span_data["start_time"]
        
        if output is not None:
            span_data["output"] = output
            
        if error is not None:
            span_data["error"] = str(error)
            span_data["status_code"] = 500
        else:
            span_data["status_code"] = 200
        
        # Format span for Respan traces API
        try:
            formatted_span = format_span_for_api(
                span_data=span_data,
                workflow_name=self.name,
                workflow_metadata=self.metadata,
            )
            if formatted_span is not None:  # Skip None (filtered spans)
                self.completed_spans.append(formatted_span)
            
            # If this is the root span (no parent), send the entire trace
            # Components nested in a pipeline will have a parent_id, but if run isolated,
            # they might have no parent. We send the trace when the top-level span finishes.
            is_root = span_data.get("parent_id") is None
            if is_root:
                logger.debug(f"Root span complete - sending trace with {len(self.completed_spans)} spans")
                self.send_trace()
        except Exception as e:
            logger.warning(f"Failed to format span: {e}")
            
        # Clean up references to prevent memory leaks in long-running pipelines
        # BUT DO NOT remove from self.spans here since they are needed for formatting
        # later or when formatting the root span that sends the batch.
        # Spans are cleared in `self.send_trace()`
        if span_id in self.span_objects:
            del self.span_objects[span_id]

    def send_trace(self):
        """Send all collected spans to Respan as a batch."""
        if not self.completed_spans:
            logger.debug("No spans to send")
            return
        
        if self.pipeline_finished:
            logger.debug("Trace already sent")
            return
            
        spans_to_send = list(self.completed_spans)
        
        def _send():
            try:
                logger.debug(f"Sending trace with {len(spans_to_send)} spans to Respan")
                response = self.kw_logger.send_trace(spans=spans_to_send)
                
                if response:
                    logger.debug(f"Trace sent successfully: {response}")
                    # Extract trace info from response
                    if "trace_ids" in response and response["trace_ids"]:
                        trace_id = response["trace_ids"][0]
                        self.trace_url = f"{self.platform_logs_base}?trace_id={trace_id}"
            except Exception as e:
                logger.warning(f"Failed to send trace to Respan: {e}")

        # Use a daemon thread to avoid blocking the main execution
        threading.Thread(target=_send, daemon=True).start()
        
        self.pipeline_finished = True
        self.completed_spans.clear()  # Free memory after sending
        self.spans.clear() # Free span lookup memory

    def get_trace_url(self) -> Optional[str]:
        """Get the URL to view this trace in Respan dashboard."""
        if self.trace_url:
            return self.trace_url
        return f"{self.platform_logs_base}?trace_id={self.trace_id}"


class RespanSpan(Span):
    """
    Span implementation for Respan tracing.
    
    Represents a single operation in the pipeline execution.
    """

    def __init__(
        self,
        tracer: RespanTracer,
        operation_name: str,
        span_id: str,
        trace_id: str,
        tags: Dict[str, Any],
        parent_id: Optional[str] = None,
    ):
        """Initialize a span."""
        self.tracer = tracer
        self.operation_name = operation_name
        self.span_id = span_id
        self.trace_id = trace_id
        self.tags = tags
        self.parent_id = parent_id
        self._is_finished = False

    def set_tag(self, key: str, value: Any) -> "RespanSpan":
        """Set a tag on the span."""
        if self.span_id in self.tracer.spans:
            if "tags" not in self.tracer.spans[self.span_id]:
                self.tracer.spans[self.span_id]["tags"] = {}
            self.tracer.spans[self.span_id]["tags"][key] = value
        return self

    def set_tags(self, tags: Dict[str, Any]) -> "RespanSpan":
        """Set multiple tags on the span."""
        for key, value in tags.items():
            self.set_tag(key=key, value=value)
        return self

    def set_content_tag(self, key: str, value: Any) -> "RespanSpan":
        """Set content data on the span."""
        if self.span_id in self.tracer.spans:
            if "data" not in self.tracer.spans[self.span_id]:
                self.tracer.spans[self.span_id]["data"] = {}
            self.tracer.spans[self.span_id]["data"][key] = value
        return self

    def raw_span(self) -> Any:
        """Get the raw span data."""
        return self.tracer.spans.get(self.span_id)

    def finish(self, output: Any = None, error: Optional[Exception] = None):
        """Finish the span and send to Respan."""
        if not self._is_finished:
            try:
                self.tracer.finalize_span(span_id=self.span_id, output=output, error=error)
            except Exception as e:
                logger.warning(f"Error finalizing span {self.span_id}: {e}")
            finally:
                self._is_finished = True

    def __enter__(self) -> "RespanSpan":
        """Context manager entry."""
        # NOTE: Do NOT use RespanSpan as a context manager natively if you don't control the flow,
        # but since Haystack controls it we just yield it back as the active context if requested.
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        error = exc_val if exc_type is not None else None
        self.finish(error=error)
