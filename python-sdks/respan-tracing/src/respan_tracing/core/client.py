from contextlib import contextmanager
from typing import Any, Callable, Dict, Generator, List, Optional, Union
from opentelemetry import trace, context as context_api
from opentelemetry.trace.span import Span
from opentelemetry.trace import Status, StatusCode

from respan_sdk import FilterParamDict
from respan_sdk.constants.span_attributes import RESPAN_METADATA, RESPAN_SPAN_ATTRIBUTES_MAP
from respan_sdk.respan_types.span_types import SpanLink
from respan_sdk.respan_types.param_types import RespanParams
from pydantic import ValidationError

from respan_tracing.core.tracer import RespanTracer
from respan_tracing.processors import SpanBuffer
from respan_tracing.processors.base import _active_span_buffer
from respan_tracing.utils.logging import get_respan_logger
from respan_tracing.utils.span_setup import setup_span, cleanup_span, LinksParam

from respan_tracing.constants.generic_constants import LOGGER_NAME_CLIENT

logger = get_respan_logger(LOGGER_NAME_CLIENT)


class RespanClient:
    """
    Client for interacting with the current trace/span context.
    Provides a clean API for getting and updating trace information.
    """
    
    def __init__(self):
        """Initialize the client. Uses the singleton tracer instance."""
        self._tracer = RespanTracer()
    
    def get_current_span(self) -> Optional[Span]:
        """
        Get the current active span.
        
        Returns:
            The current active span, or None if no span is active.
        """
        if not self._tracer.is_enabled or not RespanTracer.is_initialized():
            logger.warning("Respan Telemetry not initialized or disabled.")
            return None
            
        current_span = trace.get_current_span()
        
        if not isinstance(current_span, Span):
            return None
            
        return current_span
    
    def get_current_trace_id(self) -> Optional[str]:
        """
        Get the current trace ID.
        
        Returns:
            The current trace ID as a string, or None if no active span.
        """
        span = self.get_current_span()
        if span:
            return format(span.get_span_context().trace_id, '032x')
        return None
    
    def get_current_span_id(self) -> Optional[str]:
        """
        Get the current span ID.
        
        Returns:
            The current span ID as a string, or None if no active span.
        """
        span = self.get_current_span()
        if span:
            return format(span.get_span_context().span_id, '016x')
        return None
    
    def update_current_span(
        self, 
        respan_params: Optional[Union[Dict[str, Any], RespanParams]] = None,
        attributes: Optional[Dict[str, Any]] = None,
        status: Optional[Union[Status, StatusCode]] = None,
        status_description: Optional[str] = None,
        name: Optional[str] = None
    ) -> bool:
        """
        Update the current active span with new information.
        
        Args:
            respan_params: Respan-specific parameters to set as span attributes
            attributes: Generic attributes to set on the span
            status: Status to set on the span (Status object or StatusCode)
            status_description: Description for the status
            name: New name for the span
            
        Returns:
            True if the span was updated successfully, False otherwise.
        """
        span = self.get_current_span()
        if not span:
            logger.warning("No active span found. Cannot update span.")
            return False
        
        try:
            # Update span name if provided
            if name:
                span.update_name(name)
            
            # Set Respan-specific attributes
            if respan_params:
                self._set_respan_attributes(span, respan_params)
            
            # Set generic attributes
            if attributes:
                for key, value in attributes.items():
                    try:
                        span.set_attribute(key, value)
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Failed to set attribute {key}={value}: {str(e)}")
            
            # Set status
            if status is not None:
                if isinstance(status, StatusCode):
                    span.set_status(Status(status, status_description))
                else:
                    span.set_status(status)
            
            return True
            
        except Exception as e:
            logger.exception(f"Failed to update span: {str(e)}")
            return False
    
    def _set_respan_attributes(
        self, 
        span: Span, 
        respan_params: Union[Dict[str, Any], RespanParams]
    ):
        """Set Respan-specific attributes on a span."""
        try:
            # Validate parameters
            validated_params = (
                respan_params 
                if isinstance(respan_params, RespanParams) 
                else RespanParams.model_validate(respan_params)
            )
            
            # Set attributes based on the mapping
            for key, value in validated_params.model_dump(mode="json").items():
                attr_key = RESPAN_SPAN_ATTRIBUTES_MAP.get(key)
                if attr_key and attr_key != RESPAN_METADATA:
                    try:
                        span.set_attribute(attr_key, value)
                    except (ValueError, TypeError) as e:
                        logger.warning(
                            f"Failed to set span attribute {attr_key}={value}: {str(e)}"
                        )

                # Handle metadata specially — expand to per-key attributes
                if attr_key == RESPAN_METADATA and isinstance(value, dict):
                    for metadata_key, metadata_value in value.items():
                        try:
                            span.set_attribute(
                                f"{RESPAN_METADATA}.{metadata_key}", 
                                metadata_value
                            )
                        except (ValueError, TypeError) as e:
                            logger.warning(
                                f"Failed to set metadata attribute {metadata_key}={metadata_value}: {str(e)}"
                            )
                            
        except ValidationError as e:
            logger.warning(f"Failed to validate Respan params: {str(e.errors(include_url=False))}")
        except Exception as e:
            logger.exception(f"Unexpected error setting Respan attributes: {str(e)}")
    
    def add_event(
        self, 
        name: str, 
        attributes: Optional[Dict[str, Any]] = None,
        timestamp: Optional[int] = None
    ) -> bool:
        """
        Add an event to the current span.
        
        Args:
            name: Name of the event
            attributes: Optional attributes for the event
            timestamp: Optional timestamp (nanoseconds since epoch)
            
        Returns:
            True if the event was added successfully, False otherwise.
        """
        span = self.get_current_span()
        if not span:
            logger.warning("No active span found. Cannot add event.")
            return False
        
        try:
            span.add_event(name, attributes or {}, timestamp)
            return True
        except Exception as e:
            logger.exception(f"Failed to add event {name}: {str(e)}")
            return False
    
    def record_exception(
        self, 
        exception: Exception,
        attributes: Optional[Dict[str, Any]] = None,
        timestamp: Optional[int] = None,
        escaped: bool = False
    ) -> bool:
        """
        Record an exception on the current span.
        
        Args:
            exception: The exception to record
            attributes: Optional attributes for the exception
            timestamp: Optional timestamp (nanoseconds since epoch)
            escaped: Whether the exception escaped the span
            
        Returns:
            True if the exception was recorded successfully, False otherwise.
        """
        span = self.get_current_span()
        if not span:
            logger.warning("No active span found. Cannot record exception.")
            return False
        
        try:
            span.record_exception(exception, attributes, timestamp, escaped)
            # Also set the span status to error
            span.set_status(Status(StatusCode.ERROR, str(exception)))
            return True
        except Exception as e:
            logger.exception(f"Failed to record exception: {str(e)}")
            return False
    
    def get_context_value(self, key: str) -> Any:
        """
        Get a value from the current OpenTelemetry context.
        
        Args:
            key: The context key to retrieve
            
        Returns:
            The context value, or None if not found.
        """
        return context_api.get_value(key)
    
    def set_context_value(self, key: str, value: Any) -> bool:
        """
        Set a value in the current OpenTelemetry context.
        
        Args:
            key: The context key to set
            value: The value to set
            
        Returns:
            True if the context was updated successfully, False otherwise.
        """
        try:
            context_api.attach(context_api.set_value(key, value))
            return True
        except Exception as e:
            logger.exception(f"Failed to set context value {key}={value}: {str(e)}")
            return False
    
    def is_recording(self) -> bool:
        """
        Check if the current span is recording.
        
        Returns:
            True if the current span is recording, False otherwise.
        """
        span = self.get_current_span()
        return span.is_recording() if span else False
    
    def flush(self):
        """Force flush all pending spans."""
        self._tracer.flush()
    
    def get_tracer(self):
        """
        Get the OpenTelemetry tracer for creating custom spans.
        
        This provides access to the underlying tracer for advanced span creation,
        allowing you to manually create spans when the @workflow/@task decorators
        are not sufficient.
        
        Returns:
            opentelemetry.trace.Tracer: The OpenTelemetry tracer instance.
        
        Example:
            ```python
            from respan_tracing import get_client
            
            client = get_client()
            tracer = client.get_tracer()
            
            # Create custom spans manually
            with tracer.start_as_current_span("my_operation") as span:
                span.set_attribute("custom.attribute", "value")
                # Your code here
                pass
            ```
        """
        return self._tracer.get_tracer()

    @contextmanager
    def start_span(
        self,
        name: str,
        kind: str = "task",
        processors: Optional[Union[str, List[str]]] = None,
        export_filter: Optional[FilterParamDict] = None,
        links: LinksParam = None,
        version: Optional[int] = None,
    ) -> Generator[Span, None, None]:
        """
        Context manager for creating spans with full Respan metadata.

        This is the imperative equivalent of the @workflow/@task/@agent/@tool
        decorators. Use it when the span name or kind must be determined at
        runtime (e.g., dynamic task loops).

        Handles all the same concerns as the decorators:
        - ``processors`` attribute for FilteringSpanProcessor routing
        - ``TRACELOOP_ENTITY_NAME`` attribute and OTel context propagation
        - ``TRACELOOP_WORKFLOW_NAME`` inheritance for child spans
        - Span links (static list or callable)
        - Error recording and status propagation

        Args:
            name: Span name (equivalent to the decorator ``name`` parameter).
            kind: Span kind — ``"workflow"``, ``"task"``, ``"agent"``, or
                ``"tool"``. Controls context propagation behavior (workflow/agent
                kinds propagate entity name to children). Defaults to ``"task"``.
            processors: Processor name(s) to route this span to (e.g.,
                ``"dogfood"`` or ``["dogfood", "debug"]``).
            export_filter: Optional filter dict for conditional export.
            links: Span links — a list of ``SpanLink`` objects or a callable
                returning one.
            version: Optional version number.

        Yields:
            The active ``Span`` object.

        Example:
            ```python
            from respan_tracing import get_client

            client = get_client()

            # Imperative workflow span with processor routing
            with client.start_span("workflow_execution", kind="workflow", processors="dogfood") as span:
                span.set_attribute("workflow_count", 3)

                # Child spans inherit workflow_name automatically
                with client.start_span("step_1", kind="task", processors="dogfood") as task_span:
                    task_span.set_attribute("task_type", "condition")
                    # ... execute task ...
            ```
        """
        if not self._tracer.is_enabled or not RespanTracer.is_initialized():
            logger.warning("Respan Telemetry not initialized or disabled.")
            yield None
            return

        # Continuation mode: when inside a SpanBuffer with a parent span,
        # skip creating wrapper spans — the parent already exists. Child spans
        # (from @task/@workflow decorators via setup_span) attach directly to
        # the parent context. Only client.start_span() is affected — decorators
        # use setup_span() directly and are not skipped.
        active_buffer = _active_span_buffer.get(None)
        if active_buffer and active_buffer._parent_span_id:
            yield None
            return

        span, ctx_token, entity_name_token, entity_path_token = setup_span(
            entity_name=name,
            span_kind=kind,
            version=version,
            processors=processors,
            export_filter=export_filter,
            links=links,
        )

        try:
            yield span
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise
        finally:
            cleanup_span(span, ctx_token, entity_name_token, entity_path_token)

    def get_span_buffer(
        self,
        trace_id: str,
        parent_trace_id: Optional[str] = None,
        parent_span_id: Optional[str] = None,
    ) -> SpanBuffer:
        """
        Get an OpenTelemetry-compliant context manager for buffering spans with manual export control.

        This enables batch buffering of multiple spans for a single trace, allowing you to:
        - Create spans asynchronously (after execution completes)
        - Buffer multiple spans and export with single API call
        - Manually control when spans are exported
        - Inspect spans before exporting
        - Continue a pre-existing trace (workflow pause/resume)

        Spans created within the SpanBuffer context are isolated and only buffered
        in that buffer's local queue, without affecting other spans in the application.

        Args:
            trace_id: Trace ID for the spans being buffered
            parent_trace_id: Optional OTel trace ID (hex string) to continue.
                When provided with parent_span_id, spans created in this buffer
                inherit the parent trace, making them children of the specified
                span. Used for workflow pause/resume to keep all spans in one trace.
            parent_span_id: Optional OTel span ID (hex string) of the parent span.
                Must be provided together with parent_trace_id.

        Returns:
            SpanBuffer: Context manager for span buffering

        Example:
            ```python
            from respan_tracing import SpanLink, get_client

            client = get_client()

            # Buffer spans for batch export
            collected_spans = []

            with client.get_span_buffer("trace-123") as buffer:
                # Create multiple spans - they go to local queue
                buffer.create_span("step1", {"status": "completed", "latency": 100})
                buffer.create_span("step2", {"status": "completed", "latency": 200})
                buffer.create_span(
                    "step3",
                    {"status": "completed", "latency": 150},
                    links=[
                        SpanLink(
                            trace_id="0123456789abcdef0123456789abcdef",
                            span_id="0123456789abcdef",
                            attributes={"link.type": "resume"},
                        )
                    ],
                )

                # Optional: inspect before extracting
                print(f"Buffered {buffer.get_span_count()} spans")

                # Extract spans before context exits
                collected_spans = buffer.get_all_spans()

            # Export all spans as a single batch
            client.export_spans(collected_spans)
            ```

        Example (trace continuation for workflow resume):
            ```python
            # On resume, continue the pre-pause trace:
            with client.get_span_buffer(
                trace_id="new-run-id",
                parent_trace_id="abcdef1234567890abcdef1234567890",
                parent_span_id="abcdef1234567890",
            ) as buffer:
                # Spans inherit parent_trace_id — same trace as pre-pause
                buffer.create_span("resumed_step", {"status": "running"})
            ```
        """
        if not self._tracer.is_enabled or not RespanTracer.is_initialized():
            logger.warning("Respan Telemetry not initialized or disabled.")
            raise RuntimeError("Respan Telemetry not initialized or disabled.")

        return SpanBuffer(
            trace_id=trace_id,
            tracer_provider=self._tracer.tracer_provider,
            parent_trace_id=parent_trace_id,
            parent_span_id=parent_span_id,
        )
    
    def process_spans(self, spans) -> bool:
        """
        Process spans through the configured processors.
        
        This sends spans through the standard OTEL processing pipeline,
        allowing all registered processors to filter, transform, and export.
        
        Args:
            spans: List of ReadableSpan objects, or SpanBuffer instance
        
        Returns:
            True if processing was successful, False otherwise
        
        Example:
            ```python
            from respan_tracing import get_client
            
            client = get_client()
            
            # Collect spans in buffer
            collected_spans = []
            
            with client.get_span_buffer("trace-123") as buffer:
                buffer.create_span("step1", {"status": "completed"})
                buffer.create_span("step2", {"status": "completed"})
                
                # Extract spans before context exits
                collected_spans = buffer.get_all_spans()
            
            # Process anywhere, anytime through processors
            success = client.process_spans(collected_spans)
            ```
        
        Example (conditional export):
            ```python
            # Collect spans
            collected_spans = []
            
            with client.get_span_buffer("trace-123") as buffer:
                buffer.create_span("task", {"success": True})
                # Extract spans before context exits
                collected_spans = buffer.get_all_spans()
            
            # Later, elsewhere in code - decide whether to process
            if should_export_trace():
                client.process_spans(collected_spans)  # Process when ready
            else:
                # Just don't export (spans will be garbage collected)
                pass
            ```
        """
        from opentelemetry.sdk.trace import ReadableSpan
        from opentelemetry.sdk.trace.export import SpanExportResult
        
        # Handle SpanBuffer instance
        if hasattr(spans, 'process_spans'):
            # It's a SpanBuffer - use its process method
            count = spans.process_spans(self._tracer.tracer_provider)
            return count > 0
        else:
            # It's a list of spans - process through tracer provider
            span_list = spans
            trace_id = 'unknown'
            
            if not span_list:
                logger.debug(f"[Client] No spans to process for trace {trace_id}")
                return True
            
            logger.info(f"[Client] Processing {len(span_list)} spans for trace {trace_id}")
            
            try:
                # Send through processor pipeline
                if hasattr(self._tracer.tracer_provider, '_active_span_processor'):
                    for span in span_list:
                        self._tracer.tracer_provider._active_span_processor.on_end(span)
                    
                    logger.info(f"[Client] Successfully processed {len(span_list)} spans")
                    return True
                else:
                    logger.error("[Client] No active span processor found")
                    return False
                
            except Exception as e:
                logger.exception(f"[Client] Exception during processing: {e}")
                return False
