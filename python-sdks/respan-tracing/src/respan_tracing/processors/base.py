import json
import logging
from contextvars import ContextVar
from typing import Any, Callable, Dict, List, Optional

from opentelemetry import context, trace
from opentelemetry.sdk.trace import SpanProcessor, ReadableSpan
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor, SpanExporter, SpanExportResult
from opentelemetry.context import Context
from opentelemetry.semconv_ai import SpanAttributes

from respan_sdk.constants.llm_logging import (
    LOG_TYPE_AGENT,
    LOG_TYPE_TASK,
    LOG_TYPE_TOOL,
)
from respan_sdk.respan_types.span_types import RespanSpanAttributes
from respan_tracing.constants.generic_constants import SDK_PREFIX
from respan_tracing.constants.tracing import EXPORT_FILTER_ATTR
from respan_tracing.constants.context_constants import (
    TRACE_GROUP_ID_KEY,
    PARAMS_KEY
)
from respan_tracing.filters import evaluate_export_filter
from respan_tracing.utils.preprocessing.span_processing import is_processable_span
from respan_tracing.utils.context import get_entity_path

logger = logging.getLogger(__name__)


def _infer_respan_log_type_for_span(span: ReadableSpan) -> Optional[str]:
    """Infer a valid Respan log type for spans emitted by external instrumentors."""
    existing_log_type = span.attributes.get(RespanSpanAttributes.LOG_TYPE.value)
    if existing_log_type:
        return str(existing_log_type)

    if span.attributes.get("gen_ai.agent.name"):
        return LOG_TYPE_AGENT

    if span.attributes.get("gen_ai.tool.name"):
        return LOG_TYPE_TOOL

    if (
        span.attributes.get("tools")
        and not span.attributes.get(SpanAttributes.LLM_REQUEST_TYPE)
        and span.attributes.get("gen_ai.operation.name") is None
    ):
        return LOG_TYPE_TASK

    return None


class RespanSpanProcessor:
    """
    Custom span processor that wraps the underlying processor and adds
    Respan-specific metadata to spans.
    """

    def __init__(
        self,
        processor: SpanProcessor,
        span_postprocess_callback: Optional[Callable[[ReadableSpan], None]] = None,
    ):
        self.processor = processor
        self.span_postprocess_callback = span_postprocess_callback

        # Store original on_end method if we have a callback
        if span_postprocess_callback:
            self.original_on_end = processor.on_end
            processor.on_end = self._wrapped_on_end

    def on_start(self, span, parent_context: Optional[Context] = None):
        """Called when a span is started - add Respan metadata"""
        # Check if this span is being created within an entity context
        # If so, add the entityPath attribute so it gets preserved by our filtering
        entity_path = get_entity_path(parent_context)  # Use active context like JS version
        if entity_path and not span.attributes.get(SpanAttributes.TRACELOOP_SPAN_KIND):
            # This is an auto-instrumentation span within an entity context
            # Add the entityPath attribute so it doesn't get filtered out
            logger.debug(
                f"[Respan Debug] Adding entityPath to auto-instrumentation span: {span.name} (entityPath: {entity_path})"
            )
            span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_PATH, entity_path)

        # Add workflow name if present in context
        workflow_name = context.get_value(SpanAttributes.TRACELOOP_ENTITY_NAME)
        if workflow_name:
            span.set_attribute(SpanAttributes.TRACELOOP_WORKFLOW_NAME, workflow_name)

        # Add entity path if present in context (for redundancy)
        entity_path_from_context = context.get_value(SpanAttributes.TRACELOOP_ENTITY_PATH)
        if entity_path_from_context:
            span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_PATH, entity_path_from_context)

        # Add trace group identifier if present
        trace_group_id = context.get_value(TRACE_GROUP_ID_KEY)
        if trace_group_id:
            span.set_attribute(
                RespanSpanAttributes.RESPAN_TRACE_GROUP_ID.value, trace_group_id
            )

        # Add custom parameters if present
        respan_params = context.get_value(PARAMS_KEY)
        if respan_params and isinstance(respan_params, dict):
            for key, value in respan_params.items():
                span.set_attribute(f"{SDK_PREFIX}.{key}", value)

        inferred_log_type = _infer_respan_log_type_for_span(span=span)
        if inferred_log_type:
            span.set_attribute(
                RespanSpanAttributes.LOG_TYPE.value,
                inferred_log_type,
            )

        # Call original processor's on_start
        self.processor.on_start(span, parent_context)

    def on_end(self, span: ReadableSpan):
        """Called when a span ends - filter spans based on Respan attributes"""
        # Apply standard filtering logic (processable span check)
        if not is_processable_span(span):
            logger.debug(f"[Respan Debug] Skipping filtered span: {span.name}")
            return

        # Apply export_filter if present on this span
        filter_json = span.attributes.get(EXPORT_FILTER_ATTR) if span.attributes else None
        if filter_json:
            try:
                export_filter = json.loads(filter_json)
                span_data = dict(span.attributes) if span.attributes else {}
                span_data["status_code"] = span.status.status_code.name if span.status else "UNSET"
                span_data["name"] = span.name
                if not evaluate_export_filter(span_data=span_data, export_filter=export_filter):
                    logger.debug(f"[Respan Debug] Export filter dropped span: {span.name}")
                    return
            except (json.JSONDecodeError, Exception) as e:
                # Fail-open: invalid/malformed filters still export the span.
                # This matches the codebase's infrastructure fail-open principle —
                # a broken filter should not silently drop telemetry data.
                logger.warning(f"[Respan Debug] Failed to evaluate export filter, exporting span anyway: {e}")

        self.processor.on_end(span)

    def _wrapped_on_end(self, span: ReadableSpan):
        """Wrapped on_end method that calls custom callback first"""
        if self.span_postprocess_callback:
            self.span_postprocess_callback(span)
        self.original_on_end(span)

    def shutdown(self):
        """Shutdown the underlying processor"""
        return self.processor.shutdown()

    def force_flush(self, timeout_millis: int = 30000):
        """Force flush the underlying processor"""
        return self.processor.force_flush(timeout_millis)


# ============================================================================
# Buffering Span Processor - OTEL-compliant span buffering functionality
# ============================================================================


# Context variable to track the active SpanBuffer for the current context
_active_span_buffer: ContextVar[Optional['SpanBuffer']] = ContextVar(
    'active_span_buffer', default=None
)


class BufferingSpanProcessor(SpanProcessor):
    """
    OpenTelemetry-compliant span processor that can buffer spans when requested.
    
    This processor checks if there's an active SpanBuffer in the current context.
    If there is, spans go to that buffer's local queue. Otherwise, spans are
    passed through to the original processor for normal export.
    
    This follows OpenTelemetry patterns by using a single processor that can
    conditionally buffer spans based on context, rather than swapping processors.
    """
    
    def __init__(self, original_processor: SpanProcessor):
        """
        Initialize the buffering processor.
        
        Args:
            original_processor: The original processor to fall back to when
                              no active SpanBuffer is present
        """
        self.original_processor = original_processor
    
    def on_start(self, span, parent_context: Optional[Context] = None):
        """
        Called when a span starts.
        
        Forward to original processor (needed for proper span initialization).
        """
        self.original_processor.on_start(span, parent_context)
    
    def on_end(self, span: ReadableSpan):
        """
        Called when a span ends.
        
        If there's an active SpanBuffer in the current context, route the span
        to its local queue. Otherwise, pass through to the original processor.
        
        Args:
            span: The span that ended
        """
        # Check if there's an active SpanBuffer in this context
        buffer = _active_span_buffer.get()
        
        if buffer is not None and buffer._is_buffering:
            # Route to the buffer's local queue
            logger.debug(
                f"[SpanBuffer] Buffering span '{span.name}' "
                f"for trace {buffer.trace_id}"
            )
            buffer._local_queue.append(span)
        else:
            # No active buffer - use original processor (normal export)
            self.original_processor.on_end(span)
    
    def shutdown(self):
        """Shutdown the processor."""
        return self.original_processor.shutdown()
    
    def force_flush(self, timeout_millis: int = 30000):
        """Force flush the processor."""
        return self.original_processor.force_flush(timeout_millis)


class FilteringSpanProcessor(SpanProcessor):
    """
    OpenTelemetry-compliant span processor that filters spans based on attributes.
    
    This processor checks span attributes against filter criteria and only exports
    spans that match. This is the standard OTEL pattern for selective exporting.
    
    Example:
        # Only export spans with exporter="debug" attribute
        processor = FilteringSpanProcessor(
            exporter=debug_exporter,
            filter_fn=lambda span: span.attributes.get("exporter") == "debug"
        )
    """
    
    def __init__(
        self,
        exporter: SpanExporter,
        filter_fn: Optional[Callable[[ReadableSpan], bool]] = None,
        is_batching_enabled: bool = True,
        span_postprocess_callback: Optional[Callable[[ReadableSpan], None]] = None,
    ):
        """
        Initialize the filtering processor.
        
        Args:
            exporter: The SpanExporter to use for matching spans
            filter_fn: Optional function to determine if a span should be exported.
                      If None, all spans are exported.
            is_batching_enabled: Whether to use batch processing
            span_postprocess_callback: Optional callback for span postprocessing
        """
        
        self.filter_fn = filter_fn or (lambda span: True)
        
        # Create base processor
        if is_batching_enabled:
            base_processor = BatchSpanProcessor(exporter)
        else:
            base_processor = SimpleSpanProcessor(exporter)
        
        # Wrap with Respan processor for metadata injection
        self.processor = RespanSpanProcessor(base_processor, span_postprocess_callback)
    
    def on_start(self, span, parent_context: Optional[Context] = None):
        """Called when a span starts."""
        # Always call on_start for proper initialization
        self.processor.on_start(span, parent_context)
    
    def on_end(self, span: ReadableSpan):
        """Called when a span ends - only export if filter matches."""
        if self.filter_fn(span):
            logger.debug(f"[FilteringProcessor] Exporting span: {span.name}")
            self.processor.on_end(span)
        else:
            logger.debug(f"[FilteringProcessor] Filtering out span: {span.name}")
    
    def shutdown(self):
        """Shutdown the processor."""
        return self.processor.shutdown()
    
    def force_flush(self, timeout_millis: int = 30000):
        """Force flush the processor."""
        return self.processor.force_flush(timeout_millis)


class SpanBuffer:
    """
    OpenTelemetry-compliant context manager for buffering spans.
    
    SpanBuffer collects spans in a local queue without processing them.
    After collection, you can process them through any processor using
    the process_spans() method.
    
    This follows OpenTelemetry patterns by separating span collection
    from span processing, allowing full control over when and how spans
    are processed.
    
    This enables:
    1. Batch buffering of multiple spans
    2. Manual processing timing control
    3. Asynchronous span creation (create spans after execution completes)
    4. Route buffered spans to any processor
    5. Thread-safe isolation (each context has its own buffer)
    """
    
    def __init__(self, trace_id: str):
        """
        Initialize the span buffer.
        
        Args:
            trace_id: Trace ID for the spans being buffered
        """
        self.trace_id = trace_id
        self._local_queue: List[ReadableSpan] = []
        self._is_buffering = False
        self._context_token = None
    
    def __enter__(self):
        """
        Enter context: Set this buffer as active in the current context.
        
        Spans created within this context will be routed to this buffer's
        local queue instead of being exported immediately.
        
        Returns:
            self for context manager usage
        """
        logger.debug(f"[SpanBuffer] Entering buffering context for trace {self.trace_id}")
        
        # Mark as buffering
        self._is_buffering = True
        
        # Set this buffer as active in the context variable
        self._context_token = _active_span_buffer.set(self)
        
        logger.debug(f"[SpanBuffer] Activated buffer for trace {self.trace_id}")
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit context: Deactivate this buffer in the current context.
        
        Args:
            exc_type: Exception type if an exception was raised
            exc_val: Exception value if an exception was raised
            exc_tb: Exception traceback if an exception was raised
        """
        logger.debug(f"[SpanBuffer] Exiting buffering context for trace {self.trace_id}")
        
        # Mark as not buffering
        self._is_buffering = False
        
        # Reset the context variable
        if self._context_token is not None:
            _active_span_buffer.reset(self._context_token)
            self._context_token = None
        
        logger.debug(f"[SpanBuffer] Deactivated buffer for trace {self.trace_id}")
        
        # Note: Local queue persists for manual export or inspection
        # It will be cleaned up by garbage collection when this object is destroyed
    
    def create_span(
        self, 
        span_name: str, 
        attributes: Optional[Dict[str, Any]] = None,
        kind: Optional[trace.SpanKind] = None
    ) -> str:
        """
        Create a span that goes to the local queue (not auto-exported).
        
        Args:
            span_name: Name of the span
            attributes: Optional attributes to set on the span
            kind: Optional span kind (default: INTERNAL)
        
        Returns:
            The span ID as a hex string
        """
        tracer = trace.get_tracer("respan.span_buffer")
        
        # Set span kind
        span_kind = kind or trace.SpanKind.INTERNAL
        
        # Create span in context
        with tracer.start_as_current_span(span_name, kind=span_kind) as span:
            # Set trace ID if we can (note: trace_id is already set by the tracer)
            # We just use the provided trace_id for logging/tracking purposes
            
            # Set attributes
            if attributes:
                for key, value in attributes.items():
                    try:
                        span.set_attribute(key, value)
                    except (ValueError, TypeError) as e:
                        logger.warning(
                            f"[SpanBuffer] Failed to set attribute {key}={value}: {e}"
                        )
            
            # Span goes to local queue when this context exits
            span_id = format(span.get_span_context().span_id, '016x')
            logger.debug(f"[SpanBuffer] Created span '{span_name}' with ID {span_id}")
            
        return span_id
    
    def get_all_spans(self) -> List[ReadableSpan]:
        """
        Get all spans from the local queue.
        
        Returns:
            List of all buffered spans
        """
        return self._local_queue.copy()
    
    def process_spans(self, tracer_provider) -> int:
        """
        Process all buffered spans through the tracer's processors.
        
        This sends spans through the standard OTEL processing pipeline,
        allowing processors to filter, transform, and export as configured.
        
        Args:
            tracer_provider: The TracerProvider with registered processors
        
        Returns:
            Number of spans processed
        """
        if not self._local_queue:
            logger.debug(f"[SpanBuffer] No spans to process for trace {self.trace_id}")
            return 0
        
        span_count = len(self._local_queue)
        logger.info(
            f"[SpanBuffer] Processing {span_count} spans "
            f"for trace {self.trace_id}"
        )
        
        try:
            # Get all registered processors from tracer provider
            if hasattr(tracer_provider, '_active_span_processor'):
                # Send each span through the processor pipeline
                for span in self._local_queue:
                    tracer_provider._active_span_processor.on_end(span)
                
                logger.info(
                    f"[SpanBuffer] Successfully processed {span_count} spans"
                )
                return span_count
            else:
                logger.error("[SpanBuffer] No active span processor found")
                return 0
            
        except Exception as e:
            logger.exception(f"[SpanBuffer] Exception during processing: {e}")
            return 0
    
    def clear_spans(self):
        """
        Clear all spans from the local queue without exporting.
        
        Useful for discarding buffered spans if you decide not to export them.
        """
        span_count = len(self._local_queue)
        self._local_queue.clear()
        logger.debug(f"[SpanBuffer] Cleared {span_count} spans from queue")
    
    def get_span_count(self) -> int:
        """
        Get the number of spans in the local queue.
        
        Returns:
            Number of buffered spans
        """
        return len(self._local_queue)
