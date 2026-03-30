import json
import random
from typing import Optional, Callable, Dict, Any, List, Sequence
from contextvars import ContextVar
import logging

from opentelemetry import context as context_api, trace
from opentelemetry.sdk.trace import SpanProcessor, ReadableSpan
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor, SpanExporter, SpanExportResult
from opentelemetry.context import Context
from opentelemetry.trace import NonRecordingSpan, SpanContext, TraceFlags
from opentelemetry.semconv_ai import SpanAttributes

from respan_sdk.constants.span_attributes import RESPAN_TRACE_GROUP_ID
from respan_sdk.respan_types.span_types import SpanLink
from respan_sdk.utils.data_processing.id_processing import format_span_id
from respan_tracing.contexts.span import span_link_to_otel
from respan_tracing.constants.generic_constants import SDK_PREFIX
from respan_tracing.constants.tracing import EXPORT_FILTER_ATTR, PROCESSORS_ATTR, SAMPLE_RATE_ATTR, SPAN_BUFFER_TRACER_NAME
from respan_tracing.constants.context_constants import (
    TRACE_GROUP_ID_KEY,
    PARAMS_KEY
)
from respan_tracing.filters import evaluate_export_filter
from respan_tracing.utils.preprocessing.span_processing import is_processable_span
from respan_tracing.utils.context import get_entity_path
from respan_tracing.utils.span_factory import read_propagated_attributes

logger = logging.getLogger(__name__)


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
        workflow_name = context_api.get_value(SpanAttributes.TRACELOOP_ENTITY_NAME)
        if workflow_name:
            span.set_attribute(SpanAttributes.TRACELOOP_WORKFLOW_NAME, workflow_name)

        # Add entity path if present in context (for redundancy)
        entity_path_from_context = context_api.get_value(SpanAttributes.TRACELOOP_ENTITY_PATH)
        if entity_path_from_context:
            span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_PATH, entity_path_from_context)

        # Add trace group identifier if present
        trace_group_id = context_api.get_value(TRACE_GROUP_ID_KEY)
        if trace_group_id:
            span.set_attribute(
                RESPAN_TRACE_GROUP_ID, trace_group_id
            )

        # Inherit processors from parent span when child doesn't have its own.
        # This enables a parent @workflow(processors="dogfood,production") to
        # automatically route all child @task spans to the same processors,
        # without each child needing to repeat the processors param.
        #
        # IMPORTANT: Only inherit when child has NO processors set. If the
        # child explicitly sets processors (e.g., @workflow(processors="dogfood")),
        # respect that — merging would cause recursion when the production
        # processor's exporter creates spans with @workflow(processors="dogfood").
        if not span.attributes.get(PROCESSORS_ATTR):
            parent_span = trace.get_current_span(parent_context)
            if parent_span and hasattr(parent_span, "attributes"):
                parent_processors = (parent_span.attributes or {}).get(PROCESSORS_ATTR)
                if parent_processors:
                    span.set_attribute(PROCESSORS_ATTR, parent_processors)

            # Fallback: in continuation mode, the parent is a NonRecordingSpan
            # with no attributes. Check the active SpanBuffer for stashed
            # processors from the skipped client.start_span() call.
            if not span.attributes.get(PROCESSORS_ATTR):
                active_buffer = _active_span_buffer.get(None)
                if active_buffer:
                    buffer_processors = getattr(active_buffer, "continuation_processors", None)
                    if buffer_processors:
                        span.set_attribute(PROCESSORS_ATTR, buffer_processors)

        # Add custom parameters if present
        respan_params = context_api.get_value(PARAMS_KEY)
        if respan_params and isinstance(respan_params, dict):
            for key, value in respan_params.items():
                span.set_attribute(f"{SDK_PREFIX}.{key}", value)

        # Bridge propagated attributes (customer_identifier, thread_id, etc.)
        # from the _PROPAGATED_ATTRIBUTES ContextVar onto auto-instrumented spans.
        # This ensures spans created by OTEL auto-instrumentors (OpenAI, Anthropic, etc.)
        # carry the same user-context attributes as plugin-injected spans.
        try:
            propagated = read_propagated_attributes()
            for attr_key, attr_val in propagated.items():
                if not span.attributes.get(attr_key):
                    span.set_attribute(attr_key, attr_val)
        except Exception:
            logger.debug("Failed to bridge propagated attributes", exc_info=True)

        # Call original processor's on_start
        self.processor.on_start(span, parent_context)

    def on_end(self, span: ReadableSpan):
        """Called when a span ends - filter spans based on Respan attributes"""
        # Apply standard filtering logic (processable span check)
        if not is_processable_span(span):
            logger.debug(f"[Respan Debug] Skipping filtered span: {span.name}")
            return

        # Apply sample_rate if present on this span
        if span.attributes:
            sample_rate = span.attributes.get(SAMPLE_RATE_ATTR)
            if sample_rate is not None and random.random() >= sample_rate:
                logger.debug(f"[Respan Debug] Sample rate ({sample_rate}) dropped span: {span.name}")
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
            # Route to the buffer's local queue (deduplicated)
            buffer.buffer_span(span)
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
    
    def __init__(
        self,
        trace_id: str,
        tracer_provider=None,
        parent_trace_id: Optional[str] = None,
        parent_span_id: Optional[str] = None,
    ):
        """
        Initialize the span buffer.

        Args:
            trace_id: Trace ID for the spans being buffered
            tracer_provider: Optional TracerProvider. When provided, buffered
                spans are auto-flushed through the processor pipeline on exit.
            parent_trace_id: Optional OTel trace ID (hex string) to continue.
                When provided with parent_span_id, spans created in this buffer
                inherit the parent trace, making them children of the specified
                span. Used for workflow pause/resume to keep all spans in one trace.
            parent_span_id: Optional OTel span ID (hex string) of the parent span.
                Must be provided together with parent_trace_id.
        """
        self.trace_id = trace_id
        self._local_queue: List[ReadableSpan] = []
        self._seen_span_ids: set = set()
        self._is_buffering = False
        self._context_token = None
        self._parent_context_token = None
        self._tracer_provider = tracer_provider
        self._parent_trace_id = parent_trace_id
        self._parent_span_id = parent_span_id
    
    def __enter__(self):
        """
        Enter context: Set this buffer as active in the current context.

        Spans created within this context will be routed to this buffer's
        local queue instead of being exported immediately.

        When parent_trace_id and parent_span_id are set, injects a
        NonRecordingSpan as the current context so that all spans created
        in this buffer inherit the parent's trace_id and become children
        of the specified span. This enables workflow resume to continue
        the pre-pause trace instead of starting a new one.

        Returns:
            self for context manager usage
        """
        logger.debug(f"[SpanBuffer] Entering buffering context for trace {self.trace_id}")

        # Inject parent context for trace continuation (workflow resume)
        if self._parent_trace_id and self._parent_span_id:
            parent_ctx = SpanContext(
                trace_id=int(self._parent_trace_id, 16),
                span_id=int(self._parent_span_id, 16),
                is_remote=True,
                trace_flags=TraceFlags(TraceFlags.SAMPLED),
            )
            parent_span = NonRecordingSpan(parent_ctx)
            ctx = trace.set_span_in_context(parent_span)
            self._parent_context_token = context_api.attach(ctx)
            logger.debug(
                f"[SpanBuffer] Injected parent context: "
                f"trace_id={self._parent_trace_id}, span_id={self._parent_span_id}"
            )

        # Mark as buffering
        self._is_buffering = True

        # Set this buffer as active in the context variable
        self._context_token = _active_span_buffer.set(self)

        logger.debug(f"[SpanBuffer] Activated buffer for trace {self.trace_id}")

        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit context: Deactivate this buffer and auto-flush spans.

        Buffered spans are replayed through the tracer provider's processor
        pipeline so they reach the OTLP exporter. This does NOT clear the
        local queue — ``get_all_spans()`` still works after exit for read-only
        use cases (e.g., converting spans to unified logs).

        CRITICAL: Flush BEFORE resetting the context variable. If we reset
        first, the ContextVar restores the parent buffer. Then process_spans
        replays spans through BufferingSpanProcessor which sees the parent
        buffer (still _is_buffering=True) and re-buffers them there instead
        of forwarding to the real processor chain.
        """
        logger.debug(f"[SpanBuffer] Exiting buffering context for trace {self.trace_id}")

        # Mark as not buffering FIRST so replayed spans don't re-enter
        # THIS buffer via BufferingSpanProcessor.on_end
        self._is_buffering = False

        # Auto-flush BEFORE resetting the context variable. While this
        # buffer is still the active one in the ContextVar,
        # BufferingSpanProcessor.on_end sees _is_buffering=False and
        # falls through to the real processor (FilteringSpanProcessor →
        # exporter). If we reset first, the parent buffer would catch
        # the replayed spans instead.
        if self._tracer_provider is not None and self._local_queue:
            self.process_spans(self._tracer_provider)

        # NOW safe to reset — flush is done, no more spans to replay
        if self._context_token is not None:
            _active_span_buffer.reset(self._context_token)
            self._context_token = None

        # Detach parent context (must happen after flush so spans keep the parent)
        if self._parent_context_token is not None:
            context_api.detach(self._parent_context_token)
            self._parent_context_token = None

        logger.debug(f"[SpanBuffer] Deactivated buffer for trace {self.trace_id}")
    
    def create_span(
        self, 
        span_name: str, 
        attributes: Optional[Dict[str, Any]] = None,
        kind: Optional[trace.SpanKind] = None,
        links: Optional[Sequence[SpanLink | trace.Link]] = None,
    ) -> str:
        """
        Create a span that goes to the local queue (not auto-exported).
        
        Args:
            span_name: Name of the span
            attributes: Optional attributes to set on the span
            kind: Optional span kind (default: INTERNAL)
            links: Optional list of SDK SpanLink or OpenTelemetry Link objects
        
        Returns:
            The span ID as a hex string
        """
        tracer = trace.get_tracer(SPAN_BUFFER_TRACER_NAME)
        
        # Set span kind
        span_kind = kind or trace.SpanKind.INTERNAL
        otel_links: List[trace.Link] = []
        for link in links or []:
            if isinstance(link, SpanLink):
                otel_links.append(span_link_to_otel(link))
                continue
            if isinstance(link, trace.Link):
                otel_links.append(link)
                continue
            raise TypeError(
                "links must contain SpanLink or opentelemetry.trace.Link instances"
            )
        
        # Create span in context
        with tracer.start_as_current_span(
            span_name,
            kind=span_kind,
            links=otel_links,
        ) as span:
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
            span_id = format_span_id(span.get_span_context().span_id)
            logger.debug(f"[SpanBuffer] Created span '{span_name}' with ID {span_id}")
            
        return span_id
    
    def buffer_span(self, span: ReadableSpan) -> bool:
        """
        Add a span to the buffer, deduplicating by span_id.

        Multiple BufferingSpanProcessor instances (one per processor chain)
        all intercept on_end() and share this buffer via the ContextVar.
        Without dedup, each processor adds the same span → N copies.

        Args:
            span: The span to buffer

        Returns:
            True if the span was added, False if it was a duplicate.
        """
        span_id = span.get_span_context().span_id
        if span_id in self._seen_span_ids:
            logger.debug(
                f"[SpanBuffer] Skipping duplicate span '{span.name}' "
                f"(span_id={format_span_id(span_id)}) for trace {self.trace_id}"
            )
            return False
        self._seen_span_ids.add(span_id)
        self._local_queue.append(span)
        logger.debug(
            f"[SpanBuffer] Buffering span '{span.name}' "
            f"for trace {self.trace_id}"
        )
        return True

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
        self._seen_span_ids.clear()
        logger.debug(f"[SpanBuffer] Cleared {span_count} spans from queue")
    
    def get_span_count(self) -> int:
        """
        Get the number of spans in the local queue.
        
        Returns:
            Number of buffered spans
        """
        return len(self._local_queue)
