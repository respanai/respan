import atexit
import os
from typing import Dict, Optional, Set, Callable, Union
from threading import Lock

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource, SERVICE_NAME
from opentelemetry.sdk.trace import TracerProvider, ReadableSpan
from opentelemetry.sdk.trace.export import (
    SpanExporter,
    SimpleSpanProcessor,
    BatchSpanProcessor,
)
from opentelemetry.propagate import set_global_textmap
from opentelemetry.propagators.textmap import TextMapPropagator

from respan_tracing.processors import RespanSpanProcessor, BufferingSpanProcessor, FilteringSpanProcessor
from respan_tracing.exporters import RespanSpanExporter
from respan_tracing.instruments import Instruments
from respan_tracing.utils.notebook import is_notebook
from respan_tracing.utils.instrumentation import init_instrumentations
from respan_tracing.utils.imports import import_from_string
from respan_tracing.utils.logging import get_respan_logger
from respan_tracing.constants.tracing import TRACER_NAME, PROCESSORS_ATTR
from respan_tracing.constants.generic_constants import LOGGER_NAME_TRACER

# Use Respan logger for all logging in this module
logger = get_respan_logger(LOGGER_NAME_TRACER)

class RespanTracer:
    """
    Direct OpenTelemetry implementation for Respan tracing.
    Replaces Traceloop dependency with native OpenTelemetry components.
    """
    
    _instance = None
    _lock = Lock()
    
    def __new__(cls, *args, **kwargs):
        """Singleton pattern to ensure only one tracer instance exists"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(
        self,
        app_name: str = "respan",
        api_endpoint: str = "https://api.respan.ai/api",
        api_key: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        is_batching_enabled: bool = True,
        resource_attributes: Optional[Dict[str, str]] = None,
        instruments: Optional[Set[Instruments]] = None,
        block_instruments: Optional[Set[Instruments]] = None,
        propagator: Optional[TextMapPropagator] = None,
        span_postprocess_callback: Optional[Callable[[ReadableSpan], None]] = None,
        is_enabled: bool = True,
        is_auto_instrument: bool = True,
    ):
        # Prevent re-initialization
        if hasattr(self, '_initialized'):
            return

        self._initialized = True
        self.is_enabled = is_enabled
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.headers = headers or {}
        self.is_batching_enabled = is_batching_enabled
        self.span_postprocess_callback = span_postprocess_callback

        if not is_enabled:
            logger.info("Respan tracing is disabled")
            return

        # Setup resource attributes
        resource_attributes = resource_attributes or {}
        resource_attributes[SERVICE_NAME] = app_name

        # Initialize OpenTelemetry components
        self._setup_tracer_provider(resource_attributes)
        self._setup_propagation(propagator)
        if is_auto_instrument:
            self._setup_instrumentations(instruments, block_instruments)
        
        # Add default Respan processor for backward compatibility
        # Only if api_key is provided (user wants to send to Respan)
        if api_key:
            self._setup_default_processor()
        
        # Register cleanup
        atexit.register(self._cleanup)
        
        # Log initialization
        logger.info(f"Respan tracing initialized, sending to {api_endpoint}")
    
    def _setup_tracer_provider(self, resource_attributes: Dict[str, str]):
        """Initialize the OpenTelemetry TracerProvider"""
        resource = Resource(attributes=resource_attributes)
        self.tracer_provider = TracerProvider(resource=resource)
        trace.set_tracer_provider(self.tracer_provider)
    
    def _setup_default_processor(self):
        """Setup default Respan processor for backward compatibility"""
        from ..exporters import RespanSpanExporter
        
        logger.info("Adding default Respan processor (all spans)")
        exporter = RespanSpanExporter(
            endpoint=self.api_endpoint,
            api_key=self.api_key,
            headers=self.headers,
        )
        
        # Add without name or filter - receives ALL spans (backward compatible behavior)
        self.add_processor(
            exporter=exporter,
            name=None,  # No name = no filtering
            filter_fn=None,  # No filter = all spans
        )
    
    def add_processor(
        self,
        exporter: Union[SpanExporter, str],
        name: Optional[str] = None,
        filter_fn: Optional[Callable[[ReadableSpan], bool]] = None,
        is_batching_enabled: Optional[bool] = None,
    ) -> None:
        """
        Add a span processor with optional filtering.
        
        This is the standard OpenTelemetry way to support multiple exporters.
        You can call this method multiple times to add multiple processors.
        
        Args:
            exporter: SpanExporter instance or import string (e.g., "module.path.ExporterClass")
            name: Optional name for the processor. If provided without filter_fn, automatically
                  filters spans where span.attributes.get("processor") == name.
                  Set to None to receive all spans.
            filter_fn: Optional custom filter function. Only spans where filter_fn(span) returns True
                      will be exported. If None but name is provided, automatically creates filter
                      for that processor name. If both None, all spans are exported.
            is_batching_enabled: Whether to use batch processing (default: uses tracer default)
        
        Example:
            # Processor with name - automatically filters for processor="production" in decorator
            tracer.add_processor(
                exporter=RespanSpanExporter(...),
                name="production"
            )
            # Now @task(processor="production") will route here automatically!
            
            # Processor without name - receives ALL spans
            tracer.add_processor(
                exporter=FileExporter("./all.json")
            )
            
            # Custom filter function (overrides name-based filtering)
            tracer.add_processor(
                exporter=SlowSpanExporter(),
                name="slow_spans",
                filter_fn=lambda span: (span.end_time - span.start_time) > 1_000_000_000
            )
        """
        if not self.is_enabled:
            logger.warning("Tracer is disabled, cannot add processor")
            return
        
        # Use tracer default if not specified
        if is_batching_enabled is None:
            is_batching_enabled = self.is_batching_enabled
        
        # Handle string imports
        if isinstance(exporter, str):
            try:
                exporter_class = import_from_string(exporter)
                if name == "respan" or "Respan" in exporter:
                    # Respan exporter needs special initialization
                    exporter = exporter_class(
                        endpoint=self.api_endpoint,
                        api_key=self.api_key,
                        headers=self.headers,
                    )
                else:
                    # Other exporters use default initialization
                    exporter = exporter_class()
            except ImportError as e:
                logger.error(f"Failed to import exporter from '{exporter}': {e}")
                return
        
        # Create combined filter: name-based + custom filter
        if name is not None:
            # Name-based filter (always applied when name is provided)
            name_filter = lambda span: name in (span.attributes.get(PROCESSORS_ATTR) or "").split(",")
            
            if filter_fn is not None:
                # Combine: BOTH name filter AND custom filter must pass
                original_filter = filter_fn
                filter_fn = lambda span: name_filter(span) and original_filter(span)
                logger.debug(f"Created combined filter for processor '{name}' (name + custom)")
            else:
                # Just name filter
                filter_fn = name_filter
                logger.debug(f"Auto-created name filter for processor '{name}'")
        
        # Create filtering processor
        processor = FilteringSpanProcessor(
            exporter=exporter,
            filter_fn=filter_fn,
            is_batching_enabled=is_batching_enabled,
            span_postprocess_callback=self.span_postprocess_callback,
        )
        
        # Wrap with BufferingSpanProcessor for span collection support
        buffering_processor = BufferingSpanProcessor(processor)
        
        # Add to tracer provider (standard OTEL way!)
        self.tracer_provider.add_span_processor(buffering_processor)
        
        name_str = f" '{name}'" if name else ""
        filter_str = "auto" if (filter_fn is not None and name is not None) else ("custom" if filter_fn is not None else "none")
        logger.info(f"Added span processor{name_str} with filter: {filter_str}")
    
    def _setup_propagation(self, propagator: Optional[TextMapPropagator]):
        """Setup context propagation"""
        if propagator:
            set_global_textmap(propagator)
    
    def _setup_instrumentations(
        self,
        instruments: Optional[Set[Instruments]],
        block_instruments: Optional[Set[Instruments]],
    ):
        """Initialize library instrumentations (including threading)"""
        init_instrumentations(instruments, block_instruments)
    
    def get_tracer(self, name: str = TRACER_NAME):
        """Get OpenTelemetry tracer instance"""
        if not self.is_enabled:
            return trace.NoOpTracer()
        return self.tracer_provider.get_tracer(name)
    
    def flush(self):
        """Force flush all pending spans"""
        if hasattr(self, 'tracer_provider'):
            self.tracer_provider.force_flush()
    
    def _cleanup(self):
        """Cleanup resources on exit"""
        if hasattr(self, 'tracer_provider'):
            try:
                self.tracer_provider.force_flush()
                self.tracer_provider.shutdown()
            except Exception as e:
                logger.error(f"Error during cleanup: {e}")
    
    @classmethod
    def is_initialized(cls) -> bool:
        """Check if tracer is initialized"""
        return cls._instance is not None and hasattr(cls._instance, '_initialized')
    
    @classmethod
    def reset_instance(cls):
        """Reset the singleton instance (for testing purposes)"""
        with cls._lock:
            if cls._instance is not None:
                try:
                    cls._instance._cleanup()
                except:
                    pass
            cls._instance = None 