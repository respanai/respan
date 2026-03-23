import os
import logging
from typing import Optional, Set, Dict, Callable, Literal, Union
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter
from respan_tracing.decorators import workflow, task, agent, tool
from respan_tracing.core import RespanTracer, RespanClient
from respan_tracing.instruments import Instruments
from respan_tracing.utils.logging import get_main_logger

class RespanTelemetry:
    """
    Respan Telemetry - Direct OpenTelemetry implementation.
    
    This class initializes the OpenTelemetry tracer without any default exporters.
    Use add_processor() to add exporters after initialization.
    
    Args:
        app_name: Name of the application for telemetry identification
        api_key: Respan API key (can also be set via RESPAN_API_KEY env var)
        base_url: Respan API base URL (can also be set via RESPAN_BASE_URL env var)
        log_level: Logging level for Respan tracing (default: "INFO"). 
                  Can be "DEBUG", "INFO", "WARNING", "ERROR", or "CRITICAL".
                  Set to "DEBUG" to see detailed debug messages.
                  Can also be set via RESPAN_LOG_LEVEL environment variable.
        is_batching_enabled: Whether to enable batch span processing (default: True). 
                            When False, uses synchronous export (no background threads).
                            Useful for debugging or backends with custom exporters.
                            Can also be set via RESPAN_BATCHING_ENABLED environment variable.
        instruments: Set of instruments to enable (if None, enables default set)
        block_instruments: Set of instruments to explicitly disable
        headers: Additional headers to send with telemetry data
        resource_attributes: Additional resource attributes to attach to all spans
        span_postprocess_callback: Optional callback to process spans before export
        is_enabled: Whether telemetry is enabled (if False, becomes no-op)
    
    Example:
        ```python
        from respan_tracing import RespanTelemetry
        from respan_tracing.exporters import RespanSpanExporter
        
        # Initialize telemetry
        kai = RespanTelemetry(app_name="my-app", api_key="your-key")
        
        # Add production exporter (all spans)
        kai.add_processor(
            exporter=RespanSpanExporter(
                endpoint="https://api.respan.ai/api",
                api_key="prod-key"
            ),
            name="production"
        )
        
        # Add debug exporter (only debug spans)
        kai.add_processor(
            exporter=FileExporter("./debug.json"),
            name="debug",
            filter_fn=lambda span: span.attributes.get("exporter") == "debug"
        )
        
        # Use decorators with processor parameter
        @kai.task(name="debug_task", processor="debug")
        def debug_task():
            return "only goes to debug processor"
        
        @kai.task(name="prod_task")
        def prod_task():
            return "goes to all processors"
        ```
    
    Note:
        Threading instrumentation is ALWAYS enabled by default (even when specifying custom
        instruments) because it's critical for context propagation. To disable it explicitly:
        block_instruments={Instruments.THREADING}
    """

    def __init__(
        self,
        app_name: str = "respan",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO",
        is_batching_enabled: Optional[bool] = None,
        instruments: Optional[Set[Instruments]] = None,
        block_instruments: Optional[Set[Instruments]] = None,
        headers: Optional[Dict[str, str]] = None,
        resource_attributes: Optional[Dict[str, str]] = None,
        span_postprocess_callback: Optional[Callable[[ReadableSpan], None]] = None,
        is_enabled: bool = True,
        is_auto_instrument: bool = True,
    ):
        # Get configuration from environment variables
        api_key = api_key or os.getenv("RESPAN_API_KEY")
        base_url = base_url or os.getenv(
            "RESPAN_BASE_URL", "https://api.respan.ai/api"
        )
        # Default to True if not specified
        if is_batching_enabled is None:
            is_batching_enabled = (
                os.getenv("RESPAN_BATCHING_ENABLED", "True").lower() == "true"
            )
        
        # Get log level from environment variable if not explicitly set
        env_log_level = os.getenv("RESPAN_LOG_LEVEL")
        if env_log_level and log_level == "INFO":  # Only use env var if user didn't specify
            valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
            if env_log_level.upper() in valid_levels:
                log_level = env_log_level.upper()  # type: ignore
        
        # Configure logging level for Respan tracing
        self._configure_logging(log_level)
        
        # Initialize the tracer
        self.tracer = RespanTracer(
            app_name=app_name,
            api_endpoint=base_url,
            api_key=api_key,
            is_batching_enabled=is_batching_enabled,
            instruments=instruments,
            block_instruments=block_instruments,
            headers=headers,
            resource_attributes=resource_attributes,
            span_postprocess_callback=span_postprocess_callback,
            is_enabled=is_enabled,
            is_auto_instrument=is_auto_instrument,
        )
        
        if is_enabled:
            logging.info(f"Respan telemetry initialized")
        else:
            logging.info("Respan telemetry is disabled")

    def _configure_logging(self, log_level: Union[str, int]):
        """Configure logging level for Respan tracing"""
        # Get the Respan logger using the utility function
        respan_logger = get_main_logger()
        
        # Convert string log level to logging constant if needed
        if isinstance(log_level, str):
            log_level = getattr(logging, log_level.upper(), logging.INFO)
        
        # Set the log level
        respan_logger.setLevel(log_level)
        
        # Ensure there's a handler if none exists
        if not respan_logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            respan_logger.addHandler(handler)
        
        # Prevent duplicate messages from propagating to root logger
        # but allow child loggers to inherit the level
        respan_logger.propagate = False

    def add_processor(
        self,
        exporter: Union[SpanExporter, str],
        name: Optional[str] = None,
        filter_fn: Optional[Callable[[ReadableSpan], bool]] = None,
        is_batching_enabled: Optional[bool] = None,
    ) -> None:
        """
        Add a span processor with optional filtering (standard OpenTelemetry pattern).
        
        This method can be called multiple times to add multiple exporters.
        Each processor can have its own filter function to selectively export spans.
        
        Args:
            exporter: SpanExporter instance or import string (e.g., "module.path.ExporterClass")
            name: Optional name for the exporter (used for logging and Respan identification)
            filter_fn: Optional filter function. Only spans where filter_fn(span) returns True
                      will be exported. If None, all spans are exported to this exporter.
                      Common pattern: lambda span: span.attributes.get("processor") == "debug"
            is_batching_enabled: Whether to use batch processing (default: uses telemetry default)
        
        Example:
            ```python
            # Add production exporter (all spans)
            kai.add_processor(
                exporter=RespanSpanExporter(...),
                name="production"
            )
            
            # Add debug file exporter (only debug spans)
            kai.add_processor(
                exporter=FileExporter("./debug.json"),
                name="debug",
                filter_fn=lambda span: span.attributes.get("processor") == "debug"
            )
            ```
        """
        self.tracer.add_processor(
            exporter=exporter,
            name=name,
            filter_fn=filter_fn,
            is_batching_enabled=is_batching_enabled,
        )
    
    def flush(self):
        """Force flush all pending spans"""
        self.tracer.flush()
    
    def is_initialized(self) -> bool:
        """Check if telemetry is initialized"""
        return RespanTracer.is_initialized()

    def get_client(self) -> RespanClient:
        """
        Get a client for interacting with the current trace/span context.
        
        Returns:
            RespanClient instance for trace operations.
        """
        return RespanClient()

    # Expose decorators as instance methods for backward compatibility
    workflow = staticmethod(workflow)
    task = staticmethod(task)
    agent = staticmethod(agent)
    tool = staticmethod(tool)


# Module-level client instance for global access
_global_client: Optional[RespanClient] = None


def get_client() -> RespanClient:
    """
    Get a global Respan client instance.
    
    This function provides access to trace operations without needing to maintain
    a reference to the RespanTelemetry instance. The client uses the singleton
    tracer instance internally.
    
    Returns:
        RespanClient instance for trace operations.
        
    Example:
        ```python
        from respan_tracing import get_client
        
        client = get_client()
        
        # Get current trace information
        trace_id = client.get_current_trace_id()
        span_id = client.get_current_span_id()
        
        # Update current span
        client.update_current_span(
            respan_params={"trace_group_identifier": "my-group"},
            attributes={"custom.attribute": "value"}
        )
        
        # Add events and handle exceptions
        client.add_event("processing_started")
        try:
            # Your code here
            pass
        except Exception as e:
            client.record_exception(e)
        ```
    """
    global _global_client
    if _global_client is None:
        _global_client = RespanClient()
    return _global_client



