"""Respan Connector component for Haystack pipelines."""

import os
from typing import Any, Dict, Optional

from haystack import component, default_from_dict, default_to_dict, logging, tracing

from respan_exporter_haystack.tracer import RespanTracer
from respan_exporter_haystack.utils.config_utils import resolve_api_key, resolve_base_url

logger = logging.getLogger(__name__)


@component
class RespanConnector:
    """
    A connector component that enables Respan tracing and logging for Haystack pipelines.
    
    This component can be added to any Haystack pipeline to automatically capture execution
    traces and send them to Respan for monitoring, debugging, and analysis.
    
    The component supports two modes:
    - "tracing": Uses Haystack's content tracing system (default, requires HAYSTACK_CONTENT_TRACING_ENABLED=true)
    - "gateway": Direct logging mode without content tracing
    
    Example usage:
        ```python
        from haystack import Pipeline
        from respan_exporter_haystack.connector import RespanConnector
        
        pipeline = Pipeline()
        pipeline.add_component(
            name="tracer",
            instance=RespanConnector(name="My Pipeline"),
        )
        # Add other components and connections...
        
        response = pipeline.run(data={...})
        print(response["tracer"]["trace_url"])
        ```
    
    Args:
        name: Name of the pipeline/trace for identification in Respan dashboard
        mode: Either "tracing" (default) or "gateway" for different logging modes
        api_key: Respan API key (defaults to RESPAN_API_KEY env var)
        base_url: Respan API base URL (defaults to RESPAN_BASE_URL env var)
        metadata: Additional metadata to attach to all traces/logs
        platform_url: Optional URL for the logs UI (defaults derived from base_url)
    """

    def __init__(
        self,
        name: str,
        mode: str = "tracing",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
        platform_url: Optional[str] = None,
        timeout: float = 10.0,
    ):
        """Initialize the Respan connector."""
        self.name = name
        self.mode = mode
        self.api_key = resolve_api_key(api_key=api_key)
        self.base_url = resolve_base_url(base_url=base_url, include_api_path=True)
        self.metadata = metadata or {}
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.platform_url = platform_url
        self.timeout = timeout
        
        if not self.api_key:
            raise ValueError(
                "Respan API key is required. Set RESPAN_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        # Initialize the tracer
        self.tracer = RespanTracer(
            name=self.name,
            api_key=self.api_key,
            base_url=self.base_url,
            metadata=self.metadata,
            max_retries=self.max_retries,
            base_delay=self.base_delay,
            max_delay=self.max_delay,
            platform_url=self.platform_url,
            timeout=self.timeout,
        )
        
        # Enable content tracing if in tracing mode
        if self.mode == "tracing":
            if os.getenv("HAYSTACK_CONTENT_TRACING_ENABLED", "").lower() != "true":
                logger.warning(
                    "HAYSTACK_CONTENT_TRACING_ENABLED is not set. "
                    "Set it to 'true' to enable full tracing capabilities."
                )
            # Register the tracer with Haystack's content tracing system
            try:
                # Only set it if it's not already our tracer to avoid conflicts in multi-connector setups
                current_tracer = getattr(tracing.tracer, "actual_tracer", None)
                if not isinstance(current_tracer, RespanTracer):
                    if current_tracer is not None:
                        logger.warning(
                            f"Replacing existing tracer {type(current_tracer).__name__} with RespanTracer"
                        )
                    tracing.enable_tracing(self.tracer)
                    logger.info(f"Respan tracer registered for '{self.name}'")
            except Exception as e:
                logger.warning(f"Could not register tracer: {e}")

    @component.output_types(name=str, trace_url=Optional[str])
    def run(self) -> Dict[str, Any]:
        """
        Run method for the connector component.
        
        NOTE: The trace is automatically sent when the pipeline completes.
        This method just returns the trace info.
        
        Returns:
            Dictionary containing:
                - name: The pipeline/trace name
                - trace_url: URL to view the trace in Respan dashboard (if available)
        """
        # Don't send here - it will be sent automatically when the root span finishes.
        # Ensure we return the name so that other pipeline components can use it if desired.
        return {
            "name": self.name,
            "trace_url": self.tracer.get_trace_url(),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialize component to dictionary. API key is never serialized to avoid leaking secrets."""
        return default_to_dict(
            obj=self,
            name=self.name,
            mode=self.mode,
            api_key=None,  # Never serialize; resolved from env on from_dict
            base_url=self.base_url,
            metadata=self.metadata,
            max_retries=self.max_retries,
            base_delay=self.base_delay,
            max_delay=self.max_delay,
            platform_url=self.platform_url,
            timeout=self.timeout,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RespanConnector":
        """Deserialize component from dictionary."""
        return default_from_dict(cls=cls, data=data)
