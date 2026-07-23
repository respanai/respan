"""Respan — unified entry point for tracing and instrumentation plugins."""

from ._core import Respan
from ._auto_instrumentation_registry import (
    AUTO_INSTRUMENTATION_REGISTRY,
    AutoInstrumentationSpec,
    AutoInstrumentationStatus,
    list_auto_instrumentation_specs,
)
from ._otel_instrumentor import OTELInstrumentor
from ._types import Instrumentation

# Re-export decorators and client from respan-tracing
from respan_tracing import workflow, task, agent, tool
from respan_tracing import RespanClient, get_client, respan_span_attributes
from respan_tracing.exporters import propagate_attributes

__all__ = [
    "Respan",
    "AUTO_INSTRUMENTATION_REGISTRY",
    "AutoInstrumentationSpec",
    "AutoInstrumentationStatus",
    "list_auto_instrumentation_specs",
    "OTELInstrumentor",
    "Instrumentation",
    "workflow",
    "task",
    "agent",
    "tool",
    "RespanClient",
    "get_client",
    "respan_span_attributes",
    "propagate_attributes",
]
