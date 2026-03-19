"""Respan — unified entry point for tracing and instrumentation plugins."""

from ._core import Respan
from ._types import Instrumentation

# Re-export decorators and client from respan-tracing
from respan_tracing import workflow, task, agent, tool
from respan_tracing import RespanClient, get_client, respan_span_attributes
from respan_tracing.exporters import propagate_attributes

__all__ = [
    "Respan",
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
