"""
Span exporters for Respan tracing.

This module contains various span exporters that handle exporting spans
to different destinations like the Respan API, files, or other systems.
"""

from .respan import RespanSpanExporter
from ..utils.span_factory import propagate_attributes

__all__ = [
    "RespanSpanExporter",
    "propagate_attributes",
]
