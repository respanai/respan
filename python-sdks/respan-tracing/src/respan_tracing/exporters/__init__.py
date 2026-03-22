"""
Span exporters for Respan tracing.

This module contains various span exporters that handle exporting spans
to different destinations like the Respan API, files, or other systems.

Import directly from submodules:
    from respan_tracing.exporters.respan import RespanSpanExporter
    from respan_tracing.utils.span_factory import propagate_attributes
"""

from respan_tracing.exporters.respan import RespanSpanExporter

__all__ = [
    "RespanSpanExporter",
]
