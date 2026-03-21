"""Utility functions for Respan Google ADK instrumentation."""
from .adk_detection import ADK_SCOPE_NAMES, ADK_SPAN_NAMES, is_adk_span

__all__ = ["ADK_SCOPE_NAMES", "ADK_SPAN_NAMES", "is_adk_span"]
