"""Respan instrumentation for the Dify Python client."""

from respan_instrumentation_dify._instrumentation import DifyInstrumentor

DifyAIInstrumentor = DifyInstrumentor

__all__ = ["DifyAIInstrumentor", "DifyInstrumentor"]
