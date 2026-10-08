"""Respan Milvus plugin entry point."""

from ._native_instrumentation import MilvusInstrumentor

__all__ = ["MilvusInstrumentor"]
