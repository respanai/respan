"""Respan exporter for Pydantic AI.

Public API re-export: ``instrument_pydantic_ai`` is the sole entry point
for consumers.  This is an SDK package (not a Django app), so the re-export
follows standard Python packaging conventions.
"""

from importlib.metadata import version, PackageNotFoundError

from respan_exporter_pydantic_ai.instrument import instrument_pydantic_ai

try:
    __version__ = version("respan-exporter-pydantic-ai")
except PackageNotFoundError:
    __version__ = "0.1.0"

__all__ = ["instrument_pydantic_ai"]
