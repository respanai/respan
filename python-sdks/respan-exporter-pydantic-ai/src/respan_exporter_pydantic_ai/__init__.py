from importlib.metadata import version, PackageNotFoundError

from respan_exporter_pydantic_ai.instrument import instrument_pydantic_ai

try:
    __version__ = version("respan-exporter-pydantic-ai")
except PackageNotFoundError:
    __version__ = "0.1.0"

__all__ = ["instrument_pydantic_ai"]
