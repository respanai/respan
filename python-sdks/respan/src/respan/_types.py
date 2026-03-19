"""Instrumentation protocol for Respan plugins."""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Instrumentation(Protocol):
    """Protocol that all Respan instrumentation plugins must implement.

    Plugins are discovered via the ``respan.instrumentations`` entry-point
    group and activated by the ``Respan`` class at startup.
    """

    name: str

    def activate(self, exporter: Any) -> None:
        """Start intercepting spans and forwarding them to *exporter*.

        Args:
            exporter: A ``RespanSpanExporterV2`` instance (or compatible)
                      whose ``export([dict, ...])`` method accepts log dicts.
        """
        ...

    def deactivate(self) -> None:
        """Stop intercepting spans and clean up resources."""
        ...
