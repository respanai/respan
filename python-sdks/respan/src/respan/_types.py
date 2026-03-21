"""Instrumentation protocol for Respan plugins."""

from typing import Protocol, runtime_checkable


@runtime_checkable
class Instrumentation(Protocol):
    """Protocol that all Respan instrumentation plugins must implement.

    Plugins are discovered via the ``respan.instrumentations`` entry-point
    group and activated by the ``Respan`` class at startup.

    Plugins hook into external SDKs and emit ``ReadableSpan`` objects into
    the single OTEL pipeline via ``inject_span()``.  No exporter argument
    is needed — spans flow through the ``TracerProvider``'s processor chain
    automatically.
    """

    name: str

    def activate(self) -> None:
        """Start intercepting spans and injecting them into the OTEL pipeline."""
        ...

    def deactivate(self) -> None:
        """Stop intercepting spans and clean up resources."""
        ...
