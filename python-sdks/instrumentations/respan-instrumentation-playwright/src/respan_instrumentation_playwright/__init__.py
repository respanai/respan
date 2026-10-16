"""Respan instrumentation plugin for Playwright."""

from respan_instrumentation_playwright._instrumentation import (
    PlaywrightInstrumentor,
    secrets,
)

__all__ = ["PlaywrightInstrumentor", "secrets"]
