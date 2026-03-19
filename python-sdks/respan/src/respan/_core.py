"""Respan — unified entry point for tracing and instrumentation plugins."""

import logging
import os
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Sequence

from respan_tracing import RespanTelemetry
from respan_tracing.exporters import RespanSpanExporterV2
from respan_tracing.exporters.span_exporter_v2 import (
    propagate_attributes as _propagate_attributes,
)

from ._types import Instrumentation

logger = logging.getLogger(__name__)


class Respan:
    """Unified entry point for Respan tracing and instrumentation plugins.

    Sets up:
    1. ``RespanTelemetry`` — OTEL TracerProvider for decorators and, when no
       plugins are provided, auto-instrumentation of LLM SDKs (OpenAI,
       Anthropic, etc.) via the OTEL pipeline.
    2. ``RespanSpanExporterV2`` for plugin instrumentations to send spans
       to ``/v1/traces/ingest``.
    3. Activates any instrumentors passed via the ``instrumentations`` list.

    When ``instrumentations`` are provided, OTEL auto-instrumentation is
    disabled by default to avoid duplicate spans (plugins capture LLM calls
    themselves).  Override with ``auto_instrument=True`` if you need both.

    Args:
        api_key: Respan API key. Falls back to ``RESPAN_API_KEY`` env var.
        base_url: Respan API base URL. Falls back to ``RESPAN_BASE_URL`` env var.
        app_name: Application name for telemetry identification.
        instrumentations: List of instrumentor instances to activate.
        auto_instrument: Auto-instrument LLM SDKs (OpenAI, Anthropic, etc.)
            via OTEL.  Defaults to ``True`` when no plugins are provided,
            ``False`` when plugins are provided (to avoid duplicate spans).
        customer_identifier: Default customer/user identifier for all spans.
        thread_identifier: Default conversation thread ID for all spans.
        metadata: Default metadata dict merged into all spans.
        environment: Default environment (e.g. ``"production"``).
        **telemetry_kwargs: Extra keyword arguments forwarded to
            ``RespanTelemetry`` (e.g. ``log_level``, ``is_batching_enabled``).

    Examples::

        # Direct LLM SDK usage — auto-instruments OpenAI, Anthropic, etc.
        respan = Respan()

        # With plugins — plugins handle tracing, auto-instrumentation off
        from respan_instrumentation_openai_agents import OpenAIAgentsInstrumentor
        respan = Respan(instrumentations=[OpenAIAgentsInstrumentor()])
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        app_name: str = "respan",
        instrumentations: Optional[Sequence[object]] = None,
        auto_instrument: Optional[bool] = None,
        customer_identifier: Optional[str] = None,
        thread_identifier: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        environment: Optional[str] = None,
        **telemetry_kwargs,
    ):
        api_key = api_key or os.getenv("RESPAN_API_KEY")
        base_url = base_url or os.getenv(
            "RESPAN_BASE_URL", "https://api.respan.ai/api"
        )

        # Build default attributes from init params
        default_attributes: Dict[str, Any] = {}
        if customer_identifier:
            default_attributes["customer_identifier"] = customer_identifier
        if thread_identifier:
            default_attributes["thread_identifier"] = thread_identifier
        if metadata:
            default_attributes["metadata"] = metadata
        if environment:
            default_attributes["environment"] = environment

        # Auto-instrument LLM SDKs when no plugins are provided,
        # disable when plugins handle tracing to avoid duplicate spans.
        if auto_instrument is None:
            auto_instrument = not bool(instrumentations)

        # 1. OTEL TracerProvider + optional auto-instrumentation
        self.telemetry = RespanTelemetry(
            app_name=app_name,
            api_key=api_key,
            base_url=base_url,
            auto_instrument=auto_instrument,
            **telemetry_kwargs,
        )

        # 2. Plugin exporter → /v1/traces/ingest
        ingest_endpoint = f"{base_url.rstrip('/')}/v1/traces/ingest"
        self.exporter = RespanSpanExporterV2(
            api_key=api_key,
            endpoint=ingest_endpoint,
            default_attributes=default_attributes,
        )

        # 3. Activate instrumentations
        self._instrumentations: Dict[str, object] = {}
        for inst in instrumentations or []:
            name = getattr(inst, "name", type(inst).__name__)
            self._activate(name, inst)

    def _activate(self, name: str, inst: object) -> None:
        """Activate a single instrumentor."""
        try:
            inst.activate(self.exporter)  # type: ignore[union-attr]
            self._instrumentations[name] = inst
            logger.info("Activated instrumentation: %s", name)
        except Exception as exc:
            logger.warning("Failed to activate instrumentation %s: %s", name, exc)

    @staticmethod
    @contextmanager
    def propagate_attributes(**kwargs):
        """Attach attributes to all spans exported within this scope.

        Attributes are propagated via ``contextvars`` — safe for concurrent
        async tasks.  Nested calls merge attributes (inner wins).

        Args:
            customer_identifier: User/customer identifier.
            customer_email: Customer email address.
            customer_name: Customer display name.
            thread_identifier: Conversation thread ID.
            custom_identifier: Indexed custom identifier.
            group_identifier: Group related traces.
            environment: Environment name (e.g. ``"production"``).
            metadata: Dict of custom key-value pairs (merged, not replaced).

        Example::

            with respan.propagate_attributes(
                customer_identifier="user_123",
                thread_identifier="conv_abc",
                metadata={"plan": "pro"},
            ):
                result = await Runner.run(agent, "Hello")
        """
        with _propagate_attributes(**kwargs):
            yield

    def flush(self) -> None:
        """Flush both the OTEL pipeline and the plugin exporter."""
        self.telemetry.flush()
        self.exporter.flush()

    def shutdown(self) -> None:
        """Deactivate plugins and shut down exporters."""
        for name, inst in self._instrumentations.items():
            try:
                inst.deactivate()  # type: ignore[union-attr]
            except Exception as exc:
                logger.warning("Error deactivating %s: %s", name, exc)
        self._instrumentations.clear()
        self.exporter.shutdown()
