"""Respan — unified entry point for tracing and instrumentation plugins."""

import json
import logging
import os
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
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

        # Extract config from instrumentors (so instrumentor-level config
        # is used as fallback if Respan-level isn't set)
        for inst in instrumentations or []:
            if not customer_identifier:
                customer_identifier = getattr(inst, "customer_identifier", None)
            if not environment:
                environment = getattr(inst, "environment", None)

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
        # Pass environment and identifiers as resource attributes so they're
        # available in the OTLP pipeline (not just the V2 plugin exporter).
        ra = telemetry_kwargs.get("resource_attributes") or {}
        if environment:
            ra["deployment.environment"] = environment
        if customer_identifier:
            ra["respan.customer_params.customer_identifier"] = customer_identifier
        if thread_identifier:
            ra["respan.threads.thread_identifier"] = thread_identifier
        if ra:
            telemetry_kwargs["resource_attributes"] = ra

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
            # Register OTLP enrichers for plugins that enrich spans
            if name == "google-adk":
                from respan_tracing.exporters.adk_enrichment import _enrich_adk_spans
                self.telemetry.register_enricher(_enrich_adk_spans)

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
            prompt: Dict with ``prompt_id`` and ``variables`` for prompt
                logging.  The backend resolves the template automatically.

        Example::

            with respan.propagate_attributes(
                customer_identifier="user_123",
                thread_identifier="conv_abc",
                metadata={"plan": "pro"},
            ):
                result = await Runner.run(agent, "Hello")

            with respan.propagate_attributes(
                prompt={"prompt_id": "abc123", "variables": {"x": "y"}},
            ):
                result = await Runner.run(agent, "Hello")
        """
        with _propagate_attributes(**kwargs):
            yield

    def log_batch_results(
        self,
        requests: List[Dict[str, Any]],
        results: List[Dict[str, Any]],
        trace_id: Optional[str] = None,
    ) -> None:
        """Log OpenAI Batch API results as individual chat completion spans.

        Trace linking (in priority order):

        1. **OTEL context** — when called inside a ``@task`` / ``@workflow``
           decorated function, auto-links to the active trace and nests
           completions under the current span.
        2. **Explicit** ``trace_id`` — for async batches where results
           arrive in a separate process (e.g. 24 hours later).  Adds a
           ``batch_results`` task span to the original trace with
           completions nested underneath.
        3. **Auto-generated** — creates a new standalone trace if neither
           is available.

        Args:
            requests: Original batch request dicts (from the input JSONL).
                Each must have ``custom_id`` and ``body.messages``.
            results: Batch result dicts (from the output JSONL).
                Each must have ``custom_id`` and ``response.body``.
            trace_id: Explicit trace ID to link results to.  Use this for
                async batches where results arrive in a separate process.

        Examples::

            # Same process — auto-links to active OTEL span
            @task(name="download_results")
            def download_results(output_file_id: str):
                ...
                respan.log_batch_results(requests, results)

            # Different process (24h later) — links back to original trace
            respan.log_batch_results(requests, results, trace_id=saved_trace_id)
        """
        from respan_tracing import get_client

        # Resolve trace context: OTEL > explicit > auto-generated.
        # OTEL returns all-zero IDs when no active span — treat as absent.
        rc = get_client()
        otel_trace_id = rc.get_current_trace_id()
        otel_span_id = rc.get_current_span_id()
        if otel_trace_id and int(otel_trace_id, 16) == 0:
            otel_trace_id = None
        if otel_span_id and int(otel_span_id, 16) == 0:
            otel_span_id = None
        resolved_trace_id = otel_trace_id or trace_id or uuid.uuid4().hex

        # Determine the parent for completion spans.
        # With OTEL context: nest under the active span directly.
        # Without: create a synthetic "batch_results" task span.
        if otel_span_id:
            parent_span_id = otel_span_id
        else:
            parent_span_id = uuid.uuid4().hex

        # Index original requests by custom_id
        requests_by_id = {r["custom_id"]: r.get("body", {}) for r in requests}

        logs = []

        # When no OTEL context, create a grouping "batch_results" task span
        # so completions are nested, not floating at trace root.
        if not otel_span_id:
            logs.append({
                "trace_unique_id": resolved_trace_id,
                "span_unique_id": parent_span_id,
                "span_name": "batch_results",
                "log_type": "task",
                "status_code": 200,
                "status": "success",
            })

        completion_timestamps = []

        for result in results:
            custom_id = result.get("custom_id", "")
            response = result.get("response", {})
            body = response.get("body", {})
            status_code = response.get("status_code", 200)

            # Match with original request
            original = requests_by_id.get(custom_id, {})
            messages = original.get("messages", [])

            # Extract completion and usage
            choices = body.get("choices", [{}])
            output = choices[0].get("message", {}) if choices else {}
            usage = body.get("usage", {})

            # Extract timestamp from OpenAI response (unix epoch → ISO 8601)
            created = body.get("created")
            timestamp = None
            if created:
                ts = datetime.fromtimestamp(created, tz=timezone.utc)
                timestamp = ts.isoformat()
                completion_timestamps.append(ts)

            log: Dict[str, Any] = {
                "trace_unique_id": resolved_trace_id,
                "span_unique_id": uuid.uuid4().hex,
                "span_parent_id": parent_span_id,
                "span_name": f"batch:{custom_id}",
                "log_type": "chat",
                "model": body.get("model", original.get("model", "")),
                "input": json.dumps(messages),
                "output": output,
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "status_code": status_code,
                "status": "error" if status_code >= 400 else "success",
            }
            if timestamp:
                log["timestamp"] = timestamp
            logs.append(log)

        # Set the batch_results parent span timestamps to cover all completions
        if not otel_span_id and completion_timestamps and logs:
            earliest = min(completion_timestamps).isoformat()
            latest = max(completion_timestamps).isoformat()
            logs[0]["start_time"] = earliest
            logs[0]["timestamp"] = latest

        if logs:
            self.exporter.export(logs)

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
