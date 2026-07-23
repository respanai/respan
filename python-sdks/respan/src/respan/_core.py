"""Respan — unified entry point for tracing and instrumentation plugins."""

import json
import logging
import os
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

from respan_tracing import RespanTelemetry
from respan_tracing.utils.span_factory import (
    _PROPAGATED_ATTRIBUTES,
    build_readable_span,
    inject_span,
    propagate_attributes as _propagate_attributes,
)

from ._auto_instrumentation_registry import (
    AutoInstrumentationStatus,
    activate_auto_instrumentations,
)
from ._types import Instrumentation

logger = logging.getLogger(__name__)


class Respan:
    """Unified entry point for Respan tracing and instrumentation plugins.

    Sets up:
    1. ``RespanTelemetry`` — OTEL TracerProvider for decorators and export.
       Broad OTEL entry-point auto-discovery is disabled here so unrelated
       packages cannot create duplicate spans.
    2. Activates any instrumentors passed via the ``instrumentations`` list.
       Plugins emit ``ReadableSpan`` objects into the same OTEL pipeline.
    3. When auto-instrumentation is enabled, activates installed first-party
       Respan direct LLM SDK instrumentors from the curated registry.

    When ``instrumentations`` are provided, OTEL auto-instrumentation is
    disabled by default to avoid duplicate spans (plugins capture LLM calls
    themselves).  Override with ``is_auto_instrument=True`` if you need both.

    Args:
        api_key: Respan API key. Falls back to ``RESPAN_API_KEY`` env var.
        base_url: Respan API base URL. Falls back to ``RESPAN_BASE_URL`` env var.
        app_name: Application name for telemetry identification.
        instrumentations: List of instrumentor instances to activate.
        is_auto_instrument: Auto-instrument LLM SDKs (OpenAI, Anthropic, etc.)
            via Respan's curated direct-LLM registry.  Defaults to ``True``
            when no plugins are provided, ``False`` when plugins are provided
            (to avoid duplicate spans). Respan never forwards this flag to
            broad OTEL entry-point auto-discovery.
        customer_identifier: Default customer/user identifier for all spans.
        thread_identifier: Default conversation thread ID for all spans.
        metadata: Default metadata dict merged into all spans.
        environment: Default environment (e.g. ``"production"``).
        **telemetry_kwargs: Extra keyword arguments forwarded to
            ``RespanTelemetry`` (e.g. ``log_level``, ``is_batching_enabled``,
            ``auto_flush``).

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
        is_auto_instrument: Optional[bool] = None,
        customer_identifier: Optional[str] = None,
        session_identifier: Optional[str] = None,
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
        if session_identifier:
            default_attributes["session_identifier"] = session_identifier
        if thread_identifier:
            default_attributes["thread_identifier"] = thread_identifier
        if metadata:
            default_attributes["metadata"] = metadata
        if environment:
            default_attributes["environment"] = environment

        # When no explicit instrumentations provided, auto-instrument LLM SDKs.
        # When instrumentations=[...] is passed, disable auto-instrumentation
        # to avoid duplicate spans (plugins handle tracing themselves).
        if is_auto_instrument is None:
            is_auto_instrument = instrumentations is None

        # 1. OTEL TracerProvider. Keep respan-tracing broad OTEL entry-point
        # auto-discovery off; Respan auto mode is handled by the curated
        # direct-LLM registry below.
        self.telemetry = RespanTelemetry(
            app_name=app_name,
            api_key=api_key,
            base_url=base_url,
            is_auto_instrument=False,
            **telemetry_kwargs,
        )

        # 2. Seed propagated attributes with defaults so all spans
        #    (both auto-instrumented and plugin-injected) get them.
        if default_attributes:
            _PROPAGATED_ATTRIBUTES.set(default_attributes)

        # 3. Activate explicit instrumentations first, so user-provided
        # instances and constructor options win over auto-discovered defaults.
        self._instrumentations: Dict[str, object] = {}
        self._auto_instrumentation_status: Tuple[AutoInstrumentationStatus, ...] = ()
        for inst in instrumentations or []:
            name = getattr(inst, "name", type(inst).__name__)
            self._activate(name, inst)

        # 4. Activate installed direct LLM SDK instrumentations from Respan's
        # curated registry when auto mode is enabled.
        if is_auto_instrument:
            activations = activate_auto_instrumentations(
                already_activated=self._instrumentations.keys(),
            )
            self._auto_instrumentation_status = tuple(
                activation.status for activation in activations
            )
            for activation in activations:
                if activation.instrumentor is not None:
                    self._instrumentations[activation.status.name] = (
                        activation.instrumentor
                    )

    @property
    def auto_instrumentation_status(self) -> Tuple[AutoInstrumentationStatus, ...]:
        """Status entries from the most recent Respan auto-instrumentation run."""

        return self._auto_instrumentation_status

    def get_auto_instrumentation_status(self) -> List[Dict[str, Optional[str]]]:
        """Return auto-instrumentation status as JSON-serializable dicts."""

        return [
            {
                "id": status.id,
                "name": status.name,
                "status": status.status,
                "provider": status.provider,
                "sdk_package": status.sdk_package,
                "instrumentation_package": status.instrumentation_package,
                "reason": status.reason,
            }
            for status in self._auto_instrumentation_status
        ]

    def _activate(self, name: str, inst: object) -> bool:
        """Activate a single instrumentor."""
        if name in self._instrumentations:
            logger.info("Instrumentation already activated: %s", name)
            return True
        try:
            inst.activate()  # type: ignore[union-attr]
            if getattr(inst, "_is_instrumented", True) is False:
                logger.info("Instrumentation did not activate: %s", name)
                return False
            self._instrumentations[name] = inst
            logger.info("Activated instrumentation: %s", name)
            return True
        except Exception as exc:
            logger.warning("Failed to activate instrumentation %s: %s", name, exc)
            return False

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

        completion_timestamps = []

        # When no OTEL context, create a grouping "batch_results" task span
        # so completions are nested, not floating at trace root.
        if not otel_span_id:
            # We'll set timestamps after processing all results
            pass

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
            start_iso = end_iso = None
            if created:
                ts = datetime.fromtimestamp(created, tz=timezone.utc)
                end_iso = ts.isoformat()
                completion_timestamps.append(ts)

            model = body.get("model", original.get("model", ""))

            span = build_readable_span(
                name=f"batch:{custom_id}",
                trace_id=resolved_trace_id,
                parent_id=parent_span_id,
                end_time_iso=end_iso,
                attributes={
                    "llm.request.type": "chat",
                    "gen_ai.request.model": model,
                    "gen_ai.usage.prompt_tokens": usage.get("prompt_tokens", 0),
                    "gen_ai.usage.completion_tokens": usage.get("completion_tokens", 0),
                    "traceloop.entity.input": json.dumps(messages, default=str),
                    "traceloop.entity.output": json.dumps(output, default=str),
                    "traceloop.entity.path": "batch_results",
                    "traceloop.span.kind": "task",
                    "respan.entity.log_type": "chat",
                },
                status_code=status_code,
            )
            inject_span(span)

        # Create the grouping "batch_results" task span (when no OTEL context)
        if not otel_span_id:
            earliest_iso = latest_iso = None
            if completion_timestamps:
                earliest_iso = min(completion_timestamps).isoformat()
                latest_iso = max(completion_timestamps).isoformat()

            parent_span = build_readable_span(
                name="batch_results.task",
                trace_id=resolved_trace_id,
                span_id=parent_span_id,
                start_time_iso=earliest_iso,
                end_time_iso=latest_iso,
                attributes={
                    "traceloop.span.kind": "task",
                    "traceloop.entity.name": "batch_results",
                    "traceloop.entity.path": "",
                    "respan.entity.log_type": "task",
                },
            )
            inject_span(parent_span)

    def flush(self) -> None:
        """Flush the OTEL pipeline."""
        self.telemetry.flush()

    def shutdown(self) -> None:
        """Deactivate plugins and shut down the OTEL pipeline."""
        for name, inst in self._instrumentations.items():
            try:
                inst.deactivate()  # type: ignore[union-attr]
            except Exception as exc:
                logger.warning("Error deactivating %s: %s", name, exc)
        self._instrumentations.clear()
        self.telemetry.flush()
