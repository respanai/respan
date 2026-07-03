"""Respan — unified entry point for tracing and instrumentation plugins."""

import importlib.metadata
import importlib.util
import json
import logging
import os
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

from respan_tracing import RespanTelemetry
from respan_tracing.instruments import Instruments
from respan_tracing.utils.span_factory import (
    _PROPAGATED_ATTRIBUTES,
    build_readable_span,
    inject_span,
    propagate_attributes as _propagate_attributes,
)

from ._types import Instrumentation

logger = logging.getLogger(__name__)

# Entry-point group that native Respan instrumentation plugins register under.
_NATIVE_INSTRUMENTATION_GROUP = "respan.instrumentations"

# Only the instrumentations bundled as ``respan-ai`` dependencies auto-activate
# on a bare ``Respan()``.  Every other plugin registered under
# ``respan.instrumentations`` (litellm, langchain, crewai, cohere, mistralai,
# groq, ...) must be passed explicitly via ``Respan(instrumentations=[...])`` —
# without this allowlist a bare ``Respan()`` would silently activate whatever
# happens to be installed.
#
# Each entry maps its entry-point name to the Traceloop OTEL ``Instruments`` it
# supersedes, so auto mode can block those in the ``RespanTelemetry`` pipeline
# and never wrap a provider twice (native plugin + OTEL instrumentor = two spans
# and doubled token/cost).  To add a provider later: add its
# ``respan-instrumentation-*`` dependency in ``pyproject.toml`` and one row here.
_BUNDLED_NATIVE_INSTRUMENTATIONS: Dict[str, tuple] = {
    "openai": (Instruments.OPENAI,),  # covers OpenAI + Azure OpenAI
    "anthropic": (Instruments.ANTHROPIC,),
    "aws-bedrock": (Instruments.BEDROCK,),
    "vertexai": (Instruments.VERTEXAI,),
    "google-genai": (Instruments.GOOGLE_GENERATIVEAI,),
    "together": (Instruments.TOGETHER,),
    "ollama": (Instruments.OLLAMA,),
}

# Top-level provider SDK package for each bundled native. Used only to detect the
# "provider SDK is installed but its native instrumentor never activated" case and
# warn about it — the safety net for the allowlist/dedup logic above, so a
# bundling gap or a failed load doesn't leave an installed SDK silently untraced.
# Probed with find_spec (not import), so checking a provider the app doesn't use
# has no side effects.
_PROVIDER_SDK_MODULES: Dict[str, tuple] = {
    "openai": ("openai",),
    "anthropic": ("anthropic",),
    "aws-bedrock": ("botocore",),  # boto3's client layer
    "vertexai": ("vertexai",),
    "google-genai": ("google.genai",),
    "together": ("together",),
    "ollama": ("ollama",),
}


def _provider_sdk_installed(modules: tuple) -> bool:
    """True if any of ``modules`` is importable (installed), without importing it."""
    for name in modules:
        try:
            if importlib.util.find_spec(name) is not None:
                return True
        except (ImportError, ModuleNotFoundError, ValueError):
            # Parent package absent, name isn't a package, etc. — treat as absent.
            continue
    return False


def _discover_native_instrumentations() -> List[tuple]:
    """Instantiate the bundled native ``respan.instrumentations`` plugins.

    Only plugins in :data:`_BUNDLED_NATIVE_INSTRUMENTATIONS` are considered; any
    other plugin registered under the group must be activated explicitly via
    ``Respan(instrumentations=[...])``.  Provider SDKs are optional extras, so a
    bundled plugin whose SDK can't be imported is skipped quietly (DEBUG) rather
    than erroring — a bare ``Respan()`` never spams about providers the app
    doesn't use.

    Returns a list of ``(entry_point_name, instrumentor_instance)`` tuples.
    """
    plugins: List[tuple] = []
    try:
        entry_points = importlib.metadata.entry_points(
            group=_NATIVE_INSTRUMENTATION_GROUP
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Failed to discover native instrumentations: %s", exc)
        return plugins

    for ep in entry_points:
        if ep.name not in _BUNDLED_NATIVE_INSTRUMENTATIONS:
            # Not bundled — only auto-activates when passed explicitly.
            continue
        try:
            plugins.append((ep.name, ep.load()()))
        except (ImportError, ModuleNotFoundError) as exc:
            # Target SDK not importable — expected when that SDK isn't installed.
            logger.debug(
                "Skipping %s instrumentation (SDK not available): %s", ep.name, exc
            )
        except Exception as exc:
            logger.warning("Failed to load %s instrumentation: %s", ep.name, exc)
    return plugins


class Respan:
    """Unified entry point for Respan tracing and instrumentation plugins.

    Sets up:
    1. ``RespanTelemetry`` — OTEL TracerProvider for decorators and, when no
       plugins are provided, auto-instrumentation of LLM SDKs (OpenAI,
       Anthropic, etc.) via the OTEL pipeline.
    2. Activates any instrumentors passed via the ``instrumentations`` list.
       Plugins emit ``ReadableSpan`` objects into the same OTEL pipeline.

    When ``instrumentations`` are provided, OTEL auto-instrumentation is
    disabled by default to avoid duplicate spans (plugins capture LLM calls
    themselves).  Override with ``is_auto_instrument=True`` if you need both.

    Args:
        api_key: Respan API key. Falls back to ``RESPAN_API_KEY`` env var.
        base_url: Respan API base URL. Falls back to ``RESPAN_BASE_URL`` env var.
        app_name: Application name for telemetry identification.
        instrumentations: List of instrumentor instances to activate.
        is_auto_instrument: Auto-instrument LLM SDKs (OpenAI, Anthropic, etc.)
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

        # In auto mode, discover the bundled native plugins *before* building
        # RespanTelemetry so we can tell its OTEL (Traceloop) pipeline to skip
        # the providers those natives already cover.  Otherwise a provider that
        # has both an installed native plugin and an OTEL instrumentor gets
        # wrapped twice — two spans and doubled token/cost per LLM call.
        # Instantiating a plugin only constructs it; patching happens later in
        # _activate(), after the tracer provider exists.
        native_instrumentations = (
            _discover_native_instrumentations() if instrumentations is None else []
        )
        blocked = set(telemetry_kwargs.pop("block_instruments", None) or set())
        for ep_name, _inst in native_instrumentations:
            blocked.update(_BUNDLED_NATIVE_INSTRUMENTATIONS[ep_name])

        # 1. OTEL TracerProvider + optional auto-instrumentation
        self.telemetry = RespanTelemetry(
            app_name=app_name,
            api_key=api_key,
            base_url=base_url,
            is_auto_instrument=is_auto_instrument,
            block_instruments=blocked or None,
            **telemetry_kwargs,
        )

        # 2. Seed propagated attributes with defaults so all spans
        #    (both auto-instrumented and plugin-injected) get them.
        if default_attributes:
            _PROPAGATED_ATTRIBUTES.set(default_attributes)

        # 3. Activate instrumentations
        self._instrumentations: Dict[str, object] = {}

        # 3a. Auto mode (no explicit plugins): activate the bundled native
        #     respan-instrumentation-* plugins discovered above so a bare
        #     Respan() traces the direct LLM SDKs out of the box.  Their OTEL
        #     counterparts were blocked above, so each provider is traced once.
        activated_native_names = set()
        for ep_name, inst in native_instrumentations:
            name = getattr(inst, "name", type(inst).__name__)
            if self._activate(name, inst):
                activated_native_names.add(ep_name)

        # 3a-safety. Warn when a bundled provider's SDK is installed but its native
        #     instrumentor never activated (bundling gap, missing entry point, or a
        #     load failure).  Without this, an installed SDK that isn't covered by a
        #     native plugin — and whose OTEL twin may have been blocked above — would
        #     produce no LLM spans with no signal.  (A plugin that ran but found its
        #     SDK absent or incompatible already logs for itself, so it is not
        #     re-flagged here.)
        if instrumentations is None:
            for ep_name, sdk_modules in _PROVIDER_SDK_MODULES.items():
                if ep_name in activated_native_names:
                    continue
                if _provider_sdk_installed(sdk_modules):
                    logger.warning(
                        "%s SDK is installed but its Respan instrumentation did not "
                        "activate; its LLM calls will not be traced. Reinstall "
                        "respan-instrumentation-%s, or pass the instrumentor via "
                        "Respan(instrumentations=[...]).",
                        ep_name,
                        ep_name,
                    )

        # 3b. Explicitly-passed instrumentations.
        for inst in instrumentations or []:
            name = getattr(inst, "name", type(inst).__name__)
            self._activate(name, inst)

    def _activate(self, name: str, inst: object) -> bool:
        """Activate a single instrumentor. Returns False if activation raised."""
        try:
            inst.activate()  # type: ignore[union-attr]
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
