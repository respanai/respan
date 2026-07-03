"""Respan init + the gateway-routed OpenAI client (plan §6, §10).

Tracing uses a single instrumentor: OpenAIAgentsInstrumentor traces the Agents
SDK (Responses API), and it owns the live trace context for the whole run. The
direct embeddings call inside search_kb is nested into that same trace by an
Agents-SDK custom_span (see tools.search_kb), so it shows up under the ticket
instead of orphaning into its own trace.

The single AsyncOpenAI client points at the Respan gateway, so every LLM call
and the embedding route through Respan.
"""

import os

from openai import AsyncOpenAI
from respan import Respan
from respan_instrumentation_openai_agents import OpenAIAgentsInstrumentor
from agents import set_default_openai_client

_telemetry: Respan | None = None
_client: AsyncOpenAI | None = None


def init_telemetry() -> tuple[Respan, AsyncOpenAI]:
    """Idempotently start tracing and wire the gateway client as the SDK default."""
    global _telemetry, _client
    if _telemetry is not None and _client is not None:
        return _telemetry, _client

    _telemetry = Respan(
        instrumentations=[OpenAIAgentsInstrumentor()],
    )
    _client = AsyncOpenAI(
        api_key=os.getenv("RESPAN_API_KEY"),
        base_url=os.getenv("RESPAN_BASE_URL", "https://api.respan.ai/api"),
    )
    # Route the Agents SDK's model calls through the same gateway client.
    set_default_openai_client(_client)
    return _telemetry, _client


def get_gateway_client() -> AsyncOpenAI:
    """Return the gateway client (for the direct embeddings call in tools)."""
    if _client is None:
        raise RuntimeError("init_telemetry() must be called before get_gateway_client()")
    return _client


def flush() -> None:
    """Force-export buffered spans before the process exits."""
    if _telemetry is not None:
        _telemetry.flush()
