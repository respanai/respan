"""Streaming layer over the spine (plan §9).

run_ticket_streamed is an async generator yielding NDJSON-serializable event
dicts as a ticket is processed - the seam a UI (or the terminal animator in
run_stream.py) attaches to. It mirrors runner.run_ticket exactly (same upfront
propagation, one trace() per ticket, resolution stamp at finish) but uses
Runner.run_streamed so per-step events surface live.
"""

import uuid
from typing import AsyncIterator

import respan
from agents import ItemHelpers, Runner, trace
from respan_sdk.utils.data_processing.id_processing import ensure_trace_id

from . import prompts
from .agents_setup import build_support_agent
from .config import Tenant
from .context import TicketContext


def _short_id() -> str:
    return uuid.uuid4().hex[:6]


def _map_event(ev, ticket_id: str) -> dict | None:
    et = getattr(ev, "type", None)
    if et == "agent_updated_stream_event":
        return {"type": "agent", "ticket_id": ticket_id, "agent": ev.new_agent.name}
    if et == "run_item_stream_event":
        name, item = ev.name, ev.item
        if name == "tool_called":
            raw = getattr(item, "raw_item", None)
            return {"type": "tool_call", "ticket_id": ticket_id,
                    "tool": getattr(raw, "name", None) or "tool"}
        if name == "tool_output":
            return {"type": "tool_output", "ticket_id": ticket_id,
                    "output": str(getattr(item, "output", ""))[:800]}
        if name in ("handoff_requested", "handoff_occured"):
            return {"type": "handoff", "ticket_id": ticket_id, "event": name}
        if name == "message_output_created":
            try:
                text = ItemHelpers.text_message_output(item)
            except Exception:
                text = ""
            return {"type": "message", "ticket_id": ticket_id, "text": text[:800]}
    return None


async def run_ticket_streamed(tenant: Tenant, ticket_text: str) -> AsyncIterator[dict]:
    """Yield per-step events for one ticket; emits one nested trace, like run_ticket."""
    ticket_id = f"TKT-{tenant.id[:3].upper()}-{_short_id()}"
    ctx = TicketContext(tenant_id=tenant.id, ticket_id=ticket_id)
    support_agent = build_support_agent(tenant)
    support_prompt = prompts.resolve("support", tenant)
    prompt_meta = (
        {"prompt_id": support_prompt.prompt_id, "prompt_version_number": support_prompt.version}
        if support_prompt.prompt_id
        else {}
    )

    yield {"type": "ticket_start", "ticket_id": ticket_id,
           "tenant": tenant.display_name, "ticket": ticket_text,
           "prompt": support_prompt.source, "prompt_version": support_prompt.version}

    result = None
    trace_uid = None
    with respan.propagate_attributes(
        customer_identifier=tenant.customer_identifier,
        thread_identifier=ticket_id,
        group_identifier="support-desk",
        metadata={
            "tenant": tenant.display_name,
            "industry": tenant.industry,
            "workflow": "support",
            "ticket_id": ticket_id,
            **prompt_meta,
        },
    ):
        t = trace("support-session", group_id=tenant.customer_identifier)
        # The platform trace_unique_id is the emitter's ensure_trace_id() of the
        # SDK trace id (it MD5s the non-hex "trace_..." form) - reproduce it here.
        trace_uid = format(ensure_trace_id(t.trace_id), "032x")
        t.start(mark_as_current=True)
        try:
            streamed = Runner.run_streamed(support_agent, ticket_text, context=ctx)
            async for ev in streamed.stream_events():
                mapped = _map_event(ev, ticket_id)
                if mapped:
                    yield mapped
            result = streamed
        finally:
            resolution = ctx.resolution or (
                "escalated"
                if result is not None
                and getattr(result, "last_agent", None) is not None
                and result.last_agent.name == "billing_specialist"
                else "self_resolved"
            )
            with respan.propagate_attributes(metadata={"resolution": resolution}):
                t.finish(reset_current=True)

    yield {"type": "ticket_done", "ticket_id": ticket_id, "tenant": tenant.display_name,
           "resolution": resolution, "tools_used": ctx.tools_used, "trace_id": trace_uid,
           "final_output": getattr(result, "final_output", None)}
