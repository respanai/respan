"""One trace per ticket + outcome stamping, and the concurrent runner (plan §4.4, §8).

run_ticket:
  - propagate_attributes upfront -> customer/tenant/workflow/ticket land on every span.
  - wrap the run in an Agents SDK trace() so the whole ticket is one tree.
  - the emergent resolution is stamped at trace.finish() so it lands on the
    trace summary -> filterable via metadata__resolution::<value>.
"""

import asyncio
import uuid

import respan
from agents import Runner, trace

from . import prompts, telemetry
from .agents_setup import build_support_agent
from .config import SCENARIOS, TENANTS, Tenant
from .context import TicketContext


def _short_id() -> str:
    return uuid.uuid4().hex[:6]


async def run_ticket(tenant: Tenant, ticket_text: str) -> dict:
    """Run one customer ticket as one nested trace, stamping the outcome at finish."""
    ticket_id = f"TKT-{tenant.id[:3].upper()}-{_short_id()}"
    ctx = TicketContext(tenant_id=tenant.id, ticket_id=ticket_id)
    support_agent = build_support_agent(tenant)
    result = None

    # Which prompt drove this ticket (for the "Traffic on prompt vN" view).
    support_prompt = prompts.resolve("support", tenant)
    prompt_meta = (
        {"prompt_id": support_prompt.prompt_id,
         "prompt_version_number": support_prompt.version}
        if support_prompt.prompt_id
        else {}
    )

    # Known-upfront attributes -> land on every span (plan §3-item-1).
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
        t.start(mark_as_current=True)
        try:
            result = await Runner.run(support_agent, ticket_text, context=ctx)
        finally:
            # Emergent outcome: prefer what a terminal tool set; else infer from the
            # last agent (specialist with no terminal tool => escalated).
            resolution = ctx.resolution or (
                "escalated"
                if result is not None
                and getattr(result, "last_agent", None) is not None
                and result.last_agent.name == "billing_specialist"
                else "self_resolved"
            )
            # Stamp at trace.finish so the ROOT span (and trace summary) carries it.
            with respan.propagate_attributes(metadata={"resolution": resolution}):
                t.finish(reset_current=True)

    return {
        "ticket_id": ticket_id,
        "ticket": ticket_text,
        "tenant": tenant.display_name,
        "customer_identifier": tenant.customer_identifier,
        "resolution": resolution,
        "tools_used": ctx.tools_used,
        "prompt": f"{support_prompt.source} v{support_prompt.version}"
        if support_prompt.prompt_id
        else support_prompt.source,
        "final_output": getattr(result, "final_output", None),
    }


async def run_all() -> list[dict]:
    """Fire all scenarios concurrently across tenants (plan §8)."""
    telemetry.init_telemetry()
    tasks = [run_ticket(TENANTS[s.tenant_id], s.ticket) for s in SCENARIOS]
    results = await asyncio.gather(*tasks)
    telemetry.flush()
    return results
