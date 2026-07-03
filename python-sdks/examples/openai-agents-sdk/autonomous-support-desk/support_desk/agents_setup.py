"""Agent construction (plan §4.1).

One autonomous support_agent with all tools + a handoff edge to the
billing_specialist. The specialist's name is "billing-specialist" so the
run_ticket fallback can detect escalation from result.last_agent.
"""

from agents import Agent

from . import tools
from .config import Tenant
from .prompts import get_instructions


def build_support_agent(tenant: Tenant) -> Agent:
    """Build the per-tenant support agent (with its specialist handoff target)."""
    specialist = Agent(
        name="billing_specialist",
        model=tenant.model,
        instructions=get_instructions("specialist", tenant),
        tools=[
            tools.lookup_account,
            tools.lookup_order,
            tools.process_refund,
            tools.escalate_to_human,
        ],
    )

    # Note: the support agent has NO escalate_to_human - for contract / enterprise
    # billing disputes it must hand off to the billing specialist, who then takes
    # the terminal action. This guarantees the handoff path actually fires.
    support = Agent(
        name="support-agent",
        model=tenant.model,
        instructions=get_instructions("support", tenant),
        tools=[
            tools.search_kb,
            tools.lookup_account,
            tools.lookup_order,
            tools.process_refund,
            tools.close_resolved,
        ],
        handoffs=[specialist],
    )
    return support
