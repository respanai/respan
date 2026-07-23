"""Tenants, ticket scenarios, mock backends, and the KB corpus (plan §2, §9).

Multi-tenancy: 2 tenants, different industries, each with its own
customer_identifier and model. Scenarios are deliberately varied so outcomes
differ (refund / self-resolve / escalate) - that variety is what makes the
Views catalog look alive.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Tenant:
    id: str
    display_name: str
    industry: str
    customer_identifier: str
    model: str
    # {role: prompt_id} - empty/None means resolve by name, then inline fallback.
    prompt_ids: dict = field(default_factory=dict)


# Canonical names of the two shared managed prompts. seed_prompts.py creates them
# under these names; prompts.resolve() falls back to looking them up by name if no
# prompt_ids.json is present. Keep in sync with seed_prompts.py.
PROMPT_NAMES: dict[str, str] = {
    "support": "Support Agent — Autonomous Desk (demo)",
    "specialist": "Billing Specialist — Autonomous Desk (demo)",
}

# Prompt IDs are account-specific, so they are NOT hardcoded. `seed_prompts.py`
# creates the prompts in whatever account RESPAN_API_KEY points to and writes the
# resulting {role: id} map to prompt_ids.json (git-ignored). We load it if present;
# otherwise prompts.resolve() resolves by name, then falls back to inline prompts.
_IDS_FILE = Path(__file__).resolve().parent.parent / "prompt_ids.json"


def _load_prompt_ids() -> dict[str, str]:
    try:
        return json.loads(_IDS_FILE.read_text())
    except Exception:
        return {}


PROMPT_IDS: dict[str, str] = _load_prompt_ids()


TENANTS: dict[str, Tenant] = {
    "acme": Tenant(
        id="acme",
        display_name="Acme Retail",
        industry="consumer retail",
        customer_identifier="acme-retail",
        model="gpt-4.1-nano",
        prompt_ids=PROMPT_IDS,
    ),
    "globex": Tenant(
        id="globex",
        display_name="Globex SaaS",
        industry="B2B SaaS",
        customer_identifier="globex-saas",
        model="gpt-4.1-mini",
        prompt_ids=PROMPT_IDS,
    ),
}


@dataclass(frozen=True)
class Scenario:
    tenant_id: str
    ticket: str
    note: str  # expected path - for our reference only, never fed to the agent


# Varied on purpose: a refund, a KB self-resolve, and a billing escalation.
SCENARIOS: list[Scenario] = [
    Scenario(
        tenant_id="acme",
        ticket="I was charged twice for order A-1001 - $59.98 instead of $29.99. "
        "Please refund the duplicate charge.",
        note="expect: lookup_order -> process_refund (refunded)",
    ),
    Scenario(
        tenant_id="globex",
        ticket="I'm locked out and can't sign in. How do I reset my account password?",
        note="expect: search_kb -> close_resolved (self_resolved)",
    ),
    Scenario(
        tenant_id="acme",
        ticket="My order A-2050 arrived damaged. It was a $129.00 order - I'd like a refund.",
        note="expect: lookup_order -> process_refund (refunded)",
    ),
    Scenario(
        tenant_id="globex",
        ticket="I'm formally disputing our $4,000 enterprise renewal charge and need a "
        "human in billing to review the signed contract terms before we pay.",
        note="expect: handoff -> billing-specialist -> escalate_to_human (escalated)",
    ),
]


# --- Mock backends (tools return small dicts) -------------------------------

MOCK_ORDERS: dict[str, dict] = {
    "A-1001": {"order_id": "A-1001", "item": "Wireless Mouse", "amount": 29.99,
               "charged": 59.98, "status": "duplicate_charge"},
    "A-2050": {"order_id": "A-2050", "item": "Standing Desk Mat", "amount": 129.00,
               "charged": 129.00, "status": "delivered_damaged"},
}

MOCK_ACCOUNTS: dict[str, dict] = {
    "acme-retail": {"customer_id": "acme-retail", "tier": "consumer", "lifetime_value": 412.50},
    "globex-saas": {"customer_id": "globex-saas", "tier": "enterprise", "seats": 240,
                    "contract_value": 48000},
}


# --- KB corpus for search_kb (per tenant) -----------------------------------

KB: dict[str, list[dict]] = {
    "acme": [
        {"title": "Refunds & duplicate charges",
         "body": "Duplicate charges are refunded to the original payment method within 5-7 days. "
                 "Agents can issue refunds for verified duplicate or damaged-item orders."},
        {"title": "Damaged on arrival",
         "body": "Items damaged in transit qualify for a full refund or replacement; no return needed "
                 "for orders under $150."},
        {"title": "Order tracking",
         "body": "Track an order with its A-#### id from the Orders page."},
    ],
    "globex": [
        {"title": "Reset your password",
         "body": "Go to Sign in > Forgot password, enter your work email, and follow the reset link. "
                 "Links expire after 30 minutes. SSO users reset via their identity provider."},
        {"title": "Locked-out accounts",
         "body": "After 5 failed attempts an account locks for 15 minutes, then unlocks automatically."},
        {"title": "Enterprise billing disputes",
         "body": "Renewal and contract disputes are handled by the billing specialist team, not "
                 "front-line support. Escalate with the contract reference."},
    ],
}
