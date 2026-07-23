"""Agent instructions: managed prompts with inline fallbacks (plan §6, build step 3).

resolve() fetches a tenant's deployed managed prompt, extracts the system
message, and renders {{display_name}}/{{industry}} locally - set as the agent's
instructions. This per-agent local render avoids the one-prompt-per-client limit
of the gateway header approach. If the fetch fails or no prompt_id is configured,
it falls back to the inline strings. The resolved prompt_id/version are stamped
into trace metadata by the runner (the "Traffic on prompt vN" view).

Resolution order per role: (1) prompt_id from prompt_ids.json (written by
seed_prompts.py); (2) look the prompt up by its canonical name (PROMPT_NAMES);
(3) inline fallback. So the demo works whether or not the seeder has been run.
"""

import os
from dataclasses import dataclass

import httpx

from .config import PROMPT_NAMES, Tenant

_BASE = os.getenv("RESPAN_BASE_URL", "https://api.respan.ai/api").rstrip("/")

_INLINE = {
    "support": (
        "You are the front-line support agent for {display_name}, a {industry} company.\n"
        "You are autonomous: decide what to do for each ticket. Use your tools.\n"
        "- Use search_kb for how-to / informational questions, then close_resolved with a short answer.\n"
        "- Use lookup_order / lookup_account to verify claims before acting.\n"
        "- Use process_refund for verified duplicate charges or damaged items.\n"
        "- Hand off to the billing specialist for contract / enterprise billing disputes.\n"
        "- Use escalate_to_human only when a human must intervene.\n"
        "Always take exactly one terminal action (process_refund, close_resolved, "
        "escalate_to_human) or hand off. Be concise."
    ),
    "specialist": (
        "You are the billing specialist for {display_name} ({industry}).\n"
        "You handle escalated contract and enterprise-billing disputes that front-line "
        "support routed to you. Review the details, use lookup_account / lookup_order as needed, "
        "and take a terminal action: process_refund if clearly warranted, otherwise "
        "escalate_to_human so a person reviews the contract. Be concise."
    ),
}


@dataclass(frozen=True)
class ResolvedPrompt:
    instructions: str
    prompt_id: str | None
    version: int | None
    source: str  # "managed" | "inline"


# prompt_id -> (system_message_template, version). Prompts don't change mid-run.
_cache: dict[str, tuple[str, int | None]] = {}


def _fetch_managed(prompt_id: str) -> tuple[str, int | None] | None:
    """GET the deployed prompt; return its system-message template + version."""
    if prompt_id in _cache:
        return _cache[prompt_id]
    key = os.getenv("RESPAN_API_KEY")
    try:
        resp = httpx.get(
            f"{_BASE}/prompts/{prompt_id}/",
            headers={"Authorization": f"Bearer {key}"},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return None
    version = data.get("live_version") or data.get("current_version")
    if not isinstance(version, dict):
        return None
    texts: list[str] = []
    for msg in version.get("messages", []):
        if msg.get("role") != "system":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, list):
            texts.extend(c["text"] for c in content if isinstance(c, dict) and c.get("text"))
    if not texts:
        return None
    result = ("\n".join(texts), version.get("version"))
    _cache[prompt_id] = result
    return result


def _render(template: str, tenant: Tenant) -> str:
    return template.replace("{{display_name}}", tenant.display_name).replace(
        "{{industry}}", tenant.industry
    )


# Backup path: find a prompt's id by its canonical name (if no prompt_ids.json).
_name_cache: dict[str, str | None] = {}


def _resolve_id_by_name(name: str | None) -> str | None:
    if not name:
        return None
    if name in _name_cache:
        return _name_cache[name]
    key = os.getenv("RESPAN_API_KEY")
    result: str | None = None
    try:
        resp = httpx.post(
            f"{_BASE}/prompts/list/",
            headers={"Authorization": f"Bearer {key}"},
            json={"page_size": 100, "filters": {"name": {"operator": "iexact", "value": [name]}}},
            timeout=30,
        )
        resp.raise_for_status()
        body = resp.json()
        rows = body.get("results") or body.get("response", {}).get("results", [])
        for row in rows:
            if row.get("name") == name and row.get("id"):
                result = row["id"]
                break
    except Exception:
        result = None
    _name_cache[name] = result
    return result


def resolve(role: str, tenant: Tenant) -> ResolvedPrompt:
    """Resolve instructions for a role: prompt_ids.json -> by-name -> inline."""
    prompt_id = tenant.prompt_ids.get(role) or _resolve_id_by_name(PROMPT_NAMES.get(role))
    if prompt_id:
        fetched = _fetch_managed(prompt_id)
        if fetched:
            template, version = fetched
            return ResolvedPrompt(_render(template, tenant), prompt_id, version, "managed")
    inline = _INLINE[role].format(display_name=tenant.display_name, industry=tenant.industry)
    return ResolvedPrompt(inline, None, None, "inline")


def get_instructions(role: str, tenant: Tenant) -> str:
    return resolve(role, tenant).instructions
