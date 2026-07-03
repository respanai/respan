"""Create + deploy the demo's managed prompts (plan §6, build step 3).

Two shared, versioned prompts (support + specialist) authored with Jinja
variables ({{display_name}}, {{industry}}) so each tenant renders the same
deployed prompt with its own values. Idempotent: dedupes by name.

Sequence (verified): createPrompt -> createPromptVersion -> commit -> deploy.
A freshly created version is an uncommitted draft; deploy fails until committed.

    python seed_prompts.py        # creates the prompts and writes prompt_ids.json
"""

import json
import os
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv()

from support_desk.config import PROMPT_NAMES  # noqa: E402  (shared canonical names)

BASE = os.getenv("RESPAN_BASE_URL", "https://api.respan.ai/api").rstrip("/")
KEY = os.getenv("RESPAN_API_KEY")
HEADERS = {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}
IDS_FILE = Path(__file__).resolve().parent / "prompt_ids.json"


def _sys(text: str) -> dict:
    return {"role": "system", "content": [{"text": text, "type": "text"}]}


def _user(text: str) -> dict:
    return {"role": "user", "content": [{"text": text, "type": "text"}]}


# Same intent as the inline fallbacks in support_desk/prompts.py, now templated.
FIXTURES = [
    {
        "role": "support",
        "name": PROMPT_NAMES["support"],
        "description": "Autonomous front-line support agent for the OpenAI Agents SDK demo.",
        "version": {
            "model": "gpt-4.1-mini",
            "temperature": 0,
            "max_tokens": 1024,
            "top_p": 1,
            "variables": {"display_name": "", "industry": ""},
            "messages": [
                _sys(
                    "You are the front-line support agent for {{display_name}}, a {{industry}} "
                    "company.\n"
                    "You are autonomous: decide what to do for each ticket and use your tools.\n"
                    "- Use search_kb for how-to / informational questions, then close_resolved "
                    "with a short answer.\n"
                    "- Use lookup_order / lookup_account to verify claims before acting.\n"
                    "- Use process_refund for verified duplicate charges or damaged items.\n"
                    "- Hand off to the billing specialist for contract / enterprise billing disputes.\n"
                    "- Use escalate_to_human only when a human must intervene.\n"
                    "Always take exactly one terminal action (process_refund, close_resolved, "
                    "escalate_to_human) or hand off. Be concise."
                ),
                _user("Customer ticket: {{ticket}}"),
            ],
        },
    },
    {
        "role": "specialist",
        "name": PROMPT_NAMES["specialist"],
        "description": "Escalation target for contract / enterprise billing disputes (Agents SDK demo).",
        "version": {
            "model": "gpt-4.1-mini",
            "temperature": 0,
            "max_tokens": 1024,
            "top_p": 1,
            "variables": {"display_name": "", "industry": ""},
            "messages": [
                _sys(
                    "You are the billing specialist for {{display_name}} ({{industry}}).\n"
                    "You handle escalated contract and enterprise-billing disputes that front-line "
                    "support routed to you. Review the details, use lookup_account / lookup_order "
                    "as needed, and take a terminal action: process_refund if clearly warranted, "
                    "otherwise escalate_to_human so a person reviews the contract. Be concise."
                ),
                _user("Escalated ticket: {{ticket}}"),
            ],
        },
    },
]


def find_existing(client: httpx.Client, name: str) -> str | None:
    r = client.post(
        f"{BASE}/prompts/list/",
        json={"page_size": 100, "filters": {"name": {"operator": "iexact", "value": [name]}}},
    )
    r.raise_for_status()
    for row in r.json().get("results", []):
        if row.get("name") == name and row.get("id"):
            return row["id"]
    return None


def create_and_deploy(client: httpx.Client, fixture: dict) -> str:
    prompt = client.post(
        f"{BASE}/prompts/",
        json={"name": fixture["name"], "description": fixture["description"]},
    )
    prompt.raise_for_status()
    prompt_id = prompt.json()["id"]

    ver = client.post(
        f"{BASE}/prompts/{prompt_id}/versions/",
        json={"prompt_id": prompt_id, **fixture["version"]},
    )
    ver.raise_for_status()
    version_number = ver.json()["version"]

    client.post(f"{BASE}/prompts/{prompt_id}/commits/", json={"prompt_id": prompt_id}).raise_for_status()
    client.post(
        f"{BASE}/prompts/{prompt_id}/deployments/",
        json={"prompt_id": prompt_id, "version": version_number},
    ).raise_for_status()
    return prompt_id


def main() -> None:
    if not KEY:
        raise SystemExit("RESPAN_API_KEY not set")
    ids = {}
    with httpx.Client(headers=HEADERS, timeout=60) as client:
        for fx in FIXTURES:
            existing = find_existing(client, fx["name"])
            if existing:
                ids[fx["role"]] = existing
                print(f"[skip]    {fx['role']:10} already exists -> {existing}")
            else:
                pid = create_and_deploy(client, fx)
                ids[fx["role"]] = pid
                print(f"[created] {fx['role']:10} deployed       -> {pid}")
    IDS_FILE.write_text(json.dumps(ids, indent=2) + "\n")
    print(f"\nWrote {IDS_FILE.name}: {ids}")
    print("The demo now uses these managed prompts. (prompt_ids.json is git-ignored - "
          "it's specific to this account.)")


if __name__ == "__main__":
    main()
