"""Live dataset export: production tickets -> a Respan dataset (plan §8, build step 11).

This is the "live action" button. It runs the support desk (real traced tickets
through the gateway), then exports each handled ticket as a dataset row via the
synchronous bulk endpoint - the "production traffic -> dataset" story.

Row shape (verified contract):
  - input            : OBJECT of the prompt variables -> {"ticket": ..., "tenant": ...}
                       (key "ticket" matches the managed prompt's {{ticket}} variable)
  - expected_output  : JSON STRING -> the golden label {"resolution": ...}
  - output           : the actual agent reply (what production produced)
  - metadata         : trace linkage (ticket_id, customer_identifier, tools_used)

    python export_dataset.py

The same rows can also be exported by hand in the UI: Spans/Logs -> filter -> export to a dataset.
"""

import asyncio
import json
import os

from dotenv import load_dotenv

load_dotenv()

import httpx

from support_desk.runner import run_all  # noqa: E402

BASE = os.getenv("RESPAN_BASE_URL", "https://api.respan.ai/api").rstrip("/")
HEADERS = {"Authorization": f"Bearer {os.getenv('RESPAN_API_KEY')}", "Content-Type": "application/json"}
DATASET_NAME = "Support tickets — Autonomous Desk (demo)"
DATASET_DESC = "Production support tickets handled by the autonomous Agents-SDK desk, with golden resolutions."


def _find_dataset(client: httpx.Client) -> str | None:
    r = client.post(
        f"{BASE}/datasets/list/",
        json={"page_size": 100, "filters": {"name": {"operator": "iexact", "value": [DATASET_NAME]}}},
    )
    r.raise_for_status()
    body = r.json()
    results = body.get("results") or body.get("response", {}).get("results", [])
    for row in results:
        if row.get("name") == DATASET_NAME and row.get("id"):
            return row["id"]
    return None


def _get_or_create_dataset(client: httpx.Client) -> tuple[str, bool]:
    existing = _find_dataset(client)
    if existing:
        return existing, False
    r = client.post(
        f"{BASE}/datasets/",
        json={"name": DATASET_NAME, "is_empty": True, "description": DATASET_DESC},
    )
    r.raise_for_status()
    return r.json()["id"], True


def _to_row(result: dict) -> dict:
    return {
        "input": {"ticket": result["ticket"], "tenant": result["tenant"]},
        "expected_output": json.dumps({"resolution": result["resolution"]}),
        "output": result.get("final_output") or "",
        "metadata": {
            "ticket_id": result["ticket_id"],
            "customer_identifier": result["customer_identifier"],
            "tools_used": ", ".join(result["tools_used"]),
        },
    }


def main() -> None:
    if not os.getenv("RESPAN_API_KEY"):
        raise SystemExit("RESPAN_API_KEY not set")

    print("Running the support desk (live tickets through the gateway)...")
    results = asyncio.run(run_all())
    rows = [_to_row(r) for r in results]

    with httpx.Client(headers=HEADERS, timeout=60) as client:
        dataset_id, created = _get_or_create_dataset(client)
        print(f"Dataset {'created' if created else 'reused'}: {DATASET_NAME} ({dataset_id})")

        resp = client.post(f"{BASE}/datasets/{dataset_id}/logs/bulk/", json={"logs": rows})
        resp.raise_for_status()
        summary = resp.json()

    print(f"Exported {len(rows)} ticket rows -> dataset.")
    print(f"  success_count={summary.get('success_count')}  error_count={summary.get('error_count')}")
    if summary.get("errors"):
        print("  errors:", summary["errors"][:3])
    print(f"  View it at https://platform.respan.ai (Datasets) - '{DATASET_NAME}'.")


if __name__ == "__main__":
    main()
