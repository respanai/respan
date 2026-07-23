"""Entrypoint: run the autonomous multi-tenant support desk and print a summary.

Requires RESPAN_API_KEY (loaded from a .env found by walking up from here, or
already present in the environment). Routes through the Respan gateway.

    python run_demo.py
"""

import asyncio

from dotenv import load_dotenv

load_dotenv()  # walks up from cwd to find .env (RESPAN_API_KEY)

from support_desk.runner import run_all  # noqa: E402  (after dotenv)


def main() -> None:
    results = asyncio.run(run_all())
    print("\n=== Ticket outcomes ===")
    for r in results:
        tools = ", ".join(r["tools_used"]) or "-"
        print(f"  [{r['resolution']:12}] {r['ticket_id']}  {r['tenant']}  (prompt: {r['prompt']})")
        print(f"               tools: {tools}")
        print(f"               reply: {(r['final_output'] or '').strip()[:100]}")
    kinds = {}
    for r in results:
        kinds[r["resolution"]] = kinds.get(r["resolution"], 0) + 1
    print("\n  outcomes:", dict(kinds))
    print("  View the traces at https://platform.respan.ai (Traces) - one nested trace per ticket.")


if __name__ == "__main__":
    main()
