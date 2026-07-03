"""Streaming entrypoint (plan §9): fire all tickets concurrently and animate the
per-step events in the terminal, while writing the same events as NDJSON.

    python run_stream.py

The terminal animation is the "acceptable minimum" UI; the NDJSON file
(events.ndjson) is the stream a web UI would tail. Both consume the same
support_desk.stream.run_ticket_streamed generator - the UI attach seam.
"""

import asyncio
import json

from dotenv import load_dotenv

load_dotenv()

from support_desk import telemetry  # noqa: E402
from support_desk.config import SCENARIOS, TENANTS  # noqa: E402
from support_desk.stream import run_ticket_streamed  # noqa: E402

ICONS = {
    "ticket_start": "📨", "agent": "🤖", "tool_call": "🔧", "tool_output": "↩️ ",
    "handoff": "➡️ ", "message": "💬", "ticket_done": "✅",
}
NDJSON_PATH = "events.ndjson"


def _render(ev: dict) -> str:
    icon = ICONS.get(ev["type"], "•")
    tid = ev.get("ticket_id", "")
    if ev["type"] == "ticket_start":
        return f"{icon} {tid}  {ev['tenant']}: {ev['ticket'][:70]}"
    if ev["type"] == "agent":
        return f"{icon} {tid}  → agent: {ev['agent']}"
    if ev["type"] == "tool_call":
        return f"{icon} {tid}  tool: {ev['tool']}()"
    if ev["type"] == "tool_output":
        return f"{icon} {tid}  ↳ {ev['output']}"
    if ev["type"] == "handoff":
        return f"{icon} {tid}  handoff ({ev.get('event', 'requested')})"
    if ev["type"] == "message":
        return f"{icon} {tid}  {ev['text']}"
    if ev["type"] == "ticket_done":
        err = f"  ERROR {ev['error']}" if ev.get("error") else ""
        return f"{icon} {tid}  [{ev['resolution']}]  tools: {', '.join(ev['tools_used']) or '-'}{err}"
    return f"{icon} {tid}  {ev}"


async def _pump(tenant, ticket, queue: asyncio.Queue) -> None:
    try:
        async for ev in run_ticket_streamed(tenant, ticket):
            await queue.put(ev)
    except Exception as exc:  # surface, don't deadlock the consumer
        await queue.put({"type": "error", "ticket_id": "?", "error": f"{type(exc).__name__}: {exc}"})
    finally:
        await queue.put({"type": "_sentinel"})  # exactly one per task, always


async def main() -> None:
    telemetry.init_telemetry()
    queue: asyncio.Queue = asyncio.Queue()
    tasks = [
        asyncio.create_task(_pump(TENANTS[s.tenant_id], s.ticket, queue))
        for s in SCENARIOS
    ]

    finished = 0
    print("=== Autonomous support desk - live event stream ===\n")
    with open(NDJSON_PATH, "w") as ndjson:
        while finished < len(tasks):
            ev = await queue.get()
            if ev.get("type") == "_sentinel":
                finished += 1
                continue
            ndjson.write(json.dumps(ev) + "\n")
            ndjson.flush()
            if ev["type"] == "error":
                print(f"❌ {ev['ticket_id']}  {ev['error']}")
            else:
                print(_render(ev))
            await asyncio.sleep(0.02)  # gentle animation pacing

    await asyncio.gather(*tasks)
    telemetry.flush()
    print(f"\nWrote event stream to {NDJSON_PATH}. Traces at https://platform.respan.ai")


if __name__ == "__main__":
    asyncio.run(main())
