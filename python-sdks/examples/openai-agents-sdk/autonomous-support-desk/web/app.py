"""Tiny web server for the demo UI (plan §9 - the "live, like vercel" view).

Serves one page and streams the support desk's per-step events to the browser
over Server-Sent Events, reusing support_desk.stream.run_ticket_streamed. Zero
new dependencies - starlette + uvicorn + sse_starlette ship with the example venv.

    python web/app.py        # then open http://127.0.0.1:8000
"""

import asyncio
import json
import sys
from pathlib import Path

# Make the demo package importable when run from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

import uvicorn
from sse_starlette.sse import EventSourceResponse
from starlette.applications import Starlette
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Route

from support_desk import telemetry
from support_desk.config import SCENARIOS, TENANTS
from support_desk.stream import run_ticket_streamed

HERE = Path(__file__).resolve().parent


async def index(request):
    return FileResponse(HERE / "index.html")


async def scenarios(request):
    """The tickets that will run (so the page can show context if it wants)."""
    return JSONResponse(
        [{"tenant": TENANTS[s.tenant_id].display_name, "ticket": s.ticket} for s in SCENARIOS]
    )


async def stream(request):
    """Run every ticket concurrently and stream per-step events as SSE."""
    telemetry.init_telemetry()
    queue: asyncio.Queue = asyncio.Queue()

    async def pump(tenant, ticket):
        try:
            async for ev in run_ticket_streamed(tenant, ticket):
                await queue.put(ev)
        except Exception as exc:
            await queue.put({"type": "error", "error": f"{type(exc).__name__}: {exc}"})
        finally:
            await queue.put({"type": "_sentinel"})

    tasks = [
        asyncio.create_task(pump(TENANTS[s.tenant_id], s.ticket)) for s in SCENARIOS
    ]

    async def gen():
        finished = 0
        while finished < len(tasks):
            ev = await queue.get()
            if ev.get("type") == "_sentinel":
                finished += 1
                continue
            yield {"data": json.dumps(ev)}
        telemetry.flush()
        yield {"data": json.dumps({"type": "all_done"})}

    return EventSourceResponse(gen())


app = Starlette(
    routes=[
        Route("/", index),
        Route("/scenarios", scenarios),
        Route("/stream", stream),
    ]
)


if __name__ == "__main__":
    print("Open http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
