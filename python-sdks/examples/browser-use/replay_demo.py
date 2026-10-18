"""browser-use session replay - watch what the browser did, from the trace.

    python replay_demo.py

A short visible agent (Hacker News -> open a story -> back) with
`record_session=True`: the instrumentation records the pages with rrweb over
browser-use's own CDP connection, uploads the recording when the run ends, and
puts a replay link on every span (`metadata.session_replay_url`, ending in
`#t=<ms>`: the moment that span started). At the end this prints, for the agent
and one step / action / LLM call, the log link and its replay link; open a
replay link to play the run back from that moment in your browser. `RESPAN_RECORD_SESSION=1` turns the same
thing on without code changes.
"""

import asyncio
import os
import time

from dotenv import load_dotenv

load_dotenv(override=True)

RESPAN_API_KEY = os.getenv("RESPAN_API_KEY")
RESPAN_BASE_URL = os.getenv("RESPAN_BASE_URL", "https://api.respan.ai/api")
RESPAN_PLATFORM_URL = os.getenv("RESPAN_PLATFORM_URL", "https://platform.respan.ai").rstrip("/")
RESPAN_MODEL = os.getenv("RESPAN_MODEL", "gpt-4.1-mini")
if not RESPAN_API_KEY:
    raise RuntimeError("Set RESPAN_API_KEY (copy .env.example to .env).")

from respan import Respan
from respan_instrumentation_browser_use import BrowserUseInstrumentor
from respan_sdk.constants.span_attributes import RESPAN_METADATA
from respan_tracing.testing.live import RespanLogsClient
from respan_tracing.utils.session_recording import SESSION_REPLAY_URL_METADATA_KEY

REPLAY_URL_ATTR = f"{RESPAN_METADATA}.{SESSION_REPLAY_URL_METADATA_KEY}"
agent_spans = []


def remember_agent_span(span) -> None:
    if span.parent is None:
        agent_spans.append(span)


respan = Respan(
    api_key=RESPAN_API_KEY,
    base_url=RESPAN_BASE_URL,
    app_name="browser-use-replay-demo",
    customer_identifier=os.getenv("RESPAN_CUSTOMER_ID", "browser-use-replay-demo"),
    instrumentations=[BrowserUseInstrumentor(record_session=True)],
    span_postprocess_callback=remember_agent_span,
)

from browser_use import Agent, Browser, ChatOpenAI  # noqa: E402

llm = ChatOpenAI(model=RESPAN_MODEL, api_key=RESPAN_API_KEY, base_url=RESPAN_BASE_URL)


def stored_rows(trace_id: str) -> list[dict]:
    """Read the trace back from Respan (ingestion takes a few seconds)."""
    logs = RespanLogsClient(RESPAN_BASE_URL, RESPAN_API_KEY)
    for _ in range(20):
        rows = logs.trace_rows(trace_id)
        if any(row.get("log_type") == "agent" for row in rows):
            return rows
        time.sleep(3)
    return []


def log_link(row: dict) -> str:
    return f"{RESPAN_PLATFORM_URL}/platform/requests?unique_id={row['unique_id']}"


async def main() -> None:
    agent = Agent(
        task=(
            "Go to https://news.ycombinator.com, open the comments page of the #1 story, "
            "read its title and how many comments it has, then go back to the front page "
            "and report the title and the comment count."
        ),
        llm=llm,
        browser=Browser(headless=os.getenv("HEADLESS") == "1"),
    )
    history = await agent.run(max_steps=8)
    print("\nResult: ", history.final_result())
    print("Steps:  ", history.number_of_steps())


if __name__ == "__main__":
    try:
        asyncio.run(main())
    finally:
        respan.flush()
    for span in agent_spans:
        trace_id = format(span.context.trace_id, "032x")
        print("\nTrace id:       ", trace_id)
        print("Replay (span):  ", span.attributes.get(REPLAY_URL_ATTR) or "(none - see the warning above)")
        rows = stored_rows(trace_id)
        if not rows:
            print("Trace in Respan: (not ingested yet)")
            continue
        # Every span links to the replay at the moment it started: print one of each kind.
        for log_type in ("agent", "task", "tool", "chat"):
            row = next((r for r in rows if r.get("log_type") == log_type), None)
            if row is None:
                continue
            url = (row.get("metadata") or {}).get(SESSION_REPLAY_URL_METADATA_KEY)
            print(f"\n[{log_type}] {row.get('span_name')}")
            print("  log:   ", log_link(row))
            print("  replay:", url or "(not on the stored span)")
