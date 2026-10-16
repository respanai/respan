"""browser-use Hello World - one browser agent, traced in Respan.

    python hello_world.py

Opens a real Chrome window, lets the agent read Hacker News, and sends the
whole run to Respan: agent -> steps -> LLM calls + browser actions, with a
screenshot on every step.
"""

import asyncio
import os

from dotenv import load_dotenv

load_dotenv(override=True)

RESPAN_API_KEY = os.getenv("RESPAN_API_KEY")
RESPAN_BASE_URL = os.getenv("RESPAN_BASE_URL", "https://api.respan.ai/api")
RESPAN_MODEL = os.getenv("RESPAN_MODEL", "gpt-4.1-mini")
if not RESPAN_API_KEY:
    raise RuntimeError("Set RESPAN_API_KEY (copy .env.example to .env).")

# 1. Turn tracing on. Order does not matter, but doing it first is simplest.
from respan import Respan
from respan_instrumentation_browser_use import BrowserUseInstrumentor

respan = Respan(
    api_key=RESPAN_API_KEY,
    base_url=RESPAN_BASE_URL,
    app_name="browser-use-hello-world",
    # The end user this run is for: shows on every span and in Respan's Users view.
    customer_identifier=os.getenv("RESPAN_CUSTOMER_ID", "browser-use-hello-world"),
    instrumentations=[BrowserUseInstrumentor()],
)

# 2. Plain browser-use from here on - nothing Respan-specific.
from browser_use import Agent, Browser, ChatOpenAI

# browser-use's ChatOpenAI pointed at the Respan gateway: one key for the
# model and the traces. Any browser-use model class works (ChatAnthropic,
# ChatGoogle, ...); the instrumentation traces them all the same way.
llm = ChatOpenAI(model=RESPAN_MODEL, api_key=RESPAN_API_KEY, base_url=RESPAN_BASE_URL)


async def main() -> None:
    agent = Agent(
        task="Go to news.ycombinator.com and tell me the title of the #1 story and its points.",
        llm=llm,
        browser=Browser(headless=os.getenv("HEADLESS") == "1"),
    )
    history = await agent.run(max_steps=8)

    print("\nResult:   ", history.final_result())
    print("Success:  ", history.is_successful())
    print("Steps:    ", history.number_of_steps())
    print("Visited:  ", history.urls())


if __name__ == "__main__":
    try:
        asyncio.run(main())
    finally:
        # Short scripts exit before the background exporter's next tick.
        respan.flush()
