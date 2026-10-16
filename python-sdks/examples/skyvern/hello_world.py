"""Skyvern Hello World - one Skyvern task, traced in Respan.

    python hello_world.py

Runs Skyvern's engine in this process (`Skyvern.local()`), lets it read Hacker
News, and sends the whole run to Respan: the task -> steps -> LLM calls +
browser actions, with a screenshot on every step.
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

# 1. Turn tracing on before Skyvern runs anything.
from respan import Respan
from respan_instrumentation_skyvern import SkyvernInstrumentor

respan = Respan(
    api_key=RESPAN_API_KEY,
    base_url=RESPAN_BASE_URL,
    app_name="skyvern-hello-world",
    # The end user this run is for: shows on every span and in Respan's Users view.
    customer_identifier=os.getenv("RESPAN_CUSTOMER_ID", "skyvern-hello-world"),
    instrumentations=[SkyvernInstrumentor()],
)

# 2. Plain Skyvern from here on - nothing Respan-specific.
from skyvern import Skyvern
from skyvern.schemas.llm import LLMConfig

# Skyvern's model is any litellm model; here it goes through the Respan gateway
# (OpenAI-compatible), so one key covers the model and the traces.
skyvern = Skyvern.local(
    use_in_memory_db=True,
    llm_config=LLMConfig(
        model_name=f"openai/{RESPAN_MODEL}",
        required_env_vars=[],
        supports_vision=True,
        add_assistant_prefix=False,
        litellm_params={"api_base": RESPAN_BASE_URL, "api_key": RESPAN_API_KEY},
    ),
    settings={"BROWSER_TYPE": "chromium-headless" if os.getenv("HEADLESS") == "1" else "chromium-headful"},
)


async def main() -> None:
    try:
        result = await skyvern.run_task(
            prompt="Find the top post on Hacker News and return its title and points.",
            url="https://news.ycombinator.com",
            data_extraction_schema={
                "type": "object",
                "properties": {"title": {"type": "string"}, "points": {"type": "integer"}},
            },
            max_steps=6,
            wait_for_completion=True,
        )
    finally:
        await skyvern.aclose()
    print("\nStatus:  ", result.status)
    print("Output:  ", result.output)
    print("Steps:   ", result.step_count)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    finally:
        # Short scripts exit before the background exporter's next tick.
        respan.flush()
