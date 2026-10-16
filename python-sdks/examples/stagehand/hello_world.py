"""Stagehand + Respan in one file: one session, one act, one extract.

    python hello_world.py
"""

import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel
from respan import Respan
from respan_instrumentation_stagehand import StagehandInstrumentor
from stagehand import Stagehand, local_browser

load_dotenv(Path(__file__).resolve().parent / ".env", override=True)
sys.path.insert(0, str(Path(__file__).resolve().parent / "scenarios"))
from _harness import respan_model  # noqa: E402 - the gateway-backed client model


class Titles(BaseModel):
    titles: list[str]


async def main() -> None:
    browser = await local_browser.launch(headless=os.getenv("HEADLESS") == "1")
    # A client model: Stagehand's inferences come back to this process and go to
    # the Respan gateway. Built-in models (model="openai/gpt-4.1-mini",
    # model_api_key=...) work too; they report token totals per operation.
    stagehand = await Stagehand.create(browser=browser, model=respan_model())
    try:
        [page] = await browser.context.pages()
        await page.goto("https://news.ycombinator.com")
        await stagehand.act("click the 'new' link in the top bar")
        result = await stagehand.extract("the titles of the first 3 stories", Titles)
        print(result.data.titles)
    finally:
        await stagehand.close()
        await browser.close()


if __name__ == "__main__":
    respan = Respan(
        api_key=os.getenv("RESPAN_API_KEY"),
        base_url=os.getenv("RESPAN_BASE_URL"),
        instrumentations=[StagehandInstrumentor()],
    )
    try:
        asyncio.run(main())
    finally:
        respan.flush()  # short scripts: send spans before the process exits
