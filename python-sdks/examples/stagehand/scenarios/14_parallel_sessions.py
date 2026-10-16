"""parallel_sessions - two Stagehand sessions at once: two traces, nothing crosses over.

    python scenarios/14_parallel_sessions.py
"""

import asyncio

from pydantic import BaseModel

from _harness import Expect, Scenario, fixtures_url, main, with_stagehand


class Heading(BaseModel):
    heading: str


async def run():
    site = fixtures_url()

    async def one(path: str):
        async def flow(stagehand, page, browser):
            await page.goto(f"{site}/{path}")
            return (await stagehand.extract("extract the page heading", Heading)).data

        return await with_stagehand(flow)

    return await asyncio.gather(one("catalog-1.html"), one("catalog-2.html"))


SCENARIO = Scenario(
    name="parallel_sessions",
    title="two sessions in parallel",
    source="docs: running multiple Stagehand instances",
    look_for="2 traces, each with its own thread id",
    run=run,
    expect=Expect(traces=2, spans_include=("stagehand.extract",)),
)

if __name__ == "__main__":
    main(SCENARIO)
