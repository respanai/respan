"""multi_tab - two tabs, extracted concurrently: each LLM call stays under its own operation.

    python scenarios/06_multi_tab.py
"""

import asyncio

from pydantic import BaseModel

from _harness import Expect, Scenario, fixtures_url, main, with_stagehand


class Heading(BaseModel):
    heading: str


async def run():
    site = fixtures_url()

    async def flow(stagehand, page, browser):
        await page.goto(f"{site}/login.html")
        second = await browser.context.new_page(f"{site}/catalog-1.html")
        return await asyncio.gather(
            stagehand.extract("extract the main heading of the sign-in page", Heading, page=page),
            stagehand.extract("extract the main heading of the catalog page", Heading, page=second),
        )

    return await with_stagehand(flow)


SCENARIO = Scenario(
    name="multi_tab",
    title="new_page + concurrent extracts on two tabs",
    source="docs: multiple pages (page option)",
    look_for="context.new_page; two extract spans running in parallel, each with its own LLM calls",
    run=run,
    expect=Expect(spans_include=("context.new_page", "stagehand.extract")),
)

if __name__ == "__main__":
    main(SCENARIO)
