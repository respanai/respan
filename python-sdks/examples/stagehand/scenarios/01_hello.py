"""hello - Stagehand's quickstart: open a page, extract, act.

    python scenarios/01_hello.py
"""

from pydantic import BaseModel

from _harness import Expect, Scenario, main, with_stagehand


class Page(BaseModel):
    heading: str
    link: str


async def run():
    async def flow(stagehand, page, browser):
        await page.goto("https://example.com")
        result = await stagehand.extract("extract the page heading and the text of the link", Page)
        await stagehand.act("click the link on the page")
        return result.data

    return await with_stagehand(flow)


SCENARIO = Scenario(
    name="hello",
    title="act + extract on a live page",
    source="README quickstart",
    look_for="session -> act / extract, each with its LLM calls; a screenshot on the act",
    run=run,
    expect=Expect(spans_include=("stagehand.act", "stagehand.extract", "page.goto")),
)

if __name__ == "__main__":
    main(SCENARIO)
