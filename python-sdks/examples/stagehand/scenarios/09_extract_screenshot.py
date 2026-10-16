"""extract_screenshot - extract(screenshot=True): the viewport image goes into the prompt.

    python scenarios/09_extract_screenshot.py
"""

from pydantic import BaseModel

from _harness import Expect, Scenario, fixtures_url, main, with_stagehand


class SoldOut(BaseModel):
    sold_out: list[str]


async def run():
    async def flow(stagehand, page, browser):
        await page.goto(f"{fixtures_url()}/catalog-3.html")
        result = await stagehand.extract("which products are sold out", SoldOut, screenshot=True)
        return result.data

    return await with_stagehand(flow)


SCENARIO = Scenario(
    name="extract_screenshot",
    title="extract with a screenshot in the prompt",
    source="docs: extract screenshot option",
    look_for="the extract's LLM prompt shows the page image",
    run=run,
    expect=Expect(spans_include=("stagehand.extract",), prompt_images=True),
)

if __name__ == "__main__":
    main(SCENARIO)
