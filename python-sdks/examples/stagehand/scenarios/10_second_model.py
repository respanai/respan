"""second_model - a session on a second model: each LLM span reports its own model and cost.

    python scenarios/10_second_model.py
"""

from pydantic import BaseModel

from _harness import SMALL_MODEL, Expect, Scenario, fixtures_url, main, with_stagehand


class Heading(BaseModel):
    heading: str


async def run():
    async def flow(stagehand, page, browser):
        await page.goto(f"{fixtures_url()}/login.html")
        return (await stagehand.extract("extract the page heading", Heading)).data

    return await with_stagehand(flow, model=SMALL_MODEL)


SCENARIO = Scenario(
    name="second_model",
    title="a session on a second model",
    source="docs: model configuration",
    look_for="the chat spans name the small model",
    run=run,
    expect=Expect(spans_include=("stagehand.extract",), models=(SMALL_MODEL,)),
)

if __name__ == "__main__":
    main(SCENARIO)
