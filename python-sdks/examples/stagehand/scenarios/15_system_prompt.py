"""system_prompt - Stagehand.create(system_prompt=...): the session records it,
every prompt carries it.

    python scenarios/15_system_prompt.py
"""

from pydantic import BaseModel

from _harness import Expect, Scenario, fixtures_url, main, with_stagehand


class Cheapest(BaseModel):
    name: str
    price: float


async def run():
    async def flow(stagehand, page, browser):
        await page.goto(f"{fixtures_url()}/catalog-1.html")
        return (await stagehand.extract("extract the cheapest product", Cheapest)).data

    return await with_stagehand(
        flow,
        system_prompt="Always answer as a careful shopping assistant. Prices are in US dollars.",
    )


SCENARIO = Scenario(
    name="system_prompt",
    title="a custom system prompt",
    source="docs: system_prompt",
    look_for="the session input shows the system prompt; the LLM system messages contain it",
    run=run,
    expect=Expect(spans_include=("stagehand.extract",)),
)

if __name__ == "__main__":
    main(SCENARIO)
