"""extract_schema - a typed extract with a nested pydantic schema on a live site.

    python scenarios/04_extract_schema.py
"""

from pydantic import BaseModel, Field

from _harness import Expect, Scenario, main, with_stagehand


class Story(BaseModel):
    title: str
    points: int
    comments: int


class Stories(BaseModel):
    stories: list[Story] = Field(max_length=5)


async def run():
    async def flow(stagehand, page, browser):
        await page.goto("https://news.ycombinator.com")
        result = await stagehand.extract(
            "extract the top 5 stories with title, points and comment count", Stories
        )
        return result.data

    return await with_stagehand(flow)


SCENARIO = Scenario(
    name="extract_schema",
    title="extract a typed list from Hacker News",
    source="showcase/yc-companies-by-industry (extract with a schema)",
    look_for="extract input shows the JSON schema; output.data is the typed list",
    run=run,
    expect=Expect(spans_include=("stagehand.extract",)),
)

if __name__ == "__main__":
    main(SCENARIO)
