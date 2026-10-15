"""Structured output - mirrors browser-use's examples/getting_started/03_data_extraction.py + features/custom_output.py.

    python scenarios/03_data_extraction.py
"""

from browser_use import Agent
from pydantic import BaseModel

from _harness import Expect, Scenario, browser, llm, main


class Story(BaseModel):
    title: str
    points: int


class TopStories(BaseModel):
    stories: list[Story]


async def run():
    agent = Agent(
        task="Open https://news.ycombinator.com and return the top 5 stories with their points.",
        llm=llm(),
        browser=browser(),
        output_model_schema=TopStories,
    )
    history = await agent.run(max_steps=8)
    result = history.get_structured_output(TopStories)
    for story in result.stories if result else []:
        print(f"  {story.points:>5}  {story.title}")

SCENARIO = Scenario(
    name="data_extraction",
    title="Structured output",
    source="getting_started/03_data_extraction.py + features/custom_output.py",
    look_for="the tool.done span's input is the typed TopStories object; the agent output has final_result",
    run=run,
    expect=Expect(spans_include=("tool.done",)),
)

if __name__ == "__main__":
    main(SCENARIO)
