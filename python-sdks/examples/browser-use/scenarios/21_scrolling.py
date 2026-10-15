"""Scrolling a long page - mirrors browser-use's examples/features/scrolling_page.py.

    python scenarios/21_scrolling.py
"""

from browser_use import Agent

from _harness import Expect, Scenario, browser, llm, main


async def run():
    agent = Agent(
        task=(
            "Open https://en.wikipedia.org/wiki/Web_browser and use scroll to go down until "
            "you see the 'See also' section. List its first three links."
        ),
        llm=llm(),
        browser=browser(),
    )
    await agent.run(max_steps=12)

SCENARIO = Scenario(
    name="scrolling",
    title="Scrolling a long page",
    source="features/scrolling_page.py",
    look_for="tool.scroll spans; the step screenshots move down the page",
    run=run,
    expect=Expect(spans_include=("tool.scroll",)),
)

if __name__ == "__main__":
    main(SCENARIO)
