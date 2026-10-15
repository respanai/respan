"""Multiple tabs - mirrors browser-use's examples/features/multi_tab.py.

    python scenarios/09_multi_tab.py
"""

from browser_use import Agent

from _harness import Expect, Scenario, browser, llm, main


async def run():
    agent = Agent(
        task=(
            "Open https://news.ycombinator.com and https://en.wikipedia.org/wiki/Main_Page in "
            "two separate tabs. Then switch back to the Hacker News tab and report the #1 story."
        ),
        llm=llm(),
        browser=browser(),
    )
    await agent.run(max_steps=10)

SCENARIO = Scenario(
    name="multi_tab",
    title="Multiple tabs",
    source="features/multi_tab.py",
    look_for="navigate spans with new_tab=true and a switch span; each step's URL shows which tab was active",
    run=run,
    expect=Expect(),
)

if __name__ == "__main__":
    main(SCENARIO)
