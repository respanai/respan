"""Run out of steps - mirrors browser-use's examples/Agent.run(max_steps=...).

    python scenarios/24_max_steps.py
"""

from browser_use import Agent

from _harness import Expect, Scenario, browser, llm, main


async def run():
    agent = Agent(
        task=(
            "Open https://en.wikipedia.org/wiki/Web_browser and summarise every section "
            "of the article, one by one."
        ),
        llm=llm(),
        browser=browser(),
    )
    await agent.run(max_steps=2)

SCENARIO = Scenario(
    name="max_steps",
    title="Run out of steps",
    source="Agent.run(max_steps=...)",
    look_for="the agent span ends after 2 steps with is_done=false and no final_result",
    run=run,
    expect=Expect(success=None),
)

if __name__ == "__main__":
    main(SCENARIO)
