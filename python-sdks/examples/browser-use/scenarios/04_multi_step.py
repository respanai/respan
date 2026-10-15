"""Multi-step navigation - mirrors browser-use's examples/getting_started/04_multi_step_task.py.

    python scenarios/04_multi_step.py
"""

from browser_use import Agent

from _harness import Expect, Scenario, browser, llm, main


async def run():
    agent = Agent(
        task=(
            "Open https://en.wikipedia.org/wiki/Python_(programming_language). Find who "
            "designed Python, open that person's Wikipedia page, and report their birth year."
        ),
        llm=llm(),
        browser=browser(),
    )
    await agent.run(max_steps=10)

SCENARIO = Scenario(
    name="multi_step",
    title="Multi-step navigation",
    source="getting_started/04_multi_step_task.py",
    look_for="several steps, each with its page URL/title and screenshot; tool.click between pages",
    run=run,
    expect=Expect(),
)

if __name__ == "__main__":
    main(SCENARIO)
