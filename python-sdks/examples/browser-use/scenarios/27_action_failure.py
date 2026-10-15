"""A failing action - mirrors browser-use's examples/(failure handling).

    python scenarios/27_action_failure.py
"""

from browser_use import Agent

from _harness import Expect, Scenario, browser, llm, main


async def run():
    agent = Agent(
        task=(
            "Open https://this-domain-does-not-exist.invalid and read its title. "
            "If it cannot load, say so and finish."
        ),
        llm=llm(),
        browser=browser(),
        max_failures=2,
    )
    await agent.run(max_steps=4)

SCENARIO = Scenario(
    name="action_failure",
    title="A failing action",
    source="(failure handling)",
    look_for="tool.navigate to a dead domain is an error span with browser-use's message; the agent recovers and reports it",
    run=run,
    expect=Expect(success=None, error_spans=("tool.navigate",)),
)

if __name__ == "__main__":
    main(SCENARIO)
