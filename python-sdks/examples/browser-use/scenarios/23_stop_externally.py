"""Stop a run from outside - mirrors browser-use's examples/features/stop_externally.py.

    python scenarios/23_stop_externally.py
"""

from browser_use import Agent

from _harness import Expect, Scenario, browser, llm, main


async def run():
    agent: Agent | None = None

    async def should_stop() -> bool:
        return agent is not None and agent.state.n_steps > 2

    agent = Agent(
        task=(
            "Open https://en.wikipedia.org/wiki/Web_browser, then visit every link in the "
            "History section one by one and summarise each."
        ),
        llm=llm(),
        browser=browser(),
        register_should_stop_callback=should_stop,
    )
    await agent.run(max_steps=20)

SCENARIO = Scenario(
    name="stop_externally",
    title="Stop a run from outside",
    source="features/stop_externally.py",
    look_for="the agent span ends cleanly after 2 steps with is_done=false; no dangling spans",
    run=run,
    expect=Expect(success=None),
)

if __name__ == "__main__":
    main(SCENARIO)
