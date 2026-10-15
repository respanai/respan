"""initial_actions - mirrors browser-use's examples/features/initial_actions.py.

    python scenarios/07_initial_actions.py
"""

from browser_use import Agent

from _harness import Expect, Scenario, browser, llm, main


async def run():
    agent = Agent(
        task="Summarise the first paragraph of the current page in one sentence.",
        llm=llm(),
        browser=browser(),
        initial_actions=[{"navigate": {"url": "https://en.wikipedia.org/wiki/Web_browser"}}],
    )
    await agent.run(max_steps=6)

SCENARIO = Scenario(
    name="initial_actions",
    title="initial_actions",
    source="features/initial_actions.py",
    look_for="a tool.navigate span directly under the agent, before step 1 - browser-use runs it without the model",
    run=run,
    expect=Expect(spans_include=("tool.navigate",)),
)

if __name__ == "__main__":
    main(SCENARIO)
