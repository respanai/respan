"""Search the web - mirrors browser-use's examples/getting_started/01_basic_search.py.

    python scenarios/01_basic_search.py
"""

from browser_use import Agent

from _harness import Expect, Scenario, browser, llm, main


async def run():
    agent = Agent(
        task="Search the web for 'browser-use github' and tell me the title of the first result.",
        llm=llm(),
        browser=browser(),
    )
    await agent.run(max_steps=8)

SCENARIO = Scenario(
    name="basic_search",
    title="Search the web",
    source="getting_started/01_basic_search.py",
    look_for="agent -> steps; the search or navigate tool span, one LLM call per step, a screenshot on every step",
    run=run,
    # Search engines sometimes block automated browsers; the run may fail honestly.
    expect=Expect(success=None),
)

if __name__ == "__main__":
    main(SCENARIO)
