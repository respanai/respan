"""flash_mode (fast agent) - mirrors browser-use's examples/getting_started/05_fast_agent.py.

    python scenarios/05_fast_agent.py
"""

from browser_use import Agent

from _harness import Expect, Scenario, browser, llm, main


async def run():
    agent = Agent(
        task="What is the title of the #1 story on https://news.ycombinator.com ?",
        llm=llm(),
        browser=browser(),
        flash_mode=True,
    )
    await agent.run(max_steps=6)

SCENARIO = Scenario(
    name="fast_agent",
    title="flash_mode (fast agent)",
    source="getting_started/05_fast_agent.py",
    look_for="LLM completions have no thinking/evaluation fields, so prompts and outputs are shorter",
    run=run,
    expect=Expect(),
)

if __name__ == "__main__":
    main(SCENARIO)
