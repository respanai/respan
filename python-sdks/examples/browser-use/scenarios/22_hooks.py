"""on_step_start / on_step_end hooks - mirrors browser-use's examples/features/process_agent_output.py (lifecycle hooks).

    python scenarios/22_hooks.py
"""

from browser_use import Agent

from _harness import Expect, Scenario, browser, llm, main


async def on_step_start(agent: Agent) -> None:
    print(f"  step {agent.state.n_steps} starting")


async def on_step_end(agent: Agent) -> None:
    url = await agent.browser_session.get_current_page_url()
    print(f"  step {agent.state.n_steps} ended on {url}")


async def run():
    agent = Agent(
        task="What is the title of the #1 story on https://news.ycombinator.com ?",
        llm=llm(),
        browser=browser(),
    )
    await agent.run(max_steps=6, on_step_start=on_step_start, on_step_end=on_step_end)

SCENARIO = Scenario(
    name="hooks",
    title="on_step_start / on_step_end hooks",
    source="features/process_agent_output.py (lifecycle hooks)",
    look_for="your hook code runs inside the agent span; anything it traces lands in the same trace",
    run=run,
    expect=Expect(),
)

if __name__ == "__main__":
    main(SCENARIO)
