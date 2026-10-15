"""Replay a recorded run - mirrors browser-use's examples/features/rerun_history.py.

    python scenarios/26_rerun_history.py
"""

from browser_use import Agent

from _harness import Expect, Scenario, browser, llm, main


async def run():
    agent = Agent(
        task="Open https://news.ycombinator.com and report the title of the #1 story.",
        llm=llm(),
        browser=browser(keep_alive=True),
    )
    try:
        history = await agent.run(max_steps=6)
        await agent.rerun_history(history, delay_between_actions=0.5)
    finally:
        await agent.browser_session.kill()

SCENARIO = Scenario(
    name="rerun_history",
    title="Replay a recorded run",
    source="features/rerun_history.py",
    look_for="the first trace is the normal run; the replay is its own agent trace with the same tool spans and no step LLM calls",
    run=run,
    expect=Expect(traces=2),
)

if __name__ == "__main__":
    main(SCENARIO)
