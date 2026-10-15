"""Follow-up tasks - mirrors browser-use's examples/features/follow_up_tasks.py.

    python scenarios/08_follow_up_tasks.py
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
        await agent.run(max_steps=6)
        agent.add_new_task("Now open that story's comments page and report how many comments it has.")
        await agent.run(max_steps=6)
    finally:
        await agent.browser_session.kill()

SCENARIO = Scenario(
    name="follow_up_tasks",
    title="Follow-up tasks",
    source="features/follow_up_tasks.py",
    look_for="two traces, one per run(); both carry the same thread id (the Agent's session), so the Threads view groups them",
    run=run,
    expect=Expect(traces=2),
)

if __name__ == "__main__":
    main(SCENARIO)
