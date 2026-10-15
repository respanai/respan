"""Parallel agents - mirrors browser-use's examples/features/parallel_agents.py.

    python scenarios/10_parallel_agents.py
"""

import asyncio

from browser_use import Agent

from _harness import Expect, Scenario, browser, llm, main


async def run():
    tasks = [
        "What is the title of the #1 story on https://news.ycombinator.com ?",
        "What is the featured article on https://en.wikipedia.org/wiki/Main_Page today?",
    ]
    agents = [Agent(task=task, llm=llm(), browser=browser()) for task in tasks]
    await asyncio.gather(*(agent.run(max_steps=6) for agent in agents))

SCENARIO = Scenario(
    name="parallel_agents",
    title="Parallel agents",
    source="features/parallel_agents.py",
    look_for="two separate traces with their own thread ids; no span of one agent lands in the other's trace",
    run=run,
    expect=Expect(traces=2),
)

if __name__ == "__main__":
    main(SCENARIO)
