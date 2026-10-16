"""Quickstart - mirrors Skyvern's getting-started quickstart (`run_task`, local mode).

    python scenarios/01_quickstart.py
"""

from _harness import Expect, Scenario, main, skyvern


async def run():
    result = await skyvern().run_task(
        prompt="Find the top post on Hacker News and return its title.",
        url="https://news.ycombinator.com",
        max_steps=6,
        wait_for_completion=True,
    )
    print(f"  {result.status}: {result.output}")


SCENARIO = Scenario(
    name="quickstart",
    title="run_task on Hacker News",
    source="docs: getting-started/quickstart (run_task)",
    look_for="agent.skyvern_task -> skyvern.step -> extract-actions LLM call + actions; screenshot per step",
    run=run,
    expect=Expect(prompts_include=("extract-actions",)),
)

if __name__ == "__main__":
    main(SCENARIO)
