"""Step budget - `run_task(max_steps=1)` on a task that needs more.

    python scenarios/05_max_steps.py
"""

from _harness import Expect, Scenario, main, skyvern


async def run():
    result = await skyvern().run_task(
        prompt="Open the 'Travel' category, then the 'Mystery' category, then return the first book title.",
        url="https://books.toscrape.com",
        max_steps=1,
        wait_for_completion=True,
    )
    print(f"  {result.status}: {result.failure_reason}")


SCENARIO = Scenario(
    name="max_steps",
    title="Run stops at max_steps",
    source="docs: sdk-reference run_task (max_steps)",
    look_for="the agent span is an error with Skyvern's failure_reason; one step span",
    run=run,
    expect=Expect(statuses=("failed",), error_spans=("agent.skyvern_task",)),
)

if __name__ == "__main__":
    main(SCENARIO)
