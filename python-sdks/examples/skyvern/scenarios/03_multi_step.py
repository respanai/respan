"""Multi-step navigation - a task that needs several page loads.

    python scenarios/03_multi_step.py
"""

from _harness import Expect, Scenario, main, skyvern


async def run():
    result = await skyvern().run_task(
        prompt=(
            "Open the 'Poetry' category, open the first book in it, "
            "and return the book's title and price."
        ),
        url="https://books.toscrape.com",
        max_steps=8,
        wait_for_completion=True,
    )
    print(f"  {result.status}: {result.output}")


SCENARIO = Scenario(
    name="multi_step",
    title="Navigate across pages",
    source="docs: sdk-reference run_task (navigation)",
    look_for="several skyvern.step spans, each with its page URL, screenshot and tool.click",
    run=run,
    expect=Expect(spans_include=("click",)),
)

if __name__ == "__main__":
    main(SCENARIO)
