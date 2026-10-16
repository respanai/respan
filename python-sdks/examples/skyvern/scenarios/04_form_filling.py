"""Form filling - mirrors Skyvern's form example (navigation payload into a form).

    python scenarios/04_form_filling.py
"""

from _harness import Expect, Scenario, main, skyvern


async def run():
    result = await skyvern().run_task(
        prompt=(
            "Fill out the pizza order form with customer name 'Ada Lovelace', telephone "
            "'555-0100', email 'ada@example.com', size Medium and topping Bacon, then submit it. "
            "The task is complete once the submitted data is shown."
        ),
        url="https://httpbin.org/forms/post",
        max_steps=8,
        wait_for_completion=True,
    )
    print(f"  {result.status}: {result.output}")


SCENARIO = Scenario(
    name="form_filling",
    title="Fill and submit a form",
    source="docs: sdk-reference run_task (forms)",
    look_for="tool.input_text spans with the typed values, then the submit click",
    run=run,
    expect=Expect(spans_include=("input_text",)),
)

if __name__ == "__main__":
    main(SCENARIO)
