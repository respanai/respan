"""Structured extraction - mirrors Skyvern's `run_task(data_extraction_schema=...)` example.

    python scenarios/02_data_extraction.py
"""

from _harness import Scenario, main, skyvern

SCHEMA = {
    "type": "object",
    "properties": {
        "posts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"title": {"type": "string"}, "points": {"type": "integer"}},
            },
        }
    },
}


async def run():
    result = await skyvern().run_task(
        prompt="Return the titles and points of the top 3 posts on the page.",
        url="https://news.ycombinator.com",
        data_extraction_schema=SCHEMA,
        max_steps=6,
        wait_for_completion=True,
    )
    print(f"  {result.status}: {result.output}")


SCENARIO = Scenario(
    name="data_extraction",
    title="run_task with data_extraction_schema",
    source="docs: sdk-reference run_task (data_extraction_schema)",
    look_for="the agent span's output holds the typed posts; an extraction LLM call in the trace",
    run=run,
)

if __name__ == "__main__":
    main(SCENARIO)
