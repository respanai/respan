"""Engine selection - `run_task(engine=RunEngine.skyvern_v3)`.

    python scenarios/08_engine_v3.py
"""

from skyvern.schemas.runs import RunEngine

from _harness import Expect, Scenario, main, skyvern


async def run():
    result = await skyvern().run_task(
        prompt="Return the title of the first book on the page.",
        url="https://books.toscrape.com",
        engine=RunEngine.skyvern_v3,
        max_steps=6,
        wait_for_completion=True,
    )
    print(f"  {result.status}: {result.output}")


SCENARIO = Scenario(
    name="engine_v3",
    title="run_task on the skyvern_v3 engine",
    source="docs: sdk-reference run_task (engine)",
    look_for="skyvern.run.engine=skyvern-3.0 on the agent span; the v3 agent-loop LLM calls with their tool calls",
    run=run,
    expect=Expect(steps=False, prompts_include=("taskv3-agent-loop",)),
)

if __name__ == "__main__":
    main(SCENARIO)
