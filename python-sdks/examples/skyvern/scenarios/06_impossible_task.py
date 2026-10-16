"""Failure handling - a goal the page cannot satisfy; Skyvern terminates the run.

    python scenarios/06_impossible_task.py
"""

from _harness import Expect, Scenario, main, skyvern


async def run():
    result = await skyvern().run_task(
        prompt=(
            "Log in with the account 'nobody' and download the invoice. "
            "If there is no way to log in on this page, terminate."
        ),
        url="https://example.com",
        max_steps=3,
        wait_for_completion=True,
    )
    print(f"  {result.status}: {result.failure_reason}")


SCENARIO = Scenario(
    name="impossible_task",
    title="Agent terminates an impossible task",
    source="docs: sdk-reference run_task (terminate / failure_reason)",
    look_for="tool.terminate under the step; the agent span is an error carrying the reason",
    run=run,
    expect=Expect(
        statuses=("terminated",),
        spans_include=("terminate",),
        error_spans=("agent.skyvern_task",),
    ),
)

if __name__ == "__main__":
    main(SCENARIO)
