"""Kernel's `browser-use` template (Python): a browser-use agent on a Kernel browser.
The model goes through the Respan gateway; browser-use's own spans come from
respan-instrumentation-browser-use, Kernel's from respan-instrumentation-kernel.

    python scenarios/02_browser_use.py       # KERNEL_FAKE=1 for the local fake Kernel
"""

from typing import TypedDict

import kernel
from _harness import HEADLESS, TARGET_URL, Expect, Scenario, llm, main
from browser_use import Agent, Browser
from kernel import Kernel

client = Kernel()

app = kernel.App("python-bu")


class TaskInput(TypedDict):
    task: str


@app.action("bu-task")
async def bu_task(ctx: kernel.KernelContext, input_data: TaskInput):
    kernel_browser = client.browsers.create(invocation_id=ctx.invocation_id, stealth=True)
    print("Kernel browser live view url: ", kernel_browser.browser_live_view_url)
    try:
        browser = Browser(
            cdp_url=kernel_browser.cdp_ws_url,
            headless=HEADLESS,
            window_size={"width": 1920, "height": 1080},
            viewport={"width": 1920, "height": 1080},
            device_scale_factor=1.0,
        )
        agent = Agent(task=input_data["task"], llm=llm(), browser_session=browser)
        result = await agent.run(max_steps=6)
        if result.final_result() is not None:
            return {"final_result": result.final_result()}
        return {"errors": result.errors()}
    finally:
        client.browsers.delete_by_id(kernel_browser.session_id)


SCENARIO = Scenario(
    name="browser_use",
    title="browser-use agent on a Kernel browser",
    source="kernel/cli pkg/templates/python/browser-use",
    look_for="agent kernel.action.bu-task -> browsers.create, browser_use agent -> steps -> LLM + actions, browsers.delete_by_id",
    app=app,
    action="bu-task",
    payload={"task": f"Open {TARGET_URL} and tell me the text of the main heading."},
    expect=Expect(calls_include=("browsers.create", "browsers.delete_by_id"), has_llm=True),
)

if __name__ == "__main__":
    main(SCENARIO)
