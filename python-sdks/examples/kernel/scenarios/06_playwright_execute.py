"""Kernel's Playwright Execution API: run Playwright code inside the browser's VM.
A failed run is reported in the response (success=false), and shows as an error span.

    python scenarios/06_playwright_execute.py  # KERNEL_FAKE=1 for the local fake Kernel
"""

import kernel
from _harness import TARGET_URL, Expect, Scenario, main
from kernel import Kernel

client = Kernel()

app = kernel.App("python-playwright-execute")


@app.action("title")
async def title(ctx: kernel.KernelContext, payload: dict) -> dict:
    kernel_browser = client.browsers.create(
        invocation_id=ctx.invocation_id, start_url=payload["url"]
    )
    session = kernel_browser.session_id
    try:
        result = client.browsers.playwright.execute(session, code="return await page.title();")
        failed = client.browsers.playwright.execute(
            session, code="throw new Error('selector #missing not found');"
        )
        return {"title": result.result, "failed_run_error": failed.error}
    finally:
        client.browsers.delete_by_id(session)


SCENARIO = Scenario(
    name="playwright_execute",
    title="Playwright Execution: code in, result out",
    source="kernel.sh/docs/browsers/playwright-execution",
    look_for="playwright.execute spans with the code and result; the failing run as an error span",
    app=app,
    action="title",
    payload={"url": TARGET_URL},
    expect=Expect(calls_include=("playwright.execute",), error_calls=("playwright.execute",)),
)

if __name__ == "__main__":
    main(SCENARIO)
