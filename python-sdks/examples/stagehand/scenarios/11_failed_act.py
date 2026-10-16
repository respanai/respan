"""failed_act - a failing browser action and an act on something that is not there.

    python scenarios/11_failed_act.py
"""

import contextlib

from _harness import Expect, Scenario, fixtures_url, main, with_stagehand


async def run():
    async def flow(stagehand, page, browser):
        await page.goto(f"{fixtures_url()}/form.html")
        with contextlib.suppress(Exception):
            await page.locator("#does-not-exist").click(timeout=1000)
        with contextlib.suppress(Exception):
            await stagehand.act("click the 'Delete account' button", timeout=20000)

    return await with_stagehand(flow)


SCENARIO = Scenario(
    name="failed_act",
    title="a timed-out click and an impossible act",
    source="docs: timeouts and error handling",
    look_for="locator.click is an error span with 'Type: message'; the impossible act reports success=false",
    run=run,
    expect=Expect(error_spans=("locator.click", "stagehand.act")),
)

if __name__ == "__main__":
    main(SCENARIO)
