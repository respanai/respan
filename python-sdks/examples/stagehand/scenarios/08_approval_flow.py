"""approval_flow - mirrors cookbooks/approve-form-submission: fill, observe the
submit, approve, act on the observed action.

    python scenarios/08_approval_flow.py
"""

from _harness import Expect, Scenario, fixtures_url, main, with_stagehand


async def run():
    async def flow(stagehand, page, browser):
        await page.goto(f"{fixtures_url()}/form.html")
        await stagehand.act("type 'Grace Hopper' into the recipient field")
        await stagehand.act("type 250 into the amount field")
        actions = (await stagehand.observe("find the submit transfer button")).data
        if not actions:
            raise RuntimeError("nothing to approve")
        approved = actions[0]  # a human would review `approved.description` here
        await stagehand.act(approved)

    return await with_stagehand(flow)


SCENARIO = Scenario(
    name="approval_flow",
    title="observe -> human approval -> act(observed action)",
    source="cookbooks/approve-form-submission",
    look_for="observe returns the submit action; the next act takes that Action object (no LLM call)",
    run=run,
    expect=Expect(spans_include=("stagehand.observe", "stagehand.act")),
)

if __name__ == "__main__":
    main(SCENARIO)
