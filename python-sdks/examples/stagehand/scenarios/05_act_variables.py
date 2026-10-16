"""act_variables - act() with `variables`: the model sees %pin%, Stagehand fills the real value.

    python scenarios/05_act_variables.py
"""

from _harness import Expect, Scenario, fixtures_url, main, with_stagehand

PIN = "731904-secret"


async def run():
    async def flow(stagehand, page, browser):
        await page.goto(f"{fixtures_url()}/form.html")
        await stagehand.act("type 'Ada Lovelace' into the recipient field")
        await stagehand.act(
            "type %pin% into the approval PIN field",
            variables={"pin": {"value": PIN, "description": "the approval PIN"}},
        )

    return await with_stagehand(flow)


SCENARIO = Scenario(
    name="act_variables",
    title="act with a secret variable",
    source="docs: act variables",
    look_for="act input lists the variable name only; %pin% everywhere, the PIN nowhere",
    run=run,
    expect=Expect(forbidden=(PIN,), spans_include=("stagehand.act",)),
)

if __name__ == "__main__":
    main(SCENARIO)
