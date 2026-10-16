"""pricing_monitor - mirrors showcase/saas-pricing-monitor: extract a pricing
table from a live SaaS page.

    python scenarios/13_pricing_monitor.py
"""

from pydantic import BaseModel

from _harness import Expect, Scenario, main, with_stagehand


class Plan(BaseModel):
    name: str
    price: str


class Plans(BaseModel):
    plans: list[Plan]


async def run():
    async def flow(stagehand, page, browser):
        await page.goto("https://linear.app/pricing")
        result = await stagehand.extract(
            "extract every pricing plan with its name and monthly price as shown", Plans
        )
        return result.data

    return await with_stagehand(flow)


SCENARIO = Scenario(
    name="pricing_monitor",
    title="extract pricing tiers from a live pricing page",
    source="showcase/saas-pricing-monitor",
    look_for="extract output lists the plans with prices",
    run=run,
    expect=Expect(spans_include=("stagehand.extract",)),
)

if __name__ == "__main__":
    main(SCENARIO)
