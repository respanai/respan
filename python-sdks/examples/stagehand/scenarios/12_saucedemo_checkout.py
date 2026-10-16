"""saucedemo_checkout - mirrors showcase/qa-checkout-flow on saucedemo.com (a
public test shop), on a local browser.

    python scenarios/12_saucedemo_checkout.py
"""

from pydantic import BaseModel

from _harness import Expect, Scenario, main, with_stagehand

# saucedemo.com's demo password. The site prints it on its login page (so it
# is in every prompt); passed as a variable all the same, as with a real one.
PASSWORD = "secret_sauce"


class Order(BaseModel):
    complete: bool
    heading: str


async def run():
    async def flow(stagehand, page, browser):
        await page.goto("https://www.saucedemo.com")
        await stagehand.act("type standard_user into the username field")
        await stagehand.act("type %password% into the password field", variables={"password": PASSWORD})
        for step in (
            "click the login button",
            "add the Sauce Labs Backpack to the cart",
            "open the shopping cart",
            "click checkout",
            "type Ada into the first name field",
            "type Lovelace into the last name field",
            "type 10115 into the postal code field",
            "click continue",
            "click finish",
        ):
            await stagehand.act(step)
        result = await stagehand.extract(
            "did the order complete, and what is the confirmation heading", Order
        )
        if not result.data.complete:
            raise RuntimeError("checkout did not complete")
        return result.data

    return await with_stagehand(flow)


SCENARIO = Scenario(
    name="saucedemo_checkout",
    title="log in, add to cart, check out",
    source="showcase/qa-checkout-flow",
    look_for="a dozen act spans in order, each with a screenshot; the final extract confirms the order",
    run=run,
    expect=Expect(spans_include=("stagehand.act", "stagehand.extract")),
)

if __name__ == "__main__":
    main(SCENARIO)
