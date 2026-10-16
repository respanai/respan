/** showcase/qa-checkout-flow on saucedemo.com (a public test shop), on a local browser. */
import { z } from "zod";
import { withStagehand, type Scenario } from "../harness.js";

// saucedemo.com's demo password. The site prints it on its login page (so it
// is in every prompt); passed as a variable all the same, as with a real one.
const PASSWORD = "secret_sauce";

export const scenario: Scenario = {
  name: "saucedemo_checkout",
  title: "log in, add to cart, check out",
  source: "showcase/qa-checkout-flow",
  lookFor: "a dozen act spans in order, each with a screenshot; the final extract confirms the order",
  run: () =>
    withStagehand(async (stagehand, page) => {
      await page.goto("https://www.saucedemo.com");
      await stagehand.act("type standard_user into the username field");
      await stagehand.act("type %password% into the password field", { variables: { password: PASSWORD } });
      await stagehand.act("click the login button");
      await stagehand.act("add the Sauce Labs Backpack to the cart");
      await stagehand.act("open the shopping cart");
      await stagehand.act("click checkout");
      await stagehand.act("type Ada into the first name field");
      await stagehand.act("type Lovelace into the last name field");
      await stagehand.act("type 10115 into the postal code field");
      await stagehand.act("click continue");
      await stagehand.act("click finish");
      const { data } = await stagehand.extract("did the order complete, and what is the confirmation heading", z.object({ complete: z.boolean(), heading: z.string() }));
      if (!data.complete) throw new Error("checkout did not complete");
      return data;
    }),
  expect: { spansInclude: ["stagehand.act", "stagehand.extract"] },
};
