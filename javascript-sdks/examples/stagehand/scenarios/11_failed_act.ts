/** A failing browser action and an act on something that is not there. */
import { withStagehand, type Scenario } from "../harness.js";

export const scenario: Scenario = {
  name: "failed_act",
  title: "a timed-out click and an impossible act",
  source: "docs: timeouts and error handling",
  lookFor: "locator.click is an error span with 'Type: message'; the impossible act reports success=false",
  run: (site) =>
    withStagehand(async (stagehand, page) => {
      await page.goto(`${site}/form.html`);
      await page
        .locator("#does-not-exist")
        .click({ timeout: 1000 })
        .catch(() => undefined);
      await stagehand.act("click the 'Delete account' button", { timeout: 20000 }).catch(() => undefined);
    }),
  expect: { errorSpans: ["locator.click", "stagehand.act"] },
};
