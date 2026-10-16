/** act() with `variables`: the model sees %pin%, Stagehand fills the real value. */
import { withStagehand, type Scenario } from "../harness.js";

export const PIN = "731904-secret";

export const scenario: Scenario = {
  name: "act_variables",
  title: "act with a secret variable",
  source: "docs: act variables",
  lookFor: "act input lists the variable name only; %pin% everywhere, the PIN nowhere",
  run: (site) =>
    withStagehand(async (stagehand, page) => {
      await page.goto(`${site}/form.html`);
      await stagehand.act("type 'Ada Lovelace' into the recipient field");
      await stagehand.act("type %pin% into the approval PIN field", { variables: { pin: { value: PIN, description: "the approval PIN" } } });
    }),
  expect: { forbidden: [PIN], spansInclude: ["stagehand.act"] },
};
