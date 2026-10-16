/** cookbooks/approve-form-submission: fill, observe the submit, approve, act on the observed action. */
import { withStagehand, type Scenario } from "../harness.js";

export const scenario: Scenario = {
  name: "approval_flow",
  title: "observe -> human approval -> act(observed action)",
  source: "cookbooks/approve-form-submission",
  lookFor: "observe returns the submit action; the next act takes that Action object (no LLM call)",
  run: (site) =>
    withStagehand(async (stagehand, page) => {
      await page.goto(`${site}/form.html`);
      await stagehand.act("type 'Grace Hopper' into the recipient field");
      await stagehand.act("type 250 into the amount field");
      const { data: actions } = await stagehand.observe("find the submit transfer button");
      const approved = actions[0]; // a human would review `approved.description` here
      if (!approved) throw new Error("nothing to approve");
      await stagehand.act(approved);
    }),
  expect: { spansInclude: ["stagehand.observe", "stagehand.act"] },
};
