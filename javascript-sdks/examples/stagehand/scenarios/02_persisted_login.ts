/**
 * cookbooks/persisted-login: observe() for selectors, fill credentials without
 * the model ever seeing them, then a second session on the same profile
 * (userDataDir) reads the page state.
 */
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { z } from "zod";
import { withStagehand, type Scenario } from "../harness.js";

export const PASSWORD = "hunter2-correct-horse-4471";

export const scenario: Scenario = {
  name: "persisted_login",
  title: "observe selectors, fill credentials, reuse the profile",
  source: "cookbooks/persisted-login",
  lookFor: "2 traces; observe returns selectors; the password locator.fill shows [redacted], the email does not",
  run: async (site) => {
    const userDataDir = mkdtempSync(join(tmpdir(), "stagehand-login-"));
    try {
      await withStagehand(
        async (stagehand, page) => {
          await page.goto(`${site}/login.html`);
          const { data: email } = await stagehand.observe("find the email input");
          const { data: password } = await stagehand.observe("find the password input");
          await page.locator(email[0].selector).fill("ada@example.com");
          await page.locator(password[0].selector).fill(PASSWORD);
          await stagehand.act("click the sign in button");
        },
        { userDataDir },
      );
      return await withStagehand(
        async (stagehand, page) => {
          await page.goto(`${site}/login.html`);
          return (await stagehand.extract("is the user signed in, and what does the page say", z.object({ signedIn: z.boolean(), message: z.string() }))).data;
        },
        { userDataDir },
      );
    } finally {
      rmSync(userDataDir, { recursive: true, force: true });
    }
  },
  expect: { traces: 2, forbidden: [PASSWORD], spansInclude: ["stagehand.observe", "locator.fill"] },
};
