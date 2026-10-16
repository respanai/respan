/** Fill a sign-up form: text, password, select, checkbox, submit - mirrors Playwright's "Actions" page. */
import { expect } from "@playwright/test";
import { chromium } from "playwright";
import { launchOptions, type Scenario } from "../harness.js";

const PASSWORD = "correct-horse-battery-staple-42";

export const scenario: Scenario = {
  name: "forms",
  title: "Fill and submit a form",
  source: "docs/input",
  lookFor: "locator.fill shows the email; the password fill shows <masked>; the URL after submit shows <secret>password</secret>",
  async run(site) {
    const browser = await chromium.launch(launchOptions());
    try {
      const page = await browser.newPage();
      await page.goto(`${site}/form.html`);
      await page.getByLabel("Email").fill("ada@example.com");
      await page.getByLabel("Password").fill(PASSWORD);
      await page.getByLabel("Plan").selectOption("pro");
      await page.getByLabel("I accept the terms").check();
      await page.getByRole("button", { name: "Create account" }).click();
      // The form submits by GET: the password ends up in the page URL too.
      await expect(page).toHaveTitle("Welcome");
    } finally {
      await browser.close();
    }
  },
  expect: {
    callsInclude: ["locator.fill", "locator.selectOption", "locator.check", "locator.click", "expect.toHaveTitle"],
    forbidden: [PASSWORD],
  },
};
