/** HTTP calls with Playwright's APIRequestContext - mirrors Playwright's "API testing" page. */
import { request } from "playwright";
import { type Scenario } from "../harness.js";

export const scenario: Scenario = {
  name: "api_request",
  title: "APIRequestContext without a browser",
  source: "docs/api-testing",
  lookFor: "apiRequest.newContext, apiRequestContext.get with status + URL (Playwright reads the body with an internal call: no span)",
  async run(site) {
    const api = await request.newContext({ baseURL: site });
    try {
      const response = await api.get("/api/products.json");
      if (!response.ok()) throw new Error(`status ${response.status()}`);
      console.log(await response.json());
    } finally {
      await api.dispose();
    }
  },
  expect: { callsInclude: ["apiRequest.newContext", "apiRequestContext.get"] },
};
