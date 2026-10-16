/** Print a page to PDF - mirrors pptr.dev's "PDF generation" guide. */
import { join } from "node:path";
import puppeteer from "puppeteer-core";
import { ARTIFACTS, launchOptions, type Scenario } from "../harness.js";

export const scenario: Scenario = {
  name: "pdf",
  title: "Generate a PDF",
  source: "guides/pdf-generation",
  lookFor: "page.pdf with its options in and the PDF's byte size out",
  async run(site) {
    const browser = await puppeteer.launch(launchOptions());
    try {
      const page = await browser.newPage();
      await page.goto(`${site}/products.html`, { waitUntil: "networkidle0" });
      const pdf = await page.pdf({ path: join(ARTIFACTS, "products.pdf"), format: "A4" });
      console.log(`PDF: ${pdf.byteLength} bytes`);
    } finally {
      await browser.close();
    }
  },
  expect: { callsInclude: ["page.goto", "page.pdf"] },
};
