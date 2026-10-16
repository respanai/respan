/** cookbooks/paginated-catalog: extract every page, act to go to the next one. */
import { z } from "zod";
import { withStagehand, type Scenario } from "../harness.js";

const Products = z.object({ products: z.array(z.object({ name: z.string(), price: z.number(), inStock: z.boolean() })) });

export const scenario: Scenario = {
  name: "paginated_catalog",
  title: "extract across 3 pages with act('next page')",
  source: "cookbooks/paginated-catalog",
  lookFor: "3 extract + 2 act in order; each extract's output lists 4 products",
  run: (site) =>
    withStagehand(async (stagehand, page) => {
      await page.goto(`${site}/catalog-1.html`);
      const all = [];
      for (let index = 1; index <= 3; index++) {
        const { data } = await stagehand.extract("extract every product on this page with its price and stock status", Products);
        all.push(...data.products);
        if (index < 3) await stagehand.act("click the Next page link");
      }
      if (all.length !== 12) throw new Error(`expected 12 products, got ${all.length}`);
      return all;
    }),
  expect: { spansInclude: ["stagehand.extract", "stagehand.act"] },
};
