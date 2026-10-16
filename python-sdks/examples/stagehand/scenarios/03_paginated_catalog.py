"""paginated_catalog - mirrors cookbooks/paginated-catalog: extract every page,
act to go to the next one.

    python scenarios/03_paginated_catalog.py
"""

from pydantic import BaseModel

from _harness import Expect, Scenario, fixtures_url, main, with_stagehand


class Product(BaseModel):
    name: str
    price: float
    in_stock: bool


class Products(BaseModel):
    products: list[Product]


async def run():
    async def flow(stagehand, page, browser):
        await page.goto(f"{fixtures_url()}/catalog-1.html")
        products = []
        for index in range(1, 4):
            result = await stagehand.extract(
                "extract every product on this page with its price and stock status", Products
            )
            products += result.data.products
            if index < 3:
                await stagehand.act("click the Next page link")
        if len(products) != 12:
            raise RuntimeError(f"expected 12 products, got {len(products)}")
        return products

    return await with_stagehand(flow)


SCENARIO = Scenario(
    name="paginated_catalog",
    title="extract across 3 pages with act('next page')",
    source="cookbooks/paginated-catalog",
    look_for="3 extract + 2 act in order; each extract's output lists 4 products",
    run=run,
    expect=Expect(spans_include=("stagehand.extract", "stagehand.act")),
)

if __name__ == "__main__":
    main(SCENARIO)
