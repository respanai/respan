"""screenshot_files - mirrors cookbooks/files-to-bucket (locally): extract data,
screenshot, write both to disk.

    python scenarios/07_screenshot_files.py
"""

import json
import shutil
import tempfile
from pathlib import Path

from pydantic import BaseModel

from _harness import Expect, Scenario, fixtures_url, main, with_stagehand


class Names(BaseModel):
    names: list[str]


async def run():
    async def flow(stagehand, page, browser):
        out = Path(tempfile.mkdtemp(prefix="stagehand-files-"))
        try:
            await page.goto(f"{fixtures_url()}/catalog-2.html")
            result = await stagehand.extract("extract the product names on this page", Names)
            (out / "products.json").write_text(json.dumps(result.data.model_dump()))
            await page.screenshot(path=out / "page.png")
        finally:
            shutil.rmtree(out, ignore_errors=True)

    return await with_stagehand(flow)


SCENARIO = Scenario(
    name="screenshot_files",
    title="screenshot + JSON written to files",
    source="cookbooks/files-to-bucket",
    look_for="page.screenshot output renders the image; extract carries the JSON",
    run=run,
    expect=Expect(spans_include=("page.screenshot", "stagehand.extract")),
)

if __name__ == "__main__":
    main(SCENARIO)
