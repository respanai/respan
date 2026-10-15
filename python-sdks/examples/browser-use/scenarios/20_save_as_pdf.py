"""Save a page as PDF - mirrors browser-use's examples/features/save_as_pdf.py.

    python scenarios/20_save_as_pdf.py
"""

import tempfile

from browser_use import Agent

from _harness import Expect, Scenario, browser, llm, main


async def run():
    agent = Agent(
        task="Open https://en.wikipedia.org/wiki/Web_browser and save the page as a PDF.",
        llm=llm(),
        browser=browser(downloads_path=tempfile.mkdtemp(prefix="browser-use-pdf-")),
    )
    await agent.run(max_steps=6)

SCENARIO = Scenario(
    name="save_as_pdf",
    title="Save a page as PDF",
    source="features/save_as_pdf.py",
    look_for="a tool.save_as_pdf span whose output names the saved file",
    run=run,
    expect=Expect(spans_include=("tool.save_as_pdf",)),
)

if __name__ == "__main__":
    main(SCENARIO)
