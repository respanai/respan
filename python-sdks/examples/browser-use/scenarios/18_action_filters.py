"""Domain-scoped tools - mirrors browser-use's examples/custom-functions/action_filters.py.

    python scenarios/18_action_filters.py
"""

from browser_use import ActionResult, Agent, Tools

from _harness import Expect, Scenario, browser, llm, main

tools = Tools()


@tools.action("Record one fact from the current Wikipedia article.", domains=["*.wikipedia.org"])
async def wiki_fact(fact: str) -> ActionResult:
    return ActionResult(extracted_content=f"Recorded: {fact}", long_term_memory=f"Recorded: {fact}")


async def run():
    agent = Agent(
        task=(
            "Open https://en.wikipedia.org/wiki/Web_browser and record one fact about it "
            "with wiki_fact. Then finish."
        ),
        llm=llm(),
        browser=browser(),
        tools=tools,
    )
    await agent.run(max_steps=6)

SCENARIO = Scenario(
    name="action_filters",
    title="Domain-scoped tools",
    source="custom-functions/action_filters.py",
    look_for="tool.wiki_fact appears only on Wikipedia steps: browser-use offers it to the model only there",
    run=run,
    expect=Expect(spans_include=("tool.wiki_fact",)),
)

if __name__ == "__main__":
    main(SCENARIO)
