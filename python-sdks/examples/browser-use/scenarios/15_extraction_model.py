"""page_extraction_llm (small model) - mirrors browser-use's examples/features/small_model_for_extraction.py.

    python scenarios/15_extraction_model.py
"""

from browser_use import Agent

from _harness import SMALL_MODEL, Expect, Scenario, browser, llm, main


async def run():
    agent = Agent(
        task=(
            "Open https://en.wikipedia.org/wiki/Web_browser and use the extract action to "
            "list the browsers named in the History section."
        ),
        llm=llm(),
        page_extraction_llm=llm(SMALL_MODEL),
        browser=browser(),
    )
    await agent.run(max_steps=8)

SCENARIO = Scenario(
    name="extraction_model",
    title="page_extraction_llm (small model)",
    source="features/small_model_for_extraction.py",
    look_for="tool.extract contains its own LLM span on the small model; step LLM spans stay on the main model",
    run=run,
    expect=Expect(spans_include=("tool.extract",), models_include=(SMALL_MODEL,)),
)

if __name__ == "__main__":
    main(SCENARIO)
