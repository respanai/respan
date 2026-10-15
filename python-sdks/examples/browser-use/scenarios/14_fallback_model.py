"""fallback_llm - mirrors browser-use's examples/features/fallback_model.py.

    python scenarios/14_fallback_model.py
"""

from browser_use import Agent

from _harness import MODEL, Expect, Scenario, browser, llm, main, span_name

BROKEN = f"{MODEL}-does-not-exist"


async def run():
    agent = Agent(
        task="What is the title of the #1 story on https://news.ycombinator.com ?",
        llm=llm(BROKEN),
        fallback_llm=llm(),
        browser=browser(),
    )
    await agent.run(max_steps=6)

SCENARIO = Scenario(
    name="fallback_model",
    title="fallback_llm",
    source="features/fallback_model.py",
    look_for="the primary model's call is an error span; the next call is the fallback model, and the run still succeeds",
    run=run,
    expect=Expect(error_spans=(span_name(BROKEN),)),
)

if __name__ == "__main__":
    main(SCENARIO)
