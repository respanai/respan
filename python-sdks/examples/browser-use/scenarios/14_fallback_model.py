"""fallback_llm - mirrors browser-use's examples/features/fallback_model.py.

    python scenarios/14_fallback_model.py
"""

from browser_use import Agent

from browser_use import ChatOpenAI

from _harness import API_KEY, MODEL, Expect, Scenario, browser, llm, main, span_name

# browser-use switches to fallback_llm only on 401/402/429/5xx or a connection
# error (Respan answers a bad key with 403, which does not count). So the primary
# model points at a port nothing listens on: its first call fails to connect.
broken = ChatOpenAI(model=MODEL, api_key=API_KEY, base_url="http://127.0.0.1:9/api", max_retries=0)


async def run():
    agent = Agent(
        task="What is the title of the #1 story on https://news.ycombinator.com ?",
        llm=broken,
        fallback_llm=llm(),
        # The judge defaults to the PRIMARY model and does not fall back, so
        # without this the judge's call fails too (a second, silent error span).
        judge_llm=llm(),
        browser=browser(),
    )
    await agent.run(max_steps=6)

SCENARIO = Scenario(
    name="fallback_model",
    title="fallback_llm",
    source="features/fallback_model.py",
    look_for="the primary model's call is an error span; the next call is the fallback model, and the run still succeeds",
    run=run,
    expect=Expect(error_spans=(span_name(MODEL),)),
)

if __name__ == "__main__":
    main(SCENARIO)
