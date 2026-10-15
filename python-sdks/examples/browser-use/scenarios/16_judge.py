"""Judge with ground truth - mirrors browser-use's examples/features/judge_trace.py.

    python scenarios/16_judge.py
"""

from browser_use import Agent

from _harness import Expect, Scenario, browser, llm, main


async def run():
    agent = Agent(
        task="Open https://en.wikipedia.org/wiki/Web_browser and tell me the name of the first web browser.",
        llm=llm(),
        browser=browser(),
        judge_llm=llm(),
        ground_truth="WorldWideWeb (later renamed Nexus), created by Tim Berners-Lee in 1990",
    )
    history = await agent.run(max_steps=8)
    print("  judgement:", history.judgement())

SCENARIO = Scenario(
    name="judge",
    title="Judge with ground truth",
    source="features/judge_trace.py",
    look_for="the last LLM span (under the agent, after the steps) is the judge; the agent output carries judge_verdict and judgement",
    run=run,
    expect=Expect(),
)

if __name__ == "__main__":
    main(SCENARIO)
