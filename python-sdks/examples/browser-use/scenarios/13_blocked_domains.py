"""prohibited_domains - mirrors browser-use's examples/features/blocked_domains.py.

    python scenarios/13_blocked_domains.py
"""

from browser_use import Agent

from _harness import Expect, Scenario, browser, llm, main


async def run():
    agent = Agent(
        task=(
            "Open https://news.ycombinator.com and report the top story. "
            "If you cannot open it, say why and finish."
        ),
        llm=llm(),
        browser=browser(prohibited_domains=["news.ycombinator.com"]),
    )
    await agent.run(max_steps=4)

SCENARIO = Scenario(
    name="blocked_domains",
    title="prohibited_domains",
    source="features/blocked_domains.py",
    look_for="same as allowed_domains but with a blocklist: the navigate span fails",
    run=run,
    expect=Expect(success=None, error_spans=("tool.navigate",)),
)

if __name__ == "__main__":
    main(SCENARIO)
