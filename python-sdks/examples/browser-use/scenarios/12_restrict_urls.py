"""allowed_domains - mirrors browser-use's examples/features/restrict_urls.py.

    python scenarios/12_restrict_urls.py
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
        browser=browser(allowed_domains=["en.wikipedia.org"]),
    )
    await agent.run(max_steps=4)

SCENARIO = Scenario(
    name="restrict_urls",
    title="allowed_domains",
    source="features/restrict_urls.py",
    look_for="the blocked tool.navigate is an error span with browser-use's reason; the agent reports it could not go there",
    run=run,
    expect=Expect(success=None, error_spans=("tool.navigate",)),
)

if __name__ == "__main__":
    main(SCENARIO)
