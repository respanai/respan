"""use_vision=False - mirrors browser-use's examples/Agent(use_vision=False).

    python scenarios/25_no_vision.py
"""

from browser_use import Agent

from _harness import Expect, Scenario, browser, llm, main


async def run():
    agent = Agent(
        task="What is the title of the #1 story on https://news.ycombinator.com ?",
        llm=llm(),
        browser=browser(),
        use_vision=False,
    )
    await agent.run(max_steps=6)

SCENARIO = Scenario(
    name="no_vision",
    title="use_vision=False",
    source="Agent(use_vision=False)",
    look_for="LLM prompts contain the DOM text only - no image parts; step spans still carry the screenshot",
    run=run,
    expect=Expect(prompt_images=False),
)

if __name__ == "__main__":
    main(SCENARIO)
