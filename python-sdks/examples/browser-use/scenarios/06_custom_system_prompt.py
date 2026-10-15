"""extend_system_message - mirrors browser-use's examples/features/custom_system_prompt.py.

    python scenarios/06_custom_system_prompt.py
"""

from browser_use import Agent

from _harness import Expect, Scenario, browser, llm, main


async def run():
    agent = Agent(
        task="What is the title of the #1 story on https://news.ycombinator.com ?",
        llm=llm(),
        browser=browser(),
        extend_system_message="Always write your final answer in French.",
    )
    await agent.run(max_steps=6)

SCENARIO = Scenario(
    name="custom_system_prompt",
    title="extend_system_message",
    source="features/custom_system_prompt.py",
    look_for="the system message of each LLM span ends with the extra rule; the final answer is in French",
    run=run,
    expect=Expect(),
)

if __name__ == "__main__":
    main(SCENARIO)
