"""Fill and submit a form - mirrors browser-use's examples/getting_started/02_form_filling.py.

    python scenarios/02_form_filling.py
"""

from browser_use import Agent

from _harness import Expect, Scenario, browser, llm, main


async def run():
    agent = Agent(
        task=(
            "Go to https://httpbin.org/forms/post. Fill in customer name 'Ada Lovelace', "
            "telephone '555-0100', email 'ada@example.com', pizza size Medium, topping Bacon, "
            "delivery time 18:00 and comments 'Ring twice'. Submit the form and report the "
            "customer name the server echoes back."
        ),
        llm=llm(),
        browser=browser(),
    )
    await agent.run(max_steps=15)

SCENARIO = Scenario(
    name="form_filling",
    title="Fill and submit a form",
    source="getting_started/02_form_filling.py",
    look_for="tool.input spans carry the typed values; tool.click submits; the step after shows the server's echo",
    run=run,
    expect=Expect(spans_include=("tool.input", "tool.click")),
)

if __name__ == "__main__":
    main(SCENARIO)
