"""Custom tools - mirrors browser-use's examples/custom-functions/ (Tools + @tools.action).

    python scenarios/17_custom_tools.py
"""

from browser_use import ActionResult, Agent, BrowserSession, Tools

from _harness import Expect, Scenario, browser, llm, main

tools = Tools()
notes: list[str] = []


@tools.action("Save a short note for the user. Call once per fact worth keeping.")
async def save_note(text: str) -> ActionResult:
    notes.append(text)
    # long_term_memory is what the agent remembers about this call later on.
    return ActionResult(extracted_content=f"Saved note #{len(notes)}", long_term_memory=f"Saved: {text}")


@tools.action("Return the title of the current page.")
async def current_page_title(browser_session: BrowserSession) -> ActionResult:
    # browser_session is injected by browser-use, not chosen by the model.
    return ActionResult(extracted_content=await browser_session.get_current_page_title())


async def run():
    agent = Agent(
        task=(
            "Open https://en.wikipedia.org/wiki/Web_browser. Call current_page_title once. "
            "Then call save_note exactly twice: who made the first web browser, and the year. Then finish."
        ),
        llm=llm(),
        browser=browser(),
        tools=tools,
    )
    await agent.run(max_steps=10)
    for note in notes:
        print("  note:", note)

SCENARIO = Scenario(
    name="custom_tools",
    title="Custom tools",
    source="custom-functions/ (Tools + @tools.action)",
    look_for="tool.save_note and tool.current_page_title spans with your function's params and return value",
    run=run,
    expect=Expect(spans_include=("tool.save_note", "tool.current_page_title")),
)

if __name__ == "__main__":
    main(SCENARIO)
