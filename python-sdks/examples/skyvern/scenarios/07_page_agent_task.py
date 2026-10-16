"""Agent task on an open page - mirrors Skyvern's `page.agent.run_task` example.

    python scenarios/07_page_agent_task.py
"""

from _harness import HEADLESS, Scenario, main, skyvern


async def run():
    browser = await skyvern().launch_local_browser(headless=HEADLESS)
    try:
        page = await browser.get_working_page()
        await page.goto("https://books.toscrape.com")
        result = await page.agent.run_task(
            "Open the 'Science' category and return the title of the first book.", max_steps=6
        )
        print(f"  {result.status}: {result.output}")
    finally:
        await browser.close()


SCENARIO = Scenario(
    name="page_agent_task",
    title="page.agent.run_task",
    source="docs: sdk-reference browser page (agent.run_task)",
    look_for="agent.skyvern_task with steps driving the page you opened",
    run=run,
)

if __name__ == "__main__":
    main(SCENARIO)
