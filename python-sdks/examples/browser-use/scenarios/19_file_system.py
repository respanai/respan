"""Agent file system - mirrors browser-use's examples/file_system/file_system.py.

    python scenarios/19_file_system.py
"""

import tempfile

from browser_use import Agent

from _harness import Expect, Scenario, browser, llm, main


async def run():
    agent = Agent(
        task=(
            "Collect the titles of the top 3 stories on https://news.ycombinator.com, write "
            "them to hn.md with write_file, then read hn.md back with read_file and return it."
        ),
        llm=llm(),
        browser=browser(),
        file_system_path=tempfile.mkdtemp(prefix="browser-use-fs-"),
    )
    await agent.run(max_steps=10)

SCENARIO = Scenario(
    name="file_system",
    title="Agent file system",
    source="file_system/file_system.py",
    look_for="tool.write_file / tool.read_file spans with the file name and content",
    run=run,
    expect=Expect(spans_include=("tool.write_file", "tool.read_file")),
)

if __name__ == "__main__":
    main(SCENARIO)
