"""sensitive_data - mirrors browser-use's examples/features/sensitive_data.py.

    python scenarios/11_sensitive_data.py
"""

from browser_use import Agent

from _harness import Expect, Scenario, browser, llm, main

# A public practice site; these are its published demo credentials.
SITE = "the-internet.herokuapp.com"
PASSWORD = "SuperSecretPassword!"


async def run():
    agent = Agent(
        task=(
            f"Go to https://{SITE}/login and log in with username demo_user and "
            "password demo_pass. Report the message shown after login."
        ),
        llm=llm(),
        # browser-use refuses sensitive_data without a domain allowlist.
        browser=browser(allowed_domains=[SITE]),
        sensitive_data={"demo_user": "tomsmith", "demo_pass": PASSWORD},
    )
    await agent.run(max_steps=8)

SCENARIO = Scenario(
    name="sensitive_data",
    title="sensitive_data",
    source="features/sensitive_data.py",
    look_for="tool.input params show <secret>demo_pass</secret>; the real password appears in no span (page text included)",
    run=run,
    expect=Expect(spans_include=("tool.input",), forbidden=("SuperSecretPassword!",)),
)

if __name__ == "__main__":
    main(SCENARIO)
