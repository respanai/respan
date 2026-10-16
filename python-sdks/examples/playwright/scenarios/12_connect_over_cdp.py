"""Attach to an already-running Chrome over CDP - mirrors
BrowserType.connect_over_cdp in Playwright for Python's API docs. The
"remote" browser here is a local Chrome started with a debugging port.

    python scenarios/12_connect_over_cdp.py
"""

import socket

from playwright.sync_api import sync_playwright

from _harness import SITE, Expect, Scenario, launch_options, main


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def run():
    port = free_port()
    with sync_playwright() as p:
        remote = p.chromium.launch(**launch_options(), args=[f"--remote-debugging-port={port}"])
        browser = p.chromium.connect_over_cdp(f"http://127.0.0.1:{port}")
        context = browser.contexts[0] if browser.contexts else browser.new_context()
        page = context.new_page()
        page.goto(SITE)
        print(page.title())
        browser.close()
        remote.close()


SCENARIO = Scenario(
    name="connect_over_cdp",
    title="connect_over_cdp to a running Chrome",
    source="docs/api/class-browsertype#connect-over-cdp",
    look_for="BrowserType.connect_over_cdp, then the attached page's calls under the same workflow",
    run=run,
    expect=Expect(calls_include=("BrowserType.connect_over_cdp", "Page.goto")),
)

if __name__ == "__main__":
    main(SCENARIO)
