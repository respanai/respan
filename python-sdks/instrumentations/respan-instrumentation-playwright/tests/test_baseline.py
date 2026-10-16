"""Baseline wire contract: what a customer configures must reach every span.

These drive a real headless Chrome through Playwright's sync AND async APIs
against a local sign-in page, and capture spans *through the Respan pipeline*
(`Respan(...)` -> Respan's span processor -> what its exporter would send).
A scripted mini-agent (a fake LLM whose tool calls map to Playwright actions)
stands in for the agent a customer wraps around Playwright.
"""

from __future__ import annotations

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest
from opentelemetry import trace
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.trace import StatusCode
from playwright.async_api import async_playwright
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright
from respan_instrumentation_playwright import PlaywrightInstrumentor, secrets
from respan_instrumentation_playwright._constants import ATTR_PAGE_TITLE, ATTR_PAGE_URL
from respan_sdk.constants.span_attributes import (
    RESPAN_LOG_TYPE,
    RESPAN_METADATA,
    RESPAN_THREADS_ID,
    RESPAN_TRACE_GROUP_ID,
)
from respan_tracing.utils.agent_instrumentation import AgentScope, agent_span

from respan import agent, workflow

from .conftest import chrome_path, launch_options

pytestmark = pytest.mark.skipif(chrome_path() is None, reason="needs a local Chrome/Chromium")

CUSTOMER_ID = "respan.customer_params.customer_identifier"
CUSTOMER_METADATA = {"agent_id": "a1", "role": "browser"}
USERNAME = "ada.lovelace"
PASSWORD = "hunter2-correct-horse"
API_TOKEN = "tok-5531-secret"
OTHER_PASSWORD = "never-typed-anywhere-9921"

# The scripted model: each turn is one tool call the "agent" maps to Playwright.
SCRIPT = [
    ("goto", {"path": "/index.html"}),
    ("fill", {"selector": "#user", "text": USERNAME}),
    ("fill", {"selector": "#password", "text": PASSWORD}),
    ("click", {"selector": "#submit"}),
]


def run_sync_agent(page: Any, site: str) -> None:
    for tool, args in SCRIPT:
        if tool == "goto":
            page.goto(site + args["path"])
        elif tool == "fill":
            page.locator(args["selector"]).fill(args["text"])
        elif tool == "click":
            page.click(args["selector"])
    page.wait_for_url("**/done.html?*")


async def run_async_agent(page: Any, site: str) -> None:
    for tool, args in SCRIPT:
        if tool == "goto":
            await page.goto(site + args["path"])
        elif tool == "fill":
            await page.locator(args["selector"]).fill(args["text"])
        elif tool == "click":
            await page.click(args["selector"])
    await page.wait_for_url("**/done.html?*")


def sync_session(site: str, *, route: bool = False) -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch(**launch_options())
        try:
            page = browser.new_page()
            if route:
                # Runs in Playwright's own greenlet, not the caller's.
                page.route(
                    "**/done.html?*",
                    lambda r: r.fulfill(body="<title>Mocked</title>ok", content_type="text/html"),
                )
            run_sync_agent(page, site)
        finally:
            browser.close()


async def async_session(site: str, *, route: bool = False) -> None:
    async with async_playwright() as p:
        browser = await p.chromium.launch(**launch_options())
        try:
            page = await browser.new_page()
            if route:

                async def mock(r: Any) -> None:
                    await r.fulfill(body="<title>Mocked</title>ok", content_type="text/html")

                await page.route("**/done.html?*", mock)
            await run_async_agent(page, site)
        finally:
            await browser.close()


def playwright_spans(spans: list[Any]) -> list[Any]:
    return [s for s in spans if s.instrumentation_scope.name == "respan.instrumentation.playwright"]


def roots(spans: list[Any]) -> list[Any]:
    return [s for s in spans if s.parent is None]


def dumped(span: Any) -> str:
    return json.dumps(dict(span.attributes), default=str)


def assert_nested_under_one_root(spans: list[Any]) -> Any:
    (root,) = roots(spans)
    calls = playwright_spans(spans)
    names = {s.name for s in calls}
    assert {"Page.goto", "Locator.fill", "Page.click", "Page.wait_for_url"} <= names, names
    for span in calls:
        assert span.context.trace_id == root.context.trace_id, span.name
        assert span.parent.span_id == root.context.span_id, span.name
        # Exportable: Respan drops spans without a kind (G2).
        assert span.attributes.get(RESPAN_LOG_TYPE) == "tool", span.name
        assert span.attributes.get(SpanAttributes.TRACELOOP_SPAN_KIND) == "tool", span.name
    return root


# -- parentage and exportability, both APIs ------------------------------------


def test_baseline_sync_calls_nest_under_the_active_span(pipeline, site):
    _, spans = pipeline()
    workflow(name="login")(sync_session)(site)
    assert_nested_under_one_root(spans)


async def test_baseline_async_calls_nest_under_the_active_span(pipeline, site):
    _, spans = pipeline()
    await workflow(name="login")(async_session)(site)
    assert_nested_under_one_root(spans)


def test_baseline_sync_route_handler_keeps_parent_and_customer_context(pipeline, site):
    """The sync API runs handlers in a fresh greenlet: without carrying the
    registration context over, Route.fulfill has no parent (so it is not traced)
    and no customer identifier."""
    client, spans = pipeline()
    with client.propagate_attributes(customer_identifier="customer-1"):
        workflow(name="login")(sync_session)(site, route=True)

    root = assert_nested_under_one_root(spans)
    (fulfill,) = [s for s in spans if s.name == "Route.fulfill"]
    assert fulfill.parent.span_id == root.context.span_id
    assert fulfill.attributes.get(CUSTOMER_ID) == "customer-1"


async def test_baseline_async_route_handler_keeps_parent_and_customer_context(pipeline, site):
    client, spans = pipeline()
    with client.propagate_attributes(customer_identifier="customer-1"):
        await workflow(name="login")(async_session)(site, route=True)

    root = assert_nested_under_one_root(spans)
    (fulfill,) = [s for s in spans if s.name == "Route.fulfill"]
    assert fulfill.parent.span_id == root.context.span_id
    assert fulfill.attributes.get(CUSTOMER_ID) == "customer-1"


def test_baseline_sync_calls_inside_a_threaded_agent_nest_under_it(pipeline, site):
    """sync_playwright in a worker thread under the customer's own agent span."""
    _, spans = pipeline()

    @agent(name="browser_agent")
    def browse() -> None:
        sync_session(site)

    with ThreadPoolExecutor(1) as pool:
        pool.submit(browse).result()
    assert_nested_under_one_root(spans)


def test_baseline_no_active_span_means_no_spans(pipeline, site):
    """Standalone scripts and test suites are not flooded with one-span traces."""
    _, spans = pipeline()
    sync_session(site, route=True)
    assert spans == []


# -- customer context ----------------------------------------------------------


def test_baseline_customer_metadata_and_identifier_on_every_span(pipeline, site):
    client, spans = pipeline(metadata={"source": "constructor"}, customer_identifier="ctor")
    with client.propagate_attributes(customer_identifier="customer-1", metadata=CUSTOMER_METADATA):
        workflow(name="login")(sync_session)(site, route=True)

    assert_nested_under_one_root(spans)
    expected = {"source": "constructor", **CUSTOMER_METADATA}
    for span in spans:
        assert span.attributes.get(CUSTOMER_ID) == "customer-1", span.name
        for key, value in expected.items():
            assert span.attributes.get(f"{RESPAN_METADATA}.{key}") == value, (span.name, key)
        assert RESPAN_METADATA not in span.attributes, span.name


def test_baseline_customer_thread_and_trace_group_on_every_span(pipeline, site):
    """Playwright has no run id of its own: spans join the caller's thread/group."""
    client, spans = pipeline()
    with client.propagate_attributes(thread_identifier="my-thread", group_identifier="my-group"):
        workflow(name="login")(sync_session)(site, route=True)

    assert_nested_under_one_root(spans)
    for span in spans:
        assert span.attributes.get(RESPAN_THREADS_ID) == "my-thread", span.name
        assert span.attributes.get(RESPAN_TRACE_GROUP_ID) == "my-group", span.name


async def test_baseline_parallel_sessions_do_not_share_traces_or_identifiers(pipeline, site):
    client, spans = pipeline()

    async def one(name: str) -> None:
        with client.propagate_attributes(customer_identifier=f"customer-{name}"):
            await workflow(name=f"login-{name}")(async_session)(site, route=True)

    await asyncio.gather(one("a"), one("b"))

    trace_of = {s.attributes[CUSTOMER_ID]: s.context.trace_id for s in roots(spans)}
    assert set(trace_of) == {"customer-a", "customer-b"}
    assert len(set(trace_of.values())) == 2
    for span in spans:
        assert span.context.trace_id == trace_of[span.attributes[CUSTOMER_ID]], span.name
    for name in ("a", "b"):
        mine = [s for s in spans if s.attributes[CUSTOMER_ID] == f"customer-{name}"]
        assert sum(s.name == "Route.fulfill" for s in mine) == 1


# -- content, secrets, errors --------------------------------------------------


def test_baseline_password_text_never_reaches_a_span(pipeline, site):
    """Typed into a password field: masked in the call, and wherever it comes
    back later (this form submits by GET, so the password lands in the URL)."""
    _, spans = pipeline()
    workflow(name="login")(sync_session)(site)

    assert_nested_under_one_root(spans)
    for span in spans:
        assert PASSWORD not in dumped(span), span.name
    fills = {
        json.loads(s.attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT])["locator"]: s
        for s in spans
        if s.name == "Locator.fill"
    }
    password_input = json.loads(fills["#password"].attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT])
    user_input = json.loads(fills["#user"].attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT])
    assert password_input["value"] == "<masked>"
    assert user_input["value"] == USERNAME  # ordinary text is kept
    assert any("<secret>password</secret>" in s.attributes.get(ATTR_PAGE_URL, "") for s in spans)


def test_baseline_declared_secrets_never_reach_a_span(pipeline, site):
    _, spans = pipeline()

    @workflow(name="with_token")
    def run() -> None:
        with secrets({"api_token": API_TOKEN}), sync_playwright() as p:
            browser = p.chromium.launch(**launch_options())
            page = browser.new_page()
            page.goto(f"{site}/index.html?token={API_TOKEN}")
            page.evaluate("t => t.length", API_TOKEN)
            with pytest.raises(Exception, match=API_TOKEN):
                page.click(f"#{API_TOKEN}", timeout=300)
            browser.close()

    run()
    for span in spans:
        assert API_TOKEN not in dumped(span), span.name
        assert API_TOKEN not in (span.status.description or ""), span.name
        for event in span.events:
            assert API_TOKEN not in json.dumps(dict(event.attributes), default=str), span.name
    assert any("<secret>api_token</secret>" in dumped(s) for s in spans)


def test_baseline_failed_text_entry_is_masked(pipeline, site):
    """A fill that fails never reached a field we could inspect: fail closed."""
    _, spans = pipeline()

    @workflow(name="missing_field")
    def run() -> None:
        with sync_playwright() as p:
            browser = p.chromium.launch(**launch_options())
            page = browser.new_page()
            page.goto(f"{site}/index.html")
            with pytest.raises(PlaywrightTimeoutError):
                page.fill("#no-such-field", OTHER_PASSWORD, timeout=300)
            browser.close()

    run()
    (fill,) = [s for s in spans if s.name == "Page.fill"]
    assert fill.status.status_code == StatusCode.ERROR
    assert fill.status.description.startswith("TimeoutError: ")
    for span in spans:
        assert OTHER_PASSWORD not in dumped(span), span.name


def test_baseline_content_off_keeps_identifiers_and_drops_content(pipeline, site):
    client, spans = pipeline(instrumentor=PlaywrightInstrumentor(capture_content=False))

    @workflow(name="private")
    def run() -> None:
        with sync_playwright() as p:
            browser = p.chromium.launch(**launch_options())
            page = browser.new_page()
            run_sync_agent(page, site)
            page.screenshot()
            with pytest.raises(PlaywrightTimeoutError):
                page.click("#secret-selector-xyz", timeout=300)
            browser.close()

    with client.propagate_attributes(customer_identifier="customer-1", metadata=CUSTOMER_METADATA):
        run()

    calls = playwright_spans(spans)
    assert {"Page.goto", "Page.screenshot", "Page.click"} <= {s.name for s in calls}
    for span in calls:
        assert span.attributes.get(CUSTOMER_ID) == "customer-1", span.name
        assert span.attributes.get(f"{RESPAN_METADATA}.agent_id") == "a1", span.name
        for key in (
            SpanAttributes.TRACELOOP_ENTITY_INPUT,
            SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
            ATTR_PAGE_URL,
            ATTR_PAGE_TITLE,
        ):
            assert key not in span.attributes, (span.name, key)
        text = dumped(span) + (span.status.description or "")
        for private in (site, USERNAME, PASSWORD, "secret-selector-xyz", "image_url"):
            assert private not in text, (span.name, private)
    failed = [s for s in calls if s.status.status_code == StatusCode.ERROR]
    assert [s.status.description for s in failed] == ["TimeoutError"]


def test_baseline_spans_under_an_instrumented_agent_span(pipeline, site):
    """A customer's own agent framework span (built with the shared kit) is a parent too."""
    _, spans = pipeline()
    with agent_span(
        trace.get_tracer("my-agent"),
        "my_agent",
        "agent",
        scope=AgentScope("test"),
        content_allowed=lambda: True,
    ):
        sync_session(site)
    assert_nested_under_one_root(spans)
