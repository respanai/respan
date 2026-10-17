"""Baseline wire contract: what a customer configures must reach every span.

Kernel browsers are cloud VMs, so these run against a local fake Kernel API
(``fake_kernel.py``) whose browsers are a real headless Chrome: ``browsers.create``
hands out that Chrome's CDP URL, ``computer.*`` drives it over CDP, ``process.*``
runs commands, ``playwright.execute`` evaluates in the page. The ``kernel``
client itself is the real one, sync and async. Spans are captured *through the
Respan pipeline* (`Respan(...)` -> Respan's span processor -> what its exporter
would send). A scripted model (fixed tool calls) stands in for the computer-use
agent a customer runs on Kernel.
"""

from __future__ import annotations

import asyncio
import base64
import json
from typing import Any

import kernel
import pytest
from kernel import AsyncKernel, Kernel
from opentelemetry import trace
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.trace import StatusCode
from playwright.async_api import async_playwright
from playwright.sync_api import sync_playwright
from respan_instrumentation_kernel import KernelInstrumentor, secrets
from respan_instrumentation_kernel._constants import (
    ATTR_EXIT_CODE,
    ATTR_INVOCATION_ID,
    ATTR_LIVE_VIEW_URL,
    ATTR_REPLAY_ID,
    ATTR_REPLAY_VIEW_URL,
    ATTR_SESSION_ID,
)
from respan_sdk.constants.span_attributes import (
    RESPAN_LOG_TYPE,
    RESPAN_METADATA,
    RESPAN_SESSION_ID,
    RESPAN_THREADS_ID,
    RESPAN_TRACE_GROUP_ID,
)
from respan_tracing.core.tracer import RespanTracer
from respan_tracing.utils.agent_instrumentation import AgentScope, agent_span

from respan import agent

from .conftest import PASSWORD_FIELD, USER_FIELD, needs_chrome
from .fake_kernel import JWT

pytestmark = needs_chrome

CUSTOMER_ID = "respan.customer_params.customer_identifier"
CUSTOMER_METADATA = {"agent_id": "a1", "role": "browser"}
KERNEL_API_KEY = "sk_kernel_live_7f3a9c2e5b1d"
USERNAME = "ada.lovelace"
PASSWORD = "hunter2-correct-horse"
ENV_TOKEN = "env-token-6d1f0a9b"
INPUT = SpanAttributes.TRACELOOP_ENTITY_INPUT
OUTPUT = SpanAttributes.TRACELOOP_ENTITY_OUTPUT
GROUPING = (RESPAN_THREADS_ID, RESPAN_SESSION_ID, RESPAN_TRACE_GROUP_ID)

# The scripted computer-use model: each turn is one tool call on the Computer API.
SCRIPT = [
    ("screenshot", {}),
    ("click", {"x": USER_FIELD[0], "y": USER_FIELD[1]}),
    ("type", {"text": USERNAME}),
    ("press", {"keys": ["Tab"]}),
    ("screenshot", {}),
]


def client(fake: Any) -> Kernel:
    return Kernel(api_key=KERNEL_API_KEY, base_url=fake.base_url, max_retries=0)


def async_client(fake: Any) -> AsyncKernel:
    return AsyncKernel(api_key=KERNEL_API_KEY, base_url=fake.base_url, max_retries=0)


def run_sync_loop(kernel_client: Kernel, session_id: str) -> list[bytes]:
    shots = []
    for tool, args in SCRIPT:
        if tool == "screenshot":
            shots.append(kernel_client.browsers.computer.capture_screenshot(session_id).read())
        elif tool == "click":
            kernel_client.browsers.computer.click_mouse(session_id, **args)
        elif tool == "type":
            kernel_client.browsers.computer.type_text(session_id, **args)
        elif tool == "press":
            kernel_client.browsers.computer.press_key(session_id, **args)
    return shots


async def run_async_loop(kernel_client: AsyncKernel, session_id: str) -> list[bytes]:
    shots = []
    for tool, args in SCRIPT:
        if tool == "screenshot":
            response = await kernel_client.browsers.computer.capture_screenshot(session_id)
            shots.append(await response.read())
        elif tool == "click":
            await kernel_client.browsers.computer.click_mouse(session_id, **args)
        elif tool == "type":
            await kernel_client.browsers.computer.type_text(session_id, **args)
        elif tool == "press":
            await kernel_client.browsers.computer.press_key(session_id, **args)
    return shots


def open_page(cdp_ws_url: str, url: str):
    """Connect Playwright over CDP (as Kernel's templates do) and load the page."""
    p = sync_playwright().start()
    browser = p.chromium.connect_over_cdp(cdp_ws_url)
    context = browser.contexts[0]
    page = context.pages[0] if context.pages else context.new_page()
    page.goto(url)
    page.bring_to_front()
    return p, browser, page


def kernel_spans(spans: list[Any]) -> list[Any]:
    return [s for s in spans if s.instrumentation_scope.name == "respan.instrumentation.kernel"]


def by_name(spans: list[Any], name: str) -> list[Any]:
    return [s for s in spans if s.name == name]


def roots(spans: list[Any]) -> list[Any]:
    return [s for s in spans if s.parent is None]


def dumped(span: Any) -> str:
    events = [dict(e.attributes) for e in span.events]
    return json.dumps([dict(span.attributes), events, span.status.description], default=str)


def assert_exportable(spans: list[Any]) -> None:
    for span in kernel_spans(spans):
        # Respan drops spans without a span kind (G2).
        assert span.attributes.get(SpanAttributes.TRACELOOP_SPAN_KIND), span.name
        assert span.attributes.get(RESPAN_LOG_TYPE) in ("agent", "tool"), span.name


def assert_grouped(spans: list[Any], session_id: str) -> None:
    for span in spans:
        for key in GROUPING:
            assert span.attributes.get(key) == session_id, (span.name, key)


# -- KernelApp actions (Kernel's sample-app) ------------------------------------


def make_sample_app(kernel_client: Kernel, name: str = "python-basic") -> Any:
    """Kernel's `sample-app` template: get-page-title, Playwright over CDP."""
    app = kernel.App(name)

    @app.action("get-page-title")
    async def get_page_title(ctx: kernel.KernelContext, input_data: dict) -> dict:
        browser_info = kernel_client.browsers.create(invocation_id=ctx.invocation_id)
        async with async_playwright() as playwright:
            browser = await playwright.chromium.connect_over_cdp(browser_info.cdp_ws_url)
            context = await browser.new_context()
            page = await context.new_page()
            try:
                await page.goto(input_data["url"])
                return {"title": await page.title()}
            finally:
                await browser.close()
                kernel_client.browsers.delete_by_id(browser_info.session_id)

    return app


async def invoke(app: Any, action: str, invocation_id: str, payload: Any) -> Any:
    """What the Kernel runtime does when `kernel invoke` calls an action."""
    handler = app.get_action(action).handler
    return await handler(kernel.KernelContext(invocation_id=invocation_id), payload)


async def test_baseline_action_is_one_trace_grouped_by_its_browser(pipeline, fake_kernel, site):
    _, spans = pipeline()
    app = make_sample_app(client(fake_kernel))
    result = await invoke(app, "get-page-title", "inv-1", {"url": site})
    assert result == {"title": "Sign in"}

    (root,) = roots(spans)
    assert root.name == "kernel.action.get-page-title"
    assert root.attributes[RESPAN_LOG_TYPE] == "agent"
    assert root.attributes[ATTR_INVOCATION_ID] == "inv-1"
    assert json.loads(root.attributes[INPUT]) == {"url": site}
    assert json.loads(root.attributes[OUTPUT]) == {"title": "Sign in"}
    (create,) = by_name(spans, "kernel.browsers.create")
    (delete,) = by_name(spans, "kernel.browsers.delete_by_id")
    session_id = create.attributes[ATTR_SESSION_ID]
    assert delete.attributes[ATTR_SESSION_ID] == session_id
    for span in (create, delete):
        assert span.parent.span_id == root.context.span_id
        assert span.context.trace_id == root.context.trace_id
        assert span.attributes[RESPAN_LOG_TYPE] == "tool"
    assert_exportable(spans)
    assert_grouped(kernel_spans(spans), session_id)


async def test_baseline_other_instrumentations_in_an_action_join_its_thread(pipeline, fake_kernel):
    """browser-use / Playwright spans inside an action default to their own session
    id; the action's Kernel session must win so the trace keeps one thread."""
    _, spans = pipeline()
    kernel_client = client(fake_kernel)
    other = AgentScope("browser_agent")  # e.g. browser-use's scope, with its own session id
    app = kernel.App("python-mixed")

    @app.action("mixed")
    async def mixed(ctx: kernel.KernelContext) -> None:
        kernel_client.browsers.create(invocation_id=ctx.invocation_id)
        with other.run("browser-use-session"):
            with agent_span(
                trace.get_tracer("browser-agent"),
                "browser_use.agent",
                "agent",
                scope=other,
                content_allowed=lambda: True,
            ):
                pass

    await app.get_action("mixed").handler(kernel.KernelContext(invocation_id="inv-6"))
    (create,) = by_name(spans, "kernel.browsers.create")
    assert len(spans) == 3
    assert_grouped(spans, create.attributes[ATTR_SESSION_ID])


async def test_baseline_action_flushes_when_the_handler_returns(pipeline, fake_kernel, site):
    """A deployed Kernel action's process may stop as soon as it returns."""
    _, spans = pipeline()
    flushed_with: list[list[str]] = []
    tracer = RespanTracer._instance
    original_flush = tracer.flush
    tracer.flush = lambda: (flushed_with.append([s.name for s in spans]), original_flush())[1]
    app = make_sample_app(client(fake_kernel))

    await invoke(app, "get-page-title", "inv-flush", {"url": site})

    assert flushed_with and "kernel.action.get-page-title" in flushed_with[-1]


def test_baseline_sync_action_without_a_browser_is_grouped_by_invocation(pipeline, fake_kernel):
    _, spans = pipeline()
    app = kernel.App("python-sync")

    @app.action("ping")
    def ping(ctx: kernel.KernelContext) -> str:
        return "pong"

    assert app.get_action("ping").handler(kernel.KernelContext(invocation_id="inv-2")) == "pong"
    (root,) = spans
    assert root.name == "kernel.action.ping"
    assert_grouped(spans, "inv-2")


# -- the Computer API (a computer-use loop), sync and async ---------------------


def assert_computer_loop(spans: list[Any], session_id: str, shots: list[bytes]) -> None:
    names = [s.name for s in kernel_spans(spans)]
    assert names.count("kernel.computer.capture_screenshot") == 2, names
    assert {
        "kernel.computer.click_mouse",
        "kernel.computer.type_text",
        "kernel.computer.press_key",
    } <= set(names)
    # The caller still reads the full PNG: recording the screenshot consumed nothing.
    assert all(shot[:8] == b"\x89PNG\r\n\x1a\n" and len(shot) > 1000 for shot in shots)
    for span in by_name(spans, "kernel.computer.capture_screenshot"):
        url = json.loads(span.attributes[OUTPUT])["screenshot"]["image_url"]["url"]
        assert url.startswith("data:image/jpeg;base64,"), url[:40]
        assert base64.b64decode(url.split(",", 1)[1])[:3] == b"\xff\xd8\xff"
    (click,) = by_name(spans, "kernel.computer.click_mouse")
    assert json.loads(click.attributes[INPUT]) == {"x": USER_FIELD[0], "y": USER_FIELD[1]}
    (typed,) = by_name(spans, "kernel.computer.type_text")
    assert json.loads(typed.attributes[INPUT]) == {"text": USERNAME}  # ordinary text is kept
    assert_exportable(spans)
    assert_grouped(kernel_spans(spans), session_id)


def test_baseline_sync_computer_loop(pipeline, fake_kernel, site):
    _, spans = pipeline()
    kernel_client = client(fake_kernel)
    browser = kernel_client.browsers.create()
    p, cdp, page = open_page(browser.cdp_ws_url, site)
    try:
        shots = run_sync_loop(kernel_client, browser.session_id)
        assert page.input_value("#user") == USERNAME  # the clicks and typing were real
    finally:
        cdp.close()
        p.stop()
    assert_computer_loop(spans, browser.session_id, shots)


async def test_baseline_async_computer_loop(pipeline, fake_kernel, site):
    _, spans = pipeline()
    kernel_client = async_client(fake_kernel)
    browser = await kernel_client.browsers.create()
    async with async_playwright() as p:
        cdp = await p.chromium.connect_over_cdp(browser.cdp_ws_url)
        page = cdp.contexts[0].pages[0]
        await page.goto(site)
        await page.bring_to_front()
        shots = await run_async_loop(kernel_client, browser.session_id)
        assert await page.input_value("#user") == USERNAME
        await cdp.close()
    assert_computer_loop(spans, browser.session_id, shots)


# -- customer context -----------------------------------------------------------


async def test_baseline_customer_metadata_and_identifier_on_every_span(pipeline, fake_kernel, site):
    client_, spans = pipeline(metadata={"source": "constructor"}, customer_identifier="ctor")
    app = make_sample_app(client(fake_kernel))
    with client_.propagate_attributes(customer_identifier="customer-1", metadata=CUSTOMER_METADATA):
        await invoke(app, "get-page-title", "inv-3", {"url": site})

    assert len(kernel_spans(spans)) == 3
    expected = {"source": "constructor", **CUSTOMER_METADATA}
    for span in spans:
        assert span.attributes.get(CUSTOMER_ID) == "customer-1", span.name
        for key, value in expected.items():
            assert span.attributes.get(f"{RESPAN_METADATA}.{key}") == value, (span.name, key)
        assert RESPAN_METADATA not in span.attributes, span.name


async def test_baseline_customer_thread_and_trace_group_win(pipeline, fake_kernel, site):
    client_, spans = pipeline()
    app = make_sample_app(client(fake_kernel))
    with client_.propagate_attributes(thread_identifier="my-thread", group_identifier="my-group"):
        await invoke(app, "get-page-title", "inv-4", {"url": site})

    assert len(spans) == 3
    for span in spans:
        assert span.attributes.get(RESPAN_THREADS_ID) == "my-thread", span.name
        assert span.attributes.get(RESPAN_TRACE_GROUP_ID) == "my-group", span.name


async def test_baseline_parallel_actions_do_not_share_traces_or_identifiers(
    pipeline, fake_kernel, site
):
    client_, spans = pipeline()
    app = make_sample_app(client(fake_kernel))

    async def one(name: str) -> None:
        with client_.propagate_attributes(customer_identifier=f"customer-{name}"):
            await invoke(app, "get-page-title", f"inv-{name}", {"url": site})

    await asyncio.gather(one("a"), one("b"))

    trace_of = {s.attributes[CUSTOMER_ID]: s.context.trace_id for s in roots(spans)}
    assert set(trace_of) == {"customer-a", "customer-b"}
    assert len(set(trace_of.values())) == 2
    sessions = set()
    for name in ("a", "b"):
        mine = [s for s in spans if s.attributes[CUSTOMER_ID] == f"customer-{name}"]
        assert len(mine) == 3
        assert {s.context.trace_id for s in mine} == {trace_of[f"customer-{name}"]}
        (create,) = by_name(mine, "kernel.browsers.create")
        sessions.add(create.attributes[ATTR_SESSION_ID])
        assert_grouped(mine, create.attributes[ATTR_SESSION_ID])
    assert len(sessions) == 2


# -- process, Playwright, replays ----------------------------------------------


def test_baseline_process_output_is_decoded_and_env_values_masked(pipeline, fake_kernel):
    _, spans = pipeline()
    kernel_client = client(fake_kernel)
    browser = kernel_client.browsers.create()
    result = kernel_client.browsers.process.exec(
        browser.session_id,
        command="sh",
        args=["-c", "echo hello from the vm; echo token=$API_TOKEN; echo oops >&2; exit 3"],
        env={"API_TOKEN": ENV_TOKEN, "DEBUG": "1"},
    )
    assert base64.b64decode(result.stdout_b64).decode().endswith(f"token={ENV_TOKEN}\n")

    (span,) = by_name(spans, "kernel.process.exec")
    output = json.loads(span.attributes[OUTPUT])
    assert output["stdout"] == "hello from the vm\ntoken=<secret>API_TOKEN</secret>\n"
    assert output["stderr"] == "oops\n"
    assert span.attributes[ATTR_EXIT_CODE] == 3
    assert json.loads(span.attributes[INPUT])["env"] == {
        "API_TOKEN": "<masked>",
        "DEBUG": "<masked>",
    }
    for span in spans:
        assert ENV_TOKEN not in dumped(span), span.name


def test_baseline_playwright_execute_and_replays(pipeline, fake_kernel, site):
    _, spans = pipeline()
    kernel_client = client(fake_kernel)
    browser = kernel_client.browsers.create()
    p, cdp, _ = open_page(browser.cdp_ws_url, site)
    try:
        replay = kernel_client.browsers.replays.start(browser.session_id)
        title = kernel_client.browsers.playwright.execute(
            browser.session_id, code="return await page.title();"
        )
        failed = kernel_client.browsers.playwright.execute(
            browser.session_id, code="throw new Error('boom')"
        )
        kernel_client.browsers.replays.stop(replay.replay_id, id_or_name=browser.session_id)
        video = kernel_client.browsers.replays.download(
            replay.replay_id, id_or_name=browser.session_id
        )
    finally:
        cdp.close()
        p.stop()
    assert title.result == "Sign in" and failed.success is False
    assert video.read().startswith(b"\x00\x00\x00\x18ftyp")

    ok, error = by_name(spans, "kernel.playwright.execute")
    assert json.loads(ok.attributes[INPUT]) == {"code": "return await page.title();"}
    assert json.loads(ok.attributes[OUTPUT])["result"] == "Sign in"
    assert ok.status.status_code != StatusCode.ERROR
    assert error.status.status_code == StatusCode.ERROR
    assert error.status.description.startswith("PlaywrightExecutionError: Error: boom")
    (start,) = by_name(spans, "kernel.replays.start")
    assert start.attributes[ATTR_REPLAY_ID] == replay.replay_id
    assert start.attributes[ATTR_REPLAY_VIEW_URL].endswith("?jwt=<masked>")
    (download,) = by_name(spans, "kernel.replays.download")
    assert json.loads(download.attributes[OUTPUT]) == {"type": "bytes", "length": len(video.read())}


async def test_baseline_replay_link_reaches_the_action(pipeline, fake_kernel):
    _, spans = pipeline()
    kernel_client = client(fake_kernel)
    app = kernel.App("python-replay")

    @app.action("record")
    async def record(ctx: kernel.KernelContext) -> str:
        browser = kernel_client.browsers.create(invocation_id=ctx.invocation_id)
        replay = kernel_client.browsers.replays.start(browser.session_id)
        kernel_client.browsers.replays.stop(replay.replay_id, id_or_name=browser.session_id)
        return replay.replay_id

    replay_id = await app.get_action("record").handler(kernel.KernelContext(invocation_id="inv-5"))
    (root,) = roots(spans)
    assert root.attributes[ATTR_REPLAY_ID] == replay_id
    assert "replays.example.test" in root.attributes[ATTR_REPLAY_VIEW_URL]


# -- errors, secrets, content ---------------------------------------------------


def test_baseline_errors_are_type_and_message(pipeline, fake_kernel):
    _, spans = pipeline()
    with pytest.raises(kernel.NotFoundError):
        client(fake_kernel).browsers.retrieve("no-such-session")
    (span,) = spans
    assert span.status.status_code == StatusCode.ERROR
    assert span.status.description.startswith("NotFoundError: Error code: 404")
    assert "browser session no-such-session not found" in span.status.description


def test_baseline_api_key_and_browser_tokens_never_reach_a_span(pipeline, fake_kernel, site):
    """The API key, and the jwt in cdp_ws_url / live view / replay URLs, grant control."""
    _, spans = pipeline()
    kernel_client = client(fake_kernel)
    browser = kernel_client.browsers.create()
    assert JWT in browser.cdp_ws_url
    kernel_client.browsers.retrieve(browser.session_id)
    list(kernel_client.browsers.list())
    kernel_client.browsers.replays.start(browser.session_id)
    kernel_client.browsers.playwright.execute(
        browser.session_id, code=f"return '{KERNEL_API_KEY}'.length"
    )
    with pytest.raises(kernel.NotFoundError):
        kernel_client.browsers.retrieve(f"key-{KERNEL_API_KEY}")

    assert fake_kernel.api_keys == {KERNEL_API_KEY}
    (create,) = by_name(spans, "kernel.browsers.create")
    assert json.loads(create.attributes[OUTPUT])["cdp_ws_url"].endswith("?jwt=<masked>")
    assert create.attributes[ATTR_LIVE_VIEW_URL].endswith("?jwt=<masked>")
    for span in spans:
        assert JWT not in dumped(span), span.name
        assert KERNEL_API_KEY not in dumped(span), span.name
    assert any("<secret>kernel_api_key</secret>" in dumped(s) for s in spans)


def test_baseline_declared_secrets_never_reach_a_span(pipeline, fake_kernel, site):
    _, spans = pipeline()
    kernel_client = client(fake_kernel)
    browser = kernel_client.browsers.create()
    p, cdp, page = open_page(browser.cdp_ws_url, site)
    try:
        with secrets({"password": PASSWORD}):
            kernel_client.browsers.computer.click_mouse(
                browser.session_id, x=PASSWORD_FIELD[0], y=PASSWORD_FIELD[1]
            )
            kernel_client.browsers.computer.type_text(browser.session_id, text=PASSWORD)
            kernel_client.browsers.process.exec(browser.session_id, command="echo", args=[PASSWORD])
            kernel_client.browsers.playwright.execute(
                browser.session_id, code=f"throw new Error('bad password {PASSWORD}')"
            )
        assert page.input_value("#password") == PASSWORD
    finally:
        cdp.close()
        p.stop()

    for span in spans:
        assert PASSWORD not in dumped(span), span.name
    (typed,) = by_name(spans, "kernel.computer.type_text")
    assert json.loads(typed.attributes[INPUT]) == {"text": "<secret>password</secret>"}


def test_baseline_password_field_typing_is_masked_with_probing(pipeline, fake_kernel, site):
    """probe_password_fields: text typed into a password field is masked, other text kept."""
    _, spans = pipeline(instrumentor=KernelInstrumentor(probe_password_fields=True))
    kernel_client = client(fake_kernel)
    browser = kernel_client.browsers.create()
    p, cdp, page = open_page(browser.cdp_ws_url, site)
    try:
        computer = kernel_client.browsers.computer
        computer.click_mouse(browser.session_id, x=USER_FIELD[0], y=USER_FIELD[1])
        computer.type_text(browser.session_id, text=USERNAME)
        computer.click_mouse(browser.session_id, x=PASSWORD_FIELD[0], y=PASSWORD_FIELD[1])
        computer.type_text(browser.session_id, text=PASSWORD)
        computer.batch(
            browser.session_id, actions=[{"type": "type_text", "type_text": {"text": "-batch"}}]
        )
        assert page.input_value("#password") == PASSWORD + "-batch"
    finally:
        cdp.close()
        p.stop()

    user, password = by_name(spans, "kernel.computer.type_text")
    assert json.loads(user.attributes[INPUT]) == {"text": USERNAME}
    assert json.loads(password.attributes[INPUT]) == {"text": "<masked>"}
    (batch,) = by_name(spans, "kernel.computer.batch")
    assert json.loads(batch.attributes[INPUT])["actions"][0]["type_text"]["text"] == "<masked>"
    # The probe is not one of the customer's calls.
    assert not by_name(spans, "kernel.playwright.execute")
    for span in spans:
        assert PASSWORD not in dumped(span), span.name


def test_baseline_failed_password_probe_masks(pipeline, fake_kernel, site):
    """Fail closed: when the probe cannot answer, typed text is masked."""
    _, spans = pipeline(instrumentor=KernelInstrumentor(probe_password_fields=True))
    kernel_client = client(fake_kernel)
    browser = kernel_client.browsers.create()
    fake_kernel.execute_fails = True
    kernel_client.browsers.computer.type_text(browser.session_id, text=PASSWORD)
    (typed,) = by_name(spans, "kernel.computer.type_text")
    assert json.loads(typed.attributes[INPUT]) == {"text": "<masked>"}


def test_baseline_content_off_keeps_structure_and_drops_content(pipeline, fake_kernel, site):
    client_, spans = pipeline(instrumentor=KernelInstrumentor(capture_content=False))
    kernel_client = client(fake_kernel)

    def run() -> None:
        browser = kernel_client.browsers.create()
        p, cdp, _ = open_page(browser.cdp_ws_url, site)
        try:
            run_sync_loop(kernel_client, browser.session_id)
            kernel_client.browsers.process.exec(browser.session_id, command="echo", args=[PASSWORD])
            kernel_client.browsers.replays.start(browser.session_id)
            with pytest.raises(kernel.NotFoundError):
                kernel_client.browsers.retrieve("missing-session")  # ids are kept, not content
        finally:
            cdp.close()
            p.stop()

    with client_.propagate_attributes(customer_identifier="customer-1", metadata=CUSTOMER_METADATA):
        agent(name="computer_use")(run)()

    calls = kernel_spans(spans)
    names = {s.name for s in calls}
    assert {
        "kernel.browsers.create",
        "kernel.computer.capture_screenshot",
        "kernel.process.exec",
    } <= names
    for span in calls:
        assert span.attributes.get(CUSTOMER_ID) == "customer-1", span.name
        assert span.attributes.get(f"{RESPAN_METADATA}.agent_id") == "a1", span.name
        for key in (INPUT, OUTPUT, ATTR_LIVE_VIEW_URL, ATTR_REPLAY_VIEW_URL):
            assert key not in span.attributes, (span.name, key)
        text = dumped(span)
        for private in (site, USERNAME, PASSWORD, "image_url", "not found", "jwt"):
            assert private not in text, (span.name, private)
    (create,) = by_name(spans, "kernel.browsers.create")
    assert create.attributes[ATTR_SESSION_ID]
    (exec_span,) = by_name(spans, "kernel.process.exec")
    assert exec_span.attributes[ATTR_EXIT_CODE] == 0
    (start,) = by_name(spans, "kernel.replays.start")
    assert start.attributes[ATTR_REPLAY_ID]
    failed = [s for s in calls if s.status.status_code == StatusCode.ERROR]
    assert [s.status.description for s in failed] == ["NotFoundError"]
