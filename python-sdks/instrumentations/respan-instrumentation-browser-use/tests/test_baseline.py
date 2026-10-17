"""Baseline wire contract: what a customer configures must reach every span.

These drive a real headless Chrome through browser-use with a scripted model,
and capture spans *through the Respan pipeline* (`Respan(...)` -> Respan's span
processor -> the spans its exporter would send). Ingestion reads customer
metadata only as flat `respan.metadata.<key>` attributes, and groups runs by
thread / session / trace group, so those must be on every span.
"""

from __future__ import annotations

import asyncio
import functools
import json
import re
import threading
from contextlib import ExitStack
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest
from browser_use import Agent, BrowserProfile, BrowserSession
from opentelemetry.semconv_ai import SpanAttributes
from respan_instrumentation_browser_use import BrowserUseInstrumentor
from respan_instrumentation_browser_use._constants import ATTR_PAGE_URL
from respan_sdk.constants.span_attributes import (
    RESPAN_LOG_TYPE,
    RESPAN_METADATA,
    RESPAN_SESSION_ID,
    RESPAN_THREADS_ID,
    RESPAN_TRACE_GROUP_ID,
)
from respan_tracing.testing import UNREACHABLE_BASE_URL, capture_exported_spans
from respan_tracing.testing.recordings import recording_stub_server
from respan_tracing.utils import session_recording
from respan_tracing.utils.session_recording import (
    SESSION_REPLAY_ID_METADATA_KEY,
    SESSION_REPLAY_URL_METADATA_KEY,
)

from respan import Respan

from .conftest import ScriptedLLM, chrome_path

pytestmark = pytest.mark.skipif(
    chrome_path() is None, reason="needs a local Chrome/Chromium"
)

CUSTOMER_ID = "respan.customer_params.customer_identifier"
CUSTOMER_METADATA = {"agent_id": "a1", "role": "planner"}


@pytest.fixture
def pipeline():
    """Build `Respan(...)` with the browser-use instrumentor; return the exported spans."""
    with ExitStack() as stack:

        def make(instrumentor: BrowserUseInstrumentor | None = None, **kwargs: Any):
            return stack.enter_context(
                capture_exported_spans(
                    lambda on_export: Respan(
                        api_key="test-key",
                        base_url=kwargs.pop("base_url", UNREACHABLE_BASE_URL),
                        instrumentations=[instrumentor or BrowserUseInstrumentor()],
                        span_postprocess_callback=on_export,
                        **kwargs,
                    )
                )
            )

        yield make


def script(*actions: dict[str, Any], answer: str = "done") -> list[dict[str, Any]]:
    """One model turn per action, then `done`, then the judge's verdict."""
    turns = [
        {
            "evaluation_previous_goal": "ok",
            "memory": "",
            "next_goal": "next",
            "action": [action],
        }
        for action in actions
    ]
    turns.append(
        {
            "evaluation_previous_goal": "ok",
            "memory": "",
            "next_goal": "finish",
            "action": [{"done": {"text": answer, "success": True}}],
        }
    )
    turns.append({"reasoning": "fine", "verdict": True, "failure_reason": ""})
    return turns


def browser(tmp_path, name: str = "default", **profile: Any) -> BrowserSession:
    return BrowserSession(
        browser_profile=BrowserProfile(
            headless=True,
            executable_path=chrome_path(),
            user_data_dir=str(tmp_path / f"profile-{name}"),
            **profile,
        )
    )


async def run(agent: Agent, session: BrowserSession) -> None:
    try:
        await agent.run(max_steps=4)
    finally:
        await session.kill()


def agent_spans(spans: list[Any]) -> list[Any]:
    return [s for s in spans if s.attributes.get(RESPAN_LOG_TYPE) == "agent"]


def assert_one_full_run(spans: list[Any]) -> None:
    kinds = [s.attributes.get(RESPAN_LOG_TYPE) for s in spans]
    assert kinds.count("agent") == 1
    assert {"task", "tool", "chat"} <= set(kinds), kinds


async def test_baseline_customer_metadata_is_flat_on_every_span(pipeline, tmp_path, test_page):
    client, spans = pipeline(metadata={"source": "constructor"})
    session = browser(tmp_path)
    agent = Agent(
        task="Read",
        llm=ScriptedLLM(script({"navigate": {"url": test_page}})),
        browser_session=session,
    )
    with client.propagate_attributes(metadata=CUSTOMER_METADATA):
        await run(agent, session)

    assert_one_full_run(spans)
    expected = {"source": "constructor", **CUSTOMER_METADATA}
    for span in spans:
        for key, value in expected.items():
            assert span.attributes.get(f"{RESPAN_METADATA}.{key}") == value, (span.name, key)
        # A JSON blob under `respan.metadata` is dropped by ingestion.
        assert RESPAN_METADATA not in span.attributes, span.name


async def test_baseline_customer_identifier_on_every_span(pipeline, tmp_path, test_page):
    client, spans = pipeline(customer_identifier="from-constructor")
    session = browser(tmp_path)
    agent = Agent(
        task="Read",
        llm=ScriptedLLM(script({"navigate": {"url": test_page}})),
        browser_session=session,
    )
    with client.propagate_attributes(customer_identifier="customer-1"):
        await run(agent, session)

    assert_one_full_run(spans)
    for span in spans:
        # The per-request value wins over the constructor default.
        assert span.attributes.get(CUSTOMER_ID) == "customer-1", span.name


async def test_baseline_thread_session_and_trace_group_default_to_the_agent_session(
    pipeline, tmp_path, test_page
):
    _, spans = pipeline()
    session = browser(tmp_path)
    agent = Agent(
        task="Read",
        llm=ScriptedLLM(script({"navigate": {"url": test_page}})),
        browser_session=session,
    )
    await run(agent, session)

    assert_one_full_run(spans)
    for span in spans:
        for key in (RESPAN_THREADS_ID, RESPAN_SESSION_ID, RESPAN_TRACE_GROUP_ID):
            assert span.attributes.get(key) == str(agent.session_id), (span.name, key)


async def test_baseline_customer_thread_and_trace_group_win(pipeline, tmp_path, test_page):
    client, spans = pipeline()
    session = browser(tmp_path)
    agent = Agent(
        task="Read",
        llm=ScriptedLLM(script({"navigate": {"url": test_page}})),
        browser_session=session,
    )
    with client.propagate_attributes(
        thread_identifier="my-thread", group_identifier="my-group"
    ):
        await run(agent, session)

    assert_one_full_run(spans)
    for span in spans:
        assert span.attributes.get(RESPAN_THREADS_ID) == "my-thread", span.name
        assert span.attributes.get(RESPAN_TRACE_GROUP_ID) == "my-group", span.name
        # Not settable through propagate_attributes, so it keeps the default.
        assert span.attributes.get(RESPAN_SESSION_ID) == str(agent.session_id), span.name


async def test_baseline_follow_up_task_is_a_new_trace_in_the_same_thread(
    pipeline, tmp_path, test_page
):
    _, spans = pipeline()
    session = browser(tmp_path, keep_alive=True)
    llm = ScriptedLLM(
        [*script({"navigate": {"url": test_page}}), *script(answer="second")]
    )
    agent = Agent(task="Read", llm=llm, browser_session=session)
    try:
        await agent.run(max_steps=4)
        agent.add_new_task("Now summarise it")
        await agent.run(max_steps=4)
    finally:
        await session.kill()

    runs = agent_spans(spans)
    assert len(runs) == 2
    assert len({s.context.trace_id for s in runs}) == 2
    # Each run reports its own steps, not the history it shares with the other.
    steps = [json.loads(s.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT])["steps"] for s in runs]
    assert sorted(steps) == [1, 2]
    assert {s.attributes.get(RESPAN_THREADS_ID) for s in spans} == {str(agent.session_id)}
    assert {s.attributes.get(RESPAN_TRACE_GROUP_ID) for s in spans} == {
        str(agent.session_id)
    }


async def test_baseline_parallel_agents_do_not_share_traces_or_identifiers(
    pipeline, tmp_path, test_page
):
    client, spans = pipeline()
    agents = []
    runs = []
    for name in ("a", "b"):
        session = browser(tmp_path, name)
        agent = Agent(
            task=f"Read {name}",
            llm=ScriptedLLM(script({"navigate": {"url": test_page}}, answer=name)),
            browser_session=session,
        )
        agents.append(agent)
        runs.append((name, agent, session))

    async def one(name: str, agent: Agent, session: BrowserSession) -> None:
        with client.propagate_attributes(customer_identifier=f"customer-{name}"):
            await run(agent, session)

    await asyncio.gather(*(one(*r) for r in runs))

    roots = agent_spans(spans)
    assert len(roots) == 2
    trace_of = {s.attributes[RESPAN_THREADS_ID]: s.context.trace_id for s in roots}
    assert set(trace_of) == {str(a.session_id) for a in agents}
    for span in spans:
        thread = span.attributes.get(RESPAN_THREADS_ID)
        assert span.context.trace_id == trace_of[thread], span.name
    for name, agent, _ in runs:
        mine = [s for s in spans if s.attributes.get(RESPAN_THREADS_ID) == str(agent.session_id)]
        assert {s.attributes.get(CUSTOMER_ID) for s in mine} == {f"customer-{name}"}


async def test_baseline_content_off_keeps_identifiers_and_drops_content(
    pipeline, tmp_path, test_page
):
    client, spans = pipeline(
        instrumentor=BrowserUseInstrumentor(capture_content=False)
    )
    session = browser(tmp_path)
    agent = Agent(
        task="A task that must stay private",
        llm=ScriptedLLM(script({"navigate": {"url": test_page}})),
        browser_session=session,
    )
    with client.propagate_attributes(customer_identifier="customer-1", metadata=CUSTOMER_METADATA):
        await run(agent, session)

    assert_one_full_run(spans)
    for span in spans:
        assert span.attributes.get(CUSTOMER_ID) == "customer-1", span.name
        assert span.attributes.get(f"{RESPAN_METADATA}.agent_id") == "a1", span.name
        assert span.attributes.get(RESPAN_THREADS_ID) == str(agent.session_id), span.name
        assert ATTR_PAGE_URL not in span.attributes, span.name
        assert SpanAttributes.TRACELOOP_ENTITY_INPUT not in span.attributes, span.name
        dumped = json.dumps(dict(span.attributes), default=str)
        assert "must stay private" not in dumped, span.name
        assert test_page not in dumped, span.name


@pytest.mark.parametrize(
    "allowed_domain",
    [
        "127.0.0.1",  # navigation succeeds: the secret is in the step's page URL
        "example.com",  # navigation is blocked: the secret is in the action's error
    ],
)
async def test_baseline_sensitive_data_never_reaches_a_span(
    pipeline, tmp_path, test_page, allowed_domain
):
    secret = "s3cr3t-value-7781"
    _, spans = pipeline()
    session = browser(tmp_path, allowed_domains=[allowed_domain])
    agent = Agent(
        task="Open the page with the token",
        llm=ScriptedLLM(
            script({"navigate": {"url": f"{test_page}?token=<secret>api_token</secret>"}})
        ),
        browser_session=session,
        sensitive_data={"api_token": secret},
    )
    await run(agent, session)

    assert_one_full_run(spans)
    dumps = [json.dumps(dict(span.attributes), default=str) for span in spans]
    for span, dumped in zip(spans, dumps, strict=True):
        assert secret not in dumped, span.name
    # The real value was replaced, not just absent: the URL still shows where it went.
    assert any("<secret>api_token</secret>" in d for d in dumps)


async def test_baseline_rerun_history_is_one_trace_in_the_agent_thread(
    pipeline, tmp_path, test_page
):
    _, spans = pipeline()
    session = browser(tmp_path, keep_alive=True)
    agent = Agent(
        task="Read",
        llm=ScriptedLLM([*script({"navigate": {"url": test_page}}), *script()]),
        browser_session=session,
    )
    try:
        history = await agent.run(max_steps=4)
        first_run = {s.context.span_id for s in spans}
        await agent.rerun_history(history, delay_between_actions=0)
    finally:
        await session.kill()

    replay = [s for s in spans if s.context.span_id not in first_run]
    roots = [s for s in replay if s.parent is None]
    assert [s.name for s in roots] == ["browser_use.rerun"]
    assert {s.context.trace_id for s in replay} == {roots[0].context.trace_id}
    assert any(s.name == "navigate" for s in replay)
    assert {s.attributes.get(RESPAN_THREADS_ID) for s in replay} == {str(agent.session_id)}


# -- session replay ------------------------------------------------------------

REPLAY_SECRET = "s3cr3t-replay-4471"
TYPED_VALUE = "typed-by-the-agent-9932"
REPLAY_URL_KEY = f"{RESPAN_METADATA}.{SESSION_REPLAY_URL_METADATA_KEY}"
REPLAY_ID_KEY = f"{RESPAN_METADATA}.{SESSION_REPLAY_ID_METADATA_KEY}"


@pytest.fixture
def form_page(tmp_path):
    (tmp_path / "form.html").write_text(
        "<title>Replay form</title><h1>Sign in</h1>"
        f"<p>Your token is {REPLAY_SECRET}</p>"
        '<input id="name" type="text" placeholder="name">'
        '<a href="tab.html" target="_blank">Open the details tab</a>'
    )
    (tmp_path / "tab.html").write_text("<title>Details</title><h1>Details tab</h1>")
    handler = functools.partial(SimpleHTTPRequestHandler, directory=str(tmp_path))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}/form.html"
    server.shutdown()


def type_into_the_input(seen: str) -> dict[str, Any]:
    # The last match: the browser state comes after the system prompt's examples.
    index = int(re.findall(r"\[(\d+)\]<input", seen)[-1])
    return {
        "evaluation_previous_goal": "ok",
        "memory": "",
        "next_goal": "type",
        "action": [{"input": {"index": index, "text": TYPED_VALUE}}],
    }


def click_the_link(seen: str) -> dict[str, Any]:
    index = int(re.findall(r"\[(\d+)\]<a", seen)[-1])
    return {
        "evaluation_previous_goal": "ok",
        "memory": "",
        "next_goal": "open the tab",
        "action": [{"click": {"index": index}}],
    }


def replay_script(url: str) -> list[Any]:
    navigate, done, verdict = script({"navigate": {"url": url}})
    return [navigate, type_into_the_input, done, verdict]


async def run_recorded(
    pipeline, tmp_path, url: str, stub, turns: list[Any] | None = None, **instrumentor: Any
) -> tuple[Agent, list[Any]]:
    _, spans = pipeline(
        instrumentor=BrowserUseInstrumentor(record_session=True, **instrumentor),
        base_url=stub.base_url,
    )
    session = browser(tmp_path)
    agent = Agent(
        task="Fill in the form",
        llm=ScriptedLLM(turns or replay_script(url)),
        browser_session=session,
        sensitive_data={"token": REPLAY_SECRET},
    )
    try:
        await agent.run(max_steps=5)
    finally:
        await session.kill()
    return agent, spans


async def test_baseline_session_replay_is_recorded_masked_and_linked(pipeline, tmp_path, form_page):
    with recording_stub_server() as stub:
        agent, spans = await run_recorded(pipeline, tmp_path, form_page, stub)

    [root] = agent_spans(spans)
    [upload] = stub.uploads
    events = upload.body["events"]
    types = [e["type"] for e in events]
    assert 2 in types, types  # a full DOM snapshot ...
    assert 3 in types, types  # ... plus incremental mutations / input
    assert [e["timestamp"] for e in events] == sorted(e["timestamp"] for e in events)
    assert any(form_page in json.dumps(e) for e in events if e["type"] == 4)
    # The agent did type: the typed value reached the page, but only as a mask.
    assert any(e["type"] == 3 and e["data"].get("source") == 5 for e in events), "no input event"
    dumped = json.dumps(upload.body)
    assert TYPED_VALUE not in dumped
    assert REPLAY_SECRET not in dumped
    # One recording per run, tied to the run's root span.
    assert upload.body["trace_id"] == format(root.context.trace_id, "032x")
    assert upload.body["span_id"] == format(root.context.span_id, "016x")
    assert upload.body["session_id"] == str(agent.session_id)
    assert upload.body["framework"] == "browser-use"
    assert upload.headers["Authorization"] == "Bearer test-key"
    recording_id = root.attributes[REPLAY_ID_KEY]
    assert root.attributes[REPLAY_URL_KEY] == stub.view_url(recording_id)
    for span in spans:
        if span is not root:
            assert REPLAY_URL_KEY not in span.attributes, span.name


async def test_baseline_session_replay_upload_failure_does_not_break_the_run(
    pipeline, tmp_path, form_page, monkeypatch
):
    monkeypatch.setattr(session_recording.tempfile, "gettempdir", lambda: str(tmp_path))
    with recording_stub_server(status=500) as stub:
        agent, spans = await run_recorded(pipeline, tmp_path, form_page, stub)

    assert agent.history.is_done()
    assert_one_full_run(spans)
    [root] = agent_spans(spans)
    assert REPLAY_URL_KEY not in root.attributes
    assert len(stub.uploads) == 1
    trace_id = format(root.context.trace_id, "032x")
    assert (tmp_path / session_recording.LOCAL_REPLAY_DIR / f"{trace_id}.html").exists()


async def test_baseline_session_replay_records_nothing_with_content_off(
    pipeline, tmp_path, form_page
):
    with recording_stub_server() as stub:
        agent, spans = await run_recorded(
            pipeline, tmp_path, form_page, stub, capture_content=False
        )

    assert agent.history.is_done()
    assert stub.uploads == []
    assert not any(REPLAY_URL_KEY in s.attributes for s in spans)


async def test_baseline_session_replay_follows_the_agent_into_a_new_tab(
    pipeline, tmp_path, form_page
):
    navigate, done, verdict = script({"navigate": {"url": form_page}})
    with recording_stub_server() as stub:
        _, spans = await run_recorded(
            pipeline, tmp_path, form_page, stub, turns=[navigate, click_the_link, done, verdict]
        )

    [upload] = stub.uploads
    events = upload.body["events"]
    hrefs = [e["data"]["href"] for e in events if e["type"] == 4]
    tab = form_page.replace("form.html", "tab.html")
    assert hrefs[-1] == tab, hrefs  # the stream switched to the tab the agent opened ...
    after = events[[e.get("data", {}).get("href") for e in events].index(tab) :]
    assert after[1]["type"] == 2  # ... from that tab's own full snapshot
    assert "Details tab" in json.dumps(after)


async def test_baseline_session_replay_follow_up_run_on_a_kept_browser_records_again(
    pipeline, tmp_path, form_page
):
    with recording_stub_server() as stub:
        pipeline(instrumentor=BrowserUseInstrumentor(record_session=True), base_url=stub.base_url)
        session = browser(tmp_path, keep_alive=True)
        tab = form_page.replace("form.html", "tab.html")
        agent = Agent(
            task="Read",
            llm=ScriptedLLM([*script({"navigate": {"url": form_page}}), *script({"navigate": {"url": tab}})]),
            browser_session=session,
        )
        try:
            await agent.run(max_steps=4)
            agent.add_new_task("Now the details")
            await agent.run(max_steps=4)
        finally:
            await session.kill()

    first, second = (u.body for u in stub.uploads)
    assert first["trace_id"] != second["trace_id"]
    # The second run re-arms the recorder the first run stopped in that page.
    assert any(e["type"] == 2 for e in second["events"])
    assert "Details tab" in json.dumps(second["events"])
    assert "Details tab" not in json.dumps(first["events"])
