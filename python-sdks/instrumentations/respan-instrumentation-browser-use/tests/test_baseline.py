"""Baseline wire contract: what a customer configures must reach every span.

These drive a real headless Chrome through browser-use with a scripted model,
and capture spans *through the Respan pipeline* (`Respan(...)` -> Respan's span
processor -> the spans its exporter would send). Ingestion reads customer
metadata only as flat `respan.metadata.<key>` attributes, and groups runs by
thread / session / trace group, so those must be on every span.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from browser_use import Agent, BrowserProfile, BrowserSession
from opentelemetry import trace
from opentelemetry.semconv_ai import SpanAttributes
from respan import Respan
from respan_sdk.constants.span_attributes import (
    RESPAN_LOG_TYPE,
    RESPAN_METADATA,
    RESPAN_SESSION_ID,
    RESPAN_THREADS_ID,
    RESPAN_TRACE_GROUP_ID,
)
from respan_tracing.core.tracer import RespanTracer
from respan_tracing.utils.span_factory import _PROPAGATED_ATTRIBUTES

from respan_instrumentation_browser_use import BrowserUseInstrumentor
from respan_instrumentation_browser_use._constants import ATTR_PAGE_URL

from .conftest import ScriptedLLM, chrome_path

pytestmark = pytest.mark.skipif(
    chrome_path() is None, reason="needs a local Chrome/Chromium"
)

CUSTOMER_ID = "respan.customer_params.customer_identifier"
CUSTOMER_METADATA = {"agent_id": "a1", "role": "planner"}
# Nothing listens here: the exporter's network calls fail fast and are ignored.
UNREACHABLE = "http://127.0.0.1:9/api"


@pytest.fixture
def pipeline(monkeypatch):
    """Build `Respan(...)` with the browser-use instrumentor; return the exported spans."""
    clients: list[Respan] = []
    defaults = _PROPAGATED_ATTRIBUTES.set({})

    def make(instrumentor: BrowserUseInstrumentor | None = None, **kwargs: Any):
        exported: list[Any] = []
        client = Respan(
            api_key="test-key",
            base_url=UNREACHABLE,
            instrumentations=[instrumentor or BrowserUseInstrumentor()],
            span_postprocess_callback=exported.append,
            **kwargs,
        )
        # OTel lets the global provider be set once per process; point it at
        # this client's provider so every test exports through its own pipeline.
        monkeypatch.setattr(
            trace, "_TRACER_PROVIDER", RespanTracer._instance.tracer_provider
        )
        clients.append(client)
        return client, exported

    yield make
    for client in clients:
        client.shutdown()
    RespanTracer.reset_instance()
    _PROPAGATED_ATTRIBUTES.reset(defaults)


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
