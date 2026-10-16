"""Baseline wire contract: what a customer configures must reach every span.

These drive Skyvern's real in-process engine (`Skyvern.local()`) and a real
headless Chromium on a local page, with a scripted model behind
`litellm.acompletion`, and capture spans *through the Respan pipeline*
(`Respan(...)` -> Respan's span processor -> the spans its exporter would send).

Skyvern runs each task in a background task behind an in-process ASGI app and
opens its own OpenTelemetry spans around every step, so these also pin the
tree: one root per run, and every exported span's parent is an exported span.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest
from opentelemetry.semconv_ai import SpanAttributes
from respan_instrumentation_skyvern import SkyvernInstrumentor
from respan_instrumentation_skyvern._constants import ATTR_PAGE_URL, ATTR_PROMPT_NAME
from respan_sdk.constants.span_attributes import (
    RESPAN_LOG_TYPE,
    RESPAN_METADATA,
    RESPAN_SESSION_ID,
    RESPAN_THREADS_ID,
    RESPAN_TRACE_GROUP_ID,
)
from skyvern import Skyvern
from skyvern.forge.sdk.core import skyvern_context
from skyvern.schemas.llm import LLMConfig

from .conftest import action

CUSTOMER_ID = "respan.customer_params.customer_identifier"
CUSTOMER_METADATA = {"agent_id": "a1", "role": "planner"}
GROUPING = (RESPAN_THREADS_ID, RESPAN_SESSION_ID, RESPAN_TRACE_GROUP_ID)


def local_skyvern() -> Skyvern:
    return Skyvern.local(
        use_in_memory_db=True,
        llm_config=LLMConfig(
            model_name="openai/scripted-1",
            required_env_vars=[],
            supports_vision=True,
            add_assistant_prefix=False,
            litellm_params={"api_base": "http://127.0.0.1:9", "api_key": "test"},
        ),
        settings={"BROWSER_TYPE": "chromium-headless"},
    )


@pytest.fixture
async def skyvern():
    client = local_skyvern()
    try:
        yield client
    finally:
        await client.aclose()


async def run_task(skyvern: Skyvern, url: str, prompt: str = "Open the next page") -> Any:
    return await skyvern.run_task(prompt=prompt, url=url, max_steps=4, wait_for_completion=True)


def kinds(spans: list[Any]) -> list[str]:
    return [s.attributes.get(RESPAN_LOG_TYPE) for s in spans]


def roots(spans: list[Any]) -> list[Any]:
    return [s for s in spans if s.parent is None]


def assert_tree(spans: list[Any], runs: int = 1) -> None:
    """One agent root per run; every other span hangs off an exported span."""
    assert [s.attributes.get(RESPAN_LOG_TYPE) for s in roots(spans)] == ["agent"] * runs
    exported = {s.context.span_id for s in spans}
    for span in spans:
        if span.parent is not None:
            # Skyvern's own spans are never exported: a span parented to one
            # would be an orphan in Respan.
            assert span.parent.span_id in exported, span.name
    root_traces = {s.context.trace_id for s in roots(spans)}
    assert {s.context.trace_id for s in spans} == root_traces


def assert_full_run(spans: list[Any]) -> None:
    assert_tree(spans)
    assert {"agent", "task", "tool", "chat"} <= set(kinds(spans)), kinds(spans)


async def test_baseline_run_task_is_one_tree_across_the_background_task(pipeline, skyvern, site, model):
    _, spans = pipeline()
    model.script([action("CLICK", "Next page")], [action("COMPLETE")])
    result = await run_task(skyvern, site)

    assert_full_run(spans)
    root = roots(spans)[0]
    steps = [s for s in spans if s.name == "skyvern.step"]
    # Skyvern runs one agent step for the click; the goal check, the speculative
    # plan for the next step and the final `complete` action run after it.
    assert len(steps) == 1
    step = steps[0]
    assert step.parent.span_id == root.context.span_id
    tools = {s.name: s for s in spans if s.attributes.get(RESPAN_LOG_TYPE) == "tool"}
    assert set(tools) == {"click", "complete"}
    assert tools["click"].parent.span_id == step.context.span_id
    chats = [s for s in spans if s.attributes.get(RESPAN_LOG_TYPE) == "chat"]
    assert {s.attributes.get(ATTR_PROMPT_NAME) for s in chats} == {"extract-actions", "check-user-goal"}
    assert any(c.parent.span_id == step.context.span_id for c in chats)
    for chat in chats:
        assert chat.attributes[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] == 120
        assert chat.attributes[SpanAttributes.LLM_REQUEST_MODEL] == "scripted-1"
        assert chat.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"]
    # The step shows the page, the actions and the screenshot the model saw.
    output = json.loads(step.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT])
    assert [a["action"]["action_type"] for a in output["actions"]] == ["click"]
    assert output["screenshot"]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert step.attributes[ATTR_PAGE_URL] == site
    run = json.loads(root.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT])
    assert run["status"] == "completed" and run["run_id"] == result.run_id


async def test_baseline_failed_action_is_an_error_span(pipeline, skyvern, site, model):
    _, spans = pipeline()
    model.script([action("CLICK", "No such element")], [action("COMPLETE")])
    await run_task(skyvern, site)

    assert_tree(spans)
    clicks = [s for s in spans if s.name == "click"]
    assert clicks and clicks[0].status.status_code.name == "ERROR"
    assert clicks[0].status.description.startswith("action failed")


async def test_baseline_customer_metadata_is_flat_on_every_span(pipeline, skyvern, site, model):
    client, spans = pipeline(metadata={"source": "constructor"})
    model.script([action("CLICK", "Next page")], [action("COMPLETE")])
    with client.propagate_attributes(metadata=CUSTOMER_METADATA):
        await run_task(skyvern, site)

    assert_full_run(spans)
    expected = {"source": "constructor", **CUSTOMER_METADATA}
    for span in spans:
        for key, value in expected.items():
            assert span.attributes.get(f"{RESPAN_METADATA}.{key}") == value, (span.name, key)
        # A JSON blob under `respan.metadata` is dropped by ingestion.
        assert RESPAN_METADATA not in span.attributes, span.name


async def test_baseline_customer_identifier_on_every_span(pipeline, skyvern, site, model):
    client, spans = pipeline(customer_identifier="from-constructor")
    model.script([action("CLICK", "Next page")], [action("COMPLETE")])
    with client.propagate_attributes(customer_identifier="customer-1"):
        await run_task(skyvern, site)

    assert_full_run(spans)
    for span in spans:
        # The per-request value wins over the constructor default.
        assert span.attributes.get(CUSTOMER_ID) == "customer-1", span.name


async def test_baseline_thread_session_and_trace_group_default_to_the_run_id(
    pipeline, skyvern, site, model
):
    _, spans = pipeline()
    model.script([action("CLICK", "Next page")], [action("COMPLETE")])
    result = await run_task(skyvern, site)

    assert_full_run(spans)
    for span in spans:
        for key in GROUPING:
            assert span.attributes.get(key) == result.run_id, (span.name, key)


async def test_baseline_customer_thread_and_trace_group_win(pipeline, skyvern, site, model):
    client, spans = pipeline()
    model.script([action("CLICK", "Next page")], [action("COMPLETE")])
    with client.propagate_attributes(thread_identifier="my-thread", group_identifier="my-group"):
        result = await run_task(skyvern, site)

    assert_full_run(spans)
    for span in spans:
        assert span.attributes.get(RESPAN_THREADS_ID) == "my-thread", span.name
        assert span.attributes.get(RESPAN_TRACE_GROUP_ID) == "my-group", span.name
        # Not settable through propagate_attributes, so it keeps the default.
        assert span.attributes.get(RESPAN_SESSION_ID) == result.run_id, span.name


def remote_skyvern() -> Skyvern:
    """Skyvern's cloud/self-hosted client, against a fake server that interleaves runs.

    Skyvern's in-memory local mode cannot run tasks concurrently (its SQLite
    session is shared), so parallel isolation is pinned on the client side.
    """
    runs = iter(range(1, 1000))

    async def server(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(0.05)  # both runs are in flight at once
        body = json.loads(request.content or b"{}")
        return httpx.Response(
            200,
            json={
                "run_id": f"tsk_remote_{next(runs)}",
                "status": "completed",
                "run_type": "task_v1",
                "created_at": "2026-10-15T00:00:00",
                "modified_at": "2026-10-15T00:00:00",
                "output": {"echo": body.get("prompt")},
                "run_request": body,
                "recording_url": "https://recordings.example/run.webm?token=abc",
            },
        )

    return Skyvern(
        api_key="test",
        base_url="http://skyvern.example",
        httpx_client=httpx.AsyncClient(transport=httpx.MockTransport(server)),
    )


async def test_baseline_remote_run_is_an_agent_span_with_recording_link(pipeline):
    _, spans = pipeline()
    result = await remote_skyvern().run_task(prompt="Read the page", url="https://example.com")

    assert [s.attributes.get(RESPAN_LOG_TYPE) for s in spans] == ["agent"]
    (span,) = spans
    for key in GROUPING:
        assert span.attributes.get(key) == result.run_id, key
    output = json.loads(span.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT])
    assert output["status"] == "completed"
    assert output["recording_url"] == "https://recordings.example/run.webm?token=abc"
    assert json.loads(span.attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT])["prompt"] == "Read the page"


async def test_baseline_parallel_runs_do_not_share_traces_or_identifiers(pipeline):
    client, spans = pipeline()
    skyvern = remote_skyvern()

    async def one(name: str) -> tuple[str, Any]:
        with client.propagate_attributes(customer_identifier=f"customer-{name}"):
            return name, await skyvern.run_task(prompt=f"run {name}", url="https://example.com")

    results = dict(await asyncio.gather(*(one(name) for name in ("alpha", "beta", "gamma"))))

    assert_tree(spans, runs=3)
    by_thread = {s.attributes[RESPAN_THREADS_ID]: s for s in spans}
    assert set(by_thread) == {r.run_id for r in results.values()}
    for name, result in results.items():
        span = by_thread[result.run_id]
        assert span.attributes.get(CUSTOMER_ID) == f"customer-{name}"
        assert json.loads(span.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT])["output"] == {
            "echo": f"run {name}"
        }


async def test_baseline_sequential_local_runs_are_separate_traces_and_threads(
    pipeline, skyvern, site, model
):
    _, spans = pipeline()
    model.script([action("CLICK", "Next page")], [action("COMPLETE")])
    first = await run_task(skyvern, site, prompt="first")
    model.script([action("CLICK", "Next page")], [action("COMPLETE")])
    second = await run_task(skyvern, site, prompt="second")

    assert_tree(spans, runs=2)
    trace_of = {s.attributes[RESPAN_THREADS_ID]: s.context.trace_id for s in roots(spans)}
    assert set(trace_of) == {first.run_id, second.run_id}
    for span in spans:
        assert span.context.trace_id == trace_of[span.attributes.get(RESPAN_THREADS_ID)], span.name


async def test_baseline_content_off_keeps_identifiers_and_drops_content(
    pipeline, skyvern, site, model
):
    client, spans = pipeline(instrumentor=SkyvernInstrumentor(capture_content=False))
    model.script([action("CLICK", "Next page")], [action("COMPLETE")])
    with client.propagate_attributes(customer_identifier="customer-1", metadata=CUSTOMER_METADATA):
        result = await run_task(skyvern, site, prompt="A task that must stay private")

    assert_full_run(spans)
    for span in spans:
        assert span.attributes.get(CUSTOMER_ID) == "customer-1", span.name
        assert span.attributes.get(f"{RESPAN_METADATA}.agent_id") == "a1", span.name
        assert span.attributes.get(RESPAN_THREADS_ID) == result.run_id, span.name
        assert ATTR_PAGE_URL not in span.attributes, span.name
        dumped = json.dumps(dict(span.attributes), default=str)
        assert "must stay private" not in dumped, span.name
        assert "127.0.0.1" not in dumped, span.name
        assert "gen_ai.prompt" not in dumped, span.name
    chats = [s for s in spans if s.attributes.get(RESPAN_LOG_TYPE) == "chat"]
    assert all(s.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] == 150 for s in chats)


async def test_baseline_secrets_skyvern_resolves_never_reach_a_span(pipeline, skyvern, site, model):
    """A value Skyvern registers as secret mid-run (e.g. a resolved one-time code)."""
    secret = "s3cr3t-value-7781"
    _, spans = pipeline()

    def resolve_secret() -> None:
        skyvern_context.current().register_secret_value(secret)

    model.on_call = resolve_secret
    model.script(
        [action("INPUT_TEXT", "Token", text=secret)],
        [action("CLICK", "Send")],
        [action("COMPLETE")],
    )
    await run_task(skyvern, site, prompt="Send the token")

    assert_tree(spans)
    assert {"input_text", "click"} <= {s.name for s in spans}
    dumps = [json.dumps(dict(span.attributes), default=str) for span in spans]
    for span, dumped in zip(spans, dumps, strict=True):
        assert secret not in dumped, span.name
    # Replaced, not just absent: the typed text and the submitted URL show where it went.
    assert any("<secret>runtime_secret_0</secret>" in d for d in dumps)
