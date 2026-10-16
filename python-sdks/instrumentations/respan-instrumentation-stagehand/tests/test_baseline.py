"""Baseline wire contract: what a customer configures must reach every span.

These drive the REAL Stagehand v4 Python SDK and its Chrome extension (headless
local Chrome, a local HTTP page, a scripted client model) and capture spans
*through the Respan pipeline* (`Respan(...)` -> Respan's span processor -> the
spans its exporter would send). Ingestion reads customer metadata only as flat
`respan.metadata.<key>` attributes, and groups a session's spans by thread /
session / trace group, so those must be on every span.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from contextlib import ExitStack
from typing import Any

import pytest
from opentelemetry import context as context_api
from opentelemetry.semconv_ai import (
    SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY,
    SpanAttributes,
)
from opentelemetry.trace import StatusCode
from pydantic import BaseModel
from respan import Respan, workflow
from respan_sdk.constants.span_attributes import (
    RESPAN_LOG_TYPE,
    RESPAN_METADATA,
    RESPAN_SESSION_ID,
    RESPAN_THREADS_ID,
    RESPAN_TRACE_GROUP_ID,
)
from respan_tracing.testing import UNREACHABLE_BASE_URL, capture_exported_spans
from stagehand import Stagehand, local_browser

from respan_instrumentation_stagehand import StagehandInstrumentor, with_model_name
from respan_instrumentation_stagehand._constants import (
    ATTR_CACHE_STATUS,
    ATTR_PAGE_URL,
    ATTR_SESSION_ID,
)

from .conftest import USAGE, ScriptedModel, chrome_path

pytestmark = pytest.mark.skipif(chrome_path() is None, reason="needs a local Chrome/Chromium")

CUSTOMER_ID = "respan.customer_params.customer_identifier"
CUSTOMER_METADATA = {"agent_id": "a1", "role": "planner"}
ENTITY = SpanAttributes.TRACELOOP_ENTITY_NAME
INPUT = SpanAttributes.TRACELOOP_ENTITY_INPUT
OUTPUT = SpanAttributes.TRACELOOP_ENTITY_OUTPUT
MODEL = SpanAttributes.LLM_REQUEST_MODEL


class Heading(BaseModel):
    extraction: str


@pytest.fixture
def pipeline():
    """Build `Respan(...)` with the Stagehand instrumentor; return the exported spans."""
    with ExitStack() as stack:

        def make(instrumentor: StagehandInstrumentor | None = None, **kwargs: Any):
            return stack.enter_context(
                capture_exported_spans(
                    lambda on_export: Respan(
                        api_key="test-key",
                        base_url=UNREACHABLE_BASE_URL,
                        instrumentations=[
                            instrumentor or StagehandInstrumentor(client_model_name="scripted-1")
                        ],
                        span_postprocess_callback=on_export,
                        **kwargs,
                    )
                )
            )

        yield make


Flow = Callable[[Stagehand, Any, Any], Awaitable[Any]]


async def session(site: str, flow: Flow, *, model: Any = None) -> None:
    """One Stagehand session: create, open the page, the flow, close."""
    browser = await local_browser.launch(headless=True, executable_path=chrome_path())
    try:
        stagehand = await Stagehand.create(
            browser=browser, model=model or ScriptedModel(), logging={"level": "off"}
        )
        try:
            [page] = await browser.context.pages()
            await page.goto(f"{site}/index.html")
            await flow(stagehand, page, browser)
        finally:
            await stagehand.close()
    finally:
        await browser.close()


async def standard_flow(stagehand: Stagehand, page: Any, browser: Any) -> None:
    await stagehand.extract("get the heading", Heading)
    await stagehand.act("click the sign in button")


def by_entity(spans: list[Any], name: str) -> list[Any]:
    return [s for s in spans if s.attributes.get(ENTITY) == name]


def session_id(spans: list[Any]) -> str:
    return by_entity(spans, "stagehand.session")[0].attributes[ATTR_SESSION_ID]


def assert_full_session(spans: list[Any]) -> None:
    kinds = [s.attributes.get(RESPAN_LOG_TYPE) for s in spans]
    assert kinds.count("agent") == 1, kinds
    assert {"task", "chat", "tool"} <= set(kinds), kinds


def parent_id(span: Any) -> int | None:
    return span.parent.span_id if span.parent else None


async def test_baseline_span_tree_is_session_operation_llm_and_browser_action(pipeline, site):
    _, spans = pipeline()
    await session(site, standard_flow)

    assert_full_session(spans)
    [root] = by_entity(spans, "stagehand.session")
    [extract] = by_entity(spans, "stagehand.extract")
    [act] = by_entity(spans, "stagehand.act")
    sid = lambda s: s.context.span_id  # noqa: E731
    assert root.parent is None
    assert parent_id(extract) == sid(root)
    assert parent_id(act) == sid(root)
    llm = by_entity(spans, "stagehand.llm")
    # extract makes two inferences (extraction + completion check), act one.
    assert sorted(map(parent_id, llm)) == sorted([sid(extract), sid(extract), sid(act)])
    for chat in llm:
        assert chat.attributes[MODEL] == "scripted-1"
        assert chat.attributes[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] == USAGE["input_tokens"]
        assert chat.attributes[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS] == USAGE["output_tokens"]
        assert (
            chat.attributes[SpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS]
            == USAGE["cached_input_tokens"]
        )
        assert chat.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.role"] == "system"
        assert chat.attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"]
    [goto] = by_entity(spans, "page.goto")
    assert parent_id(goto) == sid(root)
    assert json.loads(goto.attributes[OUTPUT])["status"] == 200
    # Operation results, Stagehand's cache status and the post-act screenshot.
    assert json.loads(extract.attributes[OUTPUT])["data"]["extraction"] == "Hello browser agent"
    assert json.loads(extract.attributes[INPUT])["schema"]["properties"]["extraction"]
    assert extract.attributes[ATTR_CACHE_STATUS] == "DISABLED"
    act_output = json.loads(act.attributes[OUTPUT])
    assert act_output["data"]["success"] is True
    assert act_output["screenshot"]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    session_output = json.loads(root.attributes[OUTPUT])
    assert session_output["operations"] == 2
    assert session_output["usage"]["input_tokens"] == 3 * USAGE["input_tokens"]


async def test_baseline_customer_metadata_is_flat_on_every_span(pipeline, site):
    client, spans = pipeline(metadata={"source": "constructor"})
    with client.propagate_attributes(metadata=CUSTOMER_METADATA):
        await session(site, standard_flow)

    assert_full_session(spans)
    expected = {"source": "constructor", **CUSTOMER_METADATA}
    for span in spans:
        for key, value in expected.items():
            assert span.attributes.get(f"{RESPAN_METADATA}.{key}") == value, (span.name, key)
        # A JSON blob under `respan.metadata` is dropped by ingestion.
        assert RESPAN_METADATA not in span.attributes, span.name


async def test_baseline_customer_identifier_on_every_span(pipeline, site):
    client, spans = pipeline(customer_identifier="from-constructor")
    with client.propagate_attributes(customer_identifier="customer-1"):
        await session(site, standard_flow)

    assert_full_session(spans)
    for span in spans:
        # The per-request value wins over the constructor default.
        assert span.attributes.get(CUSTOMER_ID) == "customer-1", span.name


async def test_baseline_thread_session_and_trace_group_default_to_the_stagehand_session(
    pipeline, site
):
    _, spans = pipeline()
    await session(site, standard_flow)

    assert_full_session(spans)
    sid = session_id(spans)
    for span in spans:
        for key in (RESPAN_THREADS_ID, RESPAN_SESSION_ID, RESPAN_TRACE_GROUP_ID):
            assert span.attributes.get(key) == sid, (span.name, key)
    assert len({s.context.trace_id for s in spans}) == 1


async def test_baseline_customer_thread_and_trace_group_win(pipeline, site):
    client, spans = pipeline()
    with client.propagate_attributes(thread_identifier="my-thread", group_identifier="my-group"):
        await session(site, standard_flow)

    assert_full_session(spans)
    sid = session_id(spans)
    for span in spans:
        assert span.attributes.get(RESPAN_THREADS_ID) == "my-thread", span.name
        assert span.attributes.get(RESPAN_TRACE_GROUP_ID) == "my-group", span.name
        # Not settable through propagate_attributes, so it keeps the default.
        assert span.attributes.get(RESPAN_SESSION_ID) == sid, span.name


async def test_baseline_parallel_sessions_do_not_share_traces_or_identifiers(pipeline, site):
    client, spans = pipeline()

    async def one(name: str) -> None:
        # Session b names its own client model; a falls back to the instrumentor's.
        model = ScriptedModel()
        if name == "b":
            model = with_model_name(model, "scripted-2")
        with client.propagate_attributes(customer_identifier=f"customer-{name}"):
            await session(site, standard_flow, model=model)

    await asyncio.gather(one("a"), one("b"))

    roots = by_entity(spans, "stagehand.session")
    assert len(roots) == 2
    trace_of = {r.attributes[RESPAN_THREADS_ID]: r.context.trace_id for r in roots}
    assert len(trace_of) == 2
    for span in spans:
        assert span.context.trace_id == trace_of[span.attributes[RESPAN_THREADS_ID]], span.name
    for root in roots:
        mine = [
            s
            for s in spans
            if s.attributes.get(RESPAN_THREADS_ID) == root.attributes[RESPAN_THREADS_ID]
        ]
        customers = {s.attributes.get(CUSTOMER_ID) for s in mine}
        assert len(customers) == 1
        chats = by_entity(mine, "stagehand.llm")
        assert len(chats) == 3
        want = "scripted-2" if customers == {"customer-b"} else "scripted-1"
        assert {c.attributes[MODEL] for c in chats} == {want}


async def test_baseline_concurrent_operations_in_one_session_keep_their_own_llm_calls(
    pipeline, site
):
    _, spans = pipeline()
    # The extraction echoes the heading the model was shown, so each LLM span's
    # prompt says which page (and therefore which operation) it belongs to.
    model = ScriptedModel(
        extraction=lambda params: (
            "second"
            if "Second page heading" in json.dumps(params.model_dump(mode="json"))
            else "first"
        )
    )

    async def flow(stagehand: Stagehand, page: Any, browser: Any) -> None:
        second = await browser.context.new_page(f"{site}/second.html")
        await asyncio.gather(
            stagehand.extract("read the heading of the sign-in page", Heading, page=page),
            stagehand.extract("read the heading of the second page", Heading, page=second),
        )

    await session(site, flow, model=model)

    extracts = by_entity(spans, "stagehand.extract")
    assert len(extracts) == 2
    for op in extracts:
        answer = json.loads(op.attributes[OUTPUT])["data"]["extraction"]
        calls = [s for s in by_entity(spans, "stagehand.llm") if parent_id(s) == op.context.span_id]
        assert len(calls) == 2, answer
        saw_second = any("Second page heading" in c.attributes[INPUT] for c in calls)
        assert saw_second == (answer == "second"), answer


async def test_baseline_content_off_keeps_identifiers_and_drops_content(pipeline, site):
    client, spans = pipeline(instrumentor=StagehandInstrumentor(capture_content=False))

    async def flow(stagehand: Stagehand, page: Any, browser: Any) -> None:
        await stagehand.extract("A private instruction", Heading)
        await stagehand.act("click the sign in button")

    with client.propagate_attributes(customer_identifier="customer-1", metadata=CUSTOMER_METADATA):
        await session(site, flow)

    assert_full_session(spans)
    for span in spans:
        assert span.attributes.get(CUSTOMER_ID) == "customer-1", span.name
        assert span.attributes.get(f"{RESPAN_METADATA}.agent_id") == "a1", span.name
        assert span.attributes.get(RESPAN_THREADS_ID), span.name
        assert ATTR_PAGE_URL not in span.attributes, span.name
        dumped = json.dumps(dict(span.attributes), default=str)
        assert "A private instruction" not in dumped, span.name
        assert site not in dumped, span.name
        assert "Hello browser agent" not in dumped, span.name
        assert "base64" not in dumped, span.name
    # Structure and tokens survive.
    chats = by_entity(spans, "stagehand.llm")
    assert len(chats) == 3
    for chat in chats:
        assert chat.attributes[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] == USAGE["input_tokens"]


async def test_baseline_act_variables_and_typed_credentials_never_reach_a_span(pipeline, site):
    secret = "s3cr3t-value-7781"
    typed = "typed-password-9931"
    _, spans = pipeline()
    model = ScriptedModel(target="textbox", method="fill", arguments=["%password%"])

    async def flow(stagehand: Stagehand, page: Any, browser: Any) -> None:
        await stagehand.act(
            "type %password% into the password field",
            variables={"password": {"value": secret, "description": "the password"}},
        )
        # A credential field: the typed text is redacted, even when the selector
        # is an XPath (what observe() returns) that says nothing about the field.
        await page.locator("#password").fill(typed)
        await page.locator("xpath=/html[1]/body[1]/form[1]/label[2]/input[1]").fill(typed)
        # An ordinary field keeps its text.
        await page.locator("xpath=/html[1]/body[1]/form[1]/label[1]/input[1]").fill(
            "ada@example.com"
        )
        # An error that echoes the secret is masked too.
        with pytest.raises(Exception):  # noqa: B017 - any navigation error
            await page.goto(f"not-a-url-{secret}")

    await session(site, flow, model=model)

    assert_full_session(spans)
    for span in spans:
        dumped = json.dumps(
            [
                dict(span.attributes),
                span.status.description,
                [dict(e.attributes or {}) for e in span.events],
            ],
            default=str,
        )
        assert secret not in dumped, span.name
        assert typed not in dumped, span.name
    everything = json.dumps(
        [[dict(s.attributes), s.status.description] for s in spans], default=str
    )
    assert "%password%" in everything
    fills = [
        json.loads(s.attributes[INPUT])["arguments"]["value"]
        for s in by_entity(spans, "locator.fill")
    ]
    assert fills == ["[redacted]", "[redacted]", "ada@example.com"]


async def test_baseline_errors_carry_type_and_message(pipeline, site):
    _, spans = pipeline()
    # The model picks the heading to fill: Stagehand reports the act as failed.
    model = ScriptedModel(target="heading: Hello", method="fill", arguments=["x"])

    async def flow(stagehand: Stagehand, page: Any, browser: Any) -> None:
        with pytest.raises(Exception):  # noqa: B017 - the click times out
            await page.locator("#missing").click(timeout=300)
        result = await stagehand.act("type x into the heading")
        assert result.data.success is False

    await session(site, flow, model=model)

    [click] = by_entity(spans, "locator.click")
    assert click.status.status_code == StatusCode.ERROR
    error_type, _, message = click.status.description.partition(": ")
    assert error_type.isidentifier() and message, click.status.description
    [act] = by_entity(spans, "stagehand.act")
    assert act.status.status_code == StatusCode.ERROR
    assert act.status.description.startswith("act reported failure: Failed to perform act")
    [root] = by_entity(spans, "stagehand.session")
    assert json.loads(root.attributes[OUTPUT])["failed_operations"] == 1


async def test_baseline_inside_a_workflow_the_session_nests_and_provider_spans_are_suppressed(
    pipeline, site
):
    _, spans = pipeline()
    suppressed: list[Any] = []
    model = ScriptedModel(
        on_call=lambda _: suppressed.append(
            context_api.get_value(SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY)
        )
    )

    @workflow(name="checkout")
    async def checkout() -> None:
        await session(site, standard_flow, model=model)

    await checkout()

    # Provider instrumentations (OpenAI, Anthropic, ...) see the suppression key
    # inside generate, so the call is not logged twice.
    assert suppressed == [True, True, True]
    [flow] = [
        s for s in spans if s.attributes.get(SpanAttributes.TRACELOOP_SPAN_KIND) == "workflow"
    ]
    [root] = by_entity(spans, "stagehand.session")
    assert parent_id(root) == flow.context.span_id
    assert {s.context.trace_id for s in spans} == {flow.context.trace_id}
