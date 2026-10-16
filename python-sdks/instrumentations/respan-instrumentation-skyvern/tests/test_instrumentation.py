"""Paths Skyvern's in-memory local mode cannot run (workflows and page AI actions
need its persistent database): the workflow run and page-AI-action spans against a
fake Skyvern server, and the workflow-block wrapper on Skyvern's real `BlockResult`.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest
from opentelemetry.semconv_ai import SpanAttributes
from respan_instrumentation_skyvern import SkyvernInstrumentor, _instrumentation
from respan_instrumentation_skyvern._constants import ATTR_BLOCK_LABEL, ATTR_BLOCK_TYPE
from respan_sdk.constants.span_attributes import RESPAN_LOG_TYPE, RESPAN_THREADS_ID
from skyvern import Skyvern
from skyvern.client.types.run_sdk_action_request_action import (
    RunSdkActionRequestAction_Validate,
)
from skyvern.forge.sdk.workflow.models.block import Block
from skyvern.schemas.workflows import BlockResult, BlockStatus


def workflow_server(status: str) -> Skyvern:
    async def server(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "run_id": "wr_test_1",
                "status": status,
                "run_type": "workflow_run",
                "created_at": "2026-10-15T00:00:00",
                "modified_at": "2026-10-15T00:00:00",
                "failure_reason": "Login failed: wrong password",
                "run_request": {"workflow_id": "wpid_1"},
            },
        )

    return Skyvern(
        api_key="test",
        base_url="http://skyvern.example",
        httpx_client=httpx.AsyncClient(transport=httpx.MockTransport(server)),
    )


async def test_failed_workflow_run_is_an_error_agent_span(pipeline):
    _, spans = pipeline()
    await workflow_server("failed").run_workflow(workflow_id="wpid_1", parameters={"q": "x"})

    (span,) = spans
    assert span.name == "skyvern.run_workflow"
    assert span.attributes[RESPAN_LOG_TYPE] == "agent"
    assert span.attributes[RESPAN_THREADS_ID] == "wr_test_1"
    assert span.status.status_code.name == "ERROR"
    assert span.status.description == "run failed: Login failed: wrong password"
    inputs = json.loads(span.attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT])
    assert inputs["workflow_id"] == "wpid_1" and inputs["parameters"] == {"q": "x"}


async def test_workflow_block_is_a_task_span_with_its_outcome(pipeline):
    _, spans = pipeline()
    # The instrumentor patched Skyvern's Block.execute_safe with this wrapper.
    assert getattr(Block.__dict__["execute_safe"], "__respan_patched__", False)

    async def execute_safe(block, workflow_run_id, **kwargs) -> BlockResult:
        return BlockResult(
            success=False,
            output_parameter=None,  # type: ignore[arg-type]
            status=BlockStatus.failed,
            failure_reason="element not found",
        )

    block = SimpleNamespace(block_type="navigation", label="open_cart")
    await _instrumentation._RUNTIME.wrap_execute_safe(execute_safe)(block, "wr_1")

    (span,) = spans
    assert span.name == "skyvern.block.navigation"
    assert span.attributes[RESPAN_LOG_TYPE] == "task"
    assert span.attributes[ATTR_BLOCK_TYPE] == "navigation"
    assert span.attributes[ATTR_BLOCK_LABEL] == "open_cart"
    assert span.status.description == "block failed: element not found"


async def test_content_off_drops_run_inputs_and_failure_text(pipeline):
    _, spans = pipeline(instrumentor=SkyvernInstrumentor(capture_content=False))
    await workflow_server("failed").run_workflow(workflow_id="wpid_1", parameters={"q": "x"})

    (span,) = spans
    assert span.status.description == "run failed"
    dumped = json.dumps(dict(span.attributes), default=str)
    assert "wrong password" not in dumped and "wpid_1" not in dumped


async def test_page_ai_action_is_one_agent_span_per_call(pipeline):
    """`page.act / extract / validate / ai_*` all reach Skyvern through run_sdk_action."""

    async def server(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"workflow_run_id": "wr_sdk_1", "result": True})

    skyvern = Skyvern(
        api_key="test",
        base_url="http://skyvern.example",
        httpx_client=httpx.AsyncClient(transport=httpx.MockTransport(server)),
    )
    _, spans = pipeline()
    await skyvern.run_sdk_action(
        url="https://example.com",
        action=RunSdkActionRequestAction_Validate(prompt="The page lists books"),
    )

    (span,) = spans
    assert span.name == "skyvern.validate"
    assert span.attributes[RESPAN_LOG_TYPE] == "agent"
    assert span.attributes[RESPAN_THREADS_ID] == "wr_sdk_1"
    assert json.loads(span.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]) == {"result": True}
    assert "The page lists books" in span.attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT]


async def test_cancelled_model_call_is_not_an_error(pipeline):
    """Skyvern cancels its speculative next-step plan when the run ends first."""
    _, spans = pipeline()
    runtime = _instrumentation._RUNTIME

    async def never_answers(*args, **kwargs):
        await asyncio.sleep(60)

    async def call() -> None:
        with runtime.span("skyvern.step", "task"):
            await runtime.traced_llm_call(never_answers, None, (), {"model": "openai/m"})

    task = asyncio.ensure_future(call())
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    chat = next(s for s in spans if s.attributes.get(RESPAN_LOG_TYPE) == "chat")
    assert chat.attributes["skyvern.llm.cancelled"] is True
    assert chat.status.status_code.name != "ERROR"
