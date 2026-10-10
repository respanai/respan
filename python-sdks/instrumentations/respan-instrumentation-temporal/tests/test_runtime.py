"""Actual ephemeral-server baselines, native histories and replay acceptance."""

from __future__ import annotations

import json
import os
import uuid

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.semconv_ai import SpanAttributes
from respan_instrumentation_temporal import _instrumentation
from temporalio.client import WorkflowFailureError
from temporalio.contrib.opentelemetry import TracingInterceptor
from temporalio.exceptions import ActivityError, CancelledError
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Replayer, Worker

from ._fixtures import Approval, Echo, Failure, echo, fail


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["bare", "content", "private"])
async def test_native_sandbox_activity_signal_query_cancel_and_replay(mode):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    interceptors = (
        []
        if mode == "bare"
        else [
            _instrumentation._build_interceptor(
                TracingInterceptor,
                tracer=provider.get_tracer("temporal-runtime"),
                capture_content=mode == "content",
                max_attribute_chars=None,
                always_create_workflow_spans=True,
            )
        ]
    )
    run_id = uuid.uuid4().hex
    payload = {
        "history": [
            {"args": list(range(75)), "vectors": [0.1, 0.2, 0.3]} for _ in range(45)
        ],
        "text": "complete payload " * 2000,
    }
    try:
        async with await WorkflowEnvironment.start_time_skipping(
            interceptors=interceptors,
            test_server_existing_path=os.getenv("TEMPORAL_TEST_SERVER_PATH"),
        ) as environment:
            async with Worker(
                environment.client,
                task_queue=run_id,
                workflows=[Echo, Failure, Approval],
                activities=[echo, fail],
            ):
                result = await environment.client.execute_workflow(
                    Echo.run, payload, id=f"{run_id}-echo", task_queue=run_id
                )
                assert result == payload
                with pytest.raises(WorkflowFailureError) as failed:
                    await environment.client.execute_workflow(
                        Failure.run,
                        "expected failure",
                        id=f"{run_id}-failure",
                        task_queue=run_id,
                    )
                assert isinstance(failed.value.cause, ActivityError)
                assert failed.value.cause.cause.message == "expected failure"
                handle = await environment.client.start_workflow(
                    Approval.run,
                    "private-topic",
                    id=f"{run_id}-approval",
                    task_queue=run_id,
                )
                assert await handle.query(Approval.status) == "pending"
                await handle.signal(Approval.approve)
                assert await handle.result() == "approved:private-topic"
                history = await handle.fetch_history()
                cancelled = await environment.client.start_workflow(
                    Approval.run,
                    "private-cancel",
                    id=f"{run_id}-cancel",
                    task_queue=run_id,
                )
                assert await cancelled.query(Approval.status) == "pending"
                await cancelled.cancel()
                with pytest.raises(WorkflowFailureError) as caught:
                    await cancelled.result()
                assert isinstance(caught.value.cause, CancelledError)
            count = len(exporter.get_finished_spans())
            await Replayer(
                workflows=[Approval], interceptors=interceptors
            ).replay_workflow(history)
            assert len(exporter.get_finished_spans()) == count
        spans = exporter.get_finished_spans()
        if mode == "bare":
            assert spans == ()
        else:
            assert len(spans) == 21
            text = "\n".join(span.to_json() for span in spans)
            if mode == "private":
                assert "private-topic" not in text
                assert "private-cancel" not in text
                assert "expected failure" not in text
                assert run_id not in text
                assert "complete payload" not in text
                assert not any(span.events for span in spans)
            else:
                echo_spans = [
                    s
                    for s in spans
                    if s.name in ("RunActivity:echo", "CompleteWorkflow:Echo")
                ]
                assert len(echo_spans) == 2
                assert all(
                    json.loads(s.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT])
                    == payload
                    for s in echo_spans
                )
                failures = [
                    s
                    for s in spans
                    if s.name in ("RunActivity:fail", "CompleteWorkflow:Failure")
                ]
                assert len(failures) == 2
                assert all(
                    SpanAttributes.TRACELOOP_ENTITY_OUTPUT not in s.attributes
                    for s in failures
                )
                by_name = {s.name: s for s in spans}
                assert (
                    by_name["RunActivity:echo"].parent.span_id
                    == by_name["StartActivity:echo"].context.span_id
                )
                assert (
                    by_name["CompleteWorkflow:Echo"].parent.span_id
                    == by_name["StartWorkflow:Echo"].context.span_id
                )
    finally:
        provider.shutdown()
