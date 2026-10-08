import asyncio
import inspect
import json

import pytest
from opentelemetry import context, trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.sampling import ALWAYS_OFF
from opentelemetry.semconv_ai import SpanAttributes as A
from respan_instrumentation_superagent import SuperagentInstrumentor, _instrumentation
from respan_tracing.constants.context_constants import ENABLE_CONTENT_TRACING_KEY
from safety_agent.client import SafetyClient
from safety_agent.types import GuardOptions, RedactOptions, ScanOptions

from ._fixtures import fixture_client


@pytest.fixture
def runtime(monkeypatch):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(trace, "get_tracer_provider", lambda: provider)
    owner = SuperagentInstrumentor()
    owner.activate()
    yield owner, exporter, provider
    owner.deactivate()


def test_released_guard_chunks_redact_scan_preserve_typed_results(runtime):
    _, exporter, _ = runtime

    async def run():
        with fixture_client() as (client, requests):
            guard = await client.guard(
                GuardOptions(
                    input="safe block safe",
                    model="openai/gpt-4o-mini",
                    chunk_size=6,
                    system_prompt="fixture instructions",
                )
            )
            redact = await client.redact(
                input=RedactOptions(
                    input="Contact fixture-email@example.com",
                    model="openai/gpt-4o-mini",
                    entities=["EMAIL"],
                    rewrite=True,
                )
            )
            scan = await client.scan(
                ScanOptions(
                    repo="https://example.com/repository",
                    branch="fixture-branch",
                    model="openai/gpt-4o-mini",
                )
            )
            assert guard.classification == "block" and guard.usage.prompt_tokens == 0
            assert guard.usage.completion_tokens == 3 * (len(requests) - 1)
            assert redact.redacted == "Contact <EMAIL_REDACTED>"
            assert scan.usage.cost == 0

    asyncio.run(run())
    spans = exporter.get_finished_spans()
    assert len(spans) == 3
    assert (
        json.loads(spans[0].attributes[A.TRACELOOP_ENTITY_INPUT])["system_prompt"]
        == "fixture instructions"
    )
    assert json.loads(spans[1].attributes[A.TRACELOOP_ENTITY_INPUT])["rewrite"] is True
    assert (
        json.loads(spans[2].attributes[A.TRACELOOP_ENTITY_INPUT])["branch"]
        == "fixture-branch"
    )
    assert all("gen_ai.usage.input_tokens" not in s.attributes for s in spans)
    assert inspect.signature(SafetyClient.guard) == inspect.signature(
        SafetyClient.guard.__wrapped__
    )


@pytest.mark.parametrize("mode", ["env", "context"])
def test_private_payload_and_redaction_findings_stay_hidden(runtime, monkeypatch, mode):
    _, exporter, _ = runtime
    if mode == "env":
        monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")
    token = (
        context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, False))
        if mode == "context"
        else None
    )
    try:

        async def run():
            with fixture_client() as (client, _):
                result = await client.redact(
                    input="private fixture-email@example.com",
                    model="openai/gpt-4o-mini",
                )
                assert result.findings == ["fixture-email@example.com"]

        asyncio.run(run())
    finally:
        if token is not None:
            context.detach(token)
    attrs = dict(exporter.get_finished_spans()[0].attributes)
    assert (
        A.TRACELOOP_ENTITY_INPUT not in attrs and A.TRACELOOP_ENTITY_OUTPUT not in attrs
    )
    assert "fixture-email@example.com" not in str(attrs)
    assert not exporter.get_finished_spans()[0].events


@pytest.mark.parametrize("initial,later", [(False, True), (True, False)])
def test_environment_changes_cannot_enable_private_content(
    runtime, monkeypatch, initial, later
):
    _, exporter, _ = runtime
    monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", str(initial).lower())

    async def flip():
        monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", str(later).lower())

    async def run():
        with fixture_client(before_response=flip) as (client, _):
            await client.guard(input="private-transition", model="openai/gpt-4o-mini")

    asyncio.run(run())
    assert "private-transition" not in str(
        [dict(s.attributes) for s in exporter.get_finished_spans()]
    )


def test_suppression_keeps_parent_context_and_no_exports(runtime):
    _, exporter, provider = runtime

    async def run():
        with provider.get_tracer("test").start_as_current_span("caller") as parent:
            token = context.attach(
                context.set_value(context._SUPPRESS_INSTRUMENTATION_KEY, True)
            )
            try:

                async def check():
                    assert trace.get_current_span() is parent

                with fixture_client(before_response=check) as (client, _):
                    await client.guard(input="suppressed", model="openai/gpt-4o-mini")
            finally:
                context.detach(token)

    asyncio.run(run())
    assert [s.name for s in exporter.get_finished_spans()] == ["caller"]


def test_live_span_is_parent_during_provider_call(runtime):
    _, exporter, provider = runtime

    async def run():
        with provider.get_tracer("test").start_as_current_span("caller") as parent:

            async def check():
                current = trace.get_current_span()
                assert current is not parent and current.get_span_context().is_valid
                with provider.get_tracer("test").start_as_current_span(
                    "provider-child"
                ):
                    pass

            with fixture_client(before_response=check) as (client, _):
                await client.guard(input="safe", model="openai/gpt-4o-mini")
            assert trace.get_current_span() is parent

    asyncio.run(run())
    spans = exporter.get_finished_spans()
    child, guard, parent = spans
    assert child.parent.span_id == guard.context.span_id
    assert guard.parent.span_id == parent.context.span_id


def test_real_http_error_has_actual_status_and_no_result(runtime):
    _, exporter, _ = runtime

    async def run():
        with fixture_client(fail=True) as (client, _), pytest.raises(RuntimeError):
            await client.guard(input="failure", model="openai/gpt-4o-mini")

    asyncio.run(run())
    span = exporter.get_finished_spans()[0]
    assert span.status.status_code.name == "ERROR"
    assert A.TRACELOOP_ENTITY_OUTPUT not in span.attributes
    assert "status_code" not in span.attributes
    # SDK wraps provider errors; an HTTP attribute is emitted only if retained.
    if "http.response.status_code" in span.attributes:
        assert span.attributes["http.response.status_code"] == 401


def test_cancellation_balances_call_and_preserves_exception(runtime):
    _, exporter, _ = runtime

    async def run():
        started = asyncio.Event()

        async def pause():
            started.set()
            await asyncio.Future()

        with fixture_client(before_response=pause) as (client, _):
            task = asyncio.create_task(
                client.guard(input="cancel-input", model="openai/gpt-4o-mini")
            )
            await started.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    asyncio.run(run())
    span = exporter.get_finished_spans()[0]
    assert span.status.status_code.name == "ERROR"
    assert span.attributes["error.type"] == "CancelledError"
    assert A.TRACELOOP_ENTITY_OUTPUT not in span.attributes


def test_always_off_records_nothing(monkeypatch):
    provider = TracerProvider(sampler=ALWAYS_OFF)
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(trace, "get_tracer_provider", lambda: provider)
    owner = SuperagentInstrumentor()
    owner.activate()

    async def run():
        with fixture_client() as (client, _):
            await client.guard(input="not-sampled", model="openai/gpt-4o-mini")

    try:
        asyncio.run(run())
    finally:
        owner.deactivate()
    assert not exporter.get_finished_spans()


def test_shared_owner_rejects_other_provider(runtime, monkeypatch):
    first, _exporter, _ = runtime
    second = SuperagentInstrumentor()
    second.activate()
    first.deactivate()
    monkeypatch.setattr(trace, "get_tracer_provider", lambda: TracerProvider())
    with pytest.raises(ValueError, match="share one tracer provider"):
        SuperagentInstrumentor().activate()
    second.deactivate()
    assert _instrumentation._ACTIVE_INSTANCES == 0


def test_partial_patch_failure_rolls_back_all_bindings(runtime, monkeypatch):
    owner, _, _ = runtime
    owner.deactivate()
    original = SafetyClient.guard
    wrap = _instrumentation._wrap_method

    def fail(method_name, original, generation):
        if method_name == "redact":
            raise RuntimeError("controlled patch failure")
        return wrap(method_name, original, generation)

    monkeypatch.setattr(_instrumentation, "_wrap_method", fail)
    with pytest.raises(RuntimeError, match="controlled patch failure"):
        owner.activate()
    assert SafetyClient.guard is original
    assert not owner._is_instrumented and not _instrumentation._ORIGINAL_METHODS


def test_bytes_and_url_inputs_use_native_processing(runtime):
    import base64

    _, exporter, _ = runtime
    raw = b"\x89PNG\r\n\x1a\nfixture image"

    async def run():
        with fixture_client() as (client, requests):
            assert (
                await client.guard(input=raw, model="openai/gpt-4o-mini")
            ).classification == "pass"
            assert (
                await client.guard(
                    input="https://example.com/fixture.txt", model="openai/gpt-4o-mini"
                )
            ).classification == "pass"
            assert len(requests) == 2

    asyncio.run(run())
    byte_span, url_span = exporter.get_finished_spans()
    captured = json.loads(byte_span.attributes[A.TRACELOOP_ENTITY_INPUT])["input"]
    assert base64.b64decode(captured["base64"]) == raw
    assert (
        json.loads(url_span.attributes[A.TRACELOOP_ENTITY_INPUT])["input"]
        == "https://example.com/fixture.txt"
    )


def test_provider_fallback_preserves_result_and_one_operation(runtime):
    if "fallback_model" not in inspect.signature(SafetyClient.guard).parameters:
        pytest.skip("Fallback model API is absent in safety-agent0.1.5")
    _, exporter, _ = runtime

    async def run():
        with fixture_client(retry=True) as (client, requests):
            result = await client.guard(
                input="safe",
                model="openai/fixture-primary",
                fallback_model="openai/gpt-4o-mini",
            )
            assert result.classification == "pass" and len(requests) == 2

    asyncio.run(run())
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].status.status_code.name != "ERROR"
    assert (
        json.loads(spans[0].attributes[A.TRACELOOP_ENTITY_INPUT])["fallback_model"]
        == "openai/gpt-4o-mini"
    )


def test_serialization_failure_does_not_replace_sdk_return(runtime, monkeypatch):
    from respan_instrumentation_superagent import _span_emitter

    monkeypatch.setattr(
        _span_emitter,
        "safe_json_dumps",
        lambda value: (_ for _ in ()).throw(RuntimeError("telemetry-only")),
    )

    async def run():
        with fixture_client() as (client, _):
            result = await client.guard(input="safe", model="openai/gpt-4o-mini")
            assert result.classification == "pass"

    asyncio.run(run())
