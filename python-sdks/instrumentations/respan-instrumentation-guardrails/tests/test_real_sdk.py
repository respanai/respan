"""Exercise released Guardrails with real native OTel spans and local fixtures."""

import asyncio
import json

import pytest
from guardrails import AsyncGuard, Guard
from guardrails.settings import settings
from guardrails.validators import FailResult, PassResult, Validator, register_validator
from opentelemetry import context, trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.semconv_ai import SpanAttributes
from pydantic import BaseModel
from respan_instrumentation_guardrails import GuardrailsInstrumentor
from respan_sdk.constants.span_attributes import RESPAN_LOG_TYPE
from respan_tracing.core.tracer import RespanTracer


class Ticket(BaseModel):
    answer: str
    priority: str


@register_validator(name="respan-fixture-validator", data_type="string")
class FixtureValidator(Validator):
    def validate(self, value, metadata):
        if value == "fixture accepted":
            return PassResult()
        return FailResult(
            error_message="Fixture rejected", fix_value="fixture accepted"
        )


@pytest.fixture
def telemetry(monkeypatch):
    RespanTracer.reset_instance()
    monkeypatch.setattr(settings.rc, "enable_metrics", False)
    monkeypatch.setattr(settings, "disable_tracing", False)
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(trace, "get_tracer_provider", lambda: provider)
    plugin = GuardrailsInstrumentor()
    plugin.activate()
    yield provider, exporter, plugin
    for processor, owners in list(GuardrailsInstrumentor._providers.values()):
        for owner in list(owners):
            owner.deactivate()
    provider.shutdown()
    RespanTracer.reset_instance()


def make_guard(cls=Guard):
    guard = cls.for_pydantic(Ticket)
    guard.configure(allow_metrics_collection=False)
    return guard


def spans(exporter):
    return [
        s for s in exporter.get_finished_spans() if s.attributes.get(RESPAN_LOG_TYPE)
    ]


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("valid", [False, True])
def test_parse_success_failure_native_tree(telemetry, async_mode, valid):
    provider, exporter, _ = telemetry
    guard = make_guard(AsyncGuard if async_mode else Guard)
    payload = (
        '{"answer":"fixture answer","priority":"high"}'
        if valid
        else '{"answer":"fixture answer"}'
    )
    with provider.get_tracer("fixture").start_as_current_span("root") as root:
        result = guard.parse(payload, num_reasks=0)
        if async_mode:
            result = asyncio.run(result)
    assert result.validation_passed is valid
    emitted = spans(exporter)
    assert len(emitted) == 3
    assert {s.attributes[RESPAN_LOG_TYPE] for s in emitted} == {"guardrail"}
    guard_span = next(s for s in emitted if s.name == "guard")
    assert guard_span.parent.span_id == root.context.span_id
    assert guard_span.attributes["validation_passed"] is valid
    assert guard_span.attributes["number_of_llm_calls"] == 0
    assert all("input.value" not in s.attributes for s in emitted)
    assert all("openinference.span.kind" not in s.attributes for s in emitted)


@pytest.mark.parametrize("async_mode", [False, True])
def test_custom_generation_keeps_actual_messages_and_call_count(telemetry, async_mode):
    _, exporter, _ = telemetry
    guard = make_guard(AsyncGuard if async_mode else Guard)

    def generate(**kwargs):
        return '{"answer":"fixture generated","priority":"high"}'

    async def agenerate(**kwargs):
        return generate(**kwargs)

    result = guard(
        llm_api=agenerate if async_mode else generate,
        messages=[{"role": "user", "content": "actual fixture prompt"}],
        num_reasks=0,
    )
    if async_mode:
        result = asyncio.run(result)
    assert result.validation_passed
    emitted = spans(exporter)
    chat = next(s for s in emitted if s.attributes[RESPAN_LOG_TYPE] == "chat")
    assert (
        chat.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"]
        == "actual fixture prompt"
    )
    assert chat.attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"] == generate()
    assert SpanAttributes.LLM_USAGE_TOTAL_TOKENS not in chat.attributes
    guard_span = next(s for s in emitted if s.name == "guard")
    assert guard_span.attributes["number_of_llm_calls"] == 1


@pytest.mark.parametrize("on_fail,expected", [("noop", False), ("fix", True)])
def test_registered_validator_outcomes(telemetry, on_fail, expected):
    _, exporter, _ = telemetry
    guard = Guard().use(FixtureValidator(on_fail=on_fail))
    guard.configure(allow_metrics_collection=False)
    result = guard.validate("fixture rejected", num_reasks=0)
    assert result.validation_passed is expected
    validator = next(
        s
        for s in spans(exporter)
        if s.attributes.get("validator.name") == "respan-fixture-validator"
    )
    assert validator.attributes[RESPAN_LOG_TYPE] == "guardrail"
    output = json.loads(validator.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT])
    assert output["outcome"] == "fail"
    assert "validator.validate.output.error_message" not in validator.attributes


def test_exception_preserves_native_error(telemetry):
    _, exporter, _ = telemetry
    guard = Guard().use(FixtureValidator(on_fail="exception"))
    guard.configure(allow_metrics_collection=False)
    with pytest.raises(Exception, match="Fixture rejected"):
        guard.validate("fixture rejected", num_reasks=0)
    assert any(s.status.status_code is trace.StatusCode.ERROR for s in spans(exporter))


def test_content_opt_out_includes_validator_payloads(telemetry, monkeypatch):
    _, exporter, _ = telemetry
    monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")
    guard = Guard().use(FixtureValidator(on_fail="noop"))
    guard.configure(allow_metrics_collection=False)
    guard.validate(
        "private fixture sentinel",
        metadata={"private": "hidden metadata"},
        num_reasks=0,
    )
    emitted = spans(exporter)
    assert emitted
    for span in emitted:
        assert SpanAttributes.TRACELOOP_ENTITY_INPUT not in span.attributes
        assert SpanAttributes.TRACELOOP_ENTITY_OUTPUT not in span.attributes
        assert "private fixture sentinel" not in json.dumps(dict(span.attributes))
        assert "hidden metadata" not in json.dumps(dict(span.attributes))


def test_content_override_and_shared_owners(telemetry, monkeypatch):
    _, exporter, first = telemetry
    second = GuardrailsInstrumentor()
    second.activate()
    first.deactivate()
    monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")
    token = context.attach(context.set_value("override_enable_content_tracing", True))
    try:
        make_guard().parse(
            '{"answer":"override fixture","priority":"high"}', num_reasks=0
        )
    finally:
        context.detach(token)
    assert len(spans(exporter)) == 3
    assert any("override fixture" in str(s.attributes) for s in spans(exporter))
    second.deactivate()
    exporter.clear()
    make_guard().parse('{"answer":"untraced fixture","priority":"high"}', num_reasks=0)
    assert spans(exporter) == []


def test_foreign_span_type_is_untouched(telemetry):
    provider, exporter, _ = telemetry
    with provider.get_tracer("unrelated").start_as_current_span("call") as span:
        span.set_attribute("input.value", "unrelated payload")
    recorded = exporter.get_finished_spans()[0]
    assert dict(recorded.attributes) == {"input.value": "unrelated payload"}


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("stream", [False, True])
def test_released_litellm_fixture_generation(
    telemetry, monkeypatch, async_mode, stream
):
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    _, exporter, _ = telemetry
    guard = make_guard(AsyncGuard if async_mode else Guard)
    result = guard(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "fixture model prompt"}],
        mock_response='{"answer":"model fixture","priority":"high"}',
        stream=stream,
        num_reasks=0,
    )

    async def consume():
        value = await result
        return [chunk async for chunk in value] if stream else value

    if async_mode:
        result = asyncio.run(consume())
    elif stream:
        result = list(result)
    if stream:
        assert result[-1].validation_passed
        assert result[-1].validated_output["answer"] == "model fixture"
    else:
        assert result.validated_output["answer"] == "model fixture"
    emitted = spans(exporter)
    chat = next(s for s in emitted if s.attributes[RESPAN_LOG_TYPE] == "chat")
    assert chat.attributes[SpanAttributes.LLM_REQUEST_MODEL] == "gpt-4o-mini"
    assert (
        chat.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"]
        == "fixture model prompt"
    )
    if not stream:
        assert chat.attributes[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] > 0
        assert chat.attributes[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS] > 0
        assert (
            chat.attributes["gen_ai.usage.input_tokens"]
            == chat.attributes[SpanAttributes.LLM_USAGE_PROMPT_TOKENS]
        )
        assert (
            chat.attributes["gen_ai.usage.output_tokens"]
            == chat.attributes[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS]
        )
    assert all("input.value" not in s.attributes for s in emitted)
    if stream:
        summaries = [
            s
            for s in emitted
            if s.links
            and s.attributes.get(SpanAttributes.TRACELOOP_ENTITY_NAME)
            == "guardrails.guard"
        ]
        assert summaries
        assert all(s.attributes["number_of_llm_calls"] == 1 for s in summaries)
        assert SpanAttributes.TRACELOOP_ENTITY_OUTPUT not in chat.attributes


def test_reask_counts_llm_calls_without_usage(telemetry):
    _, exporter, _ = telemetry
    responses = iter(
        ['{"answer":"needs repair"}', '{"answer":"fixed","priority":"high"}']
    )
    guard = make_guard()
    result = guard(
        llm_api=lambda *, messages, **kwargs: next(responses),
        messages=[{"role": "user", "content": "fixture reask"}],
        num_reasks=1,
    )
    assert result.validation_passed
    emitted = spans(exporter)
    assert sum(s.attributes[RESPAN_LOG_TYPE] == "chat" for s in emitted) == 2
    guard_span = next(s for s in emitted if s.name == "guard")
    assert guard_span.attributes["number_of_llm_calls"] == 2
    assert guard_span.attributes["number_of_reasks"] == 1


@pytest.mark.parametrize("override", [False, True])
def test_runtime_content_disable_wins_over_override(telemetry, override):
    from respan_tracing.constants.context_constants import ENABLE_CONTENT_TRACING_KEY

    _, exporter, _ = telemetry
    ctx = context.set_value(ENABLE_CONTENT_TRACING_KEY, False)
    ctx = context.set_value("override_enable_content_tracing", override, ctx)
    token = context.attach(ctx)
    try:
        make_guard().parse(
            '{"answer":"runtime private sentinel","priority":"high"}', num_reasks=0
        )
    finally:
        context.detach(token)
    assert spans(exporter)
    assert all(
        "runtime private sentinel" not in str(s.attributes) for s in spans(exporter)
    )
    assert all(
        SpanAttributes.TRACELOOP_ENTITY_INPUT not in s.attributes
        for s in spans(exporter)
    )


def test_explicit_parent_context_privacy_and_foreign_scope(telemetry, monkeypatch):
    from respan_instrumentation_guardrails import _instrumentation
    from respan_tracing.constants.context_constants import ENABLE_CONTENT_TRACING_KEY

    provider, exporter, _ = telemetry
    monkeypatch.setattr(
        _instrumentation,
        "read_propagated_attributes",
        lambda: {"respan.metadata.fixture": "scope-check"},
    )
    ctx = context.set_value(ENABLE_CONTENT_TRACING_KEY, False)
    with provider.get_tracer("guardrails-ai").start_as_current_span(
        "call", context=ctx
    ) as span:
        span.set_attribute("input.value", "explicit private sentinel")
    with provider.get_tracer("unrelated").start_as_current_span("call") as span:
        span.set_attribute("input.value", "unrelated payload")
    first, second = exporter.get_finished_spans()
    assert SpanAttributes.TRACELOOP_ENTITY_INPUT not in first.attributes
    assert first.attributes["respan.metadata.fixture"] == "scope-check"
    assert dict(second.attributes) == {"input.value": "unrelated payload"}
