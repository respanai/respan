"""Real released SDK lifecycle, output and privacy acceptance."""

import asyncio
import functools
import inspect
import json
from concurrent.futures import ThreadPoolExecutor

import instructor
import pytest
from opentelemetry import context, trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.sampling import ALWAYS_OFF
from opentelemetry.semconv_ai import SpanAttributes as S
from pydantic import BaseModel, Field
from respan_instrumentation_instructor import InstructorInstrumentor
from respan_instrumentation_instructor import _instrumentation as module
from respan_sdk.constants.span_attributes import RESPAN_LOG_TYPE
from respan_tracing.constants.context_constants import ENABLE_CONTENT_TRACING_KEY
from respan_tracing.core.tracer import RespanTracer

from ._fixtures import client_context


class User(BaseModel):
    name: str
    age: int = Field(ge=0)


MESSAGES = [{"role": "user", "content": "Extract Ada, age36."}]


@pytest.fixture(autouse=True)
def cleanup(monkeypatch):
    RespanTracer.reset_instance()
    monkeypatch.delenv("TRACELOOP_TRACE_CONTENT", raising=False)
    yield
    if module._RUNTIME is not None:
        module._RUNTIME.restore()
        module._RUNTIME = None
    RespanTracer.reset_instance()


@pytest.fixture
def spans():
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    inst = InstructorInstrumentor(tracer_provider=provider)
    inst.activate()
    yield provider, exporter, inst
    inst.deactivate()
    provider.shutdown()


def call(client, **kwargs):
    return client.create(
        response_model=User, messages=MESSAGES, model="fixture-model", **kwargs
    )


def only(exporter):
    records = [
        s
        for s in exporter.get_finished_spans()
        if s.attributes.get(RESPAN_LOG_TYPE) == "chat"
    ]
    assert len(records) == 1
    return records[0]


def private(span):
    assert not any(
        k == prefix or k.startswith(prefix + ".")
        for k in span.attributes
        for prefix in module._CONTENT_KEYS
    )


def test_native_output_usage_current_calls_and_parent(spans):
    provider, exporter, _ = spans
    with provider.get_tracer("caller").start_as_current_span("parent") as parent:
        original = trace.get_current_span()
        with client_context() as (client, boundary):
            result = call(client, temperature=0)
        assert result.model_dump() == {"name": "Ada", "age": 36}
        assert trace.get_current_span() is original
    span = only(exporter)
    assert span.parent.span_id == parent.get_span_context().span_id
    assert span.attributes[S.LLM_REQUEST_TEMPERATURE] == 0
    assert span.attributes[S.LLM_USAGE_PROMPT_TOKENS] == 11
    assert span.attributes[S.LLM_USAGE_COMPLETION_TOKENS] == 7
    assert span.attributes[S.LLM_USAGE_TOTAL_TOKENS] == 18
    calls = json.loads(span.attributes[f"{S.LLM_COMPLETIONS}.0.tool_calls"])
    assert calls[0]["id"] == "call_fixture"
    assert json.loads(calls[0]["function"]["arguments"]) == result.model_dump()
    assert json.loads(span.attributes[S.TRACELOOP_ENTITY_OUTPUT]) == result.model_dump()
    assert len(boundary.requests) == 1
    assert not set(span.attributes).intersection(
        {"tools", "tool_calls", "model", "status_code", "traceloop.span.kind"}
    )


@pytest.mark.parametrize("kind", ["environment", "context", "option"])
def test_start_privacy_bound(spans, monkeypatch, kind):
    provider, exporter, inst = spans
    token = None
    if kind == "environment":
        monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")
    elif kind == "context":
        token = context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, False))
    else:
        inst.deactivate()
        inst = InstructorInstrumentor(tracer_provider=provider, trace_content=False)
        inst.activate()
    try:
        with client_context(
            during=lambda: monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "true")
        ) as (client, _):
            call(client)
    finally:
        if token:
            context.detach(token)
        if kind == "option":
            inst.deactivate()
    span = only(exporter)
    private(span)
    assert span.attributes[S.LLM_USAGE_PROMPT_TOKENS] == 11


def test_late_privacy_veto(spans, monkeypatch):
    _, exporter, _ = spans
    with client_context(
        during=lambda: monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")
    ) as (client, _):
        call(client)
    private(only(exporter))


@pytest.mark.parametrize(
    "key",
    [context._SUPPRESS_INSTRUMENTATION_KEY, "suppress_language_model_instrumentation"],
)
def test_suppression_leaves_parent_unchanged(spans, key):
    provider, exporter, _ = spans
    with provider.get_tracer("caller").start_as_current_span("parent") as parent:
        token = context.attach(context.set_value(key, True))
        try:
            with client_context() as (client, _):
                call(client)
            assert trace.get_current_span() is parent
        finally:
            context.detach(token)
    assert len(exporter.get_finished_spans()) == 1


def test_sampler_does_not_serialize(monkeypatch):
    provider = TracerProvider(sampler=ALWAYS_OFF)
    inst = InstructorInstrumentor(tracer_provider=provider)
    inst.activate()

    def forbidden(*args, **kwargs):
        raise AssertionError("nonrecording span serialized content")

    monkeypatch.setattr(module, "_build_span_attributes", forbidden)
    with client_context() as (client, _):
        assert call(client).name == "Ada"
    assert not module._RUNTIME.pending
    inst.deactivate()
    provider.shutdown()


def test_two_owners_preserve_remaining_owner_and_retained_callable(spans):
    provider, exporter, first = spans
    second = InstructorInstrumentor(tracer_provider=provider)
    second.activate()
    with client_context() as (client, _):
        call(client)
        first.deactivate()
        call(client)
        second.deactivate()
        call(client)
    assert len(exporter.get_finished_spans()) == 2


def test_incompatible_owner_settings(spans):
    provider, _, _ = spans
    with pytest.raises(ValueError):
        InstructorInstrumentor(tracer_provider=provider, trace_content=False).activate()
    with pytest.raises(ValueError):
        InstructorInstrumentor(tracer_provider=TracerProvider()).activate()


def test_foreign_method_wrapper_remains_and_retained_owned_wrapper_is_inert(
    spans, monkeypatch
):
    _, exporter, inst = spans
    owned = instructor.Instructor.create

    @functools.wraps(owned)
    def foreign(*args, **kwargs):
        return owned(*args, **kwargs)

    monkeypatch.setattr(instructor.Instructor, "create", foreign)
    inst.deactivate()
    assert instructor.Instructor.create is foreign
    with client_context() as (client, _):
        assert call(client).name == "Ada"
    assert not exporter.get_finished_spans()


def test_partial_activation_rolls_back(monkeypatch):
    original = instructor.Instructor.create
    patch = module._Runtime.patch

    def fail(self, owner, name, wrapped):
        if name == "create_partial":
            raise RuntimeError("controlled patch failure")
        return patch(self, owner, name, wrapped)

    monkeypatch.setattr(module._Runtime, "patch", fail)
    with pytest.raises(RuntimeError):
        InstructorInstrumentor().activate()
    assert instructor.Instructor.create is original
    assert module._RUNTIME is None


def test_source_missing_usage_stays_absent(spans):
    _, exporter, _ = spans
    with client_context(usage=False) as (client, _):
        call(client)
    assert not set(only(exporter).attributes).intersection(module._USAGE_KEYS)


def test_validation_retry_uses_real_aggregate(spans):
    _, exporter, _ = spans
    with client_context(invalid_attempts=1) as (client, boundary):
        assert call(client, max_retries=2).age == 36
    assert len(boundary.requests) == 2
    span = only(exporter)
    assert span.attributes[S.LLM_USAGE_PROMPT_TOKENS] == 22
    assert span.attributes[S.LLM_USAGE_COMPLETION_TOKENS] == 14
    assert span.attributes[S.LLM_USAGE_TOTAL_TOKENS] == 36


def test_native_error_has_no_invented_output_usage_or_http_status(spans):
    _, exporter, _ = spans
    error = RuntimeError("controlled native error")

    def fail(**kwargs):
        raise error

    client = instructor.Instructor(client=None, create=fail)
    with pytest.raises(RuntimeError) as caught:
        call(client)
    assert caught.value is error
    span = only(exporter)
    assert span.status.status_code is trace.StatusCode.ERROR
    assert S.TRACELOOP_ENTITY_OUTPUT not in span.attributes
    assert not set(span.attributes).intersection(module._USAGE_KEYS)
    assert "http.response.status_code" not in span.attributes


def test_cancellation_preserves_exception_identity_and_context(spans):
    provider, exporter, _ = spans
    error = asyncio.CancelledError("controlled cancellation")

    async def fail(**kwargs):
        raise error

    async def execute():
        client = instructor.AsyncInstructor(client=None, create=fail)
        with provider.get_tracer("caller").start_as_current_span("parent") as parent:
            with pytest.raises(asyncio.CancelledError) as caught:
                await call(client)
            assert caught.value is error
            assert trace.get_current_span() is parent

    asyncio.run(execute())
    assert only(exporter).status.status_code is trace.StatusCode.ERROR


def test_never_advanced_and_partial_iterator_close_preserves_native_protocol(spans):
    _, exporter, _ = spans
    observations = []

    def generate(**kwargs):
        value = yield {"name": "Ada"}
        observations.append(value)
        try:
            yield {"name": "Grace"}
        except ValueError:
            yield {"name": "Recovered"}
        return 17

    wrapped = module._RUNTIME.wrap(generate, "instructor.create_iterable")
    before = trace.get_current_span()
    stream = wrapped()
    assert trace.get_current_span() is before
    with ThreadPoolExecutor() as pool:
        assert pool.submit(next, stream).result() == {"name": "Ada"}
    assert stream.send("native-sent") == {"name": "Grace"}
    assert stream.throw(ValueError("native-throw")) == {"name": "Recovered"}
    with pytest.raises(StopIteration) as stopped:
        next(stream)
    assert stopped.value.value == 17
    assert observations == ["native-sent"]
    assert trace.get_current_span() is before
    never = wrapped()
    never.close()
    records = exporter.get_finished_spans()
    assert len(records) == 2
    assert S.TRACELOOP_ENTITY_OUTPUT not in records[-1].attributes


def test_owner_release_finishes_unadvanced_stream_without_consuming(spans):
    _, exporter, inst = spans
    observed = []

    def generate(**kwargs):
        observed.append(True)
        yield "native"

    stream = module._RUNTIME.wrap(generate, "instructor.create_iterable")()
    inst.deactivate()
    assert not observed
    assert S.TRACELOOP_ENTITY_OUTPUT not in only(exporter).attributes
    assert next(stream) == "native"
    stream.close()


def test_actual_async_client_and_tuple_return(spans):
    _, exporter, _ = spans

    async def execute():
        with client_context(async_client=True) as (client, _):
            result, raw = await client.create_with_completion(
                response_model=User, messages=MESSAGES, model="fixture-model"
            )
            assert result.name == "Ada"
            assert raw.id == "chat_fixture"
            await client.client.close()

    asyncio.run(execute())
    assert only(exporter).attributes[S.LLM_USAGE_PROMPT_TOKENS] == 11


def test_source_relocated_v2_factory_and_responses(spans):
    if not hasattr(instructor.Mode, "RESPONSES_TOOLS") or not hasattr(
        instructor, "from_provider"
    ):
        pytest.skip("native minimum SDK predates Responses/provider factory")
    _, exporter, _ = spans
    with client_context(responses=True, from_provider=True) as (client, _):
        result, raw = client.responses.create_with_completion(
            response_model=User, messages=MESSAGES
        )
    assert result.name == "Ada" and raw.id == "resp_fixture"
    span = only(exporter)
    assert span.attributes[S.LLM_USAGE_PROMPT_TOKENS] == 11
    assert span.attributes[S.LLM_USAGE_CACHE_READ_INPUT_TOKENS] == 3
    assert span.attributes[S.GEN_AI_USAGE_REASONING_TOKENS] == 2
    assert (
        json.loads(span.attributes[f"{S.LLM_COMPLETIONS}.0.tool_calls"])[0]["id"]
        == "call_responses_fixture"
    )


def test_token_budget_native_behavior(spans):
    if "token_budget" not in inspect.signature(instructor.Instructor.create).parameters:
        pytest.skip("native minimum SDK predates token budget")
    _, exporter, _ = spans
    from instructor.v2.core.errors import TokenBudgetExceeded

    with (
        client_context(invalid_attempts=10) as (client, boundary),
        pytest.raises(TokenBudgetExceeded) as caught,
    ):
        call(client, max_retries=3, token_budget=1)
    assert type(caught.value).__name__ == "TokenBudgetExceeded"
    assert len(boundary.requests) == 1
    assert only(exporter).status.status_code is trace.StatusCode.ERROR


def test_actual_stream_shape_and_source_usage(spans):
    _, exporter, _ = spans
    before = trace.get_current_span()
    with client_context() as (client, _):
        result = client.create_partial(
            response_model=User, messages=MESSAGES, model="fixture-model"
        )
        assert trace.get_current_span() is before
        items = list(result)
    assert items
    assert trace.get_current_span() is before
    span = only(exporter)
    assert span.attributes[S.GEN_AI_IS_STREAMING] is True
    if hasattr(instructor, "from_provider"):
        assert span.attributes[S.LLM_USAGE_PROMPT_TOKENS] == 11


def test_invalid_source_counts_and_complete_tool_payload(spans):
    _, _, _ = spans
    with client_context() as (client, _):
        call(client)
    provider, _, _ = spans
    with provider.get_tracer("direct-mapping").start_as_current_span("mapping") as span:
        module._set_raw_response_attributes(
            span, {"usage": {"prompt_tokens": True, "completion_tokens": -1}}
        )
        assert not set(span.attributes).intersection(module._USAGE_KEYS)
        payload = {"values": list(range(5000))}
        module._set_raw_response_attributes(
            span,
            {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "id": "actual",
                                    "function": {"name": "large", "arguments": payload},
                                }
                            ]
                        }
                    }
                ]
            },
        )
        calls = json.loads(span.attributes[f"{S.LLM_COMPLETIONS}.0.tool_calls"])
        assert json.loads(calls[0]["function"]["arguments"])["values"][-1] == 4999


def test_telemetry_serialization_failure_preserves_sdk(spans, monkeypatch):
    _, _, _ = spans

    def fail(*args, **kwargs):
        raise ValueError("controlled serializer failure")

    monkeypatch.setattr(module, "safe_json_dumps", fail)
    before = trace.get_current_span()
    with client_context() as (client, _):
        assert call(client).name == "Ada"
    assert trace.get_current_span() is before
    assert not module._RUNTIME.pending


def test_history_ids_and_arguments_are_source_only(spans):
    _, exporter, _ = spans
    history = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "past_call",
                    "type": "function",
                    "function": {"name": "prior", "arguments": {"city": "Paris"}},
                }
            ],
        },
        {"role": "tool", "content": "Paris", "tool_call_id": "past_call"},
        *MESSAGES,
    ]
    with client_context() as (client, _):
        client.create(response_model=User, messages=history, model="fixture-model")
    attrs = only(exporter).attributes
    calls = json.loads(attrs[f"{S.LLM_PROMPTS}.0.tool_calls"])
    assert json.loads(calls[0]["function"]["arguments"]) == {"city": "Paris"}
    assert attrs[f"{S.LLM_PROMPTS}.1.tool_call_id"] == "past_call"
    current = json.loads(attrs[f"{S.LLM_COMPLETIONS}.0.tool_calls"])
    assert [call["id"] for call in current] == ["call_fixture"]


def test_credential_redaction_preserves_ordinary_text(spans):
    from respan_instrumentation_instructor._serialization import safe_json_dumps

    secret = "private phrase with spaces"
    text = f"Basic Example token='{secret}' password=\"{secret}\" https://name:password@fixture.invalid Bearer credential"
    result = safe_json_dumps({"text": text, "token_budget": 1})
    assert secret not in result
    assert "name:password" not in result
    assert "Bearer credential" not in result
    assert "Basic Example" in result
    assert json.loads(result)["token_budget"] == 1


def test_observation_failure_does_not_change_native_iterator(spans, monkeypatch):
    _, exporter, _ = spans

    def native(**kwargs):
        yield {"observed": True}

    stream = module._RUNTIME.wrap(native, "instructor.create_iterable")()

    def fail(*args, **kwargs):
        raise ValueError("controlled item serializer failure")

    with monkeypatch.context() as patch:
        patch.setattr(module, "safe_json_dumps", fail)
        assert next(stream) == {"observed": True}
    stream.close()
    assert S.TRACELOOP_ENTITY_OUTPUT not in only(exporter).attributes


def test_user_completion_hooks_are_preserved(spans):
    if not hasattr(instructor, "from_provider"):
        pytest.skip("native minimum has no completion Hooks API")
    _, exporter, _ = spans
    events = []
    with client_context(invalid_attempts=1) as (client, _):
        client.on("completion:response", events.append)
        assert call(client, max_retries=2).name == "Ada"
    assert len(events) == 2
    assert only(exporter).attributes[S.LLM_USAGE_PROMPT_TOKENS] == 22


def test_complete_actual_structured_tool_arguments(spans):
    class Vectors(BaseModel):
        values: list[float]

    _, exporter, _ = spans
    values = list(range(5000))
    with client_context(payload={"values": values}) as (client, _):
        result = client.create(
            response_model=Vectors, messages=MESSAGES, model="fixture-model"
        )
    assert result.values[-1] == 4999.0
    calls = json.loads(only(exporter).attributes[f"{S.LLM_COMPLETIONS}.0.tool_calls"])
    assert json.loads(calls[0]["function"]["arguments"])["values"] == values


def test_hostile_error_string_does_not_mask_native_exception(spans):
    _, exporter, _ = spans

    class NativeError(RuntimeError):
        def __str__(self):
            raise AssertionError("error string must not run")

    error = NativeError("controlled safe error")

    def fail(**kwargs):
        raise error

    client = instructor.Instructor(client=None, create=fail)
    with pytest.raises(NativeError) as caught:
        call(client)
    assert caught.value is error
    assert only(exporter).status.status_code is trace.StatusCode.ERROR


def test_policy_failure_on_scope_exit_preserves_result_and_context(spans, monkeypatch):
    provider, exporter, _ = spans
    original_policy = module._content_allowed
    switched = []

    def policy():
        if switched:
            raise ValueError("controlled policy failure")
        return original_policy()

    def native(**kwargs):
        switched.append(True)
        return User(name="Ada", age=36)

    monkeypatch.setattr(module, "_content_allowed", policy)
    client = instructor.Instructor(client=None, create=native)
    with provider.get_tracer("caller").start_as_current_span("parent") as parent:
        assert call(client).age == 36
        assert trace.get_current_span() is parent
        assert module._CURRENT_CALL.get() is None
    assert not module._RUNTIME.pending
    private(only(exporter))


def test_failed_validation_retry_preserves_observed_native_aggregate(spans):
    if hasattr(instructor, "from_provider"):
        from instructor.v2.core.errors import InstructorRetryException
    else:
        from instructor.exceptions import InstructorRetryException

    _, exporter, _ = spans
    with (
        client_context(invalid_attempts=10) as (client, boundary),
        pytest.raises(InstructorRetryException) as caught,
    ):
        call(client, max_retries=2)
    expected_attempts = 3 if hasattr(instructor, "from_provider") else 2
    assert len(boundary.requests) == expected_attempts
    attrs = only(exporter).attributes
    assert (
        attrs[S.LLM_USAGE_PROMPT_TOKENS]
        == caught.value.total_usage.prompt_tokens
        == 11 * expected_attempts
    )
    assert attrs[S.LLM_USAGE_COMPLETION_TOKENS] == 7 * expected_attempts
    assert attrs[S.LLM_USAGE_TOTAL_TOKENS] == 18 * expected_attempts
    if hasattr(instructor, "from_provider"):
        assert attrs[S.LLM_USAGE_CACHE_READ_INPUT_TOKENS] == 9
        assert attrs[S.GEN_AI_USAGE_REASONING_TOKENS] == 6
    assert S.TRACELOOP_ENTITY_OUTPUT not in attrs


def test_hostile_callable_signature_preserves_native_patch_result(spans):
    from openai.types.chat import ChatCompletion

    provider, _, _ = spans

    class NativeCallable:
        @property
        def __signature__(self):
            raise RuntimeError("controlled signature introspection failure")

        def __call__(self, **kwargs):
            return ChatCompletion.model_validate(
                {
                    "id": "native_callable",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "fixture-model",
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "tool_calls",
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "native_call",
                                        "type": "function",
                                        "function": {
                                            "name": "User",
                                            "arguments": '{"name":"Ada","age":36}',
                                        },
                                    }
                                ],
                            },
                        }
                    ],
                }
            )

    create = instructor.patch(create=NativeCallable())
    with provider.get_tracer("caller").start_as_current_span("parent") as parent:
        result = create(response_model=User, messages=MESSAGES, model="fixture-model")
        assert result.name == "Ada"
        assert trace.get_current_span() is parent
    assert module._CURRENT_CALL.get() is None
    assert not module._RUNTIME.pending


def test_retained_native_http_status_is_preserved(spans):
    _, exporter, _ = spans
    if hasattr(instructor, "from_provider"):
        from instructor.v2.core.errors import InstructorRetryException
    else:
        from instructor.exceptions import InstructorRetryException
    with (
        client_context(status=401) as (client, _),
        pytest.raises(InstructorRetryException),
    ):
        call(client, max_retries=1)
    assert only(exporter).attributes["http.response.status_code"] == 401
