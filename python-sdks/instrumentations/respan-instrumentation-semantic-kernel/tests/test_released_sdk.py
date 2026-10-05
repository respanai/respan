"""Released Semantic Kernel and OpenAI models, with only HTTP transport mocked."""

import asyncio
import inspect
import json

import httpx2 as httpx
import pytest
from openai import AsyncOpenAI
from opentelemetry import context, trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.sampling import ALWAYS_OFF
from opentelemetry.semconv._incubating.attributes import gen_ai_attributes as G
from opentelemetry.semconv_ai import SpanAttributes as A
from respan_instrumentation_semantic_kernel import (
    SemanticKernelInstrumentor,
    _instrumentation,
    _native,
)
from respan_sdk.constants.span_attributes import RESPAN_LOG_TYPE
from semantic_kernel import Kernel
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.connectors.ai.function_choice_behavior import (
    FunctionChoiceBehavior,
)
from semantic_kernel.connectors.ai.open_ai import (
    OpenAIChatCompletion,
    OpenAIChatPromptExecutionSettings,
    OpenAITextEmbedding,
)
from semantic_kernel.contents import ChatHistory
from semantic_kernel.exceptions import ServiceResponseException
from semantic_kernel.functions import KernelArguments, kernel_function


class FixtureServer:
    def __init__(self):
        self.requests = []

    async def handle(self, request):
        body = json.loads(request.content)
        self.requests.append(body)
        if "fixture-failure" in json.dumps(body):
            return httpx.Response(
                400,
                json={
                    "error": {
                        "message": "controlled fixture failure",
                        "type": "invalid_request_error",
                        "code": "fixture_failure",
                    }
                },
            )
        if request.url.path.endswith("/embeddings"):
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "model": "fixture-embedding",
                    "data": [
                        {
                            "index": i,
                            "object": "embedding",
                            "embedding": [float(n) for n in range(128)],
                        }
                        for i, _ in enumerate(body["input"])
                    ],
                    "usage": {"prompt_tokens": 7, "total_tokens": 7},
                },
            )
        usage = {
            "prompt_tokens": 9,
            "completion_tokens": 3,
            "total_tokens": 12,
            "prompt_tokens_details": {"cached_tokens": 4},
            "completion_tokens_details": {"reasoning_tokens": 2},
        }
        if body.get("stream"):
            if "empty-stream" in json.dumps(body):
                content = "data: [DONE]\n\n"
            else:
                base = {
                    "id": "fixture-stream",
                    "object": "chat.completion.chunk",
                    "created": 0,
                    "model": "fixture-model",
                }
                chunks = [
                    {
                        **base,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"role": "assistant", "content": "fixture "},
                                "finish_reason": None,
                            }
                        ],
                    },
                    {
                        **base,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": "stream"},
                                "finish_reason": "stop",
                            }
                        ],
                    },
                    {
                        **base,
                        "choices": [],
                        "usage": {
                            "prompt_tokens": 0,
                            "completion_tokens": 0,
                            "total_tokens": 0,
                            "prompt_tokens_details": {"cached_tokens": 0},
                            "completion_tokens_details": {"reasoning_tokens": 0},
                        },
                    },
                ]
                content = (
                    "".join("data: " + json.dumps(chunk) + "\n\n" for chunk in chunks)
                    + "data: [DONE]\n\n"
                )
            return httpx.Response(
                200, text=content, headers={"content-type": "text/event-stream"}
            )
        message = {"role": "assistant", "content": "fixture answer"}
        if body.get("tools") and not any(
            row["role"] == "tool" for row in body["messages"]
        ):
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "fixture-tool-id",
                        "type": "function",
                        "function": {
                            "name": body["tools"][0]["function"]["name"],
                            "arguments": '{"a":2,"b":3}',
                        },
                    }
                ],
            }
        return httpx.Response(
            200,
            json={
                "id": "fixture-response",
                "object": "chat.completion",
                "created": 0,
                "model": "fixture-model",
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": "tool_calls"
                        if message.get("tool_calls")
                        else "stop",
                    }
                ],
                "usage": usage,
            },
        )

    def client(self):
        return AsyncOpenAI(
            api_key="fixture-only",
            base_url="https://fixture.invalid/v1",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(self.handle)),
            max_retries=0,
        )


class Calculator:
    @kernel_function(name="add", description="Add fixture integers")
    def add(self, a: int, b: int) -> int:
        return a + b


@pytest.fixture(scope="module")
def provider():
    previous = trace._TRACER_PROVIDER
    instance = TracerProvider()
    trace._TRACER_PROVIDER = instance
    # SDK module tracers cache their provider. Point the released native modules
    # at the single provider used for this isolated real-SDK test module.
    modules = [
        "semantic_kernel.utils.telemetry.model_diagnostics.decorators",
        "semantic_kernel.utils.telemetry.agent_diagnostics.decorators",
        "semantic_kernel.functions.kernel_function",
        "semantic_kernel.connectors.ai.chat_completion_client_base",
    ]
    import importlib

    originals = []
    for name in modules:
        module = importlib.import_module(name)
        originals.append((module, module.tracer))
        module.tracer = instance.get_tracer(name)
    yield instance
    for module, tracer in originals:
        module.tracer = tracer
    instance.shutdown()
    trace._TRACER_PROVIDER = previous


@pytest.fixture
def runtime(provider):
    exporter = InMemorySpanExporter()
    processor = SimpleSpanProcessor(exporter)
    provider.add_span_processor(processor)
    owners = []

    def activate(**kwargs):
        owner = SemanticKernelInstrumentor(**kwargs)
        owner.activate()
        assert owner._is_instrumented
        owners.append(owner)
        return owner

    yield activate, exporter
    for owner in reversed(owners):
        owner.deactivate()
    provider._active_span_processor._span_processors = tuple(
        p
        for p in provider._active_span_processor._span_processors
        if p is not processor
    )


def spans(exporter, kind):
    return [
        span
        for span in exporter.get_finished_spans()
        if span.attributes.get(RESPAN_LOG_TYPE) == kind
    ]


def test_released_agent_tools_current_history_ids_usage_and_parentage(runtime):
    activate, exporter = runtime
    activate()
    server = FixtureServer()

    async def run():
        async with server.client() as client:
            service = OpenAIChatCompletion(
                ai_model_id="fixture-model", async_client=client
            )
            kernel = Kernel()
            kernel.add_plugin(Calculator(), plugin_name="Calc")
            agent = ChatCompletionAgent(
                service=service,
                kernel=kernel,
                name="FixtureAgent",
                function_choice_behavior=FunctionChoiceBehavior.Auto(),
            )
            result = await agent.get_response(messages="Add fixture integers")
            assert str(result.message) == "fixture answer"

    asyncio.run(run())
    chats, tools, agents = (
        spans(exporter, "chat"),
        spans(exporter, "tool"),
        spans(exporter, "agent"),
    )
    assert (len(chats), len(tools), len(agents)) == (2, 1, 1)
    first, second = [span.attributes for span in chats]
    current = json.loads(first[f"{A.LLM_COMPLETIONS}.0.tool_calls"])
    history = json.loads(second[f"{A.LLM_PROMPTS}.1.tool_calls"])
    assert current == history
    assert (
        current[0]["id"]
        == tools[0].attributes[G.GEN_AI_TOOL_CALL_ID]
        == "fixture-tool-id"
    )
    assert (
        json.loads(second[f"{A.LLM_PROMPTS}.2.content"])["tool_call_id"]
        == "fixture-tool-id"
    )
    assert f"{A.LLM_COMPLETIONS}.0.tool_calls" not in second
    assert json.loads(tools[0].attributes[A.TRACELOOP_ENTITY_OUTPUT]) == 5
    assert all(span.attributes[G.GEN_AI_USAGE_INPUT_TOKENS] == 9 for span in chats)
    assert all(
        span.attributes[A.LLM_USAGE_CACHE_READ_INPUT_TOKENS] == 4 for span in chats
    )
    assert all(span.attributes[A.LLM_USAGE_REASONING_TOKENS] == 2 for span in chats)
    assert "Add fixture integers" in agents[0].attributes[A.TRACELOOP_ENTITY_INPUT]
    assert not any(key.startswith("gen_ai.usage") for key in agents[0].attributes)
    ids = {span.context.span_id for span in exporter.get_finished_spans()}
    assert all(
        span.parent is None or span.parent.span_id in ids
        for span in exporter.get_finished_spans()
    )
    assert spans(exporter, "task")


def test_released_embedding_batch_vectors_usage_errors_and_original_result(runtime):
    activate, exporter = runtime
    activate()
    server = FixtureServer()

    async def run():
        async with server.client() as client:
            service = OpenAITextEmbedding(
                ai_model_id="fixture-embedding", async_client=client
            )
            result = await service.generate_embeddings(
                ["first", "second", "third"], batch_size=2
            )
            assert result.shape == (3, 128)
            assert result[2, 127] == 127
            with pytest.raises(Exception, match="failed to generate embeddings"):
                await service.generate_raw_embeddings(["fixture-failure"])

    asyncio.run(run())
    embeddings = spans(exporter, "embedding")
    assert len(embeddings) == 3
    assert [
        len(json.loads(span.attributes[A.TRACELOOP_ENTITY_OUTPUT]))
        for span in embeddings[:2]
    ] == [2, 1]
    assert all(
        len(vector) == 128
        for span in embeddings[:2]
        for vector in json.loads(span.attributes[A.TRACELOOP_ENTITY_OUTPUT])
    )
    assert all(
        span.attributes[G.GEN_AI_USAGE_INPUT_TOKENS] == 7 for span in embeddings[:2]
    )
    assert embeddings[-1].status.status_code is trace.StatusCode.ERROR
    assert "http.response.status_code" not in embeddings[-1].attributes
    assert _native._EMBEDDING_SPAN.get() is None


@pytest.mark.parametrize("mode", ["capture", "environment", "context"])
def test_released_content_privacy(runtime, monkeypatch, mode):
    from respan_tracing.constants.context_constants import ENABLE_CONTENT_TRACING_KEY

    activate, exporter = runtime
    activate(capture_content=mode != "capture")
    if mode == "environment":
        monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")
    token = (
        context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, False))
        if mode == "context"
        else None
    )
    server = FixtureServer()

    async def run():
        async with server.client() as client:
            service = OpenAIChatCompletion(
                ai_model_id="fixture-model", async_client=client
            )
            agent = ChatCompletionAgent(service=service, name="PrivateAgent")
            await agent.get_response(messages="private kernel prompt")
            await OpenAITextEmbedding(
                ai_model_id="fixture-embedding", async_client=client
            ).generate_embeddings(["private kernel embedding"])

    try:
        asyncio.run(run())
    finally:
        if token is not None:
            context.detach(token)
    attrs = [dict(span.attributes) for span in exporter.get_finished_spans()]
    assert "private kernel" not in json.dumps(attrs)
    assert all(
        A.TRACELOOP_ENTITY_INPUT not in row and A.TRACELOOP_ENTITY_OUTPUT not in row
        for row in attrs
    )
    assert spans(exporter, "chat")[0].attributes[G.GEN_AI_USAGE_INPUT_TOKENS] == 9
    assert spans(exporter, "embedding")[0].attributes[G.GEN_AI_USAGE_INPUT_TOKENS] == 7


@pytest.mark.parametrize("empty", [False, True])
def test_released_streaming_and_zero_usage(runtime, empty):
    activate, exporter = runtime
    activate()
    server = FixtureServer()

    async def run():
        async with server.client() as client:
            service = OpenAIChatCompletion(
                ai_model_id="fixture-model", async_client=client
            )
            history = ChatHistory()
            history.add_user_message("empty-stream" if empty else "Stream fixture")
            chunks = [
                chunk
                async for chunk in service.get_streaming_chat_message_contents(
                    history, OpenAIChatPromptExecutionSettings()
                )
            ]
            if empty:
                assert chunks == []
            else:
                assert (
                    "".join(str(message) for chunk in chunks for message in chunk)
                    == "fixture stream"
                )

    asyncio.run(run())
    chat = spans(exporter, "chat")
    assert len(chat) == 1
    if not empty:
        assert chat[0].attributes[G.GEN_AI_USAGE_INPUT_TOKENS] == 0
        assert chat[0].attributes[G.GEN_AI_USAGE_OUTPUT_TOKENS] == 0


def test_released_two_owners_restore_settings_methods_and_suppression(runtime):
    import importlib

    activate, exporter = runtime
    agent_module = importlib.import_module(
        "semantic_kernel.utils.telemetry.agent_diagnostics.decorators"
    )
    before = (
        agent_module.MODEL_DIAGNOSTICS_SETTINGS.enable_otel_diagnostics,
        agent_module.MODEL_DIAGNOSTICS_SETTINGS.enable_otel_diagnostics_sensitive,
    )
    original = inspect.getattr_static(ChatCompletionAgent, "get_response")
    first, second = activate(), activate()
    first.deactivate()
    server = FixtureServer()

    async def run():
        async with server.client() as client:
            service = OpenAIChatCompletion(
                ai_model_id="fixture-model", async_client=client
            )
            agent = ChatCompletionAgent(service=service, name="SuppressedAgent")
            token = context.attach(
                context.set_value(context._SUPPRESS_INSTRUMENTATION_KEY, True)
            )
            try:
                await agent.get_response(messages="suppressed request")
                await OpenAITextEmbedding(
                    ai_model_id="fixture-embedding", async_client=client
                ).generate_embeddings(["suppressed input"])
                kernel = Kernel()
                kernel.add_plugin(Calculator(), plugin_name="Calc")
                await kernel.invoke(
                    plugin_name="Calc",
                    function_name="add",
                    arguments=KernelArguments(a=2, b=3),
                )
            finally:
                context.detach(token)

    asyncio.run(run())
    assert not exporter.get_finished_spans()
    second.deactivate()
    assert inspect.getattr_static(ChatCompletionAgent, "get_response") is original
    assert (
        agent_module.MODEL_DIAGNOSTICS_SETTINGS.enable_otel_diagnostics,
        agent_module.MODEL_DIAGNOSTICS_SETTINGS.enable_otel_diagnostics_sensitive,
    ) == before


def test_released_activation_failure_rolls_back_native_hooks(runtime, monkeypatch):
    from semantic_kernel.connectors.ai.open_ai.services.open_ai_handler import (
        OpenAIHandler,
    )

    original = OpenAIHandler._send_embedding_request
    monkeypatch.setattr(
        _instrumentation,
        "insert_span_processor_before_export",
        lambda *_: (_ for _ in ()).throw(
            RuntimeError("controlled installation failure")
        ),
    )
    owner = SemanticKernelInstrumentor()
    owner.activate()
    assert not owner._is_instrumented
    assert OpenAIHandler._send_embedding_request is original
    assert _instrumentation._RUNTIME_OWNER is None


def test_released_unsampled_embedding_has_no_retained_state(runtime, monkeypatch):
    activate, _ = runtime
    provider = TracerProvider(sampler=ALWAYS_OFF)
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(trace, "_TRACER_PROVIDER", provider)
    owner = activate()
    server = FixtureServer()

    async def run():
        async with server.client() as client:
            result = await OpenAITextEmbedding(
                ai_model_id="fixture-embedding", async_client=client
            ).generate_embeddings(["unsampled"])
            assert result.shape == (1, 128)

    try:
        asyncio.run(run())
        assert not exporter.get_finished_spans()
        assert _native._EMBEDDING_SPAN.get() is None
        assert owner._processor._content_policy == {}
    finally:
        owner.deactivate()
        provider.shutdown()


def test_released_failed_models_have_no_fabricated_output_or_usage(runtime):
    activate, exporter = runtime
    activate()
    server = FixtureServer()

    async def run():
        async with server.client() as client:
            history = ChatHistory()
            history.add_user_message("fixture-failure")
            with pytest.raises(ServiceResponseException):
                await OpenAIChatCompletion(
                    ai_model_id="fixture-model", async_client=client
                ).get_chat_message_contents(
                    history, OpenAIChatPromptExecutionSettings()
                )
            with pytest.raises(ServiceResponseException):
                await OpenAITextEmbedding(
                    ai_model_id="fixture-embedding", async_client=client
                ).generate_embeddings(["fixture-failure"])

    asyncio.run(run())
    failed = spans(exporter, "chat") + spans(exporter, "embedding")
    assert len(failed) == 2
    for span in failed:
        assert span.status.status_code is trace.StatusCode.ERROR
        assert span.attributes["error.message"]
        assert A.TRACELOOP_ENTITY_OUTPUT not in span.attributes
        assert not any(
            key.startswith(("gen_ai.completion", "gen_ai.usage", "llm.usage"))
            for key in span.attributes
        )


def test_released_privacy_cannot_be_reenabled_mid_request(runtime, monkeypatch, caplog):
    activate, exporter = runtime
    activate()
    monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")
    server = FixtureServer()
    original = server.handle

    async def handle(request):
        monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "true")
        return await original(request)

    server.handle = handle

    async def run():
        async with server.client() as client:
            history = ChatHistory()
            history.add_user_message("private changing prompt")
            await OpenAIChatCompletion(
                ai_model_id="fixture-model", async_client=client
            ).get_chat_message_contents(history, OpenAIChatPromptExecutionSettings())
            monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")
            await OpenAITextEmbedding(
                ai_model_id="fixture-embedding", async_client=client
            ).generate_embeddings(["private changing vector"])

    asyncio.run(run())
    for span in exporter.get_finished_spans():
        assert A.TRACELOOP_ENTITY_INPUT not in span.attributes
        assert A.TRACELOOP_ENTITY_OUTPUT not in span.attributes
        assert not any(
            key.startswith(("gen_ai.prompt.", "gen_ai.completion."))
            for key in span.attributes
        )
    assert not any(record.__dict__.get("event.name") for record in caplog.records)


def test_released_capture_failure_preserves_embedding_result_and_error(
    runtime, monkeypatch
):
    activate, exporter = runtime
    activate()
    monkeypatch.setattr(
        _native,
        "_content_json",
        lambda _: (_ for _ in ()).throw(ValueError("controlled serialization failure")),
    )
    server = FixtureServer()

    async def run():
        async with server.client() as client:
            service = OpenAITextEmbedding(
                ai_model_id="fixture-embedding", async_client=client
            )
            result = await service.generate_embeddings(["safe result"])
            assert result.shape == (1, 128)
            with pytest.raises(Exception, match="controlled fixture failure"):
                await service.generate_embeddings(["fixture-failure"])

    asyncio.run(run())
    assert len(spans(exporter, "embedding")) == 2
    assert _native._EMBEDDING_SPAN.get() is None


def test_released_retained_foreign_wrappers_inert_after_deactivation(
    runtime, monkeypatch
):
    from functools import wraps

    from semantic_kernel.connectors.ai.open_ai.services.open_ai_handler import (
        OpenAIHandler,
    )
    from semantic_kernel.utils.telemetry.agent_diagnostics import (
        decorators as agent_module,
    )
    from semantic_kernel.utils.telemetry.model_diagnostics import function_tracer
    from wrapt import FunctionWrapper

    activate, exporter = runtime
    original_embedding = inspect.getattr_static(
        OpenAIHandler, "_send_embedding_request"
    )
    original_agent = inspect.getattr_static(ChatCompletionAgent, "get_response")
    original_function = function_tracer.start_as_current_span
    owner = activate()
    retained_embedding = OpenAIHandler._send_embedding_request
    retained_function = function_tracer.start_as_current_span
    retained_check = agent_module.are_sensitive_events_enabled

    @wraps(retained_embedding)
    async def foreign_embedding(*args, **kwargs):
        return await retained_embedding(*args, **kwargs)

    @wraps(retained_function)
    def foreign_function(*args, **kwargs):
        return retained_function(*args, **kwargs)

    foreign_agent = FunctionWrapper(
        inspect.getattr_static(ChatCompletionAgent, "get_response"),
        lambda wrapped, instance, args, kwargs: wrapped(*args, **kwargs),
    )
    # A foreign library now owns these attributes; teardown must preserve it.
    OpenAIHandler._send_embedding_request = foreign_embedding
    function_tracer.start_as_current_span = foreign_function
    ChatCompletionAgent.get_response = foreign_agent
    retained_states = [
        patch[0]
        for patch in owner._native_patches
        if isinstance(patch[0], _native._HookState)
    ]
    owner.deactivate()
    assert OpenAIHandler._send_embedding_request is foreign_embedding
    assert function_tracer.start_as_current_span is foreign_function
    assert inspect.getattr_static(ChatCompletionAgent, "get_response") is foreign_agent
    assert retained_states and all(not state.active for state in retained_states)
    server = FixtureServer()

    async def run():
        async with server.client() as client:
            await OpenAITextEmbedding(
                ai_model_id="fixture-embedding", async_client=client
            ).generate_embeddings(["retained wrapper"])
            await ChatCompletionAgent(
                service=OpenAIChatCompletion(
                    ai_model_id="fixture-model", async_client=client
                ),
                name="RetainedAgent",
            ).get_response(messages="retained agent")
            kernel = Kernel()
            kernel.add_plugin(Calculator(), plugin_name="Calc")
            assert (
                str(
                    await kernel.invoke(
                        plugin_name="Calc",
                        function_name="add",
                        arguments=KernelArguments(a=2, b=3),
                    )
                )
                == "5"
            )

    try:
        asyncio.run(run())
        assert all(
            RESPAN_LOG_TYPE not in span.attributes
            for span in exporter.get_finished_spans()
        )
        assert not any(
            span.name == "embeddings" for span in exporter.get_finished_spans()
        )
        exporter.clear()
        monkeypatch.setattr(
            agent_module.MODEL_DIAGNOSTICS_SETTINGS,
            "enable_otel_diagnostics_sensitive",
            True,
        )
        token = context.attach(
            context.set_value(context._SUPPRESS_INSTRUMENTATION_KEY, True)
        )
        try:
            assert retained_check() is True
        finally:
            context.detach(token)
        next_owner = activate()
        asyncio.run(run())
        assert len(spans(exporter, "embedding")) == 1
        assert len(spans(exporter, "agent")) == 1
        assert len(spans(exporter, "chat")) == 1
        assert len(spans(exporter, "tool")) == 1
        next_owner.deactivate()
        assert OpenAIHandler._send_embedding_request is foreign_embedding
    finally:
        OpenAIHandler._send_embedding_request = original_embedding
        function_tracer.start_as_current_span = original_function
        ChatCompletionAgent.get_response = original_agent


def test_released_partial_native_installation_rolls_back(runtime, monkeypatch):
    from semantic_kernel.utils.telemetry.model_diagnostics import (
        decorators,
        function_tracer,
    )

    original_span = function_tracer.start_as_current_span
    original_response = decorators._set_completion_response
    original_import = _native.importlib.import_module

    def controlled_import(name, *args, **kwargs):
        if name == "semantic_kernel.connectors.ai.open_ai.services.open_ai_handler":
            raise RuntimeError("controlled mid-install failure")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(_native.importlib, "import_module", controlled_import)
    owner = SemanticKernelInstrumentor()
    owner.activate()
    assert not owner._is_instrumented
    assert _instrumentation._RUNTIME_OWNER is None
    assert function_tracer.start_as_current_span is original_span
    assert decorators._set_completion_response is original_response


def test_released_custom_agent_keyword_only_messages_preserves_call(runtime):
    from semantic_kernel.utils.telemetry.agent_diagnostics.decorators import (
        trace_agent_get_response,
    )

    class KeywordOnlyAgent(ChatCompletionAgent):
        @trace_agent_get_response
        async def get_response(self, *, messages=None, **kwargs):
            return await super().get_response(messages=messages, **kwargs)

    activate, exporter = runtime
    activate()
    server = FixtureServer()

    async def run():
        async with server.client() as client:
            agent = KeywordOnlyAgent(
                service=OpenAIChatCompletion(
                    ai_model_id="fixture-model", async_client=client
                ),
                name="KeywordOnlyAgent",
            )
            result = await agent.get_response(messages="keyword only request")
            assert str(result.message) == "fixture answer"

    asyncio.run(run())
    assert len(spans(exporter, "chat")) == 1
