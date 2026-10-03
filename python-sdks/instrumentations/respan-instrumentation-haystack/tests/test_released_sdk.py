"""Exercise released Haystack objects with the real OI adapter and translator."""

import asyncio
import inspect
import json
from dataclasses import replace
from importlib.metadata import version

import pytest
from haystack import Document, Pipeline, component
from haystack.components.builders import PromptBuilder
from openinference.instrumentation import TraceConfig, suppress_tracing
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.semconv_ai import SpanAttributes
from respan_instrumentation_haystack import HaystackInstrumentor
from respan_sdk.constants.span_attributes import RESPAN_LOG_TYPE

LATEST = int(version("haystack-ai").split(".")[0]) >= 3


@component
class LocalTextEmbedder:
    @component.output_types(embedding=list[float], meta=dict)
    def run(self, text: str):
        return {
            "embedding": [float(i) for i in range(128)],
            "meta": {"usage": {"prompt_tokens": 7, "total_tokens": 7}},
        }

    @component.output_types(embedding=list[float], meta=dict)
    async def run_async(self, text: str):
        return self.run(text)


@component
class LocalDocumentEmbedder:
    @component.output_types(documents=list[Document], meta=dict)
    def run(self, documents: list[Document]):
        documents = [
            replace(doc, embedding=[float(i) for i in range(128)]) for doc in documents
        ]
        return {
            "documents": documents,
            "meta": {"usage": {"prompt_tokens": 9, "total_tokens": 9}},
        }


@pytest.fixture
def runtime(monkeypatch):
    monkeypatch.setenv("HAYSTACK_TELEMETRY_ENABLED", "false")
    provider = TracerProvider()
    monkeypatch.setattr(trace, "_TRACER_PROVIDER", provider)
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    owners = []

    def activate(**kwargs):
        owner = HaystackInstrumentor(**kwargs)
        owner.activate()
        assert owner.is_instrumented
        owners.append(owner)
        return owner

    yield activate, exporter
    for owner in reversed(owners):
        owner.deactivate()
    provider.shutdown()


def _attrs(exporter, kind):
    return [
        dict(span.attributes)
        for span in exporter.get_finished_spans()
        if span.attributes.get(RESPAN_LOG_TYPE) == kind
    ]


def test_released_pipeline_graph_and_async_layout(runtime):
    activate, exporter = runtime
    owner = activate()
    pipeline = Pipeline()
    pipeline.add_component(
        "first", PromptBuilder("{{ name }}", required_variables=["name"])
    )
    pipeline.add_component(
        "second", PromptBuilder("Answer {{ prompt }}", required_variables=["prompt"])
    )
    pipeline.connect("first.prompt", "second.prompt")
    assert (
        pipeline.run({"first": {"name": "fixture"}})["second"]["prompt"]
        == "Answer fixture"
    )
    spans = exporter.get_finished_spans()
    components = [span for span in spans if span.name == "PromptBuilder.run"]
    assert len(components) == 2
    assert components[1].parent.span_id == components[0].context.span_id
    assert all(span.attributes.get(RESPAN_LOG_TYPE) != "chat" for span in components)
    exporter.clear()
    if LATEST:
        async_pipeline = pipeline
    else:
        from haystack import AsyncPipeline

        async_pipeline = AsyncPipeline()
        async_pipeline.add_component(
            "first", PromptBuilder("{{ name }}", required_variables=["name"])
        )
    result = asyncio.run(async_pipeline.run_async({"first": {"name": "async fixture"}}))
    assert result
    assert exporter.get_finished_spans()
    assert not owner._parent_processor._context_by_span_id
    assert not owner._parent_processor._active_by_trace


def test_released_embeddings_full_vectors_reported_usage_and_async_fallback(runtime):
    activate, exporter = runtime
    activate()
    LocalTextEmbedder().run("fixture text")
    asyncio.run(LocalTextEmbedder().run_async("async fixture text"))
    LocalDocumentEmbedder().run([Document(content="one"), Document(content="two")])
    spans = _attrs(exporter, "embedding")
    assert len(spans) == 3
    assert [
        len(json.loads(span[SpanAttributes.TRACELOOP_ENTITY_OUTPUT])) for span in spans
    ] == [1, 1, 2]
    assert all(
        len(vector) == 128
        for span in spans
        for vector in json.loads(span[SpanAttributes.TRACELOOP_ENTITY_OUTPUT])
    )
    assert [span[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] for span in spans] == [7, 7, 9]
    assert all(span[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] in (7, 9) for span in spans)


def test_released_content_opt_out_and_suppression(runtime):
    activate, exporter = runtime
    activate(config=TraceConfig(hide_inputs=True, hide_outputs=True))
    LocalTextEmbedder().run("private embedding fixture")
    rendered = json.dumps(
        [dict(span.attributes) for span in exporter.get_finished_spans()]
    )
    assert "private embedding fixture" not in rendered
    assert "127.0" not in rendered
    assert _attrs(exporter, "embedding")[0][SpanAttributes.LLM_USAGE_PROMPT_TOKENS] == 7
    count = len(exporter.get_finished_spans())
    with suppress_tracing():
        LocalTextEmbedder().run("suppressed")
        asyncio.run(LocalTextEmbedder().run_async("suppressed async"))
    assert len(exporter.get_finished_spans()) == count


def test_released_two_owners_restore_patches_and_late_components(runtime):
    activate, exporter = runtime
    from haystack.core.component.component import component as decorator

    originals = {
        name: inspect.getattr_static(Pipeline, name)
        for name in ("run", "_run_component")
    }
    original_registration = decorator._component
    one, two = activate(), activate()
    one.deactivate()

    @component
    class LateComponent:
        @component.output_types(value=int)
        def run(self, value: int):
            if value < 0:
                raise ValueError("controlled component failure")
            return {"value": value}

    LateComponent().run(42)
    with pytest.raises(ValueError, match="controlled component failure"):
        LateComponent().run(-1)
    assert len(exporter.get_finished_spans()) == 2
    assert (
        exporter.get_finished_spans()[-1].status.status_code is trace.StatusCode.ERROR
    )
    two.deactivate()
    assert decorator._component == original_registration
    assert all(
        getattr(
            inspect.getattr_static(Pipeline, name),
            "__func__",
            inspect.getattr_static(Pipeline, name),
        )
        is getattr(value, "__func__", value)
        for name, value in originals.items()
    )
    count = len(exporter.get_finished_spans())
    LateComponent().run(1)
    assert len(exporter.get_finished_spans()) == count
    activate()
    LateComponent().run(2)
    assert len(exporter.get_finished_spans()) == count + 1


@pytest.mark.skipif(
    not LATEST,
    reason="Native Agent tool executor and MockChatGenerator were introduced in Haystack 3",
)
@pytest.mark.parametrize("asynchronous", [False, True])
def test_released_agent_tool_ids_history_and_usage(runtime, asynchronous):
    from haystack.components.agents import Agent
    from haystack.components.generators.chat import MockChatGenerator
    from haystack.dataclasses import ChatMessage, ToolCall
    from haystack.tools import Tool

    activate, exporter = runtime
    activate()
    model = MockChatGenerator(
        responses=[
            ChatMessage.from_assistant(
                tool_calls=[
                    ToolCall(
                        tool_name="add", arguments={"a": 2, "b": 3}, id="fixture-call"
                    )
                ]
            ),
            "five",
        ],
        meta={
            "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}
        },
    )
    tool = Tool(
        name="add",
        description="Add integers",
        parameters={
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        },
        function=lambda a, b: a + b,
    )
    agent = Agent(chat_generator=model, tools=[tool])
    messages = [ChatMessage.from_user("Add two and three")]
    result = (
        asyncio.run(agent.run_async(messages=messages))
        if asynchronous
        else agent.run(messages=messages)
    )
    assert result["last_message"].text == "five"
    chats, tools, agents = (
        _attrs(exporter, "chat"),
        _attrs(exporter, "tool"),
        _attrs(exporter, "agent"),
    )
    assert (len(chats), len(tools), len(agents)) == (2, 1, 1)
    current = json.loads(chats[0][f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls"])
    history = json.loads(chats[1][f"{SpanAttributes.LLM_PROMPTS}.1.tool_calls"])
    assert current == history
    assert current[0]["id"] == tools[0]["gen_ai.tool.call.id"] == "fixture-call"
    assert json.loads(current[0]["function"]["arguments"]) == {"a": 2, "b": 3}
    assert (
        json.loads(chats[1][f"{SpanAttributes.LLM_PROMPTS}.2.content"])["result"] == "5"
    )
    assert f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls" not in chats[1]
    assert json.loads(tools[0][SpanAttributes.TRACELOOP_ENTITY_OUTPUT]) == 5
    assert all(span[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] == 7 for span in chats)
    assert not any(key.startswith("gen_ai.usage") for key in agents[0])


@pytest.mark.skipif(
    not LATEST, reason="Pipeline.stream and native mock generators require Haystack 3"
)
def test_released_pipeline_stream_preserves_handle_and_result(runtime):
    from haystack.components.generators.chat import MockChatGenerator
    from haystack.dataclasses import ChatMessage

    activate, exporter = runtime
    activate()
    pipeline = Pipeline()
    pipeline.add_component("generator", MockChatGenerator(responses="stream answer"))

    async def consume():
        handle = pipeline.stream(
            {"generator": {"messages": [ChatMessage.from_user("Stream fixture")]}}
        )
        chunks = [chunk async for chunk in handle]
        assert "".join(chunk.content for chunk in chunks) == "stream answer"
        assert handle.result["generator"]["replies"][0].text == "stream answer"

    asyncio.run(consume())
    assert len(_attrs(exporter, "chat")) == 1


def test_released_multi_query_worker_context(runtime):
    MultiQueryTextRetriever = pytest.importorskip(
        "haystack.components.retrievers.multi_query_text_retriever"
    ).MultiQueryTextRetriever
    from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
    from haystack.document_stores.in_memory import InMemoryDocumentStore

    activate, exporter = runtime
    activate()
    store = InMemoryDocumentStore()
    store.write_documents(
        [Document(content="fixture one"), Document(content="fixture two")]
    )
    retriever = MultiQueryTextRetriever(retriever=InMemoryBM25Retriever(store))
    with trace.get_tracer("fixture").start_as_current_span("root"):
        assert retriever.run(queries=["one", "two"])["documents"]
    spans = exporter.get_finished_spans()
    assert len({span.context.trace_id for span in spans}) == 1
    assert all(span.parent is not None for span in spans if span.name != "root")
    ids = {span.context.span_id for span in spans}
    assert all(span.parent.span_id in ids for span in spans if span.parent)


@pytest.mark.skipif(not LATEST, reason="MockChatGenerator requires Haystack 3")
def test_released_granular_text_privacy(runtime):
    from haystack.components.generators.chat import MockChatGenerator
    from haystack.dataclasses import ChatMessage

    activate, exporter = runtime
    activate(config=TraceConfig(hide_input_text=True, hide_output_text=True))
    MockChatGenerator(responses="private reply").run(
        messages=[ChatMessage.from_user("private prompt")]
    )
    serialized = json.dumps(
        [dict(span.attributes) for span in exporter.get_finished_spans()]
    )
    assert "private reply" not in serialized
    assert "private prompt" not in serialized


def test_released_base64_embedding_preserves_all_decoded_dimensions(runtime):
    import base64
    import struct

    activate, exporter = runtime
    activate()

    @component
    class Base64TextEmbedder:
        @component.output_types(embedding=str)
        def run(self, text: str):
            return {
                "embedding": base64.b64encode(
                    struct.pack("<128f", *range(128))
                ).decode()
            }

    result = Base64TextEmbedder().run("base64 fixture")
    assert isinstance(result["embedding"], str)
    spans = _attrs(exporter, "embedding")
    assert json.loads(spans[0][SpanAttributes.TRACELOOP_ENTITY_OUTPUT]) == [
        list(range(128))
    ]


def test_released_async_iterator_context_and_cross_task_close(runtime):
    from respan_instrumentation_haystack._context import _CURRENT_PIPELINE_RUN_CONTEXT

    activate, exporter = runtime
    owner = activate()
    if LATEST:
        pipeline = Pipeline()
    else:
        from haystack import AsyncPipeline

        pipeline = AsyncPipeline()
    pipeline.add_component(
        "prompt", PromptBuilder("{{value}}", required_variables=["value"])
    )

    async def consume():
        original = _CURRENT_PIPELINE_RUN_CONTEXT.get()
        iterator = pipeline.run_async_generator(
            {"prompt": {"value": "iterator fixture"}}, include_outputs_from={"prompt"}
        )
        assert await anext(iterator)
        assert _CURRENT_PIPELINE_RUN_CONTEXT.get() is original
        await asyncio.create_task(iterator.aclose())
        assert _CURRENT_PIPELINE_RUN_CONTEXT.get() is original
        assert not owner._parent_processor._active_by_trace

    asyncio.run(consume())
    assert exporter.get_finished_spans()


@pytest.mark.parametrize("policy", ["environment", "context"])
def test_released_respan_content_policy(runtime, monkeypatch, policy):
    from opentelemetry import context as context_api
    from respan_tracing.constants.context_constants import ENABLE_CONTENT_TRACING_KEY

    activate, exporter = runtime
    activate()
    if policy == "environment":
        monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")
    token = (
        context_api.attach(context_api.set_value(ENABLE_CONTENT_TRACING_KEY, False))
        if policy == "context"
        else None
    )
    try:
        LocalTextEmbedder().run("private policy embedding")
        pipeline = Pipeline()
        pipeline.add_component(
            "prompt",
            PromptBuilder("private policy {{ value }}", required_variables=["value"]),
        )
        pipeline.run({"prompt": {"value": "payload"}})
        if LATEST:
            from haystack.components.generators.chat import MockChatGenerator
            from haystack.dataclasses import ChatMessage

            MockChatGenerator(responses="private policy answer").run(
                messages=[ChatMessage.from_user("private policy prompt")]
            )
    finally:
        if token is not None:
            context_api.detach(token)
    attrs = [dict(span.attributes) for span in exporter.get_finished_spans()]
    assert "private policy" not in json.dumps(attrs)
    assert all(
        SpanAttributes.TRACELOOP_ENTITY_INPUT not in row
        and SpanAttributes.TRACELOOP_ENTITY_OUTPUT not in row
        for row in attrs
    )
    assert _attrs(exporter, "embedding")[0][SpanAttributes.LLM_USAGE_PROMPT_TOKENS] == 7


def test_released_partial_compatibility_install_rolls_back(runtime, monkeypatch):
    from openinference.instrumentation.haystack import _wrappers
    from respan_instrumentation_haystack import _compat

    request = _wrappers._set_component_runner_request_attributes
    response = _wrappers._set_component_runner_response_attributes
    sync = _wrappers._ComponentRunWrapper.__call__
    original_import = _compat.importlib.import_module

    def fail_during_install(name, *args, **kwargs):
        if name == "haystack.components.retrievers.multi_query_text_retriever" and any(
            frame.function == "install_compatibility" for frame in inspect.stack()
        ):
            raise RuntimeError("controlled compatibility installation failure")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(_compat.importlib, "import_module", fail_during_install)
    owner = HaystackInstrumentor()
    owner.activate()
    assert not owner.is_instrumented
    assert _wrappers._set_component_runner_request_attributes is request
    assert _wrappers._set_component_runner_response_attributes is response
    assert _wrappers._ComponentRunWrapper.__call__ is sync
    assert HaystackInstrumentor._owner is None


def test_released_unsampled_embeddings_do_not_retain_vector_cache(runtime):
    from opentelemetry.sdk.trace.sampling import ALWAYS_OFF
    from respan_instrumentation_haystack._compat import _EMBEDDINGS

    activate, exporter = runtime
    trace.get_tracer_provider().sampler = ALWAYS_OFF
    activate()
    for _ in range(3):
        assert len(LocalTextEmbedder().run("dropped fixture")["embedding"]) == 128
    assert not exporter.get_finished_spans()
    assert not _EMBEDDINGS
