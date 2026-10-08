"""Released LangChain callback/protocol regressions; providers are controlled."""

import asyncio
import json
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest
import respan_instrumentation_langchain._instrumentation as instrumentation
from langchain_core.callbacks import BaseCallbackHandler, CallbackManager
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.llms import LLM
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langchain_core.outputs import (
    ChatGeneration,
    ChatGenerationChunk,
    ChatResult,
    GenerationChunk,
    LLMResult,
)
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_tool
from opentelemetry import context, trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.sampling import ALWAYS_OFF
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import (
    GEN_AI_TOOL_CALL_ID,
)
from opentelemetry.semconv_ai import (
    SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY,
    SpanAttributes,
)
from opentelemetry.trace import StatusCode
from respan_instrumentation_langchain import (
    LangChainInstrumentor,
    RespanCallbackHandler,
    add_respan_callback,
)
from respan_instrumentation_langchain._serialization import json_value
from respan_sdk.constants.span_attributes import RESPAN_LOG_TYPE
from respan_tracing.constants.context_constants import ENABLE_CONTENT_TRACING_KEY

INPUT = SpanAttributes.TRACELOOP_ENTITY_INPUT
OUTPUT = SpanAttributes.TRACELOOP_ENTITY_OUTPUT


class ControlledChat(BaseChatModel):
    model_name: str = "fixture-chat"
    response: AIMessage = AIMessage(
        content="answer",
        usage_metadata={
            "input_tokens": 11,
            "output_tokens": 7,
            "total_tokens": 18,
            "input_token_details": {"cache_read": 3},
            "output_token_details": {"reasoning": 2},
        },
    )
    fail: bool = False

    @property
    def _llm_type(self):
        return "fixture-chat"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        if self.fail:
            raise ValueError("provider failed")
        return ChatResult(generations=[ChatGeneration(message=self.response)])

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        yield ChatGenerationChunk(message=AIMessageChunk(content="an"))
        if self.fail:
            raise ValueError("stream failed")
        yield ChatGenerationChunk(
            message=AIMessageChunk(
                content="swer", usage_metadata=self.response.usage_metadata
            )
        )

    def bind_tools(self, tools, **kwargs):
        return self.bind(tools=[convert_to_openai_tool(t) for t in tools], **kwargs)


class ControlledLLM(LLM):
    @property
    def _llm_type(self):
        return "fixture-text"

    def _call(self, prompt, stop=None, run_manager=None, **kwargs):
        return "text answer"

    def _stream(self, prompt, stop=None, run_manager=None, **kwargs):
        for value in ("text ", "answer"):
            if run_manager:
                run_manager.on_llm_new_token(value)
            yield GenerationChunk(text=value)


class Retriever(BaseRetriever):
    def _get_relevant_documents(self, query, *, run_manager):
        if query == "fail":
            raise ValueError("retriever failed")
        return [
            Document(
                page_content="actual document", metadata={"vector": list(range(5000))}
            )
        ]


@tool
def weather(city: str) -> str:
    """Get controlled weather."""
    if city == "fail":
        raise ValueError("tool failed")
    return f"sunny {city}"


@pytest.fixture
def spans(monkeypatch):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(trace, "_TRACER_PROVIDER", provider)
    monkeypatch.delenv("TRACELOOP_TRACE_CONTENT", raising=False)
    monkeypatch.setattr(instrumentation.RespanTracer, "_instance", None)
    yield exporter
    if instrumentation._RUNTIME:
        instrumentation._RUNTIME.restore()
        instrumentation._RUNTIME = None
    provider.shutdown()


def config(handler=None):
    return add_respan_callback(
        {
            "metadata": {
                "respan_params": {
                    "metadata": {"run_id": "test-run", "secret": "private metadata"}
                }
            }
        },
        handler,
    )


def chat_span(spans):
    return next(
        s for s in spans.get_finished_spans() if s.attributes[RESPAN_LOG_TYPE] == "chat"
    )


def no_content(span):
    assert INPUT not in span.attributes and OUTPUT not in span.attributes
    assert not any(
        k.startswith(
            ("gen_ai.prompt.", "gen_ai.completion.", "llm.prompts.", "llm.completions.")
        )
        for k in span.attributes
    )
    assert "private" not in json.dumps(dict(span.attributes))


def test_native_chain_parent_and_return(spans):
    value = {"v": "actual"}
    chain = RunnableLambda(lambda x: x) | RunnableLambda(lambda x: x)
    assert chain.invoke(value, config=config()) is value
    finished = spans.get_finished_spans()
    assert len(finished) == 3
    root = next(s for s in finished if s.parent is None)
    assert root.attributes[RESPAN_LOG_TYPE] == "workflow"
    assert all(
        s.parent.span_id == root.context.span_id for s in finished if s is not root
    )
    assert json.loads(root.attributes[INPUT]) == value


def test_actual_chat_usage(spans):
    response = ControlledChat().invoke("question", config=config())
    assert response.content == "answer"
    attrs = chat_span(spans).attributes
    assert (
        attrs[SpanAttributes.LLM_USAGE_PROMPT_TOKENS],
        attrs[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS],
        attrs[SpanAttributes.LLM_USAGE_TOTAL_TOKENS],
    ) == (11, 7, 18)
    assert attrs[SpanAttributes.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS] == 3
    assert attrs[SpanAttributes.GEN_AI_USAGE_REASONING_TOKENS] == 2
    assert json.loads(attrs[OUTPUT])["messages"][0]["content"] == "answer"


@pytest.mark.parametrize("method", ["stream", "batch", "batch_as_completed"])
def test_sync_model_protocol(spans, method):
    model = ControlledChat()
    cfg = config()
    if method == "stream":
        assert (
            "".join(chunk.content for chunk in model.stream("q", config=cfg))
            == "answer"
        )
    elif method == "batch":
        assert [r.content for r in model.batch(["q", "r"], config=cfg)] == [
            "answer",
            "answer",
        ]
    else:
        assert sorted(
            i for i, r in model.batch_as_completed(["q", "r"], config=cfg)
        ) == [0, 1]
    assert all(
        s.attributes[RESPAN_LOG_TYPE] == "chat" for s in spans.get_finished_spans()
    )


@pytest.mark.parametrize(
    "method", ["ainvoke", "astream", "abatch", "abatch_as_completed", "astream_events"]
)
def test_async_model_protocol(spans, method):
    async def run():
        model = ControlledChat()
        cfg = config()
        if method == "ainvoke":
            assert (await model.ainvoke("q", config=cfg)).content == "answer"
        elif method == "astream":
            assert (
                "".join([c.content async for c in model.astream("q", config=cfg)])
                == "answer"
            )
        elif method == "abatch":
            assert len(await model.abatch(["q", "r"], config=cfg)) == 2
        elif method == "abatch_as_completed":
            assert (
                len(
                    [r async for r in model.abatch_as_completed(["q", "r"], config=cfg)]
                )
                == 2
            )
        else:
            assert any(
                e["event"] == "on_chat_model_end"
                for e in [
                    e async for e in model.astream_events("q", config=cfg, version="v2")
                ]
            )

    asyncio.run(run())
    assert spans.get_finished_spans()


@pytest.mark.parametrize("stream", [False, True])
def test_native_text_llm(spans, stream):
    model = ControlledLLM()
    cfg = config()
    assert (
        "".join(model.stream("question", config=cfg))
        if stream
        else model.invoke("question", config=cfg)
    ) == "text answer"
    span = spans.get_finished_spans()[0]
    assert span.attributes[RESPAN_LOG_TYPE] == "text"
    assert not any("usage" in k for k in span.attributes)


def test_tool_actual_id_arguments_and_name(spans):
    call = {
        "name": "weather",
        "args": {"city": "Paris"},
        "id": "actual-call-id",
        "type": "tool_call",
    }
    response = weather.invoke(
        call, config={**config(), "run_name": "custom display name"}
    )
    assert (
        isinstance(response, ToolMessage) and response.tool_call_id == "actual-call-id"
    )
    attrs = spans.get_finished_spans()[0].attributes
    assert json.loads(attrs[INPUT]) == {
        "name": "weather",
        "arguments": {"city": "Paris"},
    }
    assert json.loads(attrs[OUTPUT]) == "sunny Paris"
    assert attrs[GEN_AI_TOOL_CALL_ID] == "actual-call-id"
    assert not any(
        k in attrs
        for k in (
            "gen_ai.tool.name",
            "gen_ai.tool.call.arguments",
            "gen_ai.tool.call.result",
        )
    )


def test_schema_current_calls_history_and_full_payload(spans):
    calls = [
        {
            "name": "weather",
            "args": {"city": "x" * 20000, "vector": list(range(5000))},
            "id": "c" * 600 + str(i),
        }
        for i in range(120)
    ]
    model = ControlledChat(response=AIMessage(content="", tool_calls=calls)).bind_tools(
        [weather]
    )
    history = AIMessage(
        content="", tool_calls=[{"name": "old", "args": {"x": 1}, "id": "history"}]
    )
    result = model.invoke(
        [
            history,
            ToolMessage(content="z" * 35000, tool_call_id="history"),
            HumanMessage(content="q"),
        ],
        config=config(),
    )
    attrs = chat_span(spans).attributes
    output = json.loads(attrs[OUTPUT])["messages"][0]["tool_calls"]
    assert len(output) == 120 and output[-1]["id"] == calls[-1]["id"]
    assert len(json.loads(output[-1]["function"]["arguments"])["vector"]) == 5000
    assert isinstance(output[-1]["function"]["arguments"], str)
    assert result.tool_calls == [{**call, "type": "tool_call"} for call in calls]
    assert (
        json.loads(attrs[SpanAttributes.LLM_REQUEST_FUNCTIONS])[0]["function"]["name"]
        == "weather"
    )
    assert len(json.loads(attrs[INPUT])[0][1]["content"]) == 35000
    assert len(attrs[f"{SpanAttributes.LLM_PROMPTS}.1.content"]) == 35000
    assert "history" not in [v["id"] for v in output]


def test_full_tool_artifact_and_retriever_vectors(spans):
    @tool(response_format="content_and_artifact")
    def vector_tool(x: str) -> tuple:
        """Return a full vector artifact."""
        return "v" * 35000, {"vector": list(range(5000))}

    result = vector_tool.invoke(
        {
            "name": "vector_tool",
            "args": {"x": "q"},
            "id": "vector-call",
            "type": "tool_call",
        },
        config=config(),
    )
    assert len(result.artifact["vector"]) == 5000
    assert (
        len(
            json.loads(spans.get_finished_spans()[0].attributes[OUTPUT])["artifact"][
                "vector"
            ]
        )
        == 5000
    )
    Retriever().invoke("q", config=config())
    assert (
        len(
            json.loads(spans.get_finished_spans()[-1].attributes[OUTPUT])[0][
                "metadata"
            ]["vector"]
        )
        == 5000
    )


@pytest.mark.parametrize("policy", ["env", "context", "option"])
def test_initial_content_upper_bound(spans, monkeypatch, policy):
    token = None
    if policy == "env":
        monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")
    elif policy == "context":
        token = context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, False))
    try:
        ControlledChat().invoke(
            "private prompt",
            config=config(RespanCallbackHandler(include_content=policy != "option")),
        )
    finally:
        if token:
            context.detach(token)
    no_content(chat_span(spans))
    assert chat_span(spans).attributes[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] == 11


@pytest.mark.parametrize("policy", ["env", "context"])
def test_end_veto_is_irreversible(spans, monkeypatch, policy):
    handler = RespanCallbackHandler()
    rid = uuid4()
    handler.on_chain_start({}, "private input", run_id=rid)
    token = None
    if policy == "env":
        monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")
    else:
        token = context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, False))
    handler.on_text("private token", run_id=rid)
    if token:
        context.detach(token)
    else:
        monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "true")
    handler.on_chain_end("private output", run_id=rid)
    no_content(spans.get_finished_spans()[0])


def test_private_parent_and_completed_parent_bound(spans, monkeypatch):
    handler = RespanCallbackHandler()
    root = uuid4()
    monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")
    handler.on_chain_start({}, "private root", run_id=root)
    monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "true")
    RunnableLambda(lambda x: x).invoke(
        "private child", config={"callbacks": [handler], "parent_run_id": root}
    )
    child = uuid4()
    handler.on_chain_start({}, "private child", run_id=child, parent_run_id=root)
    handler.on_chain_end("private", run_id=child)
    handler.on_chain_end("private", run_id=root)
    later = uuid4()
    handler.on_chain_start({}, "private later", run_id=later, parent_run_id=root)
    handler.on_chain_end("private", run_id=later)
    related = [
        s
        for s in spans.get_finished_spans()
        if s.attributes.get("langchain.parent_run_id") == root.hex
        or s.attributes.get("langchain.run_id") == root.hex
    ]
    assert len(related) == 3
    for span in related:
        no_content(span)
    assert all(not hasattr(v[0], "attributes") for v in handler._parents.values())


@pytest.mark.parametrize(
    "key",
    [
        context._SUPPRESS_INSTRUMENTATION_KEY,
        SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY,
    ],
)
def test_suppression(spans, key):
    token = context.attach(context.set_value(key, True))
    try:
        ControlledChat().invoke("private", config=config())
    finally:
        context.detach(token)
    assert spans.get_finished_spans() == ()


def test_sampler_honored_without_serializer(spans, monkeypatch):
    provider = TracerProvider(sampler=ALWAYS_OFF)
    monkeypatch.setattr(trace, "_TRACER_PROVIDER", provider)

    class Unsafe:
        def __repr__(self):
            raise AssertionError("repr called")

        def model_dump(self, *a, **k):
            raise AssertionError("serializer called")

    value = Unsafe()
    assert RunnableLambda(lambda x: x).invoke(value, config=config()) is value
    assert spans.get_finished_spans() == ()
    provider.shutdown()


@pytest.mark.parametrize("kind", ["chain", "chat", "tool", "retriever", "stream"])
def test_original_failures_no_invented_output(spans, kind):
    cfg = config()
    with pytest.raises(ValueError):
        if kind == "chain":
            RunnableLambda(
                lambda x: (_ for _ in ()).throw(ValueError("actual failure"))
            ).invoke("q", config=cfg)
        elif kind == "chat":
            ControlledChat(fail=True).invoke("q", config=cfg)
        elif kind == "stream":
            list(ControlledChat(fail=True).stream("q", config=cfg))
        elif kind == "tool":
            weather.invoke({"city": "fail"}, config=cfg)
        else:
            Retriever().invoke("fail", config=cfg)
    for span in spans.get_finished_spans():
        assert span.status.status_code == StatusCode.ERROR
        assert OUTPUT not in span.attributes
        assert "http.status_code" not in span.attributes
        assert not any("usage" in key for key in span.attributes)


def test_hostile_error_str_never_used(spans):
    class Bad(ValueError):
        def __str__(self):
            raise AssertionError("str called")

    handler = RespanCallbackHandler()
    rid = uuid4()
    error = Bad("actual error")
    handler.on_chain_start({}, "input", run_id=rid)
    handler.on_chain_error(error, run_id=rid)
    span = spans.get_finished_spans()[0]
    assert span.status.status_code == StatusCode.ERROR
    assert span.events[0].attributes["exception.message"] == "actual error"


def test_context_across_native_stream_yields(spans):
    tracer = trace.get_tracer("caller")
    with tracer.start_as_current_span("caller") as outer:
        for chunk in ControlledChat().stream("q", config=config()):
            assert trace.get_current_span() is outer
        assert trace.get_current_span() is outer
    assert chat_span(spans).parent.span_id == outer.get_span_context().span_id


def test_callback_manager_copy_and_private_override(spans):
    foreign = BaseCallbackHandler()
    private = RespanCallbackHandler(include_content=False)
    manager = CallbackManager(
        [foreign, private], inheritable_handlers=[foreign, private]
    )
    replacement = RespanCallbackHandler()
    copied = add_respan_callback({"callbacks": manager}, replacement)["callbacks"]
    assert manager.handlers == [foreign, private] and copied is not manager
    assert copied.handlers == [foreign, replacement]
    owner = LangChainInstrumentor()
    owner.activate()
    RunnableLambda(lambda x: x).invoke("private", config={"callbacks": manager})
    no_content(spans.get_finished_spans()[0])
    owner.deactivate()


def test_shared_owners_and_no_ghost_hook(spans):
    first = LangChainInstrumentor()
    second = LangChainInstrumentor()
    original = CallbackManager.__dict__["configure"]
    first.activate()
    second.activate()
    assert first.callback_handler is second.callback_handler
    first.deactivate()
    RunnableLambda(lambda x: x).invoke("q")
    assert len(spans.get_finished_spans()) == 1
    second.deactivate()
    assert CallbackManager.__dict__["configure"] is original
    RunnableLambda(lambda x: x).invoke("q")
    assert len(spans.get_finished_spans()) == 1


def test_incompatible_privacy_rejected(spans):
    owner = LangChainInstrumentor()
    owner.activate()
    with pytest.raises(ValueError):
        LangChainInstrumentor(include_content=False).activate()
    owner.deactivate()


def test_registration_rollback(spans, monkeypatch):
    original = CallbackManager.__dict__["configure"]
    install = instrumentation._Runtime.patch

    def fail(self, owner, name, value):
        install(self, owner, name, value)
        if len(self.patches) == 2:
            raise RuntimeError("partial install")

    monkeypatch.setattr(instrumentation._Runtime, "patch", fail)
    with pytest.raises(RuntimeError):
        LangChainInstrumentor().activate()
    assert (
        CallbackManager.__dict__["configure"] is original
        and instrumentation._RUNTIME is None
    )


def test_foreign_wrapper_survives_inert_owned_hook(spans):
    owner = LangChainInstrumentor()
    owner.activate()
    owned = CallbackManager.configure.__func__

    def foreign(cls, *args, **kwargs):
        return owned(cls, *args, **kwargs)

    CallbackManager.configure = classmethod(foreign)
    try:
        owner.deactivate()
        assert CallbackManager.configure.__func__ is foreign
        RunnableLambda(lambda x: x).invoke("q")
        assert not spans.get_finished_spans()
    finally:
        CallbackManager.configure = classmethod(owned.__wrapped__)


def test_disabled_tracing_no_hooks(spans, monkeypatch):
    monkeypatch.setattr(
        instrumentation.RespanTracer, "_instance", SimpleNamespace(is_enabled=False)
    )
    owner = LangChainInstrumentor()
    owner.activate()
    assert not owner._is_instrumented
    RunnableLambda(lambda x: x).invoke("q", config=config())
    assert not spans.get_finished_spans()


def test_safe_redaction_and_no_application_hooks(spans):
    class Unsafe:
        def __str__(self):
            raise AssertionError("str called")

        def __repr__(self):
            raise AssertionError("repr called")

        def model_dump(self, *a, **k):
            raise AssertionError("model_dump called")

    value = {
        "x": Unsafe(),
        "password": "two words",
        "text": 'https://user:password@example.org password="two words" Bearer abcdefghijk',
    }
    RunnableLambda(lambda x: x).invoke(value, config=config())
    encoded = spans.get_finished_spans()[0].attributes[INPUT]
    assert (
        "two words" not in encoded
        and "user:password" not in encoded
        and "abcdefghijk" not in encoded
    )
    cycle = {}
    cycle["x"] = cycle
    assert "CYCLE" in json_value(cycle)


def test_usage_partial_invalid_omitted(spans):
    handler = RespanCallbackHandler()
    rid = uuid4()
    handler.on_llm_start({}, ["q"], run_id=rid)
    handler.on_llm_end(
        LLMResult(
            generations=[],
            llm_output={"token_usage": {"prompt_tokens": 4, "completion_tokens": True}},
        ),
        run_id=rid,
    )
    attrs = spans.get_finished_spans()[0].attributes
    assert attrs[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] == 4
    assert (
        SpanAttributes.LLM_USAGE_COMPLETION_TOKENS not in attrs
        and SpanAttributes.LLM_USAGE_TOTAL_TOKENS not in attrs
    )


def test_provider_http_source_usage(spans):
    from langchain_openai import ChatOpenAI

    def respond(request):
        body = json.loads(request.content)
        assert body["messages"][-1]["content"] == "q"
        return httpx.Response(
            200,
            json={
                "id": "fixture",
                "object": "chat.completion",
                "created": 1,
                "model": "fixture-openai",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "actual HTTP answer",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 11,
                    "completion_tokens": 7,
                    "total_tokens": 18,
                    "prompt_tokens_details": {"cached_tokens": 3},
                    "completion_tokens_details": {"reasoning_tokens": 2},
                },
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(respond))
    model = ChatOpenAI(
        model="fixture-openai", api_key="fixture-key", http_client=client
    )
    assert model.invoke("q", config=config()).content == "actual HTTP answer"
    assert chat_span(spans).attributes[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] == 11
    client.close()


def test_native_langgraph_state_graph(spans):
    from typing import TypedDict

    from langgraph.graph import StateGraph

    class State(TypedDict):
        value: str

    graph = StateGraph(State)
    graph.add_node("echo", lambda state: {"value": state["value"] + "!"})
    graph.set_entry_point("echo")
    graph.set_finish_point("echo")
    assert graph.compile().invoke({"value": "q"}, config=config()) == {"value": "q!"}
    assert any(
        s.attributes.get("respan.metadata.framework") == "langgraph"
        for s in spans.get_finished_spans()
    )


def test_native_langgraph_interrupt_resume(spans):
    types = pytest.importorskip("langgraph.types")
    if not hasattr(types, "interrupt"):
        pytest.skip("public interrupt requires newer LangGraph")
    from typing import TypedDict

    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import StateGraph

    class State(TypedDict):
        value: str

    def node(state):
        return {"value": types.interrupt("approve?")}

    graph = StateGraph(State)
    graph.add_node("confirm", node)
    graph.set_entry_point("confirm")
    graph.set_finish_point("confirm")
    app = graph.compile(checkpointer=MemorySaver())
    cfg = {**config(), "configurable": {"thread_id": "fixture-thread"}}
    first = app.invoke({"value": "q"}, config=cfg)
    assert "__interrupt__" in first
    assert app.invoke(types.Command(resume="approved"), config=cfg) == {
        "value": "approved"
    }
    assert all(
        s.status.status_code != StatusCode.ERROR for s in spans.get_finished_spans()
    )


def test_structured_message_vector_full_before_normalization(spans):
    content = [{"type": "data", "vector": list(range(5000))}]
    ControlledChat(response=AIMessage(content=content)).invoke(
        [HumanMessage(content=content)], config=config()
    )
    attrs = chat_span(spans).attributes
    assert len(json.loads(attrs[INPUT])[0][0]["content"][0]["vector"]) == 5000
    assert len(json.loads(attrs[OUTPUT])["messages"][0]["content"][0]["vector"]) == 5000


@pytest.mark.parametrize(
    "details", ["prompt_tokens_details", "input_tokens_details", "input_token_details"]
)
def test_cache_creation_source_aliases(spans, details):
    handler = RespanCallbackHandler()
    rid = uuid4()
    handler.on_llm_start({}, ["q"], run_id=rid)
    handler.on_llm_end(
        LLMResult(
            generations=[],
            llm_output={
                "usage": {details: {"cache_write_tokens": 4, "cache_creation": 4}}
            },
        ),
        run_id=rid,
    )
    assert (
        spans.get_finished_spans()[0].attributes[
            SpanAttributes.GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS
        ]
        == 4
    )


def test_child_veto_propagates_ancestor_after_reenable(spans, monkeypatch):
    handler = RespanCallbackHandler()
    root, child = uuid4(), uuid4()
    handler.on_chain_start({}, "private root", run_id=root)
    handler.on_chain_start({}, "private child", run_id=child, parent_run_id=root)
    monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")
    handler.on_text("private", run_id=child)
    monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "true")
    handler.on_chain_end("private", run_id=child)
    handler.on_chain_end("private", run_id=root)
    for span in spans.get_finished_spans():
        no_content(span)


def test_native_stream_context_completion_veto(spans):
    iterator = ControlledChat().stream("private prompt", config=config())
    first = next(iterator)
    token = context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, False))
    try:
        assert first.content + "".join(c.content for c in iterator) == "answer"
    finally:
        context.detach(token)
    no_content(chat_span(spans))


def test_explicit_model_local_private_handler_precedes_automatic(spans):
    owner = LangChainInstrumentor()
    owner.activate()
    private = RespanCallbackHandler(include_content=False)
    assert (
        ControlledChat(callbacks=[private]).invoke("private question").content
        == "answer"
    )
    owner.deactivate()
    assert len(spans.get_finished_spans()) == 1
    no_content(spans.get_finished_spans()[0])


def test_explicit_async_model_local_private_handler_precedes_automatic(spans):
    owner = LangChainInstrumentor()
    owner.activate()
    private = RespanCallbackHandler(include_content=False)
    assert (
        asyncio.run(
            ControlledChat(callbacks=[private]).ainvoke("private question")
        ).content
        == "answer"
    )
    owner.deactivate()
    assert len(spans.get_finished_spans()) == 1
    no_content(spans.get_finished_spans()[0])


def test_actual_sparse_integer_vector_tool_artifact_full(spans):
    @tool(response_format="content_and_artifact")
    def sparse_tool(label: str) -> tuple:
        """Return a source sparse vector artifact."""
        return label, {"sparse_vector": {i * 2: i / 5000 for i in range(5000)}}

    result = sparse_tool.invoke(
        {
            "name": "sparse_tool",
            "args": {"label": "sparse"},
            "id": "source-sparse-id",
            "type": "tool_call",
        },
        config=config(),
    )
    output = json.loads(spans.get_finished_spans()[0].attributes[OUTPUT])
    assert len(result.artifact["sparse_vector"]) == 5000
    assert len(output["artifact"]["sparse_vector"]) == 5000
    assert output["artifact"]["sparse_vector"]["9998"] == 4999 / 5000
    assert len(json.loads(json_value({i: i / 5000 for i in range(5000)}))) == 5000
