import asyncio
import json

import pytest
from opentelemetry import context, trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.sampling import ALWAYS_OFF
from opentelemetry.semconv_ai import SpanAttributes as A
from pyagentspec.tracing.trace import get_trace
from respan_instrumentation_agentspec import (
    AgentSpecInstrumentor,
    _callbacks,
)
from respan_tracing.constants.context_constants import ENABLE_CONTENT_TRACING_KEY

from ._fixtures import build_agent, fixture_model


@pytest.fixture
def runtime(monkeypatch):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(trace, "get_tracer_provider", lambda: provider)
    instrumentor = AgentSpecInstrumentor(workflow_name="fixture-trace")
    instrumentor.activate()
    assert instrumentor._is_instrumented
    yield instrumentor, exporter, provider
    instrumentor.deactivate()
    assert get_trace() is None
    assert _callbacks._STATE is None


def kind(exporter, name):
    return [
        s
        for s in exporter.get_finished_spans()
        if s.attributes.get("respan.entity.log_type") == name
    ]


def test_sequential_agents_have_separate_inputs_usage_and_safe_config(runtime):
    owner, exporter, _ = runtime
    with fixture_model():
        for prompt in ["first", "second"]:
            result = build_agent("agent-" + prompt).invoke(
                {"messages": [{"role": "user", "content": prompt}]}
            )
            assert result["messages"][-1].content == "answer " + prompt
    owner.deactivate()
    agents = kind(exporter, "agent")
    chats = kind(exporter, "chat")
    assert len(chats) == 2
    if agents:
        assert len(agents) == 2
        assert "first" in agents[0].attributes[A.TRACELOOP_ENTITY_INPUT]
        assert "second" in agents[1].attributes[A.TRACELOOP_ENTITY_INPUT]
        assert "first" not in agents[1].attributes[A.TRACELOOP_ENTITY_INPUT]
    for span in chats:
        assert span.attributes["gen_ai.usage.input_tokens"] == 0
        assert span.attributes[A.LLM_SYSTEM] == "openai"
        assert span.attributes["gen_ai.response.id"] == "fixture-response"
        assert list(span.attributes["gen_ai.response.finish_reasons"]) == ["stop"]
        assert span.attributes[A.LLM_USAGE_REASONING_TOKENS] == 2
    assert "never-export-config-secret" not in str(
        [dict(s.attributes) for s in exporter.get_finished_spans()]
    )


@pytest.mark.parametrize("asynchronous", [False, True])
def test_tool_zero_output_current_history_and_correlation(runtime, asynchronous):
    _, exporter, _ = runtime
    with fixture_model():
        graph = build_agent(with_tool=True)
        request = {"messages": [{"role": "user", "content": "tool-call"}]}
        result = (
            asyncio.run(graph.ainvoke(request))
            if asynchronous
            else graph.invoke(request)
        )
    assert result["messages"][-1].content == "0.0"
    chats = kind(exporter, "chat")
    tool = kind(exporter, "tool")[0]
    assert len(chats) == 2
    definitions = json.loads(chats[0].attributes[A.LLM_REQUEST_FUNCTIONS])
    definition = definitions[0].get("function", definitions[0])
    assert definition["name"] == "subtract"
    assert set(definition["parameters"]["properties"]) == {"left", "right"}
    assert tool.attributes["gen_ai.tool.call.id"] == "fixture-call-1"
    assert list(json.loads(tool.attributes[A.TRACELOOP_ENTITY_OUTPUT]).values()) == [0]
    assert (
        json.loads(chats[0].attributes["gen_ai.completion.0.tool_calls"])[0]["id"]
        == "fixture-call-1"
    )
    call = json.loads(chats[0].attributes["gen_ai.completion.0.tool_calls"])[0]
    assert json.loads(call["function"]["arguments"]) == {"left": 3.0, "right": 3.0}
    assert "gen_ai.completion.0.tool_calls" not in chats[1].attributes
    assert "fixture-call-1" in str(
        {k: v for k, v in chats[1].attributes.items() if k.startswith("gen_ai.prompt.")}
    )


@pytest.mark.parametrize("mode", ["environment", "context"])
def test_content_policy_snapshot_and_suppression(runtime, monkeypatch, mode):
    _, exporter, _ = runtime
    if mode == "environment":
        monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")
    token = (
        context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, False))
        if mode == "context"
        else None
    )
    try:
        with fixture_model():
            build_agent().invoke(
                {"messages": [{"role": "user", "content": "private-prompt"}]}
            )
    finally:
        if token is not None:
            context.detach(token)
    for span in exporter.get_finished_spans():
        assert A.TRACELOOP_ENTITY_INPUT not in span.attributes
        assert A.TRACELOOP_ENTITY_OUTPUT not in span.attributes
        assert not span.events
    assert "private-prompt" not in str(
        [dict(s.attributes) for s in exporter.get_finished_spans()]
    )
    before = len(exporter.get_finished_spans())
    token = context.attach(
        context.set_value(context._SUPPRESS_INSTRUMENTATION_KEY, True)
    )
    try:
        with fixture_model():
            build_agent().invoke(
                {"messages": [{"role": "user", "content": "suppressed"}]}
            )
    finally:
        context.detach(token)
    assert len(exporter.get_finished_spans()) == before


@pytest.mark.parametrize("asynchronous", [False, True])
def test_native_model_error_ends_span_without_manufactured_output(
    runtime, asynchronous
):
    owner, exporter, _ = runtime
    with fixture_model():
        graph = build_agent()
        with pytest.raises(ValueError, match="controlled model failure"):
            request = {"messages": [{"role": "user", "content": "model-failure"}]}
            asyncio.run(graph.ainvoke(request)) if asynchronous else graph.invoke(
                request
            )
    chats = kind(exporter, "chat")
    assert len(chats) == 1
    assert chats[0].status.status_code.name == "ERROR"
    assert not any(
        k.startswith(("gen_ai.completion.", "gen_ai.usage."))
        for k in chats[0].attributes
    )
    assert A.TRACELOOP_ENTITY_OUTPUT not in chats[0].attributes
    assert len(owner._processor._span_registry) == 1  # root remains active
    assert all(s.status.status_code.name == "ERROR" for s in kind(exporter, "agent"))


@pytest.mark.parametrize("asynchronous", [False, True])
def test_native_tool_error_retains_original_failure(runtime, asynchronous):
    _, exporter, _ = runtime
    with fixture_model():
        graph = build_agent(with_tool=True, tool_fails=True)
        from importlib.metadata import version

        if version("pyagentspec") == "26.1.0":
            request = {"messages": [{"role": "user", "content": "tool-call"}]}
            result = (
                asyncio.run(graph.ainvoke(request))
                if asynchronous
                else graph.invoke(request)
            )
            assert "controlled tool failure" in result["messages"][-1].content
        else:
            with pytest.raises(ValueError, match="controlled tool failure"):
                request = {"messages": [{"role": "user", "content": "tool-call"}]}
                if asynchronous:
                    asyncio.run(graph.ainvoke(request))
                else:
                    graph.invoke(request)
    assert kind(exporter, "tool")[0].status.status_code.name == "ERROR"
    assert A.TRACELOOP_ENTITY_OUTPUT not in kind(exporter, "tool")[0].attributes
    assert all(s.status.status_code.name == "ERROR" for s in kind(exporter, "agent"))


def test_always_off_sampler_has_no_spans_or_retained_payload(monkeypatch):
    provider = TracerProvider(sampler=ALWAYS_OFF)
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(trace, "get_tracer_provider", lambda: provider)
    owner = AgentSpecInstrumentor()
    owner.activate()
    with fixture_model():
        build_agent().invoke({"messages": [{"role": "user", "content": "not-sampled"}]})
    processor = owner._processor
    assert not processor._span_registry
    owner.deactivate()
    assert not exporter.get_finished_spans() and not processor._policies


@pytest.mark.parametrize("asynchronous", [False, True])
def test_current_flow_builder_with_local_tool(runtime, asynchronous):
    from importlib.metadata import version

    if version("pyagentspec") == "26.1.0":
        pytest.skip("FlowBuilder and native flow tracing were added after 26.1.0")
    from pyagentspec.adapters.langgraph import AgentSpecLoader
    from pyagentspec.flows.flowbuilder import FlowBuilder
    from pyagentspec.flows.nodes.toolnode import ToolNode
    from pyagentspec.property import FloatProperty
    from pyagentspec.tools import ServerTool

    _, exporter, _ = runtime
    tool = ServerTool(
        name="subtract",
        inputs=[FloatProperty(title="left"), FloatProperty(title="right")],
        outputs=[FloatProperty(title="difference")],
    )
    node = ToolNode(name="subtract-node", tool=tool)
    builder = (
        FlowBuilder()
        .add_node(node)
        .set_entry_point(node, inputs=tool.inputs)
        .set_finish_points(node, outputs=tool.outputs)
    )
    for key in ["left", "right"]:
        builder.add_data_edge("StartNode", node, key)
    builder.add_data_edge(node, "EndNode_1", "difference")
    flow = builder.build(name="fixture-flow")
    graph = AgentSpecLoader(
        tool_registry={"subtract": lambda left, right: left - right}
    ).load_component(flow)
    result = (
        asyncio.run(graph.ainvoke({"inputs": {"left": 5.0, "right": 2.0}}))
        if asynchronous
        else graph.invoke({"inputs": {"left": 5.0, "right": 2.0}})
    )
    assert result
    flow_span = next(
        s
        for s in kind(exporter, "workflow")
        if "FlowExecution" in s.attributes.get(A.TRACELOOP_ENTITY_NAME, "")
    )
    assert json.loads(flow_span.attributes[A.TRACELOOP_ENTITY_INPUT]) == {
        "left": 5.0,
        "right": 2.0,
    }
    assert json.loads(flow_span.attributes[A.TRACELOOP_ENTITY_OUTPUT]) == {
        "difference": 3.0
    }
    assert kind(exporter, "task")
    assert any(
        "3.0" in s.attributes.get(A.TRACELOOP_ENTITY_OUTPUT, "")
        for s in exporter.get_finished_spans()
    )


def test_upstream_sensitive_mask_is_respected(runtime):
    owner, exporter, _ = runtime
    owner.deactivate()
    owner = AgentSpecInstrumentor(mask_sensitive_information=True)
    owner.activate()
    try:
        with fixture_model():
            build_agent().invoke(
                {"messages": [{"role": "user", "content": "synthetic-masked-prompt"}]}
            )
    finally:
        owner.deactivate()
    attrs = [dict(s.attributes) for s in exporter.get_finished_spans()]
    assert "synthetic-masked-prompt" not in str(attrs)
    assert all(
        A.TRACELOOP_ENTITY_INPUT not in a and A.TRACELOOP_ENTITY_OUTPUT not in a
        for a in attrs
    )


@pytest.mark.parametrize("asynchronous", [False, True])
def test_public_stream_terminal_value_and_cleanup(runtime, asynchronous):
    owner, exporter, _ = runtime
    with fixture_model():
        graph = build_agent()
        request = {"messages": [{"role": "user", "content": "stream-input"}]}
        if asynchronous:

            async def collect():
                return [
                    value
                    async for value in graph.astream(request, stream_mode="messages")
                ]

            result = asyncio.run(collect())
        else:
            result = list(graph.stream(request, stream_mode="messages"))
    assert "".join(value[0].content for value in result) == "fixture stream"
    chats = kind(exporter, "chat")
    assert len(chats) == 1
    assert "fixture stream" in chats[0].attributes[A.TRACELOOP_ENTITY_OUTPUT]
    assert chats[0].attributes["gen_ai.usage.input_tokens"] == 0
    assert chats[0].attributes["gen_ai.usage.output_tokens"] == 0
    assert chats[0].attributes[A.GEN_AI_IS_STREAMING] is True
    assert len(owner._processor._span_registry) == 1


def test_concurrent_agents_keep_inputs_separate(runtime):
    owner, exporter, _ = runtime

    async def run():
        with fixture_model():
            return await asyncio.gather(
                *(
                    build_agent("agent-" + p).ainvoke(
                        {"messages": [{"role": "user", "content": p}]}
                    )
                    for p in ["alpha", "beta"]
                )
            )

    result = asyncio.run(run())
    assert [r["messages"][-1].content for r in result] == [
        "answer alpha",
        "answer beta",
    ]
    chats = kind(exporter, "chat")
    assert len(chats) == 2
    assert sorted(
        json.loads(s.attributes[A.TRACELOOP_ENTITY_INPUT])[-1]["content"] for s in chats
    ) == ["alpha", "beta"]
    assert len(owner._processor._span_registry) == 1


def test_shared_owners_conflicting_config_and_final_release(runtime):
    owner, exporter, _ = runtime
    first_trace = get_trace()
    second = AgentSpecInstrumentor(workflow_name="fixture-trace")
    second.activate()
    with pytest.raises(ValueError, match="share provider and configuration"):
        AgentSpecInstrumentor(mask_sensitive_information=True).activate()
    owner.deactivate()
    assert get_trace() is first_trace
    with fixture_model():
        build_agent().invoke(
            {"messages": [{"role": "user", "content": "second-owner"}]}
        )
    assert kind(exporter, "chat")
    second.deactivate()
    assert get_trace() is None


def test_partial_start_failure_restores_native_trace_and_stack(runtime, monkeypatch):
    from pyagentspec.tracing.spans import span as native_spans
    from pyagentspec.tracing.trace import Trace

    owner, _, _ = runtime
    owner.deactivate()
    previous = native_spans.get_active_span_stack()
    original = Trace._start

    def fail(self, *args, **kwargs):
        original(self, *args, **kwargs)
        raise RuntimeError("controlled start failure")

    monkeypatch.setattr(Trace, "_start", fail)
    owner.activate()
    assert not owner._is_instrumented
    assert get_trace() is None
    assert native_spans.get_active_span_stack() == previous
    assert _callbacks._STATE is None


def test_retained_foreign_callback_wrapper_survives_release(runtime, monkeypatch):
    import pyagentspec.adapters.langgraph.tracing as native

    owner, _, _ = runtime
    cls = (
        getattr(native, "AgentSpecLlmCallbackHandler", None)
        or native.AgentSpecCallbackHandler
    )
    installed = cls.on_llm_end
    calls = []

    def foreign(self, *args, **kwargs):
        calls.append(True)
        return installed(self, *args, **kwargs)

    monkeypatch.setattr(cls, "on_llm_end", foreign)
    owner.deactivate()
    assert cls.on_llm_end is foreign
    owner.activate()
    with fixture_model():
        build_agent().invoke(
            {"messages": [{"role": "user", "content": "foreign-wrapper"}]}
        )
    assert calls
    owner.deactivate()
    assert cls.on_llm_end is foreign


@pytest.mark.parametrize("asynchronous", [False, True])
def test_stream_early_close_preserves_public_protocol(runtime, asynchronous):
    owner, exporter, _ = runtime
    with fixture_model():
        graph = build_agent()
        request = {"messages": [{"role": "user", "content": "early-close"}]}
        if asynchronous:

            async def collect():
                stream = graph.astream(request, stream_mode="messages")
                first = await anext(stream)
                await stream.aclose()
                return first

            first = asyncio.run(collect())
        else:
            stream = graph.stream(request, stream_mode="messages")
            first = next(stream)
            stream.close()
    assert first[0].content == "fixture "
    from importlib.metadata import version

    if version("pyagentspec") == "26.1.0":
        # This SDK predates enclosing agent spans, so no terminal parent event
        # arrives on async close; the bridge flushes observed children at release.
        processor = owner._processor
        owner.deactivate()
        assert not processor._span_registry
    else:
        assert len(owner._processor._span_registry) == 1
    assert not any(
        "early-close" in str(s.attributes.get(A.TRACELOOP_ENTITY_OUTPUT, ""))
        for s in kind(exporter, "chat")
    )


@pytest.mark.parametrize("mode", ["sampler", "suppression"])
def test_dropped_async_early_close_does_not_accumulate_policies(monkeypatch, mode):
    from importlib.metadata import version

    if version("pyagentspec") == "26.1.0":
        pytest.skip("SDK has no enclosing agent boundary until instrumentor release")
    provider = (
        TracerProvider(sampler=ALWAYS_OFF) if mode == "sampler" else TracerProvider()
    )
    monkeypatch.setattr(trace, "get_tracer_provider", lambda: provider)
    owner = AgentSpecInstrumentor()
    owner.activate()

    async def run():
        for _ in range(3):
            token = (
                context.attach(
                    context.set_value(context._SUPPRESS_INSTRUMENTATION_KEY, True)
                )
                if mode == "suppression"
                else None
            )
            try:
                with fixture_model():
                    stream = build_agent().astream(
                        {"messages": [{"role": "user", "content": "dropped-private"}]},
                        stream_mode="messages",
                    )
                    await anext(stream)
                    await stream.aclose()
            finally:
                if token is not None:
                    context.detach(token)
            assert len(owner._processor._policies) == 1
            assert all("messages" not in p for p in owner._processor._policies.values())

    try:
        asyncio.run(run())
    finally:
        owner.deactivate()


def test_async_cancellation_closes_model_without_output(runtime, monkeypatch):
    from ._fixtures import FixtureModel

    owner, exporter, _ = runtime

    async def run():
        started = asyncio.Event()

        async def wait_for_cancel(self, messages, **kwargs):
            started.set()
            await asyncio.Future()

        monkeypatch.setattr(FixtureModel, "_agenerate", wait_for_cancel)
        with fixture_model():
            task = asyncio.create_task(
                build_agent().ainvoke(
                    {"messages": [{"role": "user", "content": "cancelled-request"}]}
                )
            )
            await started.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    asyncio.run(run())
    from importlib.metadata import version

    minimum = version("pyagentspec") == "26.1.0"
    if minimum:
        owner.deactivate()
    else:
        assert len(owner._processor._span_registry) == 1
    chats = kind(exporter, "chat")
    assert len(chats) == 1
    if not minimum:
        assert chats[0].status.status_code.name == "ERROR"
        assert chats[0].attributes["error.type"] == "CancelledError"
    assert A.TRACELOOP_ENTITY_OUTPUT not in chats[0].attributes
    assert "gen_ai.usage.output_tokens" not in chats[0].attributes


def test_private_async_start_stays_private_when_environment_changes(
    runtime, monkeypatch
):
    from ._fixtures import FixtureModel

    _, exporter, _ = runtime

    async def run():
        started, released = asyncio.Event(), asyncio.Event()

        async def wait_then_return(self, messages, **kwargs):
            started.set()
            await released.wait()
            return self._generate(messages)

        monkeypatch.setattr(FixtureModel, "_agenerate", wait_then_return)
        monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")
        with fixture_model():
            task = asyncio.create_task(
                build_agent().ainvoke(
                    {"messages": [{"role": "user", "content": "deferred-private-text"}]}
                )
            )
            await started.wait()
            monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "true")
            released.set()
            await task

    asyncio.run(run())
    assert "deferred-private-text" not in str(
        [dict(s.attributes) for s in exporter.get_finished_spans()]
    )


def test_disabling_content_before_completion_vetoes_initial_capture(
    runtime, monkeypatch
):
    from ._fixtures import FixtureModel

    owner, exporter, _ = runtime
    original = FixtureModel._generate

    def finish_private(self, messages, **kwargs):
        monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")
        return original(self, messages, **kwargs)

    monkeypatch.setattr(FixtureModel, "_generate", finish_private)
    with fixture_model():
        build_agent().invoke(
            {"messages": [{"role": "user", "content": "vetoed-private-text"}]}
        )
    owner.deactivate()
    assert "vetoed-private-text" not in str(
        [dict(s.attributes) for s in exporter.get_finished_spans()]
    )
    assert kind(exporter, "chat")[0].attributes["gen_ai.usage.output_tokens"] == 3


@pytest.mark.parametrize(
    "detail",
    [
        "api_key=fixture-secret",
        '{"api_key":"fixture-secret with spaces"}',
        "{'password':'fixture-secret'}",
        "Bearer fixture-secret",
    ],
)
def test_error_details_redact_named_credentials(detail):
    from respan_instrumentation_agentspec._processor import safe_error

    assert "fixture-secret" not in safe_error(detail)
