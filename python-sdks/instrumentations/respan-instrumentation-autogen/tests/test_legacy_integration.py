"""Real legacy SDK fixtures; only the OpenAI HTTP transport is deterministic.

Run in separate environments with tests/requirements-legacy-*.txt. No AutoGen
classes or methods are mocked: chats, reply dispatch, functions and GroupChat
all execute the installed library.
"""

import asyncio
import json

import pytest

autogen = pytest.importorskip("autogen")

from opentelemetry import context as context_api, trace
from opentelemetry.instrumentation.threading import ThreadingInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from respan_instrumentation_autogen import AutoGenInstrumentor
from respan_tracing.core.tracer import RespanTracer


@pytest.fixture
def tracing(monkeypatch):
    RespanTracer.reset_instance()
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(trace, "get_tracer_provider", lambda: provider)
    plugin = AutoGenInstrumentor(api="legacy")
    plugin.activate()
    assert plugin._is_instrumented
    yield provider, exporter, plugin
    plugin.deactivate()
    provider.shutdown()
    RespanTracer.reset_instance()


def agent(name, **kwargs):
    return autogen.ConversableAgent(
        name, llm_config=False, human_input_mode="NEVER",
        code_execution_config=False, max_consecutive_auto_reply=1, **kwargs,
    )


@pytest.mark.parametrize("asynchronous", [False, True])
def test_real_chat_preserves_return_value_and_history(tracing, asynchronous):
    provider, exporter, _ = tracing
    sender = agent("sender", default_auto_reply="sender reply")
    recipient = agent("recipient", default_auto_reply="recipient reply")
    with provider.get_tracer("test").start_as_current_span("parent") as parent:
        if asynchronous:
            result = asyncio.run(sender.a_initiate_chat(recipient, message="hello", silent=True))
        else:
            result = sender.initiate_chat(recipient, message="hello", silent=True)
    if autogen.__version__ == "0.2.2":
        assert result is None
    else:
        assert isinstance(result, autogen.ChatResult)
        assert result.chat_history == sender.chat_messages[recipient]
    spans = [span for span in exporter.get_finished_spans() if span.name != "parent"]
    chat = next(span for span in spans if "initiate_chat" in span.name)
    assert chat.parent.span_id == parent.get_span_context().span_id
    assert json.loads(chat.attributes["traceloop.entity.input"])["message"] == "hello"
    assert json.loads(chat.attributes["traceloop.entity.output"]) == sender.chat_messages[recipient][-1]["content"]
    replies = [span for span in spans if "generate_reply" in span.name]
    assert replies
    assert json.loads(replies[0].attributes["traceloop.entity.input"])[0]["content"] == "hello"
    assert all(span.parent.span_id == chat.context.span_id for span in replies)
    assert {span.attributes["respan.entity.log_type"] for span in spans} == {"agent"}
    assert not any(key.startswith("gen_ai.") for span in spans for key in span.attributes)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("failure", [False, True])
def test_real_tool_outputs_and_failure_status(tracing, asynchronous, failure):
    _, exporter, _ = tracing

    def search(query):
        if failure:
            raise ValueError("search failed")
        return {"role": "assistant", "content": "ordinary tool result", "query": query}

    async def async_search(query):
        return search(query)

    executor = agent("executor", function_map={"search": async_search if asynchronous else search})
    call = {"name": "search", "arguments": '{"query":"legacy"}'}
    result = asyncio.run(executor.a_execute_function(call)) if asynchronous else executor.execute_function(call)
    assert result[0] is not failure
    span, = exporter.get_finished_spans()
    assert span.attributes["respan.entity.log_type"] == "tool"
    assert json.loads(span.attributes["traceloop.entity.input"]) == call
    output = json.loads(span.attributes["traceloop.entity.output"])
    assert output[0] is not failure
    assert "search failed" in str(output) if failure else "ordinary tool result" in str(output)
    assert span.status.status_code is (trace.StatusCode.ERROR if failure else trace.StatusCode.OK)
    assert not any(key.startswith("gen_ai.") for key in span.attributes)


@pytest.mark.parametrize("asynchronous", [False, True])
def test_suppression_preserves_results_without_spans(tracing, asynchronous):
    _, exporter, _ = tracing
    speaker = agent("speaker", default_auto_reply="reply", function_map={"echo": lambda: "tool"})
    token = context_api.attach(context_api.set_value(context_api._SUPPRESS_INSTRUMENTATION_KEY, True))
    try:
        if asynchronous:
            from respan_instrumentation_autogen._legacy import LegacyAutoGenInstrumentor
            original = LegacyAutoGenInstrumentor()._original_methods["a_generate_reply"]
            expected = asyncio.run(original(agent("control", default_auto_reply="reply"), messages=[{"role": "user", "content": "hi"}]))
            assert asyncio.run(speaker.a_generate_reply(messages=[{"role": "user", "content": "hi"}])) == expected
            assert asyncio.run(speaker.a_execute_function({"name": "echo", "arguments": "{}"}))[1]["content"] == "tool"
        else:
            assert speaker.generate_reply(messages=[{"role": "user", "content": "hi"}]) == "reply"
            assert speaker.execute_function({"name": "echo", "arguments": "{}"})[1]["content"] == "tool"
    finally:
        context_api.detach(token)
    assert not exporter.get_finished_spans()


@pytest.mark.parametrize("asynchronous", [False, True])
def test_reply_exception_propagates_once_and_context_is_restored(tracing, asynchronous):
    _, exporter, _ = tracing
    speaker = agent("speaker")

    def fail(*args, **kwargs):
        raise ValueError("reply failed")

    speaker.register_reply([autogen.Agent, None], fail, position=0)
    with pytest.raises(ValueError, match="reply failed"):
        if asynchronous:
            asyncio.run(speaker.a_generate_reply(messages=[{"role": "user", "content": "hi"}]))
        else:
            speaker.generate_reply(messages=[{"role": "user", "content": "hi"}])
    span, = exporter.get_finished_spans()
    assert span.status.status_code is trace.StatusCode.ERROR
    assert len([event for event in span.events if event.name == "exception"]) == 1
    assert not trace.get_current_span().get_span_context().is_valid


def test_legacy_plugins_share_and_restore_all_six_methods(tracing):
    _, exporter, first = tracing
    from respan_instrumentation_autogen._legacy import LegacyAutoGenInstrumentor
    originals = dict(LegacyAutoGenInstrumentor()._original_methods)
    second = AutoGenInstrumentor(api="legacy")
    second.activate()
    first.deactivate()
    agent("speaker", default_auto_reply="reply").generate_reply(messages=[{"role": "user", "content": "hi"}])
    assert len(exporter.get_finished_spans()) == 1
    second.deactivate()
    assert all(getattr(autogen.ConversableAgent, name) is method for name, method in originals.items())
    second.deactivate()
    first.activate()
    first.deactivate()
    assert all(getattr(autogen.ConversableAgent, name) is method for name, method in originals.items())


@pytest.mark.parametrize("suppressed", [False, True])
def test_invalid_tool_call_keeps_sdk_validation_without_executing(tracing, suppressed):
    _, exporter, _ = tracing
    calls = []
    executor = agent("executor", function_map={"ping": lambda: calls.append("executed")})
    token = context_api.attach(context_api.set_value(context_api._SUPPRESS_INSTRUMENTATION_KEY, suppressed))
    try:
        with pytest.raises(AttributeError):
            executor.execute_function("ping")
    finally:
        context_api.detach(token)
    assert not calls
    assert len(exporter.get_finished_spans()) == (0 if suppressed else 1)


def test_teardown_preserves_later_wrappers_and_reactivation_does_not_duplicate(tracing):
    _, exporter, plugin = tracing
    from respan_instrumentation_autogen._legacy import LegacyAutoGenInstrumentor
    original = LegacyAutoGenInstrumentor()._original_methods["generate_reply"]
    installed = autogen.ConversableAgent.generate_reply
    seen = []

    def external(self, *args, **kwargs):
        seen.append(self.name)
        return installed(self, *args, **kwargs)

    autogen.ConversableAgent.generate_reply = external
    try:
        plugin.deactivate()
        assert autogen.ConversableAgent.generate_reply is external
        agent("inactive", default_auto_reply="reply").generate_reply(messages=[{"role": "user", "content": "hi"}])
        assert not exporter.get_finished_spans()
        plugin.activate()
        agent("active", default_auto_reply="reply").generate_reply(messages=[{"role": "user", "content": "hi"}])
        assert len(exporter.get_finished_spans()) == 1
        plugin.deactivate()
        assert autogen.ConversableAgent.generate_reply is external
        assert seen == ["inactive", "active"]
    finally:
        plugin.deactivate()
        autogen.ConversableAgent.generate_reply = original


def test_real_group_chat_dispatch(tracing):
    _, exporter, _ = tracing
    writer = agent("writer", default_auto_reply="draft")
    editor = agent("editor", default_auto_reply="approved")
    group = autogen.GroupChat(agents=[writer, editor], messages=[], max_round=3, speaker_selection_method="round_robin")
    manager = autogen.GroupChatManager(groupchat=group, llm_config=False, human_input_mode="NEVER", code_execution_config=False)
    writer.initiate_chat(manager, message="write news", silent=True)
    assert [item["content"] for item in group.messages] == ["write news", "approved", "draft"]
    spans = exporter.get_finished_spans()
    chat = next(span for span in spans if span.name == "writer.initiate_chat")
    manager_span = next(span for span in spans if span.name == "chat_manager.generate_reply")
    assert manager_span.parent.span_id == chat.context.span_id
    assert {span.attributes["traceloop.entity.name"] for span in spans} >= {"writer", "editor", "chat_manager"}


def test_modern_agentchat_can_run_alongside_legacy(tracing):
    from autogen_agentchat.agents import AssistantAgent
    from autogen_ext.models.replay import ReplayChatCompletionClient

    _, exporter, legacy = tracing
    modern = AutoGenInstrumentor()
    modern.activate()
    assert modern._is_instrumented
    try:
        model_client = ReplayChatCompletionClient(["modern answer"])
        assistant = AssistantAgent("modern", model_client=model_client)
        result = asyncio.run(assistant.run(task="modern question"))
        assert result.messages[-1].content == "modern answer"
        modern_spans = exporter.get_finished_spans()
        assert any("modern answer" in span.attributes.get("traceloop.entity.output", "") for span in modern_spans)
        modern.deactivate()
        exporter.clear()
        assert legacy._is_instrumented
        agent("legacy", default_auto_reply="legacy answer").generate_reply(messages=[{"role": "user", "content": "legacy question"}])
        span, = exporter.get_finished_spans()
        assert json.loads(span.attributes["traceloop.entity.output"]) == "legacy answer"
    finally:
        modern.deactivate()


@pytest.mark.parametrize("asynchronous", [False, True])
def test_real_llm_function_roundtrip_composes_with_provider_instrumentation(tracing, asynchronous):
    httpx = pytest.importorskip("httpx")
    provider_module = pytest.importorskip("respan_instrumentation_openai")
    _, exporter, _ = tracing
    requests = []

    def transport(request):
        body = json.loads(request.content)
        requests.append(body)
        message = ({"role": "assistant", "content": None, "function_call": {"name": "search", "arguments": '{"query":"news"}'}}
                   if len(requests) == 1 else {"role": "assistant", "content": "done"})
        return httpx.Response(200, json={"id": f"chatcmpl-{len(requests)}", "object": "chat.completion", "created": 1,
            "model": "gpt-4o-mini", "choices": [{"index": 0, "message": message, "finish_reason": "function_call" if len(requests) == 1 else "stop"}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}})

    class FixtureClient(httpx.Client):
        def __deepcopy__(self, memo):
            return self

    client = FixtureClient(transport=httpx.MockTransport(transport))
    provider_plugin = provider_module.OpenAIInstrumentor()
    provider_plugin.activate()
    # RespanTracer normally installs this bridge for run_in_executor itself.
    threading = ThreadingInstrumentor()
    threading.instrument()
    try:
        assistant = autogen.AssistantAgent("assistant", llm_config={"model": "gpt-4o-mini", "api_key": "fixture-key",
            "base_url": "http://fixture.invalid/v1", "http_client": client, "cache_seed": None,
            "functions": [{"name": "search", "description": "Search news", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}]})
        user = autogen.UserProxyAgent("user", llm_config=False, human_input_mode="NEVER", code_execution_config=False,
            max_consecutive_auto_reply=3, is_termination_msg=lambda message: message.get("content") == "done",
            function_map={"search": lambda query: "tool-only news"})
        result = asyncio.run(user.a_initiate_chat(assistant, message="find news", silent=True)) if asynchronous else user.initiate_chat(assistant, message="find news", silent=True)
        assert result is None if autogen.__version__ == "0.2.2" else isinstance(result, autogen.ChatResult)
    finally:
        threading.uninstrument()
        provider_plugin.deactivate()
        client.close()
    assert len(requests) == 2
    assert any(item.get("role") == "function" and item["content"] == "tool-only news" for item in requests[1]["messages"])
    spans = exporter.get_finished_spans()
    llms = [span for span in spans if span.attributes.get("respan.entity.log_type") == "chat"]
    tools = [span for span in spans if span.attributes.get("respan.entity.log_type") == "tool"]
    assert len(llms) == 2
    assert len(tools) == 1
    assert {span.attributes["gen_ai.request.model"] for span in llms} == {"gpt-4o-mini"}
    assert all(span.attributes["gen_ai.usage.input_tokens"] == 11 for span in llms)
    by_id = {span.context.span_id: span for span in spans}
    assert all(by_id[span.parent.span_id].attributes["respan.entity.log_type"] == "agent" for span in llms + tools)
    assert len({span.context.trace_id for span in spans}) == 1
