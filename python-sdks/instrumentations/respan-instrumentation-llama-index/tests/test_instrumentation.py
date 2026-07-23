import asyncio
import json
import logging
from types import SimpleNamespace

import pytest
from llama_index.core import instrumentation
from workflows import Workflow, step
from workflows.events import Event, StartEvent, StopEvent
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.trace import StatusCode
from respan_sdk.constants.llm_logging import (
    LOG_TYPE_AGENT,
    LOG_TYPE_CHAT,
    LOG_TYPE_COMPLETION,
    LOG_TYPE_EMBEDDING,
    LOG_TYPE_TASK,
    LOG_TYPE_TOOL,
    LOG_TYPE_WORKFLOW,
)
from respan_sdk.constants.span_attributes import (
    GEN_AI_SYSTEM,
    RESPAN_LOG_TYPE,
)
from respan_tracing import RespanTelemetry
from respan_tracing.core.tracer import RespanTracer
from respan_tracing.testing import InMemorySpanExporter

from respan_instrumentation_llama_index import LlamaIndexInstrumentor
from respan_instrumentation_llama_index import _instrumentation
from respan_instrumentation_llama_index._handlers import (
    RespanLlamaIndexEventHandler,
    RespanLlamaIndexSpanHandler,
)
from respan_instrumentation_llama_index._serialization import extract_usage


@pytest.fixture(autouse=True)
def reset_tracer():
    RespanTracer.reset_instance()
    yield
    RespanTracer.reset_instance()


@pytest.fixture
def span_exporter():
    exporter = InMemorySpanExporter()
    telemetry = RespanTelemetry(
        app_name="llama-index-test",
        api_key=None,
        is_auto_instrument=False,
        is_batching_enabled=False,
    )
    telemetry.tracer.tracer_provider.add_span_processor(
        SimpleSpanProcessor(span_exporter=exporter)
    )
    yield exporter
    telemetry.flush()


def test_activate_registers_native_handlers():
    instrumentor = LlamaIndexInstrumentor()
    instrumentor.activate()

    assert instrumentor._span_handler in instrumentation.root_dispatcher.span_handlers
    assert instrumentor._event_handler in instrumentation.root_dispatcher.event_handlers
    assert instrumentor._is_instrumented is True

    instrumentor.deactivate()

    assert (
        instrumentor._span_handler not in instrumentation.root_dispatcher.span_handlers
    )
    assert (
        instrumentor._event_handler
        not in instrumentation.root_dispatcher.event_handlers
    )
    assert instrumentor._is_instrumented is False


def test_activate_is_idempotent():
    instrumentor = LlamaIndexInstrumentor()
    instrumentor.activate()
    instrumentor.activate()

    assert (
        instrumentation.root_dispatcher.span_handlers.count(instrumentor._span_handler)
        == 1
    )
    assert (
        instrumentation.root_dispatcher.event_handlers.count(
            instrumentor._event_handler
        )
        == 1
    )

    instrumentor.deactivate()


def test_activate_skips_when_respan_tracing_is_disabled(caplog):
    RespanTracer(is_enabled=False)

    instrumentor = LlamaIndexInstrumentor()
    with caplog.at_level(logging.INFO):
        instrumentor.activate()

    assert instrumentor._is_instrumented is False
    assert (
        "LlamaIndex instrumentation skipped because Respan tracing is disabled"
        in caplog.text
    )


def test_activate_logs_warning_when_dependency_is_missing(monkeypatch, caplog):
    def import_module_raises(module_name):
        raise ImportError(module_name)

    monkeypatch.setattr(
        _instrumentation.importlib,
        "import_module",
        import_module_raises,
    )
    instrumentor = LlamaIndexInstrumentor()

    with caplog.at_level(logging.WARNING):
        instrumentor.activate()

    assert "Failed to activate LlamaIndex instrumentation" in caplog.text
    assert instrumentor._is_instrumented is False


def test_span_handler_emits_workflow_and_task_spans(span_exporter):
    handler = RespanLlamaIndexSpanHandler()
    root_bound_args = SimpleNamespace(args=("question",), kwargs={})
    child_bound_args = SimpleNamespace(args=(), kwargs={"top_k": 2})

    handler.span_enter(
        id_="RetrieverQueryEngine.query-11111111-1111-1111-1111-111111111111",
        bound_args=root_bound_args,
        parent_id=None,
    )
    handler.span_enter(
        id_="BaseRetriever.retrieve-22222222-2222-2222-2222-222222222222",
        bound_args=child_bound_args,
        parent_id="RetrieverQueryEngine.query-11111111-1111-1111-1111-111111111111",
    )
    handler.span_exit(
        id_="BaseRetriever.retrieve-22222222-2222-2222-2222-222222222222",
        bound_args=child_bound_args,
        result=["node"],
    )
    handler.span_exit(
        id_="RetrieverQueryEngine.query-11111111-1111-1111-1111-111111111111",
        bound_args=root_bound_args,
        result="answer",
    )

    spans = span_exporter.get_finished_spans()
    attrs_by_name = {span.name: span.attributes for span in spans}

    assert (
        attrs_by_name["RetrieverQueryEngine.query"][RESPAN_LOG_TYPE]
        == LOG_TYPE_WORKFLOW
    )
    assert attrs_by_name["BaseRetriever.retrieve"][RESPAN_LOG_TYPE] == LOG_TYPE_TASK
    assert (
        attrs_by_name["RetrieverQueryEngine.query"][
            SpanAttributes.TRACELOOP_ENTITY_OUTPUT
        ]
        == '"answer"'
    )


def test_span_handler_error_respects_content_capture_setting(span_exporter):
    handler = RespanLlamaIndexSpanHandler(capture_content=False)
    bound_args = SimpleNamespace(args=("top-secret input",), kwargs={})
    span_id = "RetrieverQueryEngine.query-private"

    handler.span_enter(id_=span_id, bound_args=bound_args, parent_id=None)
    handler.span_drop(
        id_=span_id,
        bound_args=bound_args,
        err=RuntimeError("top-secret failure"),
    )

    span = span_exporter.get_finished_spans()[0]
    assert span.status.status_code == StatusCode.ERROR
    assert span.attributes["status_code"] == 500
    assert span.attributes["error.message"] == "RuntimeError"
    assert SpanAttributes.TRACELOOP_ENTITY_INPUT not in span.attributes
    assert json.loads(span.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]) == {
        "error": "RuntimeError",
        "status": "error",
    }
    assert "top-secret" not in json.dumps(dict(span.attributes))
    assert not span.events


def test_span_handler_uses_cached_parent_context_after_parent_exits(span_exporter):
    handler = RespanLlamaIndexSpanHandler()
    root_bound_args = SimpleNamespace(args=("question",), kwargs={})
    child_bound_args = SimpleNamespace(args=(), kwargs={"top_k": 2})
    root_id = "RetrieverQueryEngine.query-11111111-1111-1111-1111-111111111111"
    child_id = "BaseRetriever.retrieve-22222222-2222-2222-2222-222222222222"

    handler.span_enter(
        id_=root_id,
        bound_args=root_bound_args,
        parent_id=None,
    )
    handler.span_exit(
        id_=root_id,
        bound_args=root_bound_args,
        result="answer",
    )
    handler.span_enter(
        id_=child_id,
        bound_args=child_bound_args,
        parent_id=root_id,
    )
    handler.span_exit(
        id_=child_id,
        bound_args=child_bound_args,
        result=["node"],
    )

    spans_by_name = {span.name: span for span in span_exporter.get_finished_spans()}
    root_span = spans_by_name["RetrieverQueryEngine.query"]
    child_span = spans_by_name["BaseRetriever.retrieve"]

    assert child_span.context.trace_id == root_span.context.trace_id
    assert child_span.parent.span_id == root_span.context.span_id


def test_span_handler_groups_missing_parent_siblings_under_synthetic_parent(
    span_exporter,
):
    handler = RespanLlamaIndexSpanHandler()
    parent_id = "ReActAgent.run-11111111-1111-1111-1111-111111111111"
    bound_args = SimpleNamespace(args=(), kwargs={})

    handler.span_enter(
        id_="BaseWorkflowAgent.setup_agent-22222222-2222-2222-2222-222222222222",
        bound_args=bound_args,
        parent_id=parent_id,
    )
    handler.span_exit(
        id_="BaseWorkflowAgent.setup_agent-22222222-2222-2222-2222-222222222222",
        bound_args=bound_args,
        result="setup",
    )
    handler.span_enter(
        id_="BaseWorkflowAgent.run_agent_step-33333333-3333-3333-3333-333333333333",
        bound_args=bound_args,
        parent_id=parent_id,
    )
    handler.span_exit(
        id_="BaseWorkflowAgent.run_agent_step-33333333-3333-3333-3333-333333333333",
        bound_args=bound_args,
        result="step",
    )

    spans_by_name = {span.name: span for span in span_exporter.get_finished_spans()}
    synthetic_parent = spans_by_name["ReActAgent.run"]
    setup_span = spans_by_name["BaseWorkflowAgent.setup_agent"]
    step_span = spans_by_name["BaseWorkflowAgent.run_agent_step"]

    assert synthetic_parent.attributes[RESPAN_LOG_TYPE] == LOG_TYPE_AGENT
    assert not any(
        key.startswith("llama_index.") for key in synthetic_parent.attributes
    )
    assert setup_span.context.trace_id == synthetic_parent.context.trace_id
    assert step_span.context.trace_id == synthetic_parent.context.trace_id
    assert setup_span.parent.span_id == synthetic_parent.context.span_id
    assert step_span.parent.span_id == synthetic_parent.context.span_id


def test_standalone_workflows_emit_real_root_and_step_spans(span_exporter):
    class DraftEvent(Event):
        text: str

    class DemoWorkflow(Workflow):
        @step
        async def draft(self, event: StartEvent) -> DraftEvent:
            return DraftEvent(text=f"draft:{event.topic}")

        @step
        async def finish(self, event: DraftEvent) -> StopEvent:
            return StopEvent(result=event.text.upper())

    async def run_workflow():
        return await DemoWorkflow().run(topic="respan")

    instrumentor = LlamaIndexInstrumentor()
    instrumentor.activate()
    try:
        result = asyncio.run(run_workflow())
    finally:
        instrumentor.deactivate()

    assert result == "DRAFT:RESPAN"
    spans_by_name = {span.name: span for span in span_exporter.get_finished_spans()}
    root_span = spans_by_name["DemoWorkflow.run"]
    draft_span = spans_by_name["DemoWorkflow.draft"]
    finish_span = spans_by_name["DemoWorkflow.finish"]

    assert root_span.attributes[RESPAN_LOG_TYPE] == LOG_TYPE_WORKFLOW
    assert (
        "StartEvent"
        in json.loads(root_span.attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT])[
            "event"
        ]
    )
    assert json.loads(root_span.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]) == {
        "result": "DRAFT:RESPAN"
    }
    assert draft_span.attributes[RESPAN_LOG_TYPE] == LOG_TYPE_TASK
    assert finish_span.attributes[RESPAN_LOG_TYPE] == LOG_TYPE_TASK
    assert draft_span.parent.span_id == root_span.context.span_id
    assert finish_span.parent.span_id == root_span.context.span_id
    for span in (root_span, draft_span, finish_span):
        assert not any(key.startswith("llama_index.") for key in span.attributes)


def test_chat_events_emit_canonical_llm_span(span_exporter):
    handler = RespanLlamaIndexEventHandler()
    user_message = SimpleNamespace(role="user", content="Hello")
    assistant_message = SimpleNamespace(role="assistant", content="Hi")
    response = SimpleNamespace(
        message=assistant_message,
        raw={
            "usage": {
                "prompt_tokens": 3,
                "completion_tokens": 2,
                "total_tokens": 5,
            }
        },
    )

    handler.handle(
        SimpleNamespace(
            class_name=lambda: "LLMChatStartEvent",
            span_id="span-1",
            messages=[user_message],
            model_dict={"class_name": "OpenAI", "model_name": "gpt-4o-mini"},
        )
    )
    handler.handle(
        SimpleNamespace(
            class_name=lambda: "LLMChatEndEvent",
            span_id="span-1",
            response=response,
        )
    )

    spans = span_exporter.get_finished_spans()
    chat_span = next(span for span in spans if span.name == "llama_index.chat")
    attributes = chat_span.attributes

    assert attributes[RESPAN_LOG_TYPE] == LOG_TYPE_CHAT
    assert attributes[SpanAttributes.LLM_REQUEST_TYPE] == "chat"
    assert attributes[GEN_AI_SYSTEM] == "openai"
    assert attributes[SpanAttributes.LLM_REQUEST_MODEL] == "gpt-4o-mini"
    assert attributes[f"{SpanAttributes.LLM_PROMPTS}.0.role"] == "user"
    assert attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"] == "Hello"
    assert attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.role"] == "assistant"
    assert attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"] == "Hi"
    assert attributes[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] == 3
    assert attributes[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS] == 2
    assert attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] == 5


def test_react_observation_messages_are_marked_as_system(span_exporter):
    handler = RespanLlamaIndexEventHandler()
    response = SimpleNamespace(
        message=SimpleNamespace(role="assistant", content="Answer: 42"),
        raw={},
    )

    handler.handle(
        SimpleNamespace(
            class_name=lambda: "LLMChatStartEvent",
            span_id="span-react",
            messages=[
                SimpleNamespace(
                    role="user",
                    content="Use the multiply_numbers tool.",
                ),
                SimpleNamespace(
                    role="assistant",
                    content='Action: multiply_numbers\nAction Input: {"a": 7, "b": 6}',
                ),
                SimpleNamespace(role="user", content="Observation: 42"),
            ],
            model_dict={"class_name": "OpenAI", "model_name": "gpt-4o-mini"},
        )
    )
    handler.handle(
        SimpleNamespace(
            class_name=lambda: "LLMChatEndEvent",
            span_id="span-react",
            response=response,
        )
    )

    chat_span = next(
        span
        for span in span_exporter.get_finished_spans()
        if span.name == "llama_index.chat"
    )
    attributes = chat_span.attributes

    assert attributes[f"{SpanAttributes.LLM_PROMPTS}.0.role"] == "user"
    assert attributes[f"{SpanAttributes.LLM_PROMPTS}.2.role"] == "system"
    assert attributes[f"{SpanAttributes.LLM_PROMPTS}.2.content"] == "Observation: 42"


def test_generated_context_prompt_is_split_from_user_query(span_exporter):
    handler = RespanLlamaIndexEventHandler()
    generated_prompt = (
        "Context information is below.\n"
        "---------------------\n"
        "Respan traces LlamaIndex query engines.\n"
        "---------------------\n"
        "Given the context information and not prior knowledge, answer the query.\n"
        "Query: What does Respan trace?\n"
        "Answer: "
    )

    handler.handle(
        SimpleNamespace(
            class_name=lambda: "LLMChatStartEvent",
            span_id="span-context",
            messages=[
                SimpleNamespace(role="system", content="Answer from context only."),
                SimpleNamespace(role="user", content=generated_prompt),
            ],
            model_dict={"class_name": "OpenAI", "model_name": "gpt-4o-mini"},
        )
    )
    handler.handle(
        SimpleNamespace(
            class_name=lambda: "LLMChatEndEvent",
            span_id="span-context",
            response=SimpleNamespace(
                message=SimpleNamespace(role="assistant", content="It traces queries."),
                raw={},
            ),
        )
    )

    chat_span = next(
        span
        for span in span_exporter.get_finished_spans()
        if span.name == "llama_index.chat"
    )
    attributes = chat_span.attributes

    assert attributes[f"{SpanAttributes.LLM_PROMPTS}.1.role"] == "system"
    assert (
        "Context information is below."
        in attributes[f"{SpanAttributes.LLM_PROMPTS}.1.content"]
    )
    assert attributes[f"{SpanAttributes.LLM_PROMPTS}.2.role"] == "user"
    assert attributes[f"{SpanAttributes.LLM_PROMPTS}.2.content"] == (
        "What does Respan trace?"
    )


def test_generic_span_payload_marks_react_observations_as_system(span_exporter):
    handler = RespanLlamaIndexSpanHandler()
    bound_args = SimpleNamespace(
        args=(
            [
                SimpleNamespace(role="user", content="Question"),
                SimpleNamespace(
                    role="assistant",
                    content='Action: multiply_numbers\nAction Input: {"a": 7, "b": 6}',
                ),
                SimpleNamespace(role="user", content="Observation: 42"),
            ],
        ),
        kwargs={},
    )

    handler.span_enter(
        id_="OpenAI.achat-11111111-1111-1111-1111-111111111111",
        bound_args=bound_args,
        parent_id=None,
    )
    handler.span_exit(
        id_="OpenAI.achat-11111111-1111-1111-1111-111111111111",
        bound_args=bound_args,
        result="Answer: 42",
    )

    span = next(
        span
        for span in span_exporter.get_finished_spans()
        if span.name == "OpenAI.achat"
    )
    payload = json.loads(span.attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT])
    messages = payload["args"][0]

    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"
    assert messages[2]["role"] == "system"
    assert messages[2]["content"] == "Observation: 42"


def test_user_observation_prompt_remains_user(span_exporter):
    handler = RespanLlamaIndexEventHandler()

    handler.handle(
        SimpleNamespace(
            class_name=lambda: "LLMChatStartEvent",
            span_id="span-user-observation",
            messages=[
                SimpleNamespace(role="user", content="Observation: this is my note"),
            ],
            model_dict={"class_name": "OpenAI", "model_name": "gpt-4o-mini"},
        )
    )
    handler.handle(
        SimpleNamespace(
            class_name=lambda: "LLMChatEndEvent",
            span_id="span-user-observation",
            response=SimpleNamespace(
                message=SimpleNamespace(role="assistant", content="Noted."),
                raw={},
            ),
        )
    )

    chat_span = next(
        span
        for span in span_exporter.get_finished_spans()
        if span.name == "llama_index.chat"
    )
    attributes = chat_span.attributes

    assert attributes[f"{SpanAttributes.LLM_PROMPTS}.0.role"] == "user"
    assert attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"] == (
        "Observation: this is my note"
    )


def test_chat_events_can_disable_content_capture(span_exporter):
    handler = RespanLlamaIndexEventHandler(capture_content=False)
    user_message = SimpleNamespace(role="user", content="hidden")
    response = SimpleNamespace(
        message=SimpleNamespace(role="assistant", content="also hidden"),
        raw={},
    )

    handler.handle(
        SimpleNamespace(
            class_name=lambda: "LLMChatStartEvent",
            span_id="span-1",
            messages=[user_message],
            model_dict={},
        )
    )
    handler.handle(
        SimpleNamespace(
            class_name=lambda: "LLMChatEndEvent",
            span_id="span-1",
            response=response,
        )
    )

    chat_span = next(
        span
        for span in span_exporter.get_finished_spans()
        if span.name == "llama_index.chat"
    )

    assert SpanAttributes.TRACELOOP_ENTITY_INPUT not in chat_span.attributes
    assert SpanAttributes.TRACELOOP_ENTITY_OUTPUT not in chat_span.attributes
    assert not any(
        key.startswith(f"{SpanAttributes.LLM_PROMPTS}.")
        or key.startswith(f"{SpanAttributes.LLM_COMPLETIONS}.")
        for key in chat_span.attributes
    )
    assert "hidden" not in json.dumps(dict(chat_span.attributes))


def test_completion_events_can_disable_content_capture(span_exporter):
    handler = RespanLlamaIndexEventHandler(capture_content=False)
    handler.handle(
        SimpleNamespace(
            class_name=lambda: "LLMCompletionStartEvent",
            span_id="span-private",
            prompt="completion secret",
            model_dict={},
        )
    )
    handler.handle(
        SimpleNamespace(
            class_name=lambda: "LLMCompletionEndEvent",
            span_id="span-private",
            response=SimpleNamespace(
                text="private response",
                raw={"usage": {"input_tokens": 2, "output_tokens": 1}},
            ),
        )
    )

    completion_span = next(
        span
        for span in span_exporter.get_finished_spans()
        if span.name == "llama_index.completion"
    )
    assert SpanAttributes.TRACELOOP_ENTITY_INPUT not in completion_span.attributes
    assert SpanAttributes.TRACELOOP_ENTITY_OUTPUT not in completion_span.attributes
    assert not any(
        key.startswith(f"{SpanAttributes.LLM_PROMPTS}.")
        or key.startswith(f"{SpanAttributes.LLM_COMPLETIONS}.")
        for key in completion_span.attributes
    )
    assert completion_span.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] == 3
    assert "secret" not in json.dumps(dict(completion_span.attributes))
    assert "private response" not in json.dumps(dict(completion_span.attributes))


def test_completion_events_emit_text_span(span_exporter):
    handler = RespanLlamaIndexEventHandler()
    response = SimpleNamespace(
        text="A short completion.",
        raw={
            "usage": {
                "input_tokens": 4,
                "output_tokens": 3,
            }
        },
    )

    handler.handle(
        SimpleNamespace(
            class_name=lambda: "LLMCompletionStartEvent",
            span_id="span-2",
            prompt="Complete this sentence",
            model_dict={"class_name": "OpenAI", "model": "gpt-4o-mini"},
        )
    )
    handler.handle(
        SimpleNamespace(
            class_name=lambda: "LLMCompletionEndEvent",
            span_id="span-2",
            response=response,
        )
    )

    text_span = next(
        span
        for span in span_exporter.get_finished_spans()
        if span.name == "llama_index.completion"
    )
    attributes = text_span.attributes

    assert attributes[RESPAN_LOG_TYPE] == LOG_TYPE_COMPLETION
    assert attributes[SpanAttributes.LLM_REQUEST_TYPE] == "completion"
    assert (
        attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"]
        == "Complete this sentence"
    )
    assert (
        attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"]
        == "A short completion."
    )
    assert attributes[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] == 4
    assert attributes[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS] == 3
    assert attributes["gen_ai.usage.input_tokens"] == 4
    assert attributes["gen_ai.usage.output_tokens"] == 3
    assert attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] == 7
    for alias in ("prompt_tokens", "completion_tokens", "total_request_tokens"):
        assert alias not in attributes


def test_embedding_events_capture_full_vectors(span_exporter):
    handler = RespanLlamaIndexEventHandler()

    handler.handle(
        SimpleNamespace(
            class_name=lambda: "EmbeddingStartEvent",
            span_id="span-3",
            model_dict={
                "class_name": "OpenAIEmbedding",
                "model": "text-embedding-3-small",
            },
        )
    )
    handler.handle(
        SimpleNamespace(
            class_name=lambda: "EmbeddingEndEvent",
            span_id="span-3",
            chunks=["alpha", "beta"],
            embeddings=[[0.1, 0.2], [0.3, 0.4]],
        )
    )

    embedding_span = next(
        span
        for span in span_exporter.get_finished_spans()
        if span.name == "llama_index.embedding"
    )
    attributes = embedding_span.attributes

    assert attributes[RESPAN_LOG_TYPE] == LOG_TYPE_EMBEDDING
    assert attributes[SpanAttributes.LLM_REQUEST_TYPE] == "embedding"
    assert attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"] == (
        '["alpha", "beta"]'
    )
    assert json.loads(attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]) == [
        [0.1, 0.2],
        [0.3, 0.4],
    ]
    assert "llm.embeddings.0" not in attributes


def test_tool_event_emits_tool_span(span_exporter):
    handler = RespanLlamaIndexEventHandler()

    handler.handle(
        SimpleNamespace(
            class_name=lambda: "AgentToolCallEvent",
            tool=SimpleNamespace(name="lookup_order"),
            arguments='{"order_id": "ord_123"}',
        )
    )

    tool_span = next(
        span
        for span in span_exporter.get_finished_spans()
        if span.name == "llama_index.tool.lookup_order"
    )

    assert tool_span.attributes[RESPAN_LOG_TYPE] == LOG_TYPE_TOOL
    assert "ord_123" in tool_span.attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT]


def test_tool_event_respects_content_capture_setting(span_exporter):
    handler = RespanLlamaIndexEventHandler(capture_content=False)

    handler.handle(
        SimpleNamespace(
            class_name=lambda: "AgentToolCallEvent",
            tool=SimpleNamespace(name="lookup_order"),
            arguments='{"order_id": "ord_123"}',
        )
    )

    tool_span = next(
        span
        for span in span_exporter.get_finished_spans()
        if span.name == "llama_index.tool.lookup_order"
    )

    assert tool_span.attributes[RESPAN_LOG_TYPE] == LOG_TYPE_TOOL
    assert SpanAttributes.TRACELOOP_ENTITY_INPUT not in tool_span.attributes


def test_exception_event_marks_open_event_span_error(span_exporter):
    handler = RespanLlamaIndexEventHandler()

    handler.handle(
        SimpleNamespace(
            class_name=lambda: "LLMCompletionStartEvent",
            span_id="span-4",
            prompt="raise",
            model_dict={},
        )
    )
    handler.handle(
        SimpleNamespace(
            class_name=lambda: "ExceptionEvent",
            span_id="span-4",
            exception=RuntimeError("llama failure"),
        )
    )
    handler.handle(
        SimpleNamespace(
            class_name=lambda: "LLMCompletionEndEvent",
            span_id="span-4",
            response=SimpleNamespace(text="fallback", raw={}),
        )
    )

    completion_span = next(
        span
        for span in span_exporter.get_finished_spans()
        if span.name == "llama_index.completion"
    )

    assert completion_span.status.status_code == StatusCode.ERROR
    assert completion_span.attributes["error.message"] == "llama failure"
    assert completion_span.attributes["status_code"] == 500
    assert json.loads(
        completion_span.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]
    ) == {
        "error": "RuntimeError",
        "message": "llama failure",
        "status": "error",
    }


def test_exception_event_respects_content_capture_setting(span_exporter):
    handler = RespanLlamaIndexEventHandler(capture_content=False)
    handler.handle(
        SimpleNamespace(
            class_name=lambda: "LLMCompletionStartEvent",
            span_id="span-private-error",
            prompt="top-secret prompt",
            model_dict={},
        )
    )
    handler.handle(
        SimpleNamespace(
            class_name=lambda: "ExceptionEvent",
            span_id="span-private-error",
            exception=RuntimeError("top-secret failure"),
        )
    )

    completion_span = next(
        span
        for span in span_exporter.get_finished_spans()
        if span.name == "llama_index.completion"
    )
    assert completion_span.status.status_code == StatusCode.ERROR
    assert completion_span.attributes["error.message"] == "RuntimeError"
    assert json.loads(
        completion_span.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]
    ) == {"error": "RuntimeError", "status": "error"}
    serialized = json.dumps(dict(completion_span.attributes))
    assert "top-secret" not in serialized
    assert not completion_span.events


def test_extract_usage_ignores_fractional_token_counts():
    prompt_tokens, completion_tokens, total_tokens = extract_usage(
        response={
            "usage": {
                "prompt_tokens": 3.5,
                "completion_tokens": 2,
                "total_tokens": 5.5,
            }
        }
    )

    assert prompt_tokens is None
    assert completion_tokens == 2
    assert total_tokens == 2
