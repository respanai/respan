import asyncio
import json
import logging
import sys
from collections import Counter, deque
from types import ModuleType, SimpleNamespace

import pytest
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv._incubating.attributes.error_attributes import ERROR_MESSAGE
from opentelemetry.semconv._incubating.attributes.http_attributes import (
    HTTP_STATUS_CODE,
)
from opentelemetry.semconv.attributes.http_attributes import HTTP_RESPONSE_STATUS_CODE
from opentelemetry.semconv_ai import SpanAttributes
from respan_instrumentation_microsoft_agent_framework import (
    MicrosoftAgentFrameworkInstrumentor,
    _instrumentation,
)
from respan_instrumentation_microsoft_agent_framework._constants import (
    TOP_LEVEL_ALIAS_ATTRS,
)
from respan_instrumentation_microsoft_agent_framework._processor import (
    AgentFrameworkSpanProcessor,
)
from respan_sdk.constants.llm_logging import (
    LOG_TYPE_AGENT,
    LOG_TYPE_CHAT,
    LOG_TYPE_TASK,
    LOG_TYPE_TOOL,
    LOG_TYPE_WORKFLOW,
)
from respan_sdk.constants.span_attributes import (
    RESPAN_LOG_TYPE,
    RESPAN_SPAN_HANDOFFS,
    RESPAN_SPAN_TOOL_CALLS,
    RESPAN_SPAN_TOOLS,
)
from respan_tracing.core.tracer import RespanTracer


class FakeActiveProcessor:
    def __init__(self, processors=()):
        self._span_processors = processors


class FakeTracerProvider:
    def __init__(self):
        self.export_processor = object()
        self._active_span_processor = FakeActiveProcessor((self.export_processor,))
        self.added_processors = []

    def add_span_processor(self, processor):
        self.added_processors.append(processor)


class FakeSpan:
    def __init__(
        self,
        attrs,
        name="chat gpt-4.1-nano",
        scope="agent_framework",
        *,
        status=None,
        events=(),
        parent=None,
    ):
        self.name = name
        self.attributes = dict(attrs)
        self._attributes = dict(attrs)
        self.instrumentation_scope = SimpleNamespace(name=scope)
        self.status = status
        self.events = events
        self.parent = parent


@pytest.fixture(autouse=True)
def reset_respan_tracer():
    RespanTracer.reset_instance()
    while _instrumentation._chat_telemetry_patch_users:
        _instrumentation._unpatch_chat_tool_capture()
    while _instrumentation._shared_processor_users:
        _instrumentation._release_shared_processor(
            _instrumentation._shared_processor_provider,
            _instrumentation._shared_processor,
        )
    yield
    while _instrumentation._chat_telemetry_patch_users:
        _instrumentation._unpatch_chat_tool_capture()
    while _instrumentation._shared_processor_users:
        _instrumentation._release_shared_processor(
            _instrumentation._shared_processor_provider,
            _instrumentation._shared_processor,
        )
    RespanTracer.reset_instance()


@pytest.fixture
def fake_tracer_provider(monkeypatch):
    provider = FakeTracerProvider()
    monkeypatch.setattr(
        _instrumentation.trace,
        "get_tracer_provider",
        lambda: provider,
    )
    return provider


def _install_fake_agent_framework(monkeypatch):
    calls = []
    agent_framework_module = ModuleType("agent_framework")
    observability_module = ModuleType("agent_framework.observability")
    observability_module.OBSERVABILITY_SETTINGS = SimpleNamespace(
        is_user_disabled=False,
        enable_sensitive_data=False,
    )

    def get_span_attributes(**kwargs):
        return dict(kwargs)

    observability_module._get_span_attributes = get_span_attributes

    class FakeChatTelemetryLayer:
        def get_response(self, messages, *, options=None, **kwargs):
            nested = None
            if isinstance(options, dict) and "nested_options" in options:
                nested = self.get_response(
                    messages,
                    options=options["nested_options"],
                )
            return {
                "messages": messages,
                "options": options,
                "telemetry_attrs": observability_module._get_span_attributes(),
                "nested": nested,
                **kwargs,
            }

    observability_module.ChatTelemetryLayer = FakeChatTelemetryLayer

    def enable_instrumentation(**kwargs):
        calls.append(kwargs)
        observability_module.OBSERVABILITY_SETTINGS.enable_sensitive_data = kwargs.get(
            "enable_sensitive_data",
            False,
        )

    observability_module.enable_instrumentation = enable_instrumentation
    agent_framework_module.observability = observability_module
    monkeypatch.setitem(sys.modules, "agent_framework", agent_framework_module)
    monkeypatch.setitem(
        sys.modules,
        "agent_framework.observability",
        observability_module,
    )
    return calls, observability_module


def _assert_no_off_contract_aliases(attrs):
    banned = TOP_LEVEL_ALIAS_ATTRS | {
        RESPAN_SPAN_TOOLS,
        RESPAN_SPAN_TOOL_CALLS,
        RESPAN_SPAN_HANDOFFS,
    }
    for key in banned:
        assert key not in attrs


def test_activate_registers_processor_and_enables_framework_observability(
    monkeypatch,
    fake_tracer_provider,
):
    calls, observability_module = _install_fake_agent_framework(monkeypatch)
    original_get_response = observability_module.ChatTelemetryLayer.get_response

    instrumentor = MicrosoftAgentFrameworkInstrumentor(capture_content=True)
    instrumentor.activate()

    processors = fake_tracer_provider._active_span_processor._span_processors
    assert isinstance(processors[0], AgentFrameworkSpanProcessor)
    assert processors[1] is fake_tracer_provider.export_processor
    assert calls == [{"enable_sensitive_data": True}]
    assert observability_module.OBSERVABILITY_SETTINGS.enable_sensitive_data is True
    assert instrumentor._is_instrumented is True
    captured = observability_module.ChatTelemetryLayer().get_response(
        ["hello"],
        options={"tools": ["lookup_weather"]},
        client_kwargs={"timeout": 5},
    )
    assert captured["client_kwargs"] == {"timeout": 5}
    assert captured["telemetry_attrs"]["tools"] == ["lookup_weather"]

    instrumentor.deactivate()

    assert fake_tracer_provider._active_span_processor._span_processors == (
        fake_tracer_provider.export_processor,
    )
    assert instrumentor._is_instrumented is False
    assert observability_module.ChatTelemetryLayer.get_response is original_get_response


def test_tool_capture_context_is_nested_and_does_not_leak(
    monkeypatch,
    fake_tracer_provider,
):
    _calls, observability_module = _install_fake_agent_framework(monkeypatch)
    original_get_span_attributes = observability_module._get_span_attributes
    instrumentor = MicrosoftAgentFrameworkInstrumentor()
    instrumentor.activate()

    layer = observability_module.ChatTelemetryLayer()
    nested = layer.get_response(
        ["hello"],
        options={
            "tools": ["outer_tool"],
            "nested_options": {"tools": ["inner_tool"]},
        },
    )
    plain = layer.get_response(["hello"], options={})

    assert nested["telemetry_attrs"]["tools"] == ["outer_tool"]
    assert nested["nested"]["telemetry_attrs"]["tools"] == ["inner_tool"]
    assert "tools" not in plain["telemetry_attrs"]
    assert _instrumentation._chat_tool_definitions.get() is None

    instrumentor.deactivate()
    assert observability_module._get_span_attributes is original_get_span_attributes


def test_activate_is_idempotent(monkeypatch, fake_tracer_provider):
    _install_fake_agent_framework(monkeypatch)

    instrumentor = MicrosoftAgentFrameworkInstrumentor()
    instrumentor.activate()
    instrumentor.activate()

    processors = fake_tracer_provider._active_span_processor._span_processors
    assert (
        sum(isinstance(item, AgentFrameworkSpanProcessor) for item in processors) == 1
    )
    instrumentor.deactivate()


def test_two_instrumentors_share_processor_until_final_deactivation(
    monkeypatch,
    fake_tracer_provider,
):
    _calls, observability_module = _install_fake_agent_framework(monkeypatch)
    original_get_response = observability_module.ChatTelemetryLayer.get_response
    processed = []
    original_on_end = AgentFrameworkSpanProcessor.on_end

    def count_on_end(processor, span):
        processed.append(span)
        return original_on_end(processor, span)

    monkeypatch.setattr(AgentFrameworkSpanProcessor, "on_end", count_on_end)
    first = MicrosoftAgentFrameworkInstrumentor()
    second = MicrosoftAgentFrameworkInstrumentor()
    first.activate()
    second.activate()
    try:
        processors = fake_tracer_provider._active_span_processor._span_processors
        normalizers = [
            item for item in processors if isinstance(item, AgentFrameworkSpanProcessor)
        ]
        assert len(normalizers) == 1
        assert first._processor is second._processor is normalizers[0]
        assert _instrumentation._shared_processor_users == 2
        assert _instrumentation._chat_telemetry_patch_users == 2

        span = FakeSpan({"gen_ai.operation.name": "chat"})
        for processor in normalizers:
            processor.on_end(span)
        assert processed == [span]

        first.deactivate()
        assert normalizers[0] in (
            fake_tracer_provider._active_span_processor._span_processors
        )
        assert _instrumentation._shared_processor_users == 1
        assert _instrumentation._chat_telemetry_patch_users == 1
        assert observability_module.ChatTelemetryLayer.get_response is not (
            original_get_response
        )
    finally:
        first.deactivate()
        second.deactivate()

    assert fake_tracer_provider._active_span_processor._span_processors == (
        fake_tracer_provider.export_processor,
    )
    assert _instrumentation._shared_processor_users == 0
    assert _instrumentation._chat_telemetry_patch_users == 0
    assert observability_module.ChatTelemetryLayer.get_response is original_get_response


def test_activate_skips_when_dependency_missing(
    monkeypatch,
    caplog,
    fake_tracer_provider,
):
    def import_module_raises(module_name):
        if module_name == "agent_framework.observability":
            raise ImportError(module_name)
        raise AssertionError(f"unexpected import: {module_name}")

    monkeypatch.setattr(
        _instrumentation.importlib, "import_module", import_module_raises
    )

    instrumentor = MicrosoftAgentFrameworkInstrumentor()
    with caplog.at_level(logging.WARNING):
        instrumentor.activate()

    assert "missing dependency" in caplog.text
    assert instrumentor._is_instrumented is False
    assert fake_tracer_provider._active_span_processor._span_processors == (
        fake_tracer_provider.export_processor,
    )


def test_activate_skips_when_respan_tracing_disabled(
    monkeypatch,
    caplog,
    fake_tracer_provider,
):
    _install_fake_agent_framework(monkeypatch)
    RespanTracer(is_enabled=False)

    instrumentor = MicrosoftAgentFrameworkInstrumentor()
    with caplog.at_level(logging.INFO):
        instrumentor.activate()

    assert "Respan tracing is disabled" in caplog.text
    assert instrumentor._is_instrumented is False
    assert fake_tracer_provider._active_span_processor._span_processors == (
        fake_tracer_provider.export_processor,
    )


def test_processor_maps_chat_span_and_removes_aliases():
    span = FakeSpan(
        {
            "gen_ai.operation.name": "chat",
            "gen_ai.provider.name": "openai",
            SpanAttributes.LLM_REQUEST_MODEL: "gpt-4.1-nano",
            "gen_ai.usage.input_tokens": 12,
            "gen_ai.usage.output_tokens": 7,
            GenAIAttributes.GEN_AI_SYSTEM_INSTRUCTIONS: json.dumps(
                [{"type": "text", "content": "Use concise answers."}]
            ),
            GenAIAttributes.GEN_AI_INPUT_MESSAGES: json.dumps(
                [{"role": "user", "content": "Use the weather tool for Seattle."}]
            ),
            GenAIAttributes.GEN_AI_OUTPUT_MESSAGES: json.dumps(
                [
                    {
                        "role": "assistant",
                        "parts": [
                            {
                                "type": "tool_call",
                                "id": "call_1",
                                "name": "lookup_weather",
                                "arguments": {"city": "Seattle"},
                            }
                        ],
                    }
                ]
            ),
            GenAIAttributes.GEN_AI_TOOL_DEFINITIONS: json.dumps(
                [{"name": "lookup_weather", "description": "Return weather."}]
            ),
            "model": "bad-alias",
            "tool_calls": "bad-alias",
            RESPAN_SPAN_TOOLS: "bad-alias",
            SpanAttributes.TRACELOOP_SPAN_KIND: "llm",
        }
    )

    AgentFrameworkSpanProcessor().on_end(span)
    attrs = span._attributes

    assert attrs[RESPAN_LOG_TYPE] == LOG_TYPE_CHAT
    assert attrs[SpanAttributes.LLM_REQUEST_TYPE] == "chat"
    assert attrs[SpanAttributes.LLM_REQUEST_MODEL] == "gpt-4.1-nano"
    assert attrs[SpanAttributes.LLM_SYSTEM] == "openai"
    assert attrs[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] == 12
    assert attrs[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS] == 7
    assert attrs[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] == 19
    assert attrs[f"{SpanAttributes.LLM_PROMPTS}.0.role"] == "system"
    assert attrs[f"{SpanAttributes.LLM_PROMPTS}.0.content"] == "Use concise answers."
    assert attrs[f"{SpanAttributes.LLM_PROMPTS}.1.role"] == "user"
    assert attrs[f"{SpanAttributes.LLM_PROMPTS}.1.content"] == (
        "Use the weather tool for Seattle."
    )
    tool_calls = json.loads(attrs[f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls"])
    assert tool_calls == [
        {
            "type": "function",
            "function": {
                "name": "lookup_weather",
                "arguments": '{"city": "Seattle"}',
            },
            "id": "call_1",
        }
    ]
    functions = json.loads(attrs[SpanAttributes.LLM_REQUEST_FUNCTIONS])
    assert functions[0]["function"]["name"] == "lookup_weather"

    for raw_key in (
        GenAIAttributes.GEN_AI_SYSTEM_INSTRUCTIONS,
        GenAIAttributes.GEN_AI_INPUT_MESSAGES,
        GenAIAttributes.GEN_AI_OUTPUT_MESSAGES,
        GenAIAttributes.GEN_AI_TOOL_DEFINITIONS,
    ):
        assert raw_key not in attrs
    assert SpanAttributes.TRACELOOP_SPAN_KIND not in attrs
    _assert_no_off_contract_aliases(attrs)


def test_processor_maps_tool_span_and_strips_gen_ai_tool_attrs():
    span = FakeSpan(
        {
            "gen_ai.operation.name": "execute_tool",
            "gen_ai.tool.name": "lookup_weather",
            "gen_ai.tool.call.id": "call_1",
            "gen_ai.tool.call.arguments": json.dumps({"city": "Seattle"}),
            "gen_ai.tool.call.result": "Sunny and 72F.",
            "tool_calls": "bad-alias",
        },
        name="execute_tool lookup_weather",
    )

    AgentFrameworkSpanProcessor().on_end(span)
    attrs = span._attributes

    assert attrs[RESPAN_LOG_TYPE] == LOG_TYPE_TOOL
    assert attrs[SpanAttributes.TRACELOOP_ENTITY_NAME] == "lookup_weather"
    assert attrs[SpanAttributes.TRACELOOP_ENTITY_PATH] == "lookup_weather"
    assert json.loads(attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT]) == {
        "name": "lookup_weather",
        "arguments": {"city": "Seattle"},
        "id": "call_1",
    }
    assert attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] == "Sunny and 72F."
    assert attrs[GenAIAttributes.GEN_AI_TOOL_CALL_ID] == "call_1"
    assert not any(
        key.startswith("gen_ai.tool.") and key != GenAIAttributes.GEN_AI_TOOL_CALL_ID
        for key in attrs
    )
    _assert_no_off_contract_aliases(attrs)


def test_processor_keeps_error_type_without_inventing_http_status():
    span = FakeSpan(
        {
            "gen_ai.operation.name": "execute_tool",
            "gen_ai.tool.name": "always_fail",
            "error.type": "RuntimeError",
        },
        name="execute_tool always_fail",
    )

    AgentFrameworkSpanProcessor().on_end(span)
    attrs = span._attributes

    assert attrs[RESPAN_LOG_TYPE] == LOG_TYPE_TOOL
    assert "status_code" not in attrs
    assert HTTP_RESPONSE_STATUS_CODE not in attrs
    assert attrs[ERROR_MESSAGE] == "RuntimeError"


def test_processor_does_not_infer_http_status_from_provider_error_text():
    span = FakeSpan(
        {
            "gen_ai.operation.name": "chat",
            "gen_ai.provider.name": "openai",
            SpanAttributes.LLM_REQUEST_MODEL: "gpt-4.1-nano",
            "error.type": "ChatClientException",
        },
        status=SimpleNamespace(
            status_code=SimpleNamespace(name="ERROR"),
            description=(
                "ChatClientException('OpenAI failed: Error code: 401 - "
                "invalid credentials')"
            ),
        ),
    )

    AgentFrameworkSpanProcessor().on_end(span)

    assert "status_code" not in span._attributes
    assert HTTP_RESPONSE_STATUS_CODE not in span._attributes
    assert "Error code: 401" in span._attributes[ERROR_MESSAGE]


def test_processor_does_not_treat_unlabelled_number_as_http_status():
    span = FakeSpan(
        {
            "gen_ai.operation.name": "execute_tool",
            "gen_ai.tool.name": "process_rows",
            "error.type": "RuntimeError",
        },
        name="execute_tool process_rows",
        status=SimpleNamespace(
            status_code=SimpleNamespace(name="ERROR"),
            description="RuntimeError('processed 401 rows')",
        ),
    )

    AgentFrameworkSpanProcessor().on_end(span)

    assert "status_code" not in span._attributes
    assert HTTP_RESPONSE_STATUS_CODE not in span._attributes


def test_processor_does_not_parse_application_status_text_as_http_status():
    span = FakeSpan(
        {
            "gen_ai.operation.name": "execute_tool",
            "gen_ai.tool.name": "read_cache",
            "error.type": "RuntimeError",
        },
        name="execute_tool read_cache",
        status=SimpleNamespace(
            status_code=SimpleNamespace(name="ERROR"),
            description="RuntimeError('cache status code: 404')",
        ),
    )

    AgentFrameworkSpanProcessor().on_end(span)

    assert "status_code" not in span._attributes
    assert HTTP_RESPONSE_STATUS_CODE not in span._attributes


def test_processor_maps_agent_and_workflow_spans():
    agent_span = FakeSpan(
        {
            "gen_ai.operation.name": "invoke_agent",
            "gen_ai.agent.name": "weather_agent",
            "gen_ai.provider.name": "microsoft.agent_framework",
            SpanAttributes.LLM_REQUEST_MODEL: "gpt-4.1-nano",
            "gen_ai.usage.input_tokens": 12,
            GenAIAttributes.GEN_AI_INPUT_MESSAGES: json.dumps(
                [{"role": "user", "content": "Hi"}]
            ),
            GenAIAttributes.GEN_AI_OUTPUT_MESSAGES: json.dumps(
                [{"role": "assistant", "content": "Hello"}]
            ),
        },
        name="invoke_agent weather_agent",
    )
    workflow_span = FakeSpan(
        {
            "workflow.name": "weather_workflow",
            "workflow.id": "wf_123",
        },
        name="workflow.run weather_workflow",
    )

    processor = AgentFrameworkSpanProcessor()
    processor.on_end(agent_span)
    processor.on_end(workflow_span)

    assert agent_span._attributes[RESPAN_LOG_TYPE] == LOG_TYPE_AGENT
    assert (
        agent_span._attributes[SpanAttributes.TRACELOOP_ENTITY_NAME] == "weather_agent"
    )
    assert json.loads(
        agent_span._attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT]
    ) == [{"role": "user", "content": "Hi"}]
    assert not any(
        key.startswith(("gen_ai.", "llm.")) for key in agent_span._attributes
    )
    assert workflow_span._attributes[RESPAN_LOG_TYPE] == LOG_TYPE_WORKFLOW
    assert (
        workflow_span._attributes[SpanAttributes.TRACELOOP_ENTITY_NAME]
        == "weather_workflow"
    )
    assert workflow_span._attributes[SpanAttributes.TRACELOOP_ENTITY_PATH] == ""


def test_real_framework_agent_exports_canonical_tool_contract(monkeypatch):
    from agent_framework import (
        Agent,
        BaseChatClient,
        ChatMiddlewareLayer,
        ChatResponse,
        Content,
        FunctionInvocationLayer,
        Message,
        observability,
        tool,
    )
    from agent_framework.observability import ChatTelemetryLayer
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    class DeterministicChatClient(
        FunctionInvocationLayer,
        ChatMiddlewareLayer,
        ChatTelemetryLayer,
        BaseChatClient,
    ):
        OTEL_PROVIDER_NAME = "openai"

        def __init__(self):
            self.model = "gpt-4.1-nano"
            self.provider_kwargs = []
            self.responses = deque(
                [
                    ChatResponse(
                        messages=Message(
                            "assistant",
                            [
                                Content.from_function_call(
                                    "weather-call-1",
                                    "lookup_weather",
                                    arguments={"city": "Seattle"},
                                )
                            ],
                        ),
                        model=self.model,
                        usage_details={
                            "input_token_count": 18,
                            "output_token_count": 7,
                        },
                    ),
                    ChatResponse(
                        messages=Message(
                            "assistant",
                            [Content.from_text("Seattle is sunny and 72F.")],
                        ),
                        model=self.model,
                        usage_details={
                            "input_token_count": 29,
                            "output_token_count": 8,
                        },
                    ),
                ]
            )
            super().__init__()

        def service_url(self):
            return "https://deterministic.invalid/v1"

        def _inner_get_response(self, *, messages, options, stream=False, **kwargs):
            self.provider_kwargs.append(dict(kwargs))

            async def get_response():
                return self.responses.popleft()

            return get_response()

    @tool
    def lookup_weather(city: str) -> str:
        """Return deterministic weather."""
        return f"{city} is sunny and 72F."

    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(_instrumentation.trace, "get_tracer_provider", lambda: provider)
    monkeypatch.setattr(
        observability,
        "get_tracer",
        lambda: provider.get_tracer("agent_framework"),
    )
    original_get_response = ChatTelemetryLayer.get_response
    instrumentor = MicrosoftAgentFrameworkInstrumentor(capture_content=True)
    instrumentor.activate()
    client = DeterministicChatClient()

    async def run_agent():
        agent = Agent(
            client=client,
            name="weather_agent",
            instructions="Use the weather tool.",
            tools=[lookup_weather],
        )
        return await agent.run("Use the weather tool for Seattle.")

    try:
        result = asyncio.run(run_agent())
        assert str(result) == "Seattle is sunny and 72F."
        assert provider.force_flush()
        spans = list(exporter.get_finished_spans())
    finally:
        instrumentor.deactivate()
        provider.shutdown()

    assert ChatTelemetryLayer.get_response is original_get_response
    assert len(spans) == 4
    assert all("tools" not in kwargs for kwargs in client.provider_kwargs)
    assert len({span.context.span_id for span in spans}) == len(spans)
    assert Counter(span.name for span in spans) == Counter(
        {
            "chat gpt-4.1-nano": 2,
            "execute_tool lookup_weather": 1,
            "invoke_agent weather_agent": 1,
        }
    )
    assert Counter(span.attributes[RESPAN_LOG_TYPE] for span in spans) == Counter(
        {
            LOG_TYPE_CHAT: 2,
            LOG_TYPE_TOOL: 1,
            LOG_TYPE_AGENT: 1,
        }
    )

    agent_span = next(
        span for span in spans if span.attributes[RESPAN_LOG_TYPE] == LOG_TYPE_AGENT
    )
    chat_spans = [
        span for span in spans if span.attributes[RESPAN_LOG_TYPE] == LOG_TYPE_CHAT
    ]
    tool_span = next(
        span for span in spans if span.attributes[RESPAN_LOG_TYPE] == LOG_TYPE_TOOL
    )
    assert len(chat_spans) == 2
    assert all(span.parent.span_id == agent_span.context.span_id for span in chat_spans)
    assert all(
        span.attributes[SpanAttributes.TRACELOOP_ENTITY_PATH] == span.name
        for span in chat_spans
    )
    assert tool_span.parent.span_id == agent_span.context.span_id

    first_chat = next(
        span
        for span in chat_spans
        if f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls" in span.attributes
    )
    final_chat = next(span for span in chat_spans if span is not first_chat)
    definitions = json.loads(
        first_chat.attributes[SpanAttributes.LLM_REQUEST_FUNCTIONS]
    )
    assert definitions[0]["function"]["name"] == "lookup_weather"
    assert definitions[0]["function"]["parameters"]["required"] == ["city"]
    calls = json.loads(
        first_chat.attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls"]
    )
    assert calls == [
        {
            "type": "function",
            "function": {
                "name": "lookup_weather",
                "arguments": '{"city": "Seattle"}',
            },
            "id": "weather-call-1",
        }
    ]
    assert f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls" not in final_chat.attributes
    assert f"{SpanAttributes.LLM_PROMPTS}.2.tool_calls" in final_chat.attributes
    assert SpanAttributes.LLM_REQUEST_FUNCTIONS in final_chat.attributes

    assert json.loads(tool_span.attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT]) == {
        "name": "lookup_weather",
        "arguments": {"city": "Seattle"},
        "id": "weather-call-1",
    }
    assert tool_span.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] == (
        "Seattle is sunny and 72F."
    )
    assert not any(key.startswith(("gen_ai.", "llm.")) for key in agent_span.attributes)
    assert (
        json.loads(agent_span.attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT])[0][
            "role"
        ]
        == "user"
    )
    assert (
        "Seattle is sunny and 72F."
        in agent_span.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]
    )
    for span in spans:
        assert SpanAttributes.TRACELOOP_SPAN_KIND not in span.attributes
        _assert_no_off_contract_aliases(span.attributes)
        assert GenAIAttributes.GEN_AI_TOOL_DEFINITIONS not in span.attributes


def test_real_framework_provider_failure_propagates_typed_http_status(monkeypatch):
    from agent_framework import (
        Agent,
        BaseChatClient,
        ChatMiddlewareLayer,
        FunctionInvocationLayer,
        WorkflowBuilder,
        WorkflowContext,
        executor,
        observability,
    )
    from agent_framework.exceptions import ChatClientInvalidAuthException
    from agent_framework.observability import ChatTelemetryLayer
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    class FailingChatClient(
        FunctionInvocationLayer,
        ChatMiddlewareLayer,
        ChatTelemetryLayer,
        BaseChatClient,
    ):
        OTEL_PROVIDER_NAME = "openai"

        def __init__(self):
            self.model = "gpt-4.1-nano"
            super().__init__()

        def service_url(self):
            return "https://deterministic.invalid/v1"

        def _inner_get_response(
            self,
            *,
            messages,
            options,
            stream=False,
            **kwargs,
        ):
            async def fail():
                raise ChatClientInvalidAuthException(
                    "OpenAI failed: Error code: 401 - invalid credentials"
                )

            return fail()

    client = FailingChatClient()
    agent = Agent(
        client=client,
        name="failing_agent",
        instructions="Fail deterministically.",
    )

    @executor(id="failing_executor", input=str, workflow_output=str)
    async def failing_executor(query: str, ctx: WorkflowContext):
        await ctx.yield_output(str(await agent.run(query)))

    native_workflow = WorkflowBuilder(
        start_executor=failing_executor,
        output_from=[failing_executor],
        name="failing_provider_workflow",
    ).build()

    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(_instrumentation.trace, "get_tracer_provider", lambda: provider)
    monkeypatch.setattr(
        observability,
        "get_tracer",
        lambda: provider.get_tracer("agent_framework"),
    )
    instrumentor = MicrosoftAgentFrameworkInstrumentor(capture_content=True)
    instrumentor.activate()
    try:
        with pytest.raises(
            ChatClientInvalidAuthException,
            match="Error code: 401",
        ):
            asyncio.run(native_workflow.run("Trigger provider authentication failure."))
        assert provider.force_flush()
        spans = list(exporter.get_finished_spans())
    finally:
        instrumentor.deactivate()
        provider.shutdown()

    assert Counter(
        span.name for span in spans if not span.name.startswith("edge_group.process")
    ) == Counter(
        {
            "chat gpt-4.1-nano": 1,
            "executor.process failing_executor": 1,
            "invoke_agent failing_agent": 1,
            "workflow.run": 1,
        }
    )
    assert len({span.context.span_id for span in spans}) == len(spans)
    failed_spans = [span for span in spans if span.status.status_code.name == "ERROR"]
    assert Counter(
        span.attributes[RESPAN_LOG_TYPE] for span in failed_spans
    ) == Counter(
        {
            LOG_TYPE_CHAT: 1,
            LOG_TYPE_AGENT: 1,
            LOG_TYPE_TASK: 1,
            LOG_TYPE_WORKFLOW: 1,
        }
    )
    assert all("status_code" not in span.attributes for span in failed_spans)
    assert all(
        HTTP_RESPONSE_STATUS_CODE not in span.attributes for span in failed_spans
    )
    assert all(
        "ChatClientInvalidAuthException" in span.attributes[ERROR_MESSAGE]
        for span in failed_spans
    )


@pytest.mark.parametrize("result", [False, 0])
def test_tool_result_history_is_content_not_another_tool_call(result):
    span = FakeSpan(
        {
            "gen_ai.operation.name": "chat",
            GenAIAttributes.GEN_AI_INPUT_MESSAGES: json.dumps(
                [
                    {
                        "role": "tool",
                        "parts": [
                            {
                                "type": "tool_call_response",
                                "name": "lookup",
                                "id": "past-call",
                                "response": result,
                            }
                        ],
                    }
                ]
            ),
        }
    )
    AgentFrameworkSpanProcessor().on_end(span)
    assert span._attributes["gen_ai.prompt.0.content"] == json.dumps(result)
    assert "gen_ai.prompt.0.tool_calls" not in span._attributes
    assert "gen_ai.completion.0.tool_calls" not in span._attributes


@pytest.mark.parametrize("native_key", [HTTP_RESPONSE_STATUS_CODE, HTTP_STATUS_CODE])
def test_processor_preserves_reported_http_status_canonically(native_key):
    span = FakeSpan(
        {
            GenAIAttributes.GEN_AI_OPERATION_NAME: "chat",
            native_key: 429,
            "error.type": "ChatClientException",
            "status_code": 500,
        },
        status=SimpleNamespace(
            status_code=SimpleNamespace(name="ERROR"),
            description="Provider reported HTTP 429",
        ),
    )
    AgentFrameworkSpanProcessor().on_end(span)
    assert span._attributes[HTTP_RESPONSE_STATUS_CODE] == 429
    assert HTTP_STATUS_CODE not in span._attributes
    assert "status_code" not in span._attributes
    assert span.status.status_code.name == "ERROR"
    assert span._attributes[ERROR_MESSAGE] == "Provider reported HTTP 429"
