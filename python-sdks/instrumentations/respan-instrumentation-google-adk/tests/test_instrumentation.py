import json
import logging
import sys
from types import ModuleType, SimpleNamespace

import pytest

from respan_instrumentation_google_adk import GoogleADKInstrumentor
from respan_instrumentation_google_adk import _instrumentation
from respan_instrumentation_google_adk._instrumentation import (
    OPENINFERENCE_GOOGLE_ADK_MODULE,
)
from respan_instrumentation_google_adk._processor import (
    GOOGLE_ADK_SCOPE_NAME,
    GoogleADKSpanProcessor,
)
from respan_tracing.core.tracer import RespanTracer


class BufferingSpanProcessor:
    pass


class FakeActiveProcessor:
    def __init__(self):
        self.export_processor = BufferingSpanProcessor()
        self._span_processors = (self.export_processor,)


class FakeTracerProvider:
    def __init__(self):
        self._active_span_processor = FakeActiveProcessor()
        self.added_processors = []

    def add_span_processor(self, processor):
        self.added_processors.append(processor)


class FakeSpan:
    def __init__(self, attrs, name="call_llm", trace_id=None):
        self.name = name
        self._attributes = dict(attrs)
        self.attributes = self._attributes
        self.instrumentation_scope = SimpleNamespace(name=GOOGLE_ADK_SCOPE_NAME)
        self.context = SimpleNamespace(trace_id=trace_id)


def _install_fake_modules(monkeypatch):
    monkeypatch.setattr(_instrumentation, "patch_legacy_agent_iterator", lambda: None)

    class FakeGoogleADKInstrumentor:
        created = []

        def __init__(self):
            self.instrument_kwargs = None
            self.is_instrumented = False
            self.is_uninstrumented = False
            self.__class__.created.append(self)

        def instrument(self, **kwargs):
            self.instrument_kwargs = kwargs
            self.is_instrumented = True

        def uninstrument(self):
            self.is_uninstrumented = True

    openinference_module = ModuleType("openinference")
    openinference_instrumentation_module = ModuleType("openinference.instrumentation")
    openinference_google_adk_module = ModuleType(OPENINFERENCE_GOOGLE_ADK_MODULE)
    openinference_google_adk_module.GoogleADKInstrumentor = FakeGoogleADKInstrumentor
    openinference_instrumentation_module.google_adk = openinference_google_adk_module

    monkeypatch.setitem(sys.modules, "openinference", openinference_module)
    monkeypatch.setitem(
        sys.modules,
        "openinference.instrumentation",
        openinference_instrumentation_module,
    )
    monkeypatch.setitem(
        sys.modules,
        OPENINFERENCE_GOOGLE_ADK_MODULE,
        openinference_google_adk_module,
    )

    return SimpleNamespace(
        google_adk_instrumentor_class=FakeGoogleADKInstrumentor,
    )


@pytest.fixture(autouse=True)
def reset_tracer():
    RespanTracer.reset_instance()
    yield
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


def test_activate_uses_openinference_google_adk(monkeypatch, fake_tracer_provider):
    fake = _install_fake_modules(monkeypatch)

    instrumentor = GoogleADKInstrumentor()
    instrumentor.activate()

    upstream = fake.google_adk_instrumentor_class.created[0]
    assert upstream.instrument_kwargs == {"tracer_provider": fake_tracer_provider}
    assert upstream.is_instrumented is True
    assert instrumentor._is_instrumented is True

    processors = fake_tracer_provider._active_span_processor._span_processors
    assert isinstance(processors[0], GoogleADKSpanProcessor)
    assert processors[1] is fake_tracer_provider._active_span_processor.export_processor

    instrumentor.deactivate()

    assert upstream.is_uninstrumented is True
    assert instrumentor._is_instrumented is False
    assert fake_tracer_provider._active_span_processor._span_processors == (
        fake_tracer_provider._active_span_processor.export_processor,
    )


def test_activate_passes_custom_openinference_kwargs(monkeypatch, fake_tracer_provider):
    fake = _install_fake_modules(monkeypatch)

    instrumentor = GoogleADKInstrumentor(trace_content=False)
    instrumentor.activate()

    upstream = fake.google_adk_instrumentor_class.created[0]
    assert upstream.instrument_kwargs == {
        "tracer_provider": fake_tracer_provider,
        "trace_content": False,
    }


def test_activate_cleans_up_when_upstream_declines(monkeypatch, fake_tracer_provider):
    fake = _install_fake_modules(monkeypatch)
    fake.google_adk_instrumentor_class.is_instrumented_by_opentelemetry = False
    instrumentor = GoogleADKInstrumentor()
    instrumentor.activate()
    assert instrumentor._is_instrumented is False
    assert instrumentor._processor is None
    assert instrumentor._instrumentor is None
    assert len(fake_tracer_provider._active_span_processor._span_processors) == 1


@pytest.mark.parametrize("completion,thoughts", [(3, 0), (3, 2), (0, 0)])
def test_processor_replaces_adk15_total_as_output_tokens(completion, thoughts):
    span = FakeSpan(
        {
            "openinference.span.kind": "LLM",
            "gen_ai.usage.input_tokens": 11,
            "gen_ai.usage.output_tokens": 11 + completion + thoughts,
            "gcp.vertex.agent.llm_response": json.dumps(
                {
                    "usage_metadata": {
                        "prompt_token_count": 11,
                        "candidates_token_count": completion,
                        "thoughts_token_count": thoughts,
                        "total_token_count": 11 + completion + thoughts,
                    }
                }
            ),
        }
    )
    GoogleADKSpanProcessor().on_end(span)
    assert span._attributes["gen_ai.usage.input_tokens"] == 11
    assert span._attributes["gen_ai.usage.prompt_tokens"] == 11
    assert span._attributes["gen_ai.usage.output_tokens"] == completion + thoughts
    assert span._attributes["gen_ai.usage.completion_tokens"] == completion + thoughts
    assert span._attributes["llm.usage.total_tokens"] == 11 + completion + thoughts


def test_activate_is_idempotent(monkeypatch, fake_tracer_provider):
    fake = _install_fake_modules(monkeypatch)

    instrumentor = GoogleADKInstrumentor()
    instrumentor.activate()
    instrumentor.activate()

    assert len(fake.google_adk_instrumentor_class.created) == 1
    processors = fake_tracer_provider._active_span_processor._span_processors
    assert sum(isinstance(item, GoogleADKSpanProcessor) for item in processors) == 1


def test_activate_cleans_up_processor_when_activation_fails(
    monkeypatch,
    caplog,
    fake_tracer_provider,
):
    fake = _install_fake_modules(monkeypatch)

    def instrument_raises(self, **kwargs):
        self.instrument_kwargs = kwargs
        raise RuntimeError("boom")

    monkeypatch.setattr(
        fake.google_adk_instrumentor_class,
        "instrument",
        instrument_raises,
    )

    instrumentor = GoogleADKInstrumentor()
    with caplog.at_level(logging.ERROR):
        instrumentor.activate()

    upstream = fake.google_adk_instrumentor_class.created[0]
    assert upstream.is_uninstrumented is True
    assert instrumentor._instrumentor is None
    assert instrumentor._processor is None
    assert instrumentor._is_instrumented is False
    assert fake_tracer_provider._active_span_processor._span_processors == (
        fake_tracer_provider._active_span_processor.export_processor,
    )
    assert "Failed to activate Google ADK instrumentation" in caplog.text


def test_activate_skips_when_respan_tracing_is_disabled(
    monkeypatch,
    caplog,
    fake_tracer_provider,
):
    fake = _install_fake_modules(monkeypatch)
    RespanTracer(is_enabled=False)

    instrumentor = GoogleADKInstrumentor()
    with caplog.at_level(logging.INFO):
        instrumentor.activate()

    assert fake.google_adk_instrumentor_class.created == []
    assert instrumentor._is_instrumented is False
    assert fake_tracer_provider._active_span_processor._span_processors == (
        fake_tracer_provider._active_span_processor.export_processor,
    )
    assert (
        "Google ADK instrumentation skipped because Respan tracing is disabled"
        in caplog.text
    )


def test_activate_logs_warning_when_dependencies_are_missing(
    monkeypatch,
    caplog,
    fake_tracer_provider,
):
    def import_module_raises(module_name):
        if module_name == OPENINFERENCE_GOOGLE_ADK_MODULE:
            raise ImportError(module_name)
        raise AssertionError(f"unexpected import: {module_name}")

    monkeypatch.setattr(
        _instrumentation.importlib,
        "import_module",
        import_module_raises,
    )
    instrumentor = GoogleADKInstrumentor()

    with caplog.at_level(logging.WARNING):
        instrumentor.activate()

    assert "Failed to activate Google ADK instrumentation" in caplog.text
    assert instrumentor._is_instrumented is False
    assert fake_tracer_provider._active_span_processor._span_processors == (
        fake_tracer_provider._active_span_processor.export_processor,
    )


def test_processor_promotes_google_adk_payloads_and_strips_local_noise():
    span = FakeSpan(
        {
            "openinference.span.kind": "LLM",
            "gen_ai.system": "gcp.vertex.agent",
            "llm.provider": "google",
            "gcp.vertex.agent.llm_request": json.dumps(
                {
                    "model": "openai/gpt-4o",
                    "config": {
                        "system_instruction": "You are concise.",
                        "tools": [
                            {
                                "function_declarations": [
                                    {
                                        "name": "get_weather",
                                        "description": "Get weather.",
                                        "parameters": {"type": "OBJECT"},
                                    }
                                ]
                            }
                        ],
                    },
                    "contents": [
                        {
                            "role": "user",
                            "parts": [{"text": "Weather in Tokyo?"}],
                        }
                    ],
                }
            ),
            "gcp.vertex.agent.llm_response": json.dumps(
                {
                    "content": {
                        "role": "model",
                        "parts": [{"text": "It is sunny."}],
                    },
                    "usage_metadata": {
                        "prompt_token_count": 12,
                        "candidates_token_count": 5,
                        "total_token_count": 17,
                    },
                }
            ),
        }
    )

    GoogleADKSpanProcessor().on_end(span)

    assert span._attributes["respan.entity.log_type"] == "chat"
    assert span._attributes["gen_ai.system"] == "google"
    assert span._attributes["gen_ai.request.model"] == "openai/gpt-4o"
    assert span._attributes["gen_ai.prompt.0.content"] == "You are concise."
    assert span._attributes["gen_ai.prompt.1.content"] == "Weather in Tokyo?"
    assert span._attributes["gen_ai.completion.0.role"] == "assistant"
    assert span._attributes["gen_ai.completion.0.content"] == "It is sunny."
    assert span._attributes["gen_ai.usage.prompt_tokens"] == 12
    assert span._attributes["gen_ai.usage.completion_tokens"] == 5
    assert span._attributes["llm.usage.total_tokens"] == 17
    assert json.loads(span._attributes["llm.request.functions"]) == [
        {
            "name": "get_weather",
            "description": "Get weather.",
            "parameters": {"type": "OBJECT"},
        }
    ]
    assert "traceloop.span.kind" not in span._attributes
    assert "model" not in span._attributes
    assert "tools" not in span._attributes
    assert "tool_calls" not in span._attributes
    assert "respan.span.tools" not in span._attributes
    assert "gcp.vertex.agent.llm_request" not in span._attributes
    assert "gcp.vertex.agent.llm_response" not in span._attributes


def test_processor_promotes_openinference_message_content_blocks():
    span = FakeSpan(
        {
            "openinference.span.kind": "LLM",
            "llm.input_messages.0.message.role": "user",
            "llm.input_messages.0.message.contents.0.message_content.type": "text",
            "llm.input_messages.0.message.contents.0.message_content.text": "Hello",
            "llm.output_messages.0.message.role": "model",
            "llm.output_messages.0.message.contents.0.message_content.type": "text",
            "llm.output_messages.0.message.contents.0.message_content.text": "Hi there",
            "llm.output_messages.0.message.tool_calls.0.tool_call.function.name": "lookup",
            "llm.output_messages.0.message.tool_calls.0.tool_call.function.arguments": (
                '{"query":"weather"}'
            ),
        }
    )

    GoogleADKSpanProcessor().on_end(span)

    assert span._attributes["gen_ai.prompt.0.role"] == "user"
    assert span._attributes["gen_ai.prompt.0.content"] == "Hello"
    assert span._attributes["gen_ai.completion.0.role"] == "assistant"
    assert span._attributes["gen_ai.completion.0.content"] == "Hi there"
    assert json.loads(span._attributes["gen_ai.completion.0.tool_calls"]) == [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "arguments": '{"query":"weather"}',
            },
        }
    ]
    assert "gen_ai.completion.0.tool_calls.0.function.name" not in span._attributes
    assert "llm.output_messages.0.message.contents.0.message_content.text" not in (
        span._attributes
    )


def test_processor_cleans_google_adk_tool_attrs():
    span = FakeSpan(
        {
            "openinference.span.kind": "TOOL",
            "tool.name": "get_weather",
            "gen_ai.tool.name": "get_weather",
            "gcp.vertex.agent.tool_call_args": '{"city":"Paris"}',
            "gcp.vertex.agent.tool_response": '{"result":"sunny"}',
        },
        name="execute_tool get_weather",
    )

    GoogleADKSpanProcessor().on_end(span)

    assert span._attributes["respan.entity.log_type"] == "tool"
    assert json.loads(span._attributes["traceloop.entity.input"]) == {
        "name": "get_weather",
        "arguments": {"city": "Paris"},
    }
    assert span._attributes["traceloop.entity.output"] == '{"result":"sunny"}'
    assert "gen_ai.system" not in span._attributes
    assert "traceloop.span.kind" not in span._attributes
    assert "tool.name" not in span._attributes
    assert "gen_ai.tool.name" not in span._attributes
    assert "gcp.vertex.agent.tool_call_args" not in span._attributes
    assert "gcp.vertex.agent.tool_response" not in span._attributes


def test_processor_preserves_tool_result_identity_in_chat_history():
    span = FakeSpan(
        {
            "openinference.span.kind": "LLM",
            "llm.input_messages.0.message.role": "tool",
            "llm.input_messages.0.message.content": '{"result":"sunny"}',
            "gcp.vertex.agent.llm_request": json.dumps(
                {
                    "model": "test-model",
                    "contents": [
                        {
                            "role": "user",
                            "parts": [
                                {
                                    "function_response": {
                                        "id": "call-weather-1",
                                        "name": "get_weather",
                                        "response": {"result": "sunny"},
                                    }
                                }
                            ],
                        }
                    ],
                }
            ),
        }
    )

    GoogleADKSpanProcessor().on_end(span)

    assert span._attributes["gen_ai.prompt.0.role"] == "tool"
    assert json.loads(span._attributes["gen_ai.prompt.0.content"]) == {
        "id": "call-weather-1",
        "name": "get_weather",
        "response": {"result": "sunny"},
    }


def test_processor_aligns_adk_history_after_existing_system_prompt():
    span = FakeSpan(
        {
            "openinference.span.kind": "LLM",
            "llm.input_messages.0.message.role": "system",
            "llm.input_messages.0.message.content": "Use the weather tool.",
            "llm.input_messages.1.message.role": "user",
            "llm.input_messages.1.message.content": "Weather in Paris?",
            "llm.input_messages.2.message.role": "assistant",
            "llm.input_messages.2.message.tool_calls.0.tool_call.id": (
                "call-weather-1"
            ),
            "llm.input_messages.2.message.tool_calls.0.tool_call.function.name": (
                "get_weather"
            ),
            "llm.input_messages.2.message.tool_calls.0.tool_call.function.arguments": (
                '{"city":"Paris"}'
            ),
            "llm.input_messages.3.message.role": "tool",
            "llm.input_messages.3.message.content": '{"result":"sunny"}',
            "gcp.vertex.agent.llm_request": json.dumps(
                {
                    "model": "test-model",
                    "contents": [
                        {"role": "user", "parts": [{"text": "Weather in Paris?"}]},
                        {
                            "role": "model",
                            "parts": [
                                {
                                    "function_call": {
                                        "id": "call-weather-1",
                                        "name": "get_weather",
                                        "args": {"city": "Paris"},
                                    }
                                }
                            ],
                        },
                        {
                            "role": "user",
                            "parts": [
                                {
                                    "function_response": {
                                        "id": "call-weather-1",
                                        "name": "get_weather",
                                        "response": {"result": "sunny"},
                                    }
                                }
                            ],
                        },
                    ],
                }
            ),
        }
    )

    GoogleADKSpanProcessor().on_end(span)

    assert span._attributes["gen_ai.prompt.0.role"] == "system"
    assert span._attributes["gen_ai.prompt.1.role"] == "user"
    assert span._attributes["gen_ai.prompt.2.role"] == "assistant"
    assert span._attributes["gen_ai.prompt.3.role"] == "tool"
    assert json.loads(span._attributes["gen_ai.prompt.3.content"]) == {
        "id": "call-weather-1",
        "name": "get_weather",
        "response": {"result": "sunny"},
    }
    assert "gen_ai.prompt.4.role" not in span._attributes


def test_processor_promotes_session_and_carries_prompt_to_agent():
    processor = GoogleADKSpanProcessor()
    llm_span = FakeSpan(
        {
            "openinference.span.kind": "LLM",
            "session.id": "session-123",
            "llm.input_messages.0.message.role": "user",
            "llm.input_messages.0.message.content": "Plan a Tokyo day trip.",
            "llm.output_messages.0.message.role": "model",
            "llm.output_messages.0.message.content": "Here is a plan.",
        },
        trace_id=1234,
    )
    agent_span = FakeSpan(
        {
            "openinference.span.kind": "AGENT",
            "input.value": "",
            "output.value": '{"answer":"Here is a plan."}',
        },
        name="travel_agent",
        trace_id=1234,
    )

    processor.on_end(llm_span)
    processor.on_end(agent_span)

    assert llm_span._attributes["respan.sessions.session_identifier"] == "session-123"
    assert agent_span._attributes["respan.sessions.session_identifier"] == "session-123"
    assert json.loads(agent_span._attributes["traceloop.entity.input"]) == {
        "prompt": "Plan a Tokyo day trip."
    }
    assert "session.id" not in llm_span._attributes
    assert "gen_ai.request.model" not in agent_span._attributes
    assert processor._trace_context == {}
