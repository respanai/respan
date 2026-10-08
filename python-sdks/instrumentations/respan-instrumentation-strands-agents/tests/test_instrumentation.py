import json
import logging
import os
import sys
from types import ModuleType, SimpleNamespace

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.semconv._incubating.attributes import gen_ai_attributes as gen_ai
from opentelemetry.semconv_ai import LLMRequestTypeValues, SpanAttributes
from opentelemetry.trace import Status, StatusCode
from respan_instrumentation_strands_agents import (
    StrandsAgentsInstrumentor,
    _instrumentation,
)
from respan_instrumentation_strands_agents._constants import (
    STRANDS_EVENT_CHOICE,
    STRANDS_EVENT_TOOL_MESSAGE,
    STRANDS_EVENT_USER_MESSAGE,
    STRANDS_OPERATION_CHAT,
    STRANDS_OPERATION_EXECUTE_TOOL,
    STRANDS_OPERATION_INVOKE_AGENT,
    STRANDS_SEMCONV_TOOL_DEFINITIONS_OPT_IN,
    STRANDS_SYSTEM_NAME,
    STRANDS_TOOL_JSON_SCHEMA_ATTR,
    STRANDS_TOP_LEVEL_ALIAS_ATTRS_TO_STRIP,
)
from respan_instrumentation_strands_agents._processor import (
    StrandsAgentsSpanProcessor,
    enrich_strands_agents_span,
)
from respan_instrumentation_strands_agents._serialization import json_dumps, safe_text
from respan_sdk.constants.llm_logging import (
    LOG_TYPE_AGENT,
    LOG_TYPE_CHAT,
    LOG_TYPE_TOOL,
    LogMethodChoices,
)
from respan_sdk.constants.span_attributes import (
    RESPAN_LOG_METHOD,
    RESPAN_LOG_TYPE,
    RESPAN_SPAN_HANDOFFS,
    RESPAN_SPAN_TOOL_CALLS,
    RESPAN_SPAN_TOOLS,
)


def _make_fake_tracer_provider():
    tracer_provider = SimpleNamespace(
        _active_span_processor=SimpleNamespace(_span_processors=()),
        added_processors=[],
        get_tracer=lambda service_name: f"tracer:{service_name}",
    )

    def add_span_processor(processor):
        tracer_provider.added_processors.append(processor)

    tracer_provider.add_span_processor = add_span_processor
    return tracer_provider


def _install_fake_strands_modules(monkeypatch):
    tracer_instance = SimpleNamespace(
        service_name="strands.telemetry.tracer",
        tracer_provider=None,
        tracer=None,
        _include_tool_definitions=False,
    )

    def _parse_semconv_opt_in():
        opt_in = os.getenv("OTEL_SEMCONV_STABILITY_OPT_IN", "")
        return {value.strip() for value in opt_in.split(",") if value.strip()}

    tracer_instance._parse_semconv_opt_in = _parse_semconv_opt_in

    strands_module = ModuleType("strands")
    strands_module.__path__ = []
    telemetry_module = ModuleType("strands.telemetry")
    telemetry_module.__path__ = []
    tracer_module = ModuleType("strands.telemetry.tracer")
    tracer_module._tracer_instance = tracer_instance

    monkeypatch.setitem(sys.modules, "strands", strands_module)
    monkeypatch.setitem(sys.modules, "strands.telemetry", telemetry_module)
    monkeypatch.setitem(sys.modules, "strands.telemetry.tracer", tracer_module)
    return tracer_instance


def _make_span(name, attributes=None, events=None):
    attrs = dict(attributes or {})
    return SimpleNamespace(
        name=name,
        _attributes=attrs,
        attributes=attrs,
        events=tuple(events or ()),
    )


def _event(name, attributes):
    return SimpleNamespace(name=name, attributes=attributes)


def _assert_no_off_contract_aliases(attrs):
    for key in (
        RESPAN_SPAN_TOOLS,
        RESPAN_SPAN_TOOL_CALLS,
        RESPAN_SPAN_HANDOFFS,
        *STRANDS_TOP_LEVEL_ALIAS_ATTRS_TO_STRIP,
    ):
        assert key not in attrs


def test_package_exports_instrumentor():
    assert StrandsAgentsInstrumentor is _instrumentation.StrandsAgentsInstrumentor
    assert StrandsAgentsInstrumentor.name == "strands-agents"


def test_activate_registers_processor_first_and_refreshes_existing_tracer(
    monkeypatch,
):
    tracer_provider = _make_fake_tracer_provider()
    existing_processor = object()
    tracer_provider._active_span_processor._span_processors = (existing_processor,)
    tracer_instance = _install_fake_strands_modules(monkeypatch)
    monkeypatch.setattr(trace, "get_tracer_provider", lambda: tracer_provider)
    monkeypatch.delenv("OTEL_SEMCONV_STABILITY_OPT_IN", raising=False)

    instrumentor = StrandsAgentsInstrumentor()
    instrumentor.activate()

    processors = tracer_provider._active_span_processor._span_processors
    assert isinstance(processors[0], StrandsAgentsSpanProcessor)
    assert processors[1] is existing_processor
    assert tracer_instance.tracer_provider is tracer_provider
    assert tracer_instance.tracer == "tracer:strands.telemetry.tracer"
    assert tracer_instance._include_tool_definitions is True
    assert (
        os.environ["OTEL_SEMCONV_STABILITY_OPT_IN"]
        == STRANDS_SEMCONV_TOOL_DEFINITIONS_OPT_IN
    )

    instrumentor.deactivate()

    assert tracer_provider._active_span_processor._span_processors == (
        existing_processor,
    )
    assert tracer_instance.tracer_provider is None
    assert tracer_instance.tracer is None
    assert tracer_instance._include_tool_definitions is False
    assert "OTEL_SEMCONV_STABILITY_OPT_IN" not in os.environ


def test_activation_rejects_config_mismatch_and_preserves_later_env_owner(
    monkeypatch,
):
    tracer_provider = _make_fake_tracer_provider()
    _install_fake_strands_modules(monkeypatch)
    monkeypatch.setattr(trace, "get_tracer_provider", lambda: tracer_provider)
    monkeypatch.delenv("OTEL_SEMCONV_STABILITY_OPT_IN", raising=False)

    owner = StrandsAgentsInstrumentor(include_tool_definitions=True)
    owner.activate()
    with pytest.raises(ValueError, match="different include_tool_definitions"):
        StrandsAgentsInstrumentor(include_tool_definitions=False).activate()

    os.environ["OTEL_SEMCONV_STABILITY_OPT_IN"] = "foreign-owner"
    owner.deactivate()
    assert os.environ["OTEL_SEMCONV_STABILITY_OPT_IN"] == "foreign-owner"


def test_real_strands_tracer_moves_agent_tools_to_child_chat(monkeypatch):
    from strands.telemetry import tracer as tracer_module

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(trace, "get_tracer_provider", lambda: provider)
    monkeypatch.delenv("OTEL_SEMCONV_STABILITY_OPT_IN", raising=False)

    monkeypatch.setattr(tracer_module, "_tracer_instance", None)
    instrumentor = StrandsAgentsInstrumentor()
    try:
        instrumentor.activate()
        tracer_instance = tracer_module.get_tracer()
        agent_span = tracer_instance.start_agent_span(
            messages=[{"role": "user", "content": [{"text": "weather?"}]}],
            agent_name="WeatherAgent",
            tools_config={
                "get_weather": {
                    "description": "Get the weather.",
                    "inputSchema": {
                        "json": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                        }
                    },
                }
            },
        )
        chat_span = tracer_instance.start_model_invoke_span(
            messages=[{"role": "user", "content": [{"text": "weather?"}]}],
            parent_span=agent_span,
            model_id="gpt-4o-mini",
        )
        chat_span.end()
        tracer_instance.end_agent_span(agent_span)

        spans = exporter.get_finished_spans()
        assert [span.name for span in spans] == ["chat", "invoke_agent WeatherAgent"]
        chat_attrs = dict(spans[0].attributes)
        agent_attrs = dict(spans[1].attributes)
        tools = json.loads(chat_attrs[SpanAttributes.LLM_REQUEST_FUNCTIONS])
        assert tools == [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the weather.",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    },
                },
            }
        ]
        assert SpanAttributes.LLM_REQUEST_FUNCTIONS not in agent_attrs
        assert spans[0].parent.span_id == spans[1].context.span_id
    finally:
        instrumentor.deactivate()
        assert tracer_instance._include_tool_definitions is False


def test_real_export_drops_raw_events_and_preserves_private_bounded_error():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(StrandsAgentsSpanProcessor())
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    span = provider.get_tracer("strands.telemetry.tracer").start_span(
        STRANDS_OPERATION_CHAT,
        attributes={
            SpanAttributes.LLM_SYSTEM: STRANDS_SYSTEM_NAME,
            gen_ai.GEN_AI_OPERATION_NAME: STRANDS_OPERATION_CHAT,
            "http.response.status_code": 429,
        },
    )
    provider._active_span_processor._span_processors[0].register_native(
        span, True, span.attributes
    )
    span.record_exception(RuntimeError('{"api_key":"plain-secret"}'))
    span.set_status(
        Status(StatusCode.ERROR, '{"api_key":"plain-secret"}' + "😀" * 10_000)
    )
    span.end()

    exported = exporter.get_finished_spans()
    assert len(exported) == 1
    result = exported[0]
    attrs = dict(result.attributes)
    assert len(result.events) == 1
    assert result.events[0].name == "exception"
    assert result.status.status_code is StatusCode.ERROR
    assert len((result.status.description or "").encode("utf-8")) <= 4_000
    assert "plain-secret" not in (result.status.description or "")
    assert "status_code" not in attrs
    assert "error.message" not in attrs
    assert SpanAttributes.TRACELOOP_ENTITY_OUTPUT not in attrs


def test_activate_logs_warning_when_dependency_missing(monkeypatch, caplog):
    def raise_import_error(module_name):
        if module_name == "strands.telemetry.tracer":
            raise ImportError(module_name)
        return __import__(module_name)

    monkeypatch.setattr(
        _instrumentation.importlib,
        "import_module",
        raise_import_error,
    )
    instrumentor = StrandsAgentsInstrumentor()

    with caplog.at_level(logging.WARNING):
        instrumentor.activate()

    assert "Failed to activate Strands Agents instrumentation" in caplog.text


def test_enrich_agent_span_maps_common_fields_and_tool_definitions():
    span = _make_span(
        name=f"{STRANDS_OPERATION_INVOKE_AGENT} WeatherAgent",
        attributes={
            SpanAttributes.LLM_SYSTEM: STRANDS_SYSTEM_NAME,
            gen_ai.GEN_AI_OPERATION_NAME: STRANDS_OPERATION_INVOKE_AGENT,
            gen_ai.GEN_AI_AGENT_NAME: "WeatherAgent",
            SpanAttributes.LLM_REQUEST_MODEL: "gpt-4o-mini",
            gen_ai.GEN_AI_TOOL_DEFINITIONS: json.dumps(
                [
                    {
                        "name": "get_weather",
                        "description": "Get weather.",
                        "inputSchema": {"type": "object"},
                    }
                ]
            ),
            gen_ai.GEN_AI_USAGE_INPUT_TOKENS: 30,
            gen_ai.GEN_AI_USAGE_OUTPUT_TOKENS: 6,
        },
        events=[
            _event(
                STRANDS_EVENT_USER_MESSAGE,
                {"content": json.dumps([{"text": "weather in Seattle"}])},
            ),
            _event(STRANDS_EVENT_CHOICE, {"message": "It is sunny."}),
        ],
    )

    enrich_strands_agents_span(span)

    attrs = span._attributes
    assert attrs[RESPAN_LOG_TYPE] == LOG_TYPE_AGENT
    assert attrs[RESPAN_LOG_METHOD] == LogMethodChoices.TRACING_INTEGRATION.value
    assert attrs[SpanAttributes.TRACELOOP_ENTITY_NAME] == "WeatherAgent"
    assert attrs[SpanAttributes.TRACELOOP_ENTITY_PATH] == "WeatherAgent"
    assert attrs[SpanAttributes.TRACELOOP_WORKFLOW_NAME] == "WeatherAgent"
    assert json.loads(attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT]) == [
        {"role": "user", "content": "weather in Seattle"}
    ]
    assert json.loads(attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]) == [
        {"role": "assistant", "content": "It is sunny."}
    ]
    assert SpanAttributes.LLM_REQUEST_FUNCTIONS not in attrs
    assert SpanAttributes.LLM_REQUEST_MODEL not in attrs
    assert SpanAttributes.LLM_SYSTEM not in attrs
    assert SpanAttributes.LLM_REQUEST_TYPE not in attrs
    assert gen_ai.GEN_AI_USAGE_INPUT_TOKENS not in attrs
    assert SpanAttributes.TRACELOOP_SPAN_KIND not in attrs
    assert gen_ai.GEN_AI_AGENT_NAME not in attrs
    _assert_no_off_contract_aliases(attrs)


def test_enrich_chat_span_maps_messages_usage_and_tool_calls():
    span = _make_span(
        name=STRANDS_OPERATION_CHAT,
        attributes={
            SpanAttributes.LLM_SYSTEM: STRANDS_SYSTEM_NAME,
            gen_ai.GEN_AI_OPERATION_NAME: STRANDS_OPERATION_CHAT,
            SpanAttributes.LLM_REQUEST_MODEL: "gpt-4o-mini",
            gen_ai.GEN_AI_USAGE_INPUT_TOKENS: 12,
            gen_ai.GEN_AI_USAGE_OUTPUT_TOKENS: 4,
            SpanAttributes.GEN_AI_USAGE_TOTAL_TOKENS: 16,
        },
        events=[
            _event(
                STRANDS_EVENT_USER_MESSAGE,
                {"content": json.dumps([{"text": "use the tool"}])},
            ),
            _event(
                STRANDS_EVENT_CHOICE,
                {
                    "message": json.dumps(
                        [
                            {
                                "toolUse": {
                                    "toolUseId": "tool_1",
                                    "name": "lookup",
                                    "input": {"query": "tracing"},
                                }
                            }
                        ]
                    )
                },
            ),
        ],
    )

    enrich_strands_agents_span(span)

    attrs = span._attributes
    assert attrs[RESPAN_LOG_TYPE] == LOG_TYPE_CHAT
    assert attrs[SpanAttributes.LLM_REQUEST_TYPE] == LLMRequestTypeValues.CHAT.value
    assert attrs[SpanAttributes.LLM_REQUEST_MODEL] == "gpt-4o-mini"
    assert attrs[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] == 12
    assert attrs[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS] == 4
    assert attrs[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] == 16
    assert attrs[f"{SpanAttributes.LLM_PROMPTS}.0.role"] == "user"
    assert attrs[f"{SpanAttributes.LLM_PROMPTS}.0.content"] == "use the tool"
    assert attrs[f"{SpanAttributes.LLM_COMPLETIONS}.0.role"] == "assistant"
    tool_calls = json.loads(attrs[f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls"])
    assert tool_calls == [
        {
            "id": "tool_1",
            "type": "function",
            "function": {
                "name": "lookup",
                "arguments": '{"query":"tracing"}',
            },
        }
    ]
    assert SpanAttributes.TRACELOOP_SPAN_KIND not in attrs
    assert gen_ai.GEN_AI_OPERATION_NAME not in attrs
    _assert_no_off_contract_aliases(attrs)


def test_enrich_tool_span_maps_input_and_output():
    span = _make_span(
        name=f"{STRANDS_OPERATION_EXECUTE_TOOL} get_weather",
        attributes={
            SpanAttributes.LLM_SYSTEM: STRANDS_SYSTEM_NAME,
            gen_ai.GEN_AI_OPERATION_NAME: STRANDS_OPERATION_EXECUTE_TOOL,
            gen_ai.GEN_AI_TOOL_NAME: "get_weather",
            gen_ai.GEN_AI_TOOL_CALL_ID: "tool_1",
            gen_ai.GEN_AI_TOOL_DESCRIPTION: "Get weather.",
            STRANDS_TOOL_JSON_SCHEMA_ATTR: json.dumps({"type": "object"}),
        },
        events=[
            _event(STRANDS_EVENT_TOOL_MESSAGE, {"content": '{"city":"Seattle"}'}),
            _event(
                STRANDS_EVENT_CHOICE,
                {"message": json.dumps([{"text": "Sunny and 72F."}])},
            ),
        ],
    )

    enrich_strands_agents_span(span)

    attrs = span._attributes
    assert attrs[RESPAN_LOG_TYPE] == LOG_TYPE_TOOL
    assert attrs[SpanAttributes.TRACELOOP_ENTITY_NAME] == "get_weather"
    assert attrs[SpanAttributes.TRACELOOP_ENTITY_PATH] == "get_weather"
    assert json.loads(attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT]) == {
        "name": "get_weather",
        "arguments": {"city": "Seattle"},
    }
    assert json.loads(attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]) == "Sunny and 72F."
    assert gen_ai.GEN_AI_TOOL_NAME not in attrs
    assert SpanAttributes.LLM_SYSTEM not in attrs
    assert SpanAttributes.LLM_REQUEST_TYPE not in attrs
    assert attrs[gen_ai.GEN_AI_TOOL_CALL_ID] == "tool_1"
    assert gen_ai.GEN_AI_TOOL_DESCRIPTION not in attrs
    assert STRANDS_TOOL_JSON_SCHEMA_ATTR not in attrs
    assert SpanAttributes.TRACELOOP_SPAN_KIND not in attrs
    _assert_no_off_contract_aliases(attrs)


def test_chat_inherits_normalized_agent_tool_definitions():
    span = _make_span(
        name=STRANDS_OPERATION_CHAT,
        attributes={
            SpanAttributes.LLM_SYSTEM: STRANDS_SYSTEM_NAME,
            gen_ai.GEN_AI_OPERATION_NAME: STRANDS_OPERATION_CHAT,
        },
    )
    inherited = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
            },
        }
    ]
    enrich_strands_agents_span(span, inherited_tool_definitions=inherited)
    assert (
        json.loads(span._attributes[SpanAttributes.LLM_REQUEST_FUNCTIONS]) == inherited
    )


def test_tool_schema_json_envelope_is_unwrapped():
    span = _make_span(
        name=STRANDS_OPERATION_CHAT,
        attributes={
            SpanAttributes.LLM_SYSTEM: STRANDS_SYSTEM_NAME,
            gen_ai.GEN_AI_OPERATION_NAME: STRANDS_OPERATION_CHAT,
            gen_ai.GEN_AI_TOOL_DEFINITIONS: json.dumps(
                [{"name": "lookup", "inputSchema": {"json": {"type": "object"}}}]
            ),
        },
    )
    enrich_strands_agents_span(span)
    tools = json.loads(span._attributes[SpanAttributes.LLM_REQUEST_FUNCTIONS])
    assert tools[0]["function"]["parameters"] == {"type": "object"}


def test_scalar_and_json_serialization_are_bounded_private_and_hostile_safe():
    class Hostile:
        @property
        def model_dump(self):
            raise AssertionError("must not inspect hostile property")

        def __str__(self):
            raise AssertionError("must not stringify")

    text = safe_text('{"api_key":"plain-secret"}' + "😀" * 10_000)
    payload = json_dumps(
        {
            "client_secret": "nested-secret",
            "hostile": Hostile(),
            "text": "😀" * 10_000,
        }
    )
    assert len(text.encode("utf-8")) <= 4_000
    assert "plain-secret" not in text
    assert len(payload.encode("utf-8")) <= 16_000
    assert "nested-secret" not in payload
    assert "Hostile" in payload

    span = _make_span(
        name=STRANDS_OPERATION_CHAT,
        attributes={
            SpanAttributes.LLM_SYSTEM: STRANDS_SYSTEM_NAME,
            gen_ai.GEN_AI_OPERATION_NAME: STRANDS_OPERATION_CHAT,
            SpanAttributes.LLM_REQUEST_MODEL: "api_key=plain-secret",
        },
    )
    enrich_strands_agents_span(span)
    assert "plain-secret" not in span._attributes[SpanAttributes.LLM_REQUEST_MODEL]


def test_non_strands_span_is_unchanged():
    span = _make_span(
        name="other",
        attributes={
            SpanAttributes.LLM_SYSTEM: "openai",
            SpanAttributes.LLM_REQUEST_MODEL: "gpt-4o-mini",
        },
    )

    enrich_strands_agents_span(span)

    assert span._attributes == {
        SpanAttributes.LLM_SYSTEM: "openai",
        SpanAttributes.LLM_REQUEST_MODEL: "gpt-4o-mini",
    }


def test_basic_auth_redaction_preserves_ordinary_agent_names():
    assert safe_text("Strands Basic Example") == "Strands Basic Example"
    assert "dXNlcjpwYXNz" not in safe_text("Basic dXNlcjpwYXNz")
    assert "password-value" not in safe_text(
        "https://user:password-value@example.invalid/path"
    )
