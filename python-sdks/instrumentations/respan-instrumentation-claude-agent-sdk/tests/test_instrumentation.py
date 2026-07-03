import asyncio
import builtins
import json
import sys
from types import ModuleType, SimpleNamespace

from opentelemetry import trace
from opentelemetry.semconv_ai import (
    LLMRequestTypeValues,
    SpanAttributes,
    TraceloopSpanKindValues,
)
from respan_tracing.exporters.respan import _prepare_spans_for_export

from respan_instrumentation_claude_agent_sdk import (
    ClaudeAgentSDKInstrumentor,
    _instrumentation,
    _processor,
)
from respan_sdk.constants.llm_logging import (
    LOG_TYPE_AGENT,
    LOG_TYPE_CHAT,
    LOG_TYPE_TOOL,
    LogMethodChoices,
)
from respan_sdk.constants.span_attributes import (
    GEN_AI_SYSTEM,
    GEN_AI_TOOL_CALL_ARGUMENTS,
    GEN_AI_TOOL_CALL_RESULT,
    GEN_AI_TOOL_NAME,
    LLM_REQUEST_TYPE,
    RESPAN_LOG_METHOD,
    RESPAN_LOG_TYPE,
    RESPAN_SESSION_ID,
    RESPAN_SPAN_TOOL_CALLS,
    RESPAN_SPAN_TOOLS,
)


def _make_fake_tracer_provider(*, composite: bool = True) -> SimpleNamespace:
    added_processors: list[object] = []
    tracer_provider = SimpleNamespace(added_processors=added_processors)

    if composite:
        tracer_provider._active_span_processor = SimpleNamespace(_span_processors=())

    def add_span_processor(processor: object) -> None:
        added_processors.append(processor)

    tracer_provider.add_span_processor = add_span_processor
    return tracer_provider


def _install_fake_claude_agent_sdk_modules(
    monkeypatch,
    *,
    instrument_error: Exception | None = None,
) -> SimpleNamespace:
    output_messages_attr = "gen_ai.output.messages"
    usage_input_tokens_attr = "gen_ai.usage.input_tokens"
    usage_output_tokens_attr = "gen_ai.usage.output_tokens"
    usage_cache_creation_tokens_attr = "gen_ai.usage.cache_creation_input_tokens"
    usage_cache_read_tokens_attr = "gen_ai.usage.cache_read_input_tokens"

    class FakeClaudeAgentSdkInstrumentor:
        # Mirrors upstream's BaseInstrumentor singleton: instrument() applies the
        # module-level `query` wrap once and a second call is a no-op.
        _module_query_wrapped = False

        def __init__(self):
            self.instrument_kwargs = None
            self.uninstrument_calls = 0

        def instrument(self, **kwargs):
            self.instrument_kwargs = kwargs
            if instrument_error is not None:
                raise instrument_error
            if FakeClaudeAgentSdkInstrumentor._module_query_wrapped:
                return
            import wrapt

            wrapt.wrap_function_wrapper("claude_agent_sdk", "query", self._wrap_query)
            FakeClaudeAgentSdkInstrumentor._module_query_wrapped = True

        def uninstrument(self):
            self.uninstrument_calls += 1
            if not FakeClaudeAgentSdkInstrumentor._module_query_wrapped:
                return
            FakeClaudeAgentSdkInstrumentor._module_query_wrapped = False
            import claude_agent_sdk

            module_query = getattr(claude_agent_sdk, "query", None)
            if hasattr(module_query, "__wrapped__"):
                claude_agent_sdk.query = module_query.__wrapped__

        def _wrap_query(self, wrapped, instance, args, kwargs):
            return wrapped(*args, **kwargs)

        def _wrap_client_query(self, wrapped, instance, args, kwargs):
            instance._otel_invocation_ctx = kwargs.get("otel_invocation_ctx")
            return wrapped(*args, **kwargs)

        async def _instrumented_receive_response(
            self,
            wrapped,
            instance,
            args,
            kwargs,
        ):
            async for message in wrapped(*args, **kwargs):
                yield message

    def _original_set_response_content(span, content):
        span.set_attribute(
            output_messages_attr,
            json.dumps([{"role": "assistant", "content": content}], default=str),
        )

    def _original_set_result_attributes(span, result_message):
        usage = getattr(result_message, "usage", None) or {}
        total_input_tokens = (
            (usage.get("input_tokens", 0) or 0)
            + (usage.get("cache_creation_input_tokens", 0) or 0)
            + (usage.get("cache_read_input_tokens", 0) or 0)
        )
        span.set_attribute(usage_input_tokens_attr, total_input_tokens)
        span.set_attribute(usage_output_tokens_attr, usage.get("output_tokens", 0) or 0)
        if usage.get("cache_creation_input_tokens", 0):
            span.set_attribute(
                usage_cache_creation_tokens_attr,
                usage["cache_creation_input_tokens"],
            )
        if usage.get("cache_read_input_tokens", 0):
            span.set_attribute(
                usage_cache_read_tokens_attr,
                usage["cache_read_input_tokens"],
            )

    claude_package = ModuleType("opentelemetry.instrumentation.claude_agent_sdk")
    claude_package.__path__ = []
    claude_package.ClaudeAgentSdkInstrumentor = FakeClaudeAgentSdkInstrumentor

    constants_module = ModuleType(
        "opentelemetry.instrumentation.claude_agent_sdk._constants"
    )
    constants_module.GEN_AI_OUTPUT_MESSAGES = output_messages_attr
    constants_module.GEN_AI_USAGE_INPUT_TOKENS = usage_input_tokens_attr
    constants_module.GEN_AI_USAGE_OUTPUT_TOKENS = usage_output_tokens_attr
    constants_module.GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS = (
        usage_cache_creation_tokens_attr
    )
    constants_module.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS = (
        usage_cache_read_tokens_attr
    )

    context_state = {"value": None}

    context_module = ModuleType(
        "opentelemetry.instrumentation.claude_agent_sdk._context"
    )

    def get_invocation_context():
        return context_state["value"]

    def set_invocation_context(value):
        context_state["value"] = value

    context_module.get_invocation_context = get_invocation_context
    context_module.set_invocation_context = set_invocation_context

    spans_module = ModuleType("opentelemetry.instrumentation.claude_agent_sdk._spans")
    spans_module._to_serializable = lambda value: value
    spans_module.set_response_content = _original_set_response_content
    spans_module.set_result_attributes = _original_set_result_attributes

    instrumentor_module = ModuleType(
        "opentelemetry.instrumentation.claude_agent_sdk._instrumentor"
    )
    instrumentor_module.ClaudeAgentSdkInstrumentor = FakeClaudeAgentSdkInstrumentor
    instrumentor_module.set_response_content = _original_set_response_content
    instrumentor_module.set_result_attributes = _original_set_result_attributes

    claude_sdk_module = ModuleType("claude_agent_sdk")
    claude_sdk_module.__path__ = []

    # Standalone `query()` module attribute — what upstream wraps and what
    # `from claude_agent_sdk import query` binds. The A6 seam covers it via
    # InternalClient.process_query below.
    def _standalone_query(*args, **kwargs):
        return "standalone-query-result"

    claude_sdk_module.query = _standalone_query

    internal_module = ModuleType("claude_agent_sdk._internal")
    internal_module.__path__ = []
    query_module = ModuleType("claude_agent_sdk._internal.query")

    class FakeQuery:
        def __init__(self):
            self._otel_invocation_ctx = None

        async def _handle_control_request(self, request):
            return context_module.get_invocation_context()

    query_module.Query = FakeQuery

    # Internal seam that the standalone query() delegates to (A6). Wrapping this
    # is how respan traces `from claude_agent_sdk import query`.
    client_module = ModuleType("claude_agent_sdk._internal.client")

    class FakeInternalClient:
        def process_query(self, *args, **kwargs):
            return "process-query-result"

    client_module.InternalClient = FakeInternalClient

    monkeypatch.setitem(
        sys.modules,
        "opentelemetry.instrumentation.claude_agent_sdk",
        claude_package,
    )
    monkeypatch.setitem(
        sys.modules,
        "opentelemetry.instrumentation.claude_agent_sdk._constants",
        constants_module,
    )
    monkeypatch.setitem(
        sys.modules,
        "opentelemetry.instrumentation.claude_agent_sdk._context",
        context_module,
    )
    monkeypatch.setitem(
        sys.modules,
        "opentelemetry.instrumentation.claude_agent_sdk._spans",
        spans_module,
    )
    monkeypatch.setitem(
        sys.modules,
        "opentelemetry.instrumentation.claude_agent_sdk._instrumentor",
        instrumentor_module,
    )
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", claude_sdk_module)
    monkeypatch.setitem(sys.modules, "claude_agent_sdk._internal", internal_module)
    monkeypatch.setitem(
        sys.modules,
        "claude_agent_sdk._internal.query",
        query_module,
    )
    monkeypatch.setitem(
        sys.modules,
        "claude_agent_sdk._internal.client",
        client_module,
    )

    return SimpleNamespace(
        instrumentor_class=FakeClaudeAgentSdkInstrumentor,
        context_module=context_module,
        query_class=FakeQuery,
        spans_module=spans_module,
        instrumentor_module=instrumentor_module,
        original_set_response_content=_original_set_response_content,
        original_set_result_attributes=_original_set_result_attributes,
        claude_sdk_module=claude_sdk_module,
        standalone_query=_standalone_query,
        internal_client=FakeInternalClient,
    )


def _make_span(
    *,
    name: str,
    attributes: dict[str, object] | None = None,
    trace_id: int = 1,
    span_id: int = 1,
    parent_span_id: int | None = None,
    start_time: int = 10,
    scope_name: str | None = None,
) -> SimpleNamespace:
    attrs = dict(attributes or {})
    span_context = SimpleNamespace(trace_id=trace_id, span_id=span_id)
    return SimpleNamespace(
        name=name,
        _attributes=attrs,
        attributes=attrs,
        parent=(
            SimpleNamespace(span_id=parent_span_id)
            if parent_span_id is not None
            else None
        ),
        start_time=start_time,
        end_time=start_time + 1,
        instrumentation_scope=(
            SimpleNamespace(name=scope_name, version="1.0.0")
            if scope_name is not None
            else None
        ),
        get_span_context=lambda: span_context,
    )


def test_package_exports_instrumentor():
    assert ClaudeAgentSDKInstrumentor is _instrumentation.ClaudeAgentSDKInstrumentor
    assert ClaudeAgentSDKInstrumentor.name == "claude-agent-sdk"


def test_instrumentation_helpers_read_attrs_and_parse_json():
    span_with_public_attrs = SimpleNamespace(attributes={"key": "value"})
    span_with_private_attrs = SimpleNamespace(_attributes={"key": "private"})

    assert _instrumentation._safe_json_loads('{"a": 1}') == {"a": 1}
    assert _instrumentation._safe_json_loads("plain-text") is None
    assert _instrumentation._get_span_attr_value(span_with_public_attrs, "key") == "value"
    assert _instrumentation._get_span_attr_value(span_with_private_attrs, "key") == "private"


def test_register_and_unregister_processor_keep_processor_first():
    tracer_provider = _make_fake_tracer_provider()
    existing_processor = object()
    tracer_provider._active_span_processor._span_processors = (existing_processor,)
    processor = object()

    ClaudeAgentSDKInstrumentor._register_processor(
        tracer_provider=tracer_provider,
        processor=processor,
    )
    ClaudeAgentSDKInstrumentor._register_processor(
        tracer_provider=tracer_provider,
        processor=processor,
    )

    assert tracer_provider._active_span_processor._span_processors == (
        processor,
        existing_processor,
    )

    ClaudeAgentSDKInstrumentor._unregister_processor(
        tracer_provider=tracer_provider,
        processor=processor,
    )

    assert tracer_provider._active_span_processor._span_processors == (
        existing_processor,
    )


def test_register_processor_falls_back_to_add_span_processor():
    tracer_provider = _make_fake_tracer_provider(composite=False)
    processor = object()

    ClaudeAgentSDKInstrumentor._register_processor(
        tracer_provider=tracer_provider,
        processor=processor,
    )

    assert tracer_provider.added_processors == [processor]


def test_activate_patches_helpers_and_restores_originals(monkeypatch):
    tracer_provider = _make_fake_tracer_provider()
    monkeypatch.setattr(trace, "get_tracer_provider", lambda: tracer_provider)
    fake = _install_fake_claude_agent_sdk_modules(monkeypatch)

    instrumentor = ClaudeAgentSDKInstrumentor(
        agent_name="demo-agent",
        capture_content=True,
    )
    instrumentor.activate()

    assert instrumentor._is_instrumented is True
    assert instrumentor._otel_instrumentor.instrument_kwargs == {
        "tracer_provider": tracer_provider,
        "agent_name": "demo-agent",
        "capture_content": True,
    }
    assert fake.spans_module.set_response_content is not fake.original_set_response_content
    assert fake.spans_module.set_result_attributes is not fake.original_set_result_attributes

    fake_query = fake.query_class()
    fake_query._otel_invocation_ctx = SimpleNamespace(marker="client-session")
    seen_ctx = asyncio.run(fake_query._handle_control_request({"request": {}}))
    assert seen_ctx.marker == "client-session"

    span = SimpleNamespace(attributes={})
    span.set_attribute = lambda key, value: span.attributes.__setitem__(key, value)

    fake.spans_module.set_response_content(
        span,
        [{"type": "tool_use", "id": "toolu_123", "name": "calculator"}],
    )
    fake.spans_module.set_response_content(
        span,
        [{"type": "text", "text": "The tip is $18.00."}],
    )
    fake.spans_module.set_result_attributes(
        span,
        SimpleNamespace(
            usage={
                "input_tokens": 4,
                "cache_creation_input_tokens": 291,
                "cache_read_input_tokens": 39025,
                "output_tokens": 121,
            },
            total_cost_usd=0.04241955,
        ),
    )

    assert json.loads(span.attributes["gen_ai.output.messages"]) == [
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "toolu_123", "name": "calculator"}
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "The tip is $18.00."}],
        },
    ]
    assert span.attributes["gen_ai.usage.input_tokens"] == 4
    assert span.attributes["gen_ai.usage.output_tokens"] == 121
    # Cost is emitted as respan.metadata.response_cost (string), matching the
    # LiteLLM/OpenAI instrumentors, not a bare "cost" attribute (A7).
    assert span.attributes["respan.metadata.response_cost"] == "0.04241955"
    assert "cost" not in span.attributes
    # The helper writes only raw input/output; the span processor owns
    # prompt/completion/total, so the helper must not pre-empt them here.
    assert "gen_ai.usage.total_tokens" not in span.attributes

    instrumentor.deactivate()

    assert instrumentor._is_instrumented is False
    assert tracer_provider._active_span_processor._span_processors == ()
    assert fake.spans_module.set_response_content is fake.original_set_response_content
    assert fake.spans_module.set_result_attributes is fake.original_set_result_attributes
    restored_ctx = asyncio.run(fake_query._handle_control_request({"request": {}}))
    assert restored_ctx is None


def test_activate_traces_standalone_query_via_internal_seam(monkeypatch):
    """A6: the from-import query() path is instrumented at InternalClient.process_query."""
    tracer_provider = _make_fake_tracer_provider()
    monkeypatch.setattr(trace, "get_tracer_provider", lambda: tracer_provider)
    fake = _install_fake_claude_agent_sdk_modules(monkeypatch)

    instrumentor = ClaudeAgentSDKInstrumentor()
    instrumentor.activate()

    # The internal seam is wrapped, so `from claude_agent_sdk import query` traces...
    assert hasattr(fake.internal_client.process_query, "__wrapped__")
    # ...and the bypassable module-level `query` wrap is dropped, so module-qualified
    # and from-imported calls both hit exactly the seam (one span, no double-count).
    assert fake.claude_sdk_module.query is fake.standalone_query
    assert not hasattr(fake.claude_sdk_module.query, "__wrapped__")


def test_double_activation_does_not_stack_the_query_seam(monkeypatch):
    """A second instrumentor must not wrap the seam twice (would emit two spans)."""
    tracer_provider = _make_fake_tracer_provider()
    monkeypatch.setattr(trace, "get_tracer_provider", lambda: tracer_provider)
    fake = _install_fake_claude_agent_sdk_modules(monkeypatch)

    ClaudeAgentSDKInstrumentor().activate()
    ClaudeAgentSDKInstrumentor().activate()

    wrapped = fake.internal_client.process_query
    assert hasattr(wrapped, "__wrapped__")
    # Exactly one wrapper layer: __wrapped__ is the pristine original, not a second wrapper.
    assert not hasattr(wrapped.__wrapped__, "__wrapped__")


def test_deactivate_restores_the_query_seam_and_module_query(monkeypatch):
    tracer_provider = _make_fake_tracer_provider()
    monkeypatch.setattr(trace, "get_tracer_provider", lambda: tracer_provider)
    fake = _install_fake_claude_agent_sdk_modules(monkeypatch)

    original_process_query = fake.internal_client.process_query

    instrumentor = ClaudeAgentSDKInstrumentor()
    instrumentor.activate()
    instrumentor.deactivate()

    assert fake.internal_client.process_query is original_process_query
    assert not hasattr(fake.internal_client.process_query, "__wrapped__")
    assert fake.claude_sdk_module.query is fake.standalone_query


def test_activate_logs_warning_when_dependency_missing(monkeypatch, caplog):
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "opentelemetry.instrumentation.claude_agent_sdk":
            raise ImportError("missing claude agent sdk instrumentation")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    instrumentor = ClaudeAgentSDKInstrumentor()
    instrumentor.activate()

    assert instrumentor._is_instrumented is False
    assert "missing dependency" in caplog.text


def test_activate_cleans_up_when_upstream_instrument_fails(monkeypatch, caplog):
    tracer_provider = _make_fake_tracer_provider()
    monkeypatch.setattr(trace, "get_tracer_provider", lambda: tracer_provider)
    fake = _install_fake_claude_agent_sdk_modules(
        monkeypatch,
        instrument_error=RuntimeError("boom"),
    )

    instrumentor = ClaudeAgentSDKInstrumentor()
    instrumentor.activate()

    assert instrumentor._is_instrumented is False
    assert instrumentor._otel_instrumentor is None
    assert tracer_provider._active_span_processor._span_processors == ()
    assert fake.spans_module.set_response_content is fake.original_set_response_content
    assert "Failed to activate Claude Agent SDK instrumentation" in caplog.text


def test_deactivate_is_a_noop_when_not_instrumented():
    instrumentor = ClaudeAgentSDKInstrumentor()
    instrumentor.deactivate()

    assert instrumentor._is_instrumented is False


def test_processor_helpers_parse_and_normalize_values():
    assert _processor._safe_json_loads('{"a": 1}') == {"a": 1}
    assert _processor._safe_json_loads("({'a': 1})") == {"a": 1}
    assert _processor._safe_json_loads("plain-text") is None
    assert _processor._json_string({"a": 1}) == '{"a": 1}'
    assert _processor._json_string('[{"a":1}]') == '[{"a": 1}]'
    assert _processor._json_string("plain-text") == "plain-text"


def test_extract_usage_normalizes_cached_prompt_tokens():
    prompt_tokens, completion_tokens, cache_hit_tokens, cache_creation_tokens = (
        _processor._extract_usage(
            {
                "gen_ai.usage.input_tokens": 39320,
                "gen_ai.usage.cache_read_input_tokens": 39025,
                "gen_ai.usage.cache_creation_input_tokens": 291,
                "gen_ai.usage.output_tokens": 121,
            }
        )
    )

    assert prompt_tokens == 4
    assert completion_tokens == 121
    assert cache_hit_tokens == 39025
    assert cache_creation_tokens == 291


def test_extract_input_output_prefers_messages_and_value_fallbacks():
    input_value, output_value = _processor._extract_input_output(
        {
            "gen_ai.system_instructions": "Always be concise.",
            "gen_ai.input.messages": '[{"role":"user","content":"hi"}]',
            "gen_ai.output.messages": '[{"role":"assistant","content":"hello"}]',
        }
    )

    assert json.loads(input_value) == [
        {"role": "system", "content": "Always be concise."},
        {"role": "user", "content": "hi"},
    ]
    assert json.loads(output_value) == [{"role": "assistant", "content": "hello"}]

    fallback_input, fallback_output = _processor._extract_input_output(
        {
            "input.value": {"prompt": "fallback"},
            "output.value": {"answer": "ok"},
        }
    )

    assert json.loads(fallback_input) == {"prompt": "fallback"}
    assert json.loads(fallback_output) == {"answer": "ok"}


def test_extract_tools_and_tool_calls_normalize_supported_shapes():
    tools = _processor._extract_tools(
        {
            "gen_ai.tool.definitions": json.dumps(
                [
                    "get_weather",
                    {"name": "calculator", "input_schema": {"type": "object"}},
                    {
                        "type": "function",
                        "function": {
                            "name": "lookup_customer",
                            "description": "Find a customer.",
                            "parameters": {"type": "object"},
                            "strict": True,
                        },
                    },
                ]
            )
        }
    )

    assert tools == [
        {"type": "function", "function": {"name": "get_weather"}},
        {
            "type": "function",
            "function": {
                "name": "calculator",
                "parameters": {"type": "object"},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "lookup_customer",
                "description": "Find a customer.",
                "parameters": {"type": "object"},
                "strict": True,
            },
        },
    ]

    tool_calls = _processor._extract_tool_calls(
        {
            "gen_ai.output.messages": json.dumps(
                [
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "toolu_123",
                                "name": "calculator",
                                "input": {"expression": "120 * 0.15"},
                            },
                            {"type": "text", "text": "The tip is $18.00."},
                        ],
                    }
                ]
            )
        }
    )

    assert tool_calls == [
        {
            "id": "toolu_123",
            "type": "function",
            "function": {
                "name": "calculator",
                "arguments": '{"expression": "120 * 0.15"}',
            },
        }
    ]


def test_extract_existing_tool_calls_and_key_helpers():
    span = _make_span(name="tool-span", trace_id=22, span_id=33, parent_span_id=11)

    assert _processor._extract_existing_tool_calls(
        {"tool_calls": [{"id": "override"}]}
    ) == [{"id": "override"}]
    assert _processor._extract_existing_tool_calls(
        {RESPAN_SPAN_TOOL_CALLS: '[{"id":"parsed"}]'}
    ) == [{"id": "parsed"}]
    assert _processor._get_span_key(span) == (22, 33)
    assert _processor._get_parent_span_key(span) == (22, 11)


def test_build_tool_call_from_tool_span_attrs_and_merge_tool_calls():
    built_tool_call = _processor._build_tool_call_from_tool_span_attrs(
        {
            SpanAttributes.TRACELOOP_ENTITY_NAME: "calculator",
            SpanAttributes.TRACELOOP_ENTITY_INPUT: {"expression": "120 * 0.15"},
            "gen_ai.tool.call.id": "toolu_123",
        }
    )

    assert built_tool_call == {
        "id": "toolu_123",
        "type": "function",
        "function": {
            "name": "calculator",
            "arguments": '{"expression": "120 * 0.15"}',
        },
    }

    merged_tool_calls = _processor._merge_tool_calls(
        [built_tool_call],
        [built_tool_call, {"id": "toolu_456", "function": {"name": "search", "arguments": "{}"}}],
    )

    assert merged_tool_calls == [
        built_tool_call,
        {"id": "toolu_456", "function": {"name": "search", "arguments": "{}"}},
    ]


def test_is_claude_agent_sdk_span_recognizes_supported_shapes():
    invoke_span = _make_span(name="invoke_agent weather")
    tool_span = _make_span(name="execute_tool calculator")
    other_span = _make_span(name="http.request")

    assert _processor.is_claude_agent_sdk_span(
        invoke_span,
        {"gen_ai.operation.name": "invoke_agent"},
    )
    assert _processor.is_claude_agent_sdk_span(
        tool_span,
        {"gen_ai.tool.name": "calculator"},
    )
    assert _processor.is_claude_agent_sdk_span(other_span, {}) is False


def test_enrich_claude_agent_sdk_span_maps_agent_fields():
    span = _make_span(
        name="invoke_agent weather_agent",
        attributes={
            "gen_ai.operation.name": "invoke_agent",
            "gen_ai.system": "anthropic",
            "gen_ai.agent.name": "weather_agent",
            "gen_ai.conversation.id": "session-123",
            "gen_ai.system_instructions": "Always call tools first.",
            "gen_ai.input.messages": '[{"role":"user","content":"weather?"}]',
            "gen_ai.output.messages": json.dumps(
                [
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "toolu_123",
                                "name": "get_weather",
                                "input": {"city": "Tokyo"},
                            }
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "Tokyo is sunny."}],
                    },
                ]
            ),
            "gen_ai.tool.definitions": '[{"name":"get_weather"}]',
            "gen_ai.response.model": "claude-sonnet-4-5",
            "gen_ai.usage.input_tokens": 19,
            "gen_ai.usage.output_tokens": 7,
        },
    )

    _processor.enrich_claude_agent_sdk_span(span)

    assert span._attributes[RESPAN_LOG_METHOD] == LogMethodChoices.TRACING_INTEGRATION.value
    assert span._attributes[RESPAN_LOG_TYPE] == LOG_TYPE_AGENT
    assert span._attributes[SpanAttributes.TRACELOOP_SPAN_KIND] == TraceloopSpanKindValues.AGENT.value
    assert span._attributes[SpanAttributes.TRACELOOP_ENTITY_NAME] == "weather_agent"
    assert span._attributes[SpanAttributes.TRACELOOP_WORKFLOW_NAME] == "weather_agent"
    assert json.loads(span._attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT]) == [
        {"role": "system", "content": "Always call tools first."},
        {"role": "user", "content": "weather?"},
    ]
    assert span._attributes["model"] == "claude-sonnet-4-5"
    assert span._attributes["prompt_tokens"] == 19
    assert span._attributes["completion_tokens"] == 7
    assert span._attributes["total_request_tokens"] == 26
    assert span._attributes[RESPAN_SESSION_ID] == "session-123"
    assert json.loads(span._attributes[RESPAN_SPAN_TOOLS]) == [
        {"type": "function", "function": {"name": "get_weather"}}
    ]
    assert json.loads(span._attributes[RESPAN_SPAN_TOOL_CALLS]) == [
        {
            "id": "toolu_123",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": '{"city": "Tokyo"}',
            },
        }
    ]
    assert "gen_ai.agent.name" not in span._attributes
    assert "gen_ai.input.messages" not in span._attributes
    assert "gen_ai.output.messages" not in span._attributes
    assert "tool_calls" not in span._attributes


def test_enrich_claude_agent_sdk_span_maps_tool_fields():
    span = _make_span(
        name="execute_tool mcp__demo__calculator",
        attributes={
            "gen_ai.operation.name": "execute_tool",
            "gen_ai.tool.name": "calculator",
            "gen_ai.tool.call.arguments": {"expression": "120 * 0.15"},
            "gen_ai.tool.call.result": {"content": [{"type": "text", "text": "18"}]},
            "gen_ai.response.model": "claude-sonnet-4-5",
            "gen_ai.usage.input_tokens": 6,
            "gen_ai.usage.output_tokens": 2,
            LLM_REQUEST_TYPE: "chat",
        },
    )

    _processor.enrich_claude_agent_sdk_span(span)

    assert span._attributes[RESPAN_LOG_TYPE] == LOG_TYPE_TOOL
    assert span._attributes[SpanAttributes.TRACELOOP_SPAN_KIND] == TraceloopSpanKindValues.TOOL.value
    assert span._attributes[SpanAttributes.TRACELOOP_ENTITY_NAME] == "mcp__demo__calculator"
    assert span._attributes[SpanAttributes.TRACELOOP_ENTITY_PATH] == "mcp__demo__calculator"
    assert json.loads(span._attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT]) == {
        "expression": "120 * 0.15"
    }
    assert json.loads(span._attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]) == {
        "content": [{"type": "text", "text": "18"}]
    }
    assert LLM_REQUEST_TYPE not in span._attributes
    assert span._attributes["tools"] == [
        {"type": "function", "function": {"name": "mcp__demo__calculator"}}
    ]
    assert span._attributes["span_tools"] == ["mcp__demo__calculator"]
    assert "model" not in span._attributes
    assert "prompt_tokens" not in span._attributes
    assert "completion_tokens" not in span._attributes
    assert "total_request_tokens" not in span._attributes
    assert "cost" not in span._attributes
    assert GEN_AI_TOOL_NAME not in span._attributes
    assert GEN_AI_TOOL_CALL_ARGUMENTS not in span._attributes
    assert GEN_AI_TOOL_CALL_RESULT not in span._attributes


def test_enrich_claude_agent_sdk_span_overrides_upstream_tool_chat_defaults():
    span = _make_span(
        name="execute_tool mcp__demo__get_weather",
        attributes={
            "gen_ai.operation.name": "execute_tool",
            GEN_AI_SYSTEM: "anthropic",
            "gen_ai.tool.name": "get_weather",
            "gen_ai.tool.call.arguments": {"city": "Tokyo", "unit": "celsius"},
            "gen_ai.tool.call.result": {"content": [{"type": "text", "text": "22C"}]},
            "gen_ai.input.messages": '[{"role":"user","content":"{\\"city\\": \\"Tokyo\\", \\"unit\\": \\"celsius\\"}"}]',
            "gen_ai.output.messages": '[{"role":"assistant","content":""}]',
            "gen_ai.prompt.0.role": "user",
            "gen_ai.prompt.0.content": '{"city": "Tokyo", "unit": "celsius"}',
            "gen_ai.completion.0.role": "assistant",
            "gen_ai.completion.0.content": "",
            RESPAN_LOG_TYPE: LOG_TYPE_CHAT,
            SpanAttributes.TRACELOOP_SPAN_KIND: LLMRequestTypeValues.CHAT.value,
            SpanAttributes.TRACELOOP_ENTITY_NAME: "placeholder-chat",
            SpanAttributes.TRACELOOP_ENTITY_PATH: "placeholder-chat",
            SpanAttributes.TRACELOOP_ENTITY_INPUT: '[{"role":"user","content":"{\\"city\\": \\"Tokyo\\"}"}]',
            SpanAttributes.TRACELOOP_ENTITY_OUTPUT: '{"_is_placeholder": true, "content": "", "role": "assistant"}',
            LLM_REQUEST_TYPE: "chat",
            "model": "gpt-4",
            "prompt_tokens": 20,
            "completion_tokens": 0,
            "total_request_tokens": 20,
            "cost": 0.0006,
        },
    )

    _processor.enrich_claude_agent_sdk_span(span)

    assert span._attributes[RESPAN_LOG_TYPE] == LOG_TYPE_TOOL
    assert span._attributes[SpanAttributes.TRACELOOP_SPAN_KIND] == TraceloopSpanKindValues.TOOL.value
    assert span._attributes[SpanAttributes.TRACELOOP_ENTITY_NAME] == "mcp__demo__get_weather"
    assert span._attributes[SpanAttributes.TRACELOOP_ENTITY_PATH] == "mcp__demo__get_weather"
    assert json.loads(span._attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT]) == {
        "city": "Tokyo",
        "unit": "celsius",
    }
    assert json.loads(span._attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]) == {
        "content": [{"type": "text", "text": "22C"}]
    }
    assert span._attributes["tools"] == [
        {"type": "function", "function": {"name": "mcp__demo__get_weather"}}
    ]
    assert "model" not in span._attributes
    assert "prompt_tokens" not in span._attributes
    assert "completion_tokens" not in span._attributes
    assert "total_request_tokens" not in span._attributes
    assert "cost" not in span._attributes
    assert GEN_AI_SYSTEM not in span._attributes
    assert "gen_ai.prompt.0.role" not in span._attributes
    assert "gen_ai.prompt.0.content" not in span._attributes
    assert "gen_ai.completion.0.role" not in span._attributes
    assert "gen_ai.completion.0.content" not in span._attributes
    assert RESPAN_SPAN_TOOL_CALLS not in span._attributes
    assert "tool_calls" not in span._attributes


def test_enrich_claude_agent_sdk_span_reconciles_tools_with_namespaced_tool_calls():
    span = _make_span(
        name="invoke_agent weather_agent",
        attributes={
            "gen_ai.operation.name": "invoke_agent",
            "gen_ai.agent.name": "weather_agent",
            "gen_ai.system": "anthropic",
            "gen_ai.input.messages": '[{"role":"user","content":"weather?"}]',
            "gen_ai.output.messages": json.dumps(
                [
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "toolu_123",
                                "name": "mcp__demo__get_weather",
                                "input": {"city": "Tokyo"},
                            }
                        ],
                    }
                ]
            ),
            "gen_ai.tool.definitions": json.dumps(
                [
                    {"name": "get_weather"},
                    {"name": "calculator"},
                ]
            ),
        },
    )

    _processor.enrich_claude_agent_sdk_span(span)

    assert json.loads(span._attributes[RESPAN_SPAN_TOOLS]) == [
        {"type": "function", "function": {"name": "mcp__demo__get_weather"}},
        {"type": "function", "function": {"name": "mcp__demo__calculator"}},
    ]
    assert json.loads(span._attributes[RESPAN_SPAN_TOOL_CALLS]) == [
        {
            "id": "toolu_123",
            "type": "function",
            "function": {
                "name": "mcp__demo__get_weather",
                "arguments": '{"city": "Tokyo"}',
            },
        }
    ]


def test_enrich_claude_agent_sdk_span_leaves_unrelated_spans_untouched():
    span = _make_span(
        name="http.request",
        attributes={"http.method": "POST"},
    )

    _processor.enrich_claude_agent_sdk_span(span)

    assert span._attributes == {"http.method": "POST"}


def test_span_processor_on_end_merges_pending_tool_calls_into_parent_agent_span():
    processor = _processor.ClaudeAgentSDKSpanProcessor()

    tool_span = _make_span(
        name="execute_tool mcp__demo__get_weather",
        trace_id=55,
        span_id=2,
        parent_span_id=1,
        start_time=20,
        attributes={
            "gen_ai.operation.name": "execute_tool",
            "gen_ai.tool.name": "get_weather",
            "gen_ai.tool.call.arguments": {"city": "Tokyo"},
            "gen_ai.tool.call.result": {"temperature": "22C"},
            "gen_ai.tool.call.id": "toolu_123",
        },
    )
    agent_span = _make_span(
        name="invoke_agent weather_agent",
        trace_id=55,
        span_id=1,
        start_time=10,
        attributes={
            "gen_ai.operation.name": "invoke_agent",
            "gen_ai.agent.name": "weather_agent",
            "gen_ai.tool.definitions": '[{"name":"get_weather"}]',
            "gen_ai.output.messages": json.dumps(
                [
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "toolu_123",
                                "name": "mcp__demo__get_weather",
                                "input": {"city": "Tokyo"},
                            }
                        ],
                    }
                ]
            ),
        },
    )

    processor.on_start(tool_span)
    processor.on_end(tool_span)
    processor.on_end(agent_span)

    assert json.loads(agent_span._attributes[RESPAN_SPAN_TOOL_CALLS]) == [
        {
            "id": "toolu_123",
            "type": "function",
            "function": {
                "name": "mcp__demo__get_weather",
                "arguments": '{"city": "Tokyo"}',
            },
        }
    ]
    assert agent_span._attributes["tool_calls"] == [
        {
            "id": "toolu_123",
            "type": "function",
            "function": {
                "name": "mcp__demo__get_weather",
                "arguments": '{"city": "Tokyo"}',
            },
        }
    ]
    assert agent_span._attributes["tools"] == [
        {"type": "function", "function": {"name": "mcp__demo__get_weather"}}
    ]


def test_span_processor_on_end_discards_pending_tool_calls_for_unrelated_parent_span():
    processor = _processor.ClaudeAgentSDKSpanProcessor()

    tool_span = _make_span(
        name="execute_tool calculator",
        trace_id=56,
        span_id=2,
        parent_span_id=1,
        start_time=20,
        attributes={
            "gen_ai.operation.name": "execute_tool",
            "gen_ai.tool.name": "calculator",
            "gen_ai.tool.call.arguments": {"expression": "2 + 2"},
        },
    )
    parent_span = _make_span(
        name="POST /chat",
        trace_id=56,
        span_id=1,
        start_time=10,
        attributes={"http.method": "POST"},
    )

    processor.on_end(tool_span)

    assert processor._pending_tool_calls_by_parent

    processor.on_end(parent_span)

    assert parent_span._attributes == {"http.method": "POST"}
    assert processor._pending_tool_calls_by_parent == {}


def test_span_processor_on_end_discards_pending_tool_calls_for_tool_parent_span():
    processor = _processor.ClaudeAgentSDKSpanProcessor()

    child_tool_span = _make_span(
        name="execute_tool sub_task",
        trace_id=57,
        span_id=3,
        parent_span_id=2,
        start_time=30,
        attributes={
            "gen_ai.operation.name": "execute_tool",
            "gen_ai.tool.name": "sub_task",
            "gen_ai.tool.call.arguments": {"step": 1},
        },
    )
    parent_tool_span = _make_span(
        name="execute_tool workflow_tool",
        trace_id=57,
        span_id=2,
        parent_span_id=1,
        start_time=20,
        attributes={
            "gen_ai.operation.name": "execute_tool",
            "gen_ai.tool.name": "workflow_tool",
            "gen_ai.tool.call.arguments": {"step": 0},
        },
    )

    processor.on_end(child_tool_span)
    processor.on_end(parent_tool_span)

    assert (57, 2) not in processor._pending_tool_calls_by_parent
    assert (57, 1) in processor._pending_tool_calls_by_parent


def test_span_processor_on_end_leaves_final_chat_child_to_shared_exporter():
    processor = _processor.ClaudeAgentSDKSpanProcessor()

    agent_span = _make_span(
        name="ClaudeAgentSDK.query",
        trace_id=88,
        span_id=7,
        start_time=100,
        scope_name="openinference.instrumentation.claude_agent_sdk",
        attributes={
            "gen_ai.operation.name": "invoke_agent",
            "gen_ai.agent.name": "weather_agent",
            "gen_ai.system": "anthropic",
            "gen_ai.input.messages": '[{"role":"user","content":"weather?"}]',
            "gen_ai.output.messages": json.dumps(
                [
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "toolu_123",
                                "name": "get_weather",
                                "input": {"city": "Tokyo"},
                            }
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "Tokyo is sunny."}],
                    },
                ]
            ),
            "gen_ai.response.model": "claude-sonnet-4-5",
        },
    )

    processor.on_end(agent_span)
    agent_span.attributes = agent_span._attributes

    assert json.loads(agent_span._attributes[RESPAN_SPAN_TOOL_CALLS]) == [
        {
            "id": "toolu_123",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": '{"city": "Tokyo"}',
            },
        }
    ]
    assert agent_span._attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] == json.dumps(
        [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_123",
                        "name": "get_weather",
                        "input": {"city": "Tokyo"},
                    }
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "Tokyo is sunny."}],
            },
        ]
    )

    prepared_spans = _prepare_spans_for_export(spans=[agent_span])
    assert [span.name for span in prepared_spans] == [
        "ClaudeAgentSDK.query",
        "assistant_message",
    ]


def test_span_processor_shutdown_clears_pending_calls_and_force_flush_returns_true():
    processor = _processor.ClaudeAgentSDKSpanProcessor()
    tool_span = _make_span(
        name="execute_tool calculator",
        trace_id=77,
        span_id=4,
        parent_span_id=3,
        attributes={
            "gen_ai.operation.name": "execute_tool",
            "gen_ai.tool.name": "calculator",
            "gen_ai.tool.call.arguments": {"expression": "2 + 2"},
        },
    )

    processor.on_end(tool_span)

    assert processor._pending_tool_calls_by_parent

    processor.shutdown()

    assert processor._pending_tool_calls_by_parent == {}
    assert processor.force_flush() is True
