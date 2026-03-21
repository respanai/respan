"""Pipeline integration and end-to-end tests for GoogleAdkInstrumentor.

These tests exercise the real instrumentation pipeline:
  OTel TracerProvider -> SimpleSpanProcessor -> instrumentor wrapping -> exporter

External APIs are mocked, but the instrumentation and converter code paths are real.
"""

import json
from types import SimpleNamespace

import pytest
from opentelemetry import trace as trace_api
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter

import respan_instrumentation_google_adk.instrumentor as inst_module
from respan_instrumentation_google_adk import GoogleAdkInstrumentor
from respan_instrumentation_google_adk.converter import AdkSpanConverter
from respan_instrumentation_google_adk.utils import otel_span_to_dict
from respan_sdk.constants.llm_logging import (
    LOG_TYPE_AGENT,
    LOG_TYPE_GENERATION,
    LOG_TYPE_TOOL,
    LOG_TYPE_WORKFLOW,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_instrumentor_state():
    """Reset global instrumentor state between tests."""
    original_exporter = inst_module._ACTIVE_EXPORTER
    original_converter = inst_module._ACTIVE_CONVERTER
    original_dedupe = inst_module._ACTIVE_DEDUPE
    original_passthrough = inst_module._ACTIVE_PASSTHROUGH
    original_max_buf = inst_module._ACTIVE_TRACE_BUFFER_MAX_SPANS
    yield
    inst_module._ACTIVE_EXPORTER = original_exporter
    inst_module._ACTIVE_CONVERTER = original_converter
    inst_module._ACTIVE_DEDUPE = original_dedupe
    inst_module._ACTIVE_PASSTHROUGH = original_passthrough
    inst_module._ACTIVE_TRACE_BUFFER_MAX_SPANS = original_max_buf
    with inst_module._TRACE_BUFFER_LOCK:
        inst_module._TRACE_BUFFER.clear()


@pytest.fixture
def captured_payloads():
    """Set up instrumentor with a mock exporter that captures payloads."""
    captured = []

    mock_exporter = SimpleNamespace(export=lambda payloads: captured.extend(payloads))
    instrumentor = GoogleAdkInstrumentor()
    instrumentor.activate(mock_exporter)

    yield captured

    instrumentor.deactivate()


def _create_adk_tracer():
    """Create a tracer that produces spans with google_adk scope name."""
    provider = trace_sdk.TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    return provider.get_tracer("google_adk", "1.0.0")


# ---------------------------------------------------------------------------
# Protocol conformance tests
# ---------------------------------------------------------------------------


class TestInstrumentationProtocol:
    """Verify GoogleAdkInstrumentor satisfies the Instrumentation protocol."""

    def test_has_name_attribute(self):
        instrumentor = GoogleAdkInstrumentor()
        assert instrumentor.name == "google-adk"

    def test_has_activate_method(self):
        assert callable(getattr(GoogleAdkInstrumentor, "activate", None))

    def test_has_deactivate_method(self):
        assert callable(getattr(GoogleAdkInstrumentor, "deactivate", None))

    def test_activate_sets_globals(self):
        mock_exporter = SimpleNamespace(export=lambda payloads: None)
        instrumentor = GoogleAdkInstrumentor()
        instrumentor.activate(mock_exporter)

        assert inst_module._ACTIVE_EXPORTER is mock_exporter
        assert inst_module._ACTIVE_CONVERTER is not None
        assert inst_module._ACTIVE_DEDUPE is not None

        instrumentor.deactivate()

    def test_deactivate_clears_globals(self):
        mock_exporter = SimpleNamespace(export=lambda payloads: None)
        instrumentor = GoogleAdkInstrumentor()
        instrumentor.activate(mock_exporter)
        instrumentor.deactivate()

        assert inst_module._ACTIVE_EXPORTER is None
        assert inst_module._ACTIVE_CONVERTER is None
        assert inst_module._ACTIVE_DEDUPE is None

    def test_deactivate_flushes_and_clears_trace_buffer(self):
        mock_exporter = SimpleNamespace(export=lambda payloads: None)
        instrumentor = GoogleAdkInstrumentor()
        instrumentor.activate(mock_exporter)
        with inst_module._TRACE_BUFFER_LOCK:
            inst_module._TRACE_BUFFER["deadbeef"] = []
        instrumentor.deactivate()
        assert inst_module._TRACE_BUFFER == {}

    def test_trace_buffer_max_spans_applied_on_activate(self):
        mock_exporter = SimpleNamespace(export=lambda payloads: None)
        instrumentor = GoogleAdkInstrumentor(trace_buffer_max_spans=128)
        instrumentor.activate(mock_exporter)
        assert inst_module._ACTIVE_TRACE_BUFFER_MAX_SPANS == 128
        instrumentor.deactivate()

    def test_constructor_params_passed_to_converter(self):
        instrumentor = GoogleAdkInstrumentor(
            environment="staging",
            customer_identifier="user-42",
        )
        mock_exporter = SimpleNamespace(export=lambda payloads: None)
        instrumentor.activate(mock_exporter)

        converter = inst_module._ACTIVE_CONVERTER
        assert converter.environment == "staging"
        assert converter.customer_identifier == "user-42"

        instrumentor.deactivate()


# ---------------------------------------------------------------------------
# Multi-turn pipeline test
# ---------------------------------------------------------------------------


class TestMultiTurnPipeline:
    def test_multi_turn_spans_reach_exporter(self, captured_payloads):
        tracer = _create_adk_tracer()

        turns = [
            ("Hi! My name is Alex.", "Hello Alex! Nice to meet you."),
            ("What's my name?", "Your name is Alex!"),
            ("Tell me a fun fact about 42.", "42 is the answer to everything."),
        ]

        for i, (user_msg, assistant_msg) in enumerate(turns):
            inv = tracer.start_span("invocation", attributes={
                "gen_ai.conversation.id": "session-mt-pipeline",
            })
            agent = tracer.start_span("agent_run", attributes={
                "gen_ai.agent.name": "chat_agent",
            }, context=trace_api.set_span_in_context(inv))
            llm = tracer.start_span("call_llm", attributes={
                "gen_ai.request.model": "gemini-2.0-flash",
                "gen_ai.response.model": "gemini-2.0-flash",
                "gen_ai.usage.input_tokens": 10 + i * 20,
                "gen_ai.usage.output_tokens": 15 + i * 5,
                "gen_ai.input.messages": json.dumps(
                    [{"role": "user", "content": user_msg}]
                ),
                "gen_ai.output.messages": json.dumps(
                    [{"role": "model", "content": assistant_msg}]
                ),
            }, context=trace_api.set_span_in_context(agent))

            llm.end()
            agent.end()
            inv.end()

        assert len(captured_payloads) >= 9

        log_types = [p["log_type"] for p in captured_payloads]
        assert log_types.count(LOG_TYPE_WORKFLOW) >= 3
        assert log_types.count(LOG_TYPE_AGENT) >= 3
        assert log_types.count(LOG_TYPE_GENERATION) >= 3

    def test_multi_turn_conversation_id_preserved(self, captured_payloads):
        tracer = _create_adk_tracer()

        inv = tracer.start_span("invocation", attributes={
            "gen_ai.conversation.id": "conv-preserved-test",
        })
        agent = tracer.start_span("agent_run", attributes={
            "gen_ai.agent.name": "chat_agent",
        }, context=trace_api.set_span_in_context(inv))
        llm = tracer.start_span("call_llm", attributes={
            "gen_ai.request.model": "gemini-2.0-flash",
            "gen_ai.usage.input_tokens": 10,
            "gen_ai.usage.output_tokens": 15,
        }, context=trace_api.set_span_in_context(agent))

        llm.end()
        agent.end()
        inv.end()

        workflows = [p for p in captured_payloads if p["log_type"] == LOG_TYPE_WORKFLOW]
        assert len(workflows) >= 1
        assert workflows[0].get("session_identifier") == "conv-preserved-test"

    def test_multi_turn_increasing_token_counts(self, captured_payloads):
        tracer = _create_adk_tracer()

        for i in range(3):
            inv = tracer.start_span("invocation", attributes={})
            agent = tracer.start_span("agent_run", attributes={
                "gen_ai.agent.name": "chat_agent",
            }, context=trace_api.set_span_in_context(inv))
            llm = tracer.start_span("call_llm", attributes={
                "gen_ai.request.model": "gemini-2.0-flash",
                "gen_ai.usage.input_tokens": 10 + i * 30,
                "gen_ai.usage.output_tokens": 20,
            }, context=trace_api.set_span_in_context(agent))
            llm.end()
            agent.end()
            inv.end()

        gens = [p for p in captured_payloads if p["log_type"] == LOG_TYPE_GENERATION]
        token_counts = sorted([g.get("prompt_tokens", 0) for g in gens])
        assert token_counts == [10, 40, 70]


# ---------------------------------------------------------------------------
# Streaming pipeline test
# ---------------------------------------------------------------------------


class TestStreamingPipeline:
    def test_streaming_spans_flow_through_pipeline(self, captured_payloads):
        tracer = _create_adk_tracer()

        long_story = (
            "Once upon a time, in a lab filled with brushes and canvases, "
            "there lived a robot named Pixel who dreamed of painting..."
        )

        inv = tracer.start_span("invocation", attributes={})
        agent = tracer.start_span("agent_run", attributes={
            "gen_ai.agent.name": "storyteller",
        }, context=trace_api.set_span_in_context(inv))
        llm = tracer.start_span("call_llm", attributes={
            "gen_ai.request.model": "gemini-2.0-flash",
            "gen_ai.response.model": "gemini-2.0-flash",
            "gen_ai.usage.input_tokens": 25,
            "gen_ai.usage.output_tokens": 200,
            "gen_ai.input.messages": json.dumps(
                [{"role": "user", "content": "Tell me a short story about a robot learning to paint."}]
            ),
            "gen_ai.output.messages": json.dumps(
                [{"role": "model", "content": long_story}]
            ),
            "gen_ai.request.temperature": 0.9,
        }, context=trace_api.set_span_in_context(agent))

        llm.end()
        agent.end()
        inv.end()

        assert len(captured_payloads) >= 3

        gen = next(p for p in captured_payloads if p["log_type"] == LOG_TYPE_GENERATION)
        assert gen["prompt_tokens"] == 25
        assert gen["completion_tokens"] == 200
        assert gen["model"] == "gemini-2.0-flash"
        assert "robot" in str(gen.get("output", "")).lower() or "pixel" in str(gen.get("output", "")).lower()

    def test_batch_path_propagates_output(self):
        """When spans are grouped (as in batch export), output propagates to
        workflow and agent spans."""
        from respan_instrumentation_google_adk.instrumentor import _export_adk_spans

        captured = []
        mock_exporter = SimpleNamespace(export=lambda payloads: captured.extend(payloads))
        converter = AdkSpanConverter()

        old_exporter = inst_module._ACTIVE_EXPORTER
        old_converter = inst_module._ACTIVE_CONVERTER
        old_dedupe = inst_module._ACTIVE_DEDUPE
        inst_module._ACTIVE_EXPORTER = mock_exporter
        inst_module._ACTIVE_CONVERTER = converter
        inst_module._ACTIVE_DEDUPE = inst_module._SpanDedupeCache()

        try:
            trace_id = 0xABCDEF1234567890ABCDEF1234567890
            ctx_inv = SimpleNamespace(trace_id=trace_id, span_id=0x1111111111111111)
            ctx_agent = SimpleNamespace(trace_id=trace_id, span_id=0x2222222222222222)
            ctx_llm = SimpleNamespace(trace_id=trace_id, span_id=0x3333333333333333)
            scope = SimpleNamespace(name="google_adk", version="1.0.0")
            kind = SimpleNamespace(name="INTERNAL")
            ok_status = SimpleNamespace(
                status_code=SimpleNamespace(name="OK"), description=None
            )
            base = 1_700_000_000_000_000_000

            inv = SimpleNamespace(
                name="invocation", context=ctx_inv, parent=None,
                instrumentation_scope=scope, kind=kind, status=ok_status,
                attributes={}, start_time=base, end_time=base + 3_000_000_000,
            )
            agent = SimpleNamespace(
                name="agent_run", context=ctx_agent,
                parent=SimpleNamespace(span_id=0x1111111111111111, trace_id=trace_id, is_remote=False),
                instrumentation_scope=scope, kind=kind, status=ok_status,
                attributes={"gen_ai.agent.name": "storyteller"},
                start_time=base + 100_000_000, end_time=base + 2_900_000_000,
            )
            llm = SimpleNamespace(
                name="call_llm", context=ctx_llm,
                parent=SimpleNamespace(span_id=0x2222222222222222, trace_id=trace_id, is_remote=False),
                instrumentation_scope=scope, kind=kind, status=ok_status,
                attributes={
                    "gen_ai.request.model": "gemini-2.0-flash",
                    "gen_ai.output.messages": json.dumps(
                        [{"role": "model", "content": "A story about painting."}]
                    ),
                },
                start_time=base + 200_000_000, end_time=base + 2_800_000_000,
            )

            _export_adk_spans([inv, agent, llm])

            wf = next(p for p in captured if p["log_type"] == LOG_TYPE_WORKFLOW)
            ag = next(p for p in captured if p["log_type"] == LOG_TYPE_AGENT)
            assert wf.get("output") is not None
            assert ag.get("output") is not None
        finally:
            inst_module._ACTIVE_EXPORTER = old_exporter
            inst_module._ACTIVE_CONVERTER = old_converter
            inst_module._ACTIVE_DEDUPE = old_dedupe

    def test_streaming_llm_config_captured(self, captured_payloads):
        tracer = _create_adk_tracer()

        inv = tracer.start_span("invocation", attributes={})
        agent = tracer.start_span("agent_run", attributes={
            "gen_ai.agent.name": "storyteller",
        }, context=trace_api.set_span_in_context(inv))
        llm = tracer.start_span("call_llm", attributes={
            "gen_ai.request.model": "gemini-2.0-flash",
            "gen_ai.request.temperature": 0.9,
            "gen_ai.request.top_p": 0.95,
        }, context=trace_api.set_span_in_context(agent))

        llm.end()
        agent.end()
        inv.end()

        gen = next(p for p in captured_payloads if p["log_type"] == LOG_TYPE_GENERATION)
        assert gen["metadata"]["llm_config"]["temperature"] == 0.9
        assert gen["metadata"]["llm_config"]["top_p"] == 0.95


# ---------------------------------------------------------------------------
# Tool calls pipeline test
# ---------------------------------------------------------------------------


class TestToolCallsPipeline:
    def test_tool_call_full_pipeline(self, captured_payloads):
        tracer = _create_adk_tracer()

        inv = tracer.start_span("invocation", attributes={})
        agent = tracer.start_span("agent_run", attributes={
            "gen_ai.agent.name": "weather_agent",
        }, context=trace_api.set_span_in_context(inv))

        llm1 = tracer.start_span("call_llm", attributes={
            "gen_ai.request.model": "gemini-2.0-flash",
            "gen_ai.response.model": "gemini-2.0-flash",
            "gen_ai.usage.input_tokens": 30,
            "gen_ai.usage.output_tokens": 15,
            "gen_ai.input.messages": json.dumps(
                [{"role": "user", "content": "What's the weather in Tokyo and London?"}]
            ),
            "gen_ai.output.messages": json.dumps(
                [{"role": "model", "content": "I'll check the weather for you."}]
            ),
        }, context=trace_api.set_span_in_context(agent))
        llm1.end()

        tool = tracer.start_span("execute_tool", attributes={
            "gen_ai.tool.name": "get_weather",
        }, context=trace_api.set_span_in_context(agent))
        tool.end()

        llm2 = tracer.start_span("call_llm", attributes={
            "gen_ai.request.model": "gemini-2.0-flash",
            "gen_ai.response.model": "gemini-2.0-flash",
            "gen_ai.usage.input_tokens": 60,
            "gen_ai.usage.output_tokens": 40,
            "gen_ai.input.messages": json.dumps([
                {"role": "user", "content": "What's the weather in Tokyo and London?"},
                {"role": "model", "content": "I'll check the weather for you."},
                {"role": "tool", "content": json.dumps({"city": "Tokyo", "temp_f": 80})},
            ]),
            "gen_ai.output.messages": json.dumps(
                [{"role": "model", "content": "Tokyo is 80°F. London is 59°F."}]
            ),
        }, context=trace_api.set_span_in_context(agent))
        llm2.end()

        agent.end()
        inv.end()

        assert len(captured_payloads) >= 5

        type_counts = {}
        for p in captured_payloads:
            lt = p["log_type"]
            type_counts[lt] = type_counts.get(lt, 0) + 1

        assert type_counts.get(LOG_TYPE_WORKFLOW, 0) >= 1
        assert type_counts.get(LOG_TYPE_AGENT, 0) >= 1
        assert type_counts.get(LOG_TYPE_GENERATION, 0) >= 2
        assert type_counts.get(LOG_TYPE_TOOL, 0) >= 1

    def test_tool_name_captured(self, captured_payloads):
        tracer = _create_adk_tracer()

        inv = tracer.start_span("invocation", attributes={})
        agent = tracer.start_span("agent_run", attributes={
            "gen_ai.agent.name": "weather_agent",
        }, context=trace_api.set_span_in_context(inv))
        tool = tracer.start_span("execute_tool", attributes={
            "gen_ai.tool.name": "get_weather",
        }, context=trace_api.set_span_in_context(agent))

        tool.end()
        agent.end()
        inv.end()

        tool_payloads = [p for p in captured_payloads if p["log_type"] == LOG_TYPE_TOOL]
        assert len(tool_payloads) >= 1
        assert tool_payloads[0]["span_tools"] == ["get_weather"]
        assert tool_payloads[0]["log_type"] == LOG_TYPE_TOOL

    def test_all_spans_share_trace_id(self, captured_payloads):
        tracer = _create_adk_tracer()

        inv = tracer.start_span("invocation", attributes={})
        agent = tracer.start_span("agent_run", attributes={
            "gen_ai.agent.name": "weather_agent",
        }, context=trace_api.set_span_in_context(inv))
        llm = tracer.start_span("call_llm", attributes={
            "gen_ai.request.model": "gemini-2.0-flash",
        }, context=trace_api.set_span_in_context(agent))
        tool = tracer.start_span("execute_tool", attributes={
            "gen_ai.tool.name": "get_weather",
        }, context=trace_api.set_span_in_context(agent))

        tool.end()
        llm.end()
        agent.end()
        inv.end()

        trace_ids = {p["trace_id"] for p in captured_payloads}
        assert len(trace_ids) == 1, f"Expected 1 trace_id, got {len(trace_ids)}: {trace_ids}"


# ---------------------------------------------------------------------------
# Deduplication test
# ---------------------------------------------------------------------------


class TestDeduplication:
    def test_same_span_not_exported_twice(self, captured_payloads):
        tracer = _create_adk_tracer()

        # Create a full trace (root + child) so the buffer flushes on root end
        root = tracer.start_span("invocation")
        child = tracer.start_span(
            "call_llm",
            context=trace_api.set_span_in_context(root),
            attributes={
                "gen_ai.request.model": "gemini-2.0-flash",
            },
        )
        child.end()
        root.end()

        initial_count = len(captured_payloads)
        assert initial_count > 0, "Trace should have been exported"

        # Re-exporting the same spans should be deduplicated
        from respan_instrumentation_google_adk.instrumentor import _export_adk_spans
        _export_adk_spans([root, child])

        assert len(captured_payloads) == initial_count


# ---------------------------------------------------------------------------
# E2E tests with OpenLLMetry conventions
# ---------------------------------------------------------------------------


TRACE_ID = 0xABCDEF1234567890ABCDEF1234567890
BASE_NS = 1_700_000_000_000_000_000


def _mock_span(
    name,
    span_id,
    parent_span_id=None,
    attributes=None,
    start_ns=BASE_NS,
    end_ns=None,
):
    end_ns = end_ns or start_ns + 1_000_000_000
    ctx = SimpleNamespace(trace_id=TRACE_ID, span_id=span_id)
    parent = None
    if parent_span_id is not None:
        parent = SimpleNamespace(
            span_id=parent_span_id, trace_id=TRACE_ID, is_remote=False
        )
    return SimpleNamespace(
        name=name,
        context=ctx,
        parent=parent,
        instrumentation_scope=SimpleNamespace(name="google_adk", version="1.0.0"),
        kind=SimpleNamespace(name="INTERNAL"),
        status=SimpleNamespace(
            status_code=SimpleNamespace(name="OK"), description=None
        ),
        attributes=attributes or {},
        start_time=start_ns,
        end_time=end_ns,
    )


def _adk_tracer():
    provider = trace_sdk.TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    return provider.get_tracer("google_adk", "1.0.0")


class TestMultiTurnE2E:
    def _emit_turn(self, tracer, turn_idx, user_msg, assistant_msg, conv_id="session-e2e"):
        inv = tracer.start_span("invocation", attributes={
            "gen_ai.conversation.id": conv_id,
        })
        agent = tracer.start_span("agent_run", attributes={
            "gen_ai.agent.name": "chat_agent",
        }, context=trace_api.set_span_in_context(inv))
        llm = tracer.start_span("call_llm", attributes={
            "gen_ai.request.model": "gemini-2.0-flash",
            "gen_ai.response.model": "gemini-2.0-flash",
            "gen_ai.usage.prompt_tokens": 10 + turn_idx * 25,
            "gen_ai.usage.completion_tokens": 15 + turn_idx * 5,
            "gen_ai.input.messages": json.dumps(
                [{"role": "user", "content": user_msg}]
            ),
            "gen_ai.output.messages": json.dumps(
                [{"role": "model", "content": assistant_msg}]
            ),
        }, context=trace_api.set_span_in_context(agent))
        llm.end()
        agent.end()
        inv.end()

    def test_first_class_fields_replace_traceloop_metadata(self, captured_payloads):
        tracer = _adk_tracer()
        self._emit_turn(tracer, 0, "Hi", "Hello")

        wf = next(p for p in captured_payloads if p["log_type"] == LOG_TYPE_WORKFLOW)
        agent = next(p for p in captured_payloads if p["log_type"] == LOG_TYPE_AGENT)
        gen = next(p for p in captured_payloads if p["log_type"] == LOG_TYPE_GENERATION)

        assert wf["log_type"] == LOG_TYPE_WORKFLOW
        assert agent["log_type"] == LOG_TYPE_AGENT
        assert gen["log_type"] == LOG_TYPE_GENERATION

        assert wf["status"] == "success"
        assert agent["status"] == "success"
        assert gen["status"] == "success"

        assert agent["span_name"] is not None
        assert wf["span_workflow_name"] is not None

    def test_gen_ai_system_not_in_metadata(self, captured_payloads):
        tracer = _adk_tracer()
        self._emit_turn(tracer, 0, "Hi", "Hello")

        for p in captured_payloads:
            metadata = p.get("metadata") or {}
            assert "gen_ai.system" not in metadata

    def test_parent_child_chain(self, captured_payloads):
        tracer = _adk_tracer()
        self._emit_turn(tracer, 0, "Hi", "Hello")

        wf = next(p for p in captured_payloads if p["log_type"] == LOG_TYPE_WORKFLOW)
        agent = next(p for p in captured_payloads if p["log_type"] == LOG_TYPE_AGENT)
        gen = next(p for p in captured_payloads if p["log_type"] == LOG_TYPE_GENERATION)

        assert wf.get("parent_id") is None
        assert agent["parent_id"] == wf["span_id"]
        assert gen["parent_id"] == agent["span_id"]

    def test_output_propagates_to_workflow_and_agent_batch(self):
        from respan_instrumentation_google_adk.instrumentor import _export_adk_spans

        captured = []
        mock_exporter = SimpleNamespace(export=lambda payloads: captured.extend(payloads))
        converter = AdkSpanConverter()

        old_exp = inst_module._ACTIVE_EXPORTER
        old_conv = inst_module._ACTIVE_CONVERTER
        old_ded = inst_module._ACTIVE_DEDUPE
        inst_module._ACTIVE_EXPORTER = mock_exporter
        inst_module._ACTIVE_CONVERTER = converter
        inst_module._ACTIVE_DEDUPE = inst_module._SpanDedupeCache()

        try:
            inv = _mock_span("invocation", 0x1001, attributes={
                "gen_ai.conversation.id": "conv-batch",
            })
            ag = _mock_span("agent_run", 0x1002, parent_span_id=0x1001, attributes={
                "gen_ai.agent.name": "chat_agent",
            }, start_ns=BASE_NS + 100_000_000)
            llm = _mock_span("call_llm", 0x1003, parent_span_id=0x1002, attributes={
                "gen_ai.request.model": "gemini-2.0-flash",
                "gen_ai.usage.prompt_tokens": 10,
                "gen_ai.usage.completion_tokens": 15,
                "gen_ai.output.messages": json.dumps(
                    [{"role": "model", "content": "Hello Alex!"}]
                ),
            }, start_ns=BASE_NS + 200_000_000)

            _export_adk_spans([inv, ag, llm])

            wf = next(p for p in captured if p["log_type"] == LOG_TYPE_WORKFLOW)
            agent_p = next(p for p in captured if p["log_type"] == LOG_TYPE_AGENT)
            assert wf["output"] is not None
            assert agent_p["output"] is not None
            assert "Alex" in str(wf["output"])
        finally:
            inst_module._ACTIVE_EXPORTER = old_exp
            inst_module._ACTIVE_CONVERTER = old_conv
            inst_module._ACTIVE_DEDUPE = old_ded


class TestOpenLLMetryConventionFallbacks:
    def setup_method(self):
        self.converter = AdkSpanConverter()

    def test_prompt_tokens_convention(self):
        span = _mock_span("call_llm", 0xC001, attributes={
            "gen_ai.request.model": "gemini-2.0-flash",
            "gen_ai.usage.prompt_tokens": 42,
            "gen_ai.usage.completion_tokens": 18,
        })
        payloads = self.converter.convert(
            trace_or_spans=[otel_span_to_dict(span)]
        )
        gen = payloads[0]
        assert gen["prompt_tokens"] == 42
        assert gen["completion_tokens"] == 18

    def test_input_tokens_fallback(self):
        span = _mock_span("call_llm", 0xC002, attributes={
            "gen_ai.request.model": "gemini-2.0-flash",
            "gen_ai.usage.input_tokens": 50,
            "gen_ai.usage.output_tokens": 20,
        })
        payloads = self.converter.convert(
            trace_or_spans=[otel_span_to_dict(span)]
        )
        gen = payloads[0]
        assert gen["prompt_tokens"] == 50
        assert gen["completion_tokens"] == 20

    def test_total_tokens_computed(self):
        span = _mock_span("call_llm", 0xC003, attributes={
            "gen_ai.request.model": "gemini-2.0-flash",
            "gen_ai.usage.prompt_tokens": 100,
            "gen_ai.usage.completion_tokens": 50,
        })
        payloads = self.converter.convert(
            trace_or_spans=[otel_span_to_dict(span)]
        )
        assert payloads[0]["total_request_tokens"] == 150
