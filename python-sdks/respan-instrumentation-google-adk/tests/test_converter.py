"""Unit tests for AdkSpanConverter and utility functions."""

import json
from types import SimpleNamespace
import pytest

from respan_instrumentation_google_adk.utils import (
    extract_genai_messages,
    is_adk_span,
    otel_span_to_dict,
)
from respan_instrumentation_google_adk.converter import AdkSpanConverter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TRACE_ID = 0xABCDEF1234567890ABCDEF1234567890


def _make_span(
    name="call_llm",
    trace_id=0xABCDEF1234567890ABCDEF1234567890,
    span_id=0x1234567890ABCDEF,
    parent_span_id=None,
    attributes=None,
    scope_name="google_adk",
    start_time=1_700_000_000_000_000_000,
    end_time=1_700_000_001_000_000_000,
    status_name="OK",
):
    """Create a mock OTel span resembling an ADK span."""
    ctx = SimpleNamespace(trace_id=trace_id, span_id=span_id)
    parent = None
    if parent_span_id is not None:
        parent = SimpleNamespace(span_id=parent_span_id, trace_id=trace_id, is_remote=False)
    scope = SimpleNamespace(name=scope_name, version="1.0.0")
    kind = SimpleNamespace(name="INTERNAL")
    status_code_enum = SimpleNamespace(name=status_name)
    status = SimpleNamespace(status_code=status_code_enum, description=None)
    return SimpleNamespace(
        name=name,
        context=ctx,
        parent=parent,
        instrumentation_scope=scope,
        kind=kind,
        status=status,
        attributes=attributes or {},
        start_time=start_time,
        end_time=end_time,
    )


def _spans_to_dicts(spans):
    return [otel_span_to_dict(s) for s in spans]


# ---------------------------------------------------------------------------
# is_adk_span tests
# ---------------------------------------------------------------------------


class TestIsAdkSpan:
    def test_positive_by_scope_name(self):
        span = _make_span(scope_name="google_adk")
        assert is_adk_span(span) is True

    def test_positive_by_scope_name_hyphen(self):
        span = _make_span(scope_name="google-adk")
        assert is_adk_span(span) is True

    def test_positive_by_span_name_and_genai_attrs(self):
        span = _make_span(
            scope_name="other_lib",
            name="call_llm",
            attributes={"gen_ai.request.model": "gemini-2.0-flash"},
        )
        assert is_adk_span(span) is True

    def test_negative_unknown_scope_no_genai(self):
        span = _make_span(scope_name="some_other_lib", name="random_op", attributes={})
        assert is_adk_span(span) is False

    def test_negative_adk_name_but_no_genai_attrs(self):
        span = _make_span(scope_name="other", name="call_llm", attributes={"foo": "bar"})
        assert is_adk_span(span) is False


# ---------------------------------------------------------------------------
# otel_span_to_dict tests
# ---------------------------------------------------------------------------


class TestOtelSpanToDict:
    def test_basic_conversion(self):
        span = _make_span(
            name="agent_run",
            attributes={"gen_ai.agent.name": "my_agent"},
        )
        result = otel_span_to_dict(span)
        assert result["name"] == "agent_run"
        assert result["trace_id"] is not None
        assert result["span_id"] is not None
        assert result["parent_id"] is None
        assert result["attributes"]["gen_ai.agent.name"] == "my_agent"
        assert result["status_code"] == 200

    def test_with_parent(self):
        span = _make_span(parent_span_id=0xFEDCBA0987654321)
        result = otel_span_to_dict(span)
        assert result["parent_id"] is not None

    def test_error_status(self):
        span = _make_span(status_name="ERROR")
        span.status.description = "Something went wrong"
        result = otel_span_to_dict(span)
        assert result["status_code"] == 500
        assert result["error"] == "Something went wrong"


# ---------------------------------------------------------------------------
# _map_log_type tests
# ---------------------------------------------------------------------------


class TestMapLogType:
    def setup_method(self):
        self.converter = AdkSpanConverter()

    def test_invocation(self):
        assert self.converter._map_log_type("invocation", None, None) == "workflow"

    def test_agent_run(self):
        assert self.converter._map_log_type("agent_run", "parent", None) == "agent"

    def test_call_llm(self):
        assert self.converter._map_log_type("call_llm", "parent", None) == "generation"

    def test_execute_tool(self):
        assert self.converter._map_log_type("execute_tool", "parent", None) == "tool"

    def test_fallback_model(self):
        assert self.converter._map_log_type("unknown_span", None, "gemini-2.0") == "generation"

    def test_fallback_no_parent(self):
        assert self.converter._map_log_type("unknown_span", None, None) == "workflow"

    def test_fallback_with_parent(self):
        assert self.converter._map_log_type("unknown_span", "some-parent", None) == "task"


# ---------------------------------------------------------------------------
# extract_genai_messages tests
# ---------------------------------------------------------------------------


class TestExtractGenaiMessages:
    def test_json_string_list(self):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "model", "content": "Hi there!"},
        ]
        attrs = {"gen_ai.input.messages": json.dumps(messages)}
        result = extract_genai_messages(attrs, "gen_ai.input.messages")
        assert result is not None
        assert len(result) == 2
        assert result[0]["role"] == "user"
        assert result[1]["content"] == "Hi there!"

    def test_single_dict(self):
        attrs = {"gen_ai.output.messages": json.dumps({"role": "model", "content": "Done"})}
        result = extract_genai_messages(attrs, "gen_ai.output.messages")
        assert result is not None
        assert len(result) == 1
        assert result[0]["role"] == "model"

    def test_missing_key(self):
        result = extract_genai_messages({}, "gen_ai.input.messages")
        assert result is None

    def test_none_value(self):
        result = extract_genai_messages({"gen_ai.input.messages": None}, "gen_ai.input.messages")
        assert result is None

    def test_invalid_json(self):
        result = extract_genai_messages(
            {"gen_ai.input.messages": "not valid json {{"}, "gen_ai.input.messages"
        )
        assert result is None


# ---------------------------------------------------------------------------
# convert tests (was build_payload)
# ---------------------------------------------------------------------------


class TestConvert:
    def setup_method(self):
        self.converter = AdkSpanConverter()

    def test_single_llm_span(self):
        span = _make_span(
            name="call_llm",
            attributes={
                "gen_ai.request.model": "gemini-2.0-flash",
                "gen_ai.response.model": "gemini-2.0-flash",
                "gen_ai.usage.input_tokens": 10,
                "gen_ai.usage.output_tokens": 20,
                "gen_ai.input.messages": json.dumps(
                    [{"role": "user", "content": "What is 2+2?"}]
                ),
                "gen_ai.output.messages": json.dumps(
                    [{"role": "model", "content": "4"}]
                ),
            },
        )
        span_dict = otel_span_to_dict(span)
        payloads = self.converter.convert(trace_or_spans=[span_dict])

        assert len(payloads) == 1
        payload = payloads[0]
        assert payload["log_type"] == "generation"
        assert payload["model"] == "gemini-2.0-flash"
        assert payload["prompt_tokens"] == 10
        assert payload["completion_tokens"] == 20
        assert payload["total_request_tokens"] == 30
        assert payload["prompt_messages"] is not None
        assert payload["completion_message"] is not None
        assert payload["trace_id"] is not None
        assert payload["span_id"] is not None

    def test_agent_run_span(self):
        span = _make_span(
            name="agent_run",
            attributes={
                "gen_ai.agent.name": "weather_agent",
                "gen_ai.agent.id": "agent-123",
            },
        )
        span_dict = otel_span_to_dict(span)
        payloads = self.converter.convert(trace_or_spans=[span_dict])

        assert len(payloads) == 1
        payload = payloads[0]
        assert payload["log_type"] == "agent"
        assert payload["span_name"] == "agent_run"
        assert payload["metadata"]["adk_agent_id"] == "agent-123"

    def test_tool_span(self):
        span = _make_span(
            name="execute_tool",
            parent_span_id=0xAAAABBBBCCCCDDDD,
            attributes={
                "gen_ai.tool.name": "get_weather",
            },
        )
        span_dict = otel_span_to_dict(span)
        payloads = self.converter.convert(trace_or_spans=[span_dict])

        assert len(payloads) == 1
        payload = payloads[0]
        assert payload["log_type"] == "tool"
        assert payload["span_tools"] == ["get_weather"]
        parent_keys = ("parent_id", "span_parent_id")
        assert any(k in payload and payload[k] is not None for k in parent_keys)

    def test_llm_config_in_metadata(self):
        span = _make_span(
            name="call_llm",
            attributes={
                "gen_ai.request.model": "gemini-2.0-flash",
                "gen_ai.request.temperature": 0.7,
                "gen_ai.request.top_k": 40,
                "gen_ai.request.top_p": 0.9,
                "gen_ai.request.max_output_tokens": 1024,
            },
        )
        span_dict = otel_span_to_dict(span)
        payloads = self.converter.convert(trace_or_spans=[span_dict])

        payload = payloads[0]
        llm_config = payload["metadata"]["llm_config"]
        assert llm_config["temperature"] == 0.7
        assert llm_config["top_k"] == 40
        assert llm_config["top_p"] == 0.9
        assert llm_config["max_tokens"] == 1024

    def test_conversation_id(self):
        span = _make_span(
            name="invocation",
            attributes={
                "gen_ai.conversation.id": "conv-abc-123",
            },
        )
        span_dict = otel_span_to_dict(span)
        payloads = self.converter.convert(trace_or_spans=[span_dict])

        payload = payloads[0]
        assert payload["session_identifier"] == "conv-abc-123"

    def test_multi_span_trace(self):
        """Test a realistic multi-span ADK trace."""
        invocation = _make_span(
            name="invocation",
            span_id=0x1111111111111111,
            attributes={"gen_ai.conversation.id": "session-1"},
        )
        agent = _make_span(
            name="agent_run",
            span_id=0x2222222222222222,
            parent_span_id=0x1111111111111111,
            attributes={"gen_ai.agent.name": "my_agent"},
        )
        llm = _make_span(
            name="call_llm",
            span_id=0x3333333333333333,
            parent_span_id=0x2222222222222222,
            attributes={
                "gen_ai.request.model": "gemini-2.0-flash",
                "gen_ai.usage.input_tokens": 50,
                "gen_ai.usage.output_tokens": 100,
                "gen_ai.output.messages": json.dumps(
                    [{"role": "model", "content": "The answer is 42."}]
                ),
            },
        )

        span_dicts = [otel_span_to_dict(s) for s in [invocation, agent, llm]]
        payloads = self.converter.convert(trace_or_spans=span_dicts)

        assert len(payloads) == 3
        log_types = {p["log_type"] for p in payloads}
        assert log_types == {"workflow", "agent", "generation"}

        # Output should propagate from generation to workflow/agent
        gen_payload = next(p for p in payloads if p["log_type"] == "generation")
        assert gen_payload["output"] is not None

        wf_payload = next(p for p in payloads if p["log_type"] == "workflow")
        assert wf_payload["output"] is not None


# ---------------------------------------------------------------------------
# Example pattern tests (multi_turn, streaming, tool_calls)
# ---------------------------------------------------------------------------


class TestMultiTurnPattern:
    def setup_method(self):
        self.converter = AdkSpanConverter()

    def _make_turn(self, turn_index, user_msg, assistant_msg, conv_id="session-mt-1"):
        base_time = 1_700_000_000_000_000_000 + turn_index * 3_000_000_000
        inv_id = 0x1000000000000000 + turn_index * 0x100
        agent_id = inv_id + 1
        llm_id = inv_id + 2

        invocation = _make_span(
            name="invocation",
            span_id=inv_id,
            attributes={"gen_ai.conversation.id": conv_id},
            start_time=base_time,
            end_time=base_time + 2_500_000_000,
        )
        agent_run = _make_span(
            name="agent_run",
            span_id=agent_id,
            parent_span_id=inv_id,
            attributes={"gen_ai.agent.name": "chat_agent"},
            start_time=base_time + 100_000_000,
            end_time=base_time + 2_400_000_000,
        )
        call_llm = _make_span(
            name="call_llm",
            span_id=llm_id,
            parent_span_id=agent_id,
            attributes={
                "gen_ai.request.model": "gemini-2.0-flash",
                "gen_ai.response.model": "gemini-2.0-flash",
                "gen_ai.usage.input_tokens": 10 + turn_index * 20,
                "gen_ai.usage.output_tokens": 15 + turn_index * 5,
                "gen_ai.input.messages": json.dumps(
                    [{"role": "user", "content": user_msg}]
                ),
                "gen_ai.output.messages": json.dumps(
                    [{"role": "model", "content": assistant_msg}]
                ),
            },
            start_time=base_time + 200_000_000,
            end_time=base_time + 2_300_000_000,
        )
        return [invocation, agent_run, call_llm]

    def test_single_turn_produces_correct_log_types(self):
        from respan_sdk.constants.llm_logging import LOG_TYPE_WORKFLOW, LOG_TYPE_AGENT, LOG_TYPE_GENERATION
        spans = self._make_turn(0, "Hi! My name is Alex.", "Hello Alex!")
        payloads = self.converter.convert(trace_or_spans=_spans_to_dicts(spans))
        assert len(payloads) == 3
        log_types = {p["log_type"] for p in payloads}
        assert log_types == {LOG_TYPE_WORKFLOW, LOG_TYPE_AGENT, LOG_TYPE_GENERATION}

    def test_conversation_id_propagates(self):
        conv_id = "session-mt-abc"
        spans = self._make_turn(0, "Hello", "Hi!", conv_id=conv_id)
        payloads = self.converter.convert(trace_or_spans=_spans_to_dicts(spans))
        wf = next(p for p in payloads if p["log_type"] == "workflow")
        assert wf["session_identifier"] == conv_id

    def test_multi_turn_token_counts_increase(self):
        turn0 = self._make_turn(0, "Hi! My name is Alex.", "Hello Alex!")
        turn2 = self._make_turn(2, "Tell me a fun fact.", "42 is the answer.")
        p0 = self.converter.convert(trace_or_spans=_spans_to_dicts(turn0))
        p2 = self.converter.convert(trace_or_spans=_spans_to_dicts(turn2))
        gen0 = next(p for p in p0 if p["log_type"] == "generation")
        gen2 = next(p for p in p2 if p["log_type"] == "generation")
        assert gen2["prompt_tokens"] > gen0["prompt_tokens"]

    def test_output_propagates_to_workflow_and_agent(self):
        spans = self._make_turn(0, "What's my name?", "Your name is Alex.")
        payloads = self.converter.convert(trace_or_spans=_spans_to_dicts(spans))
        wf = next(p for p in payloads if p["log_type"] == "workflow")
        agent = next(p for p in payloads if p["log_type"] == "agent")
        gen = next(p for p in payloads if p["log_type"] == "generation")
        assert gen["output"] is not None
        assert wf["output"] is not None
        assert agent["output"] is not None

    def test_parent_child_relationships(self):
        spans = self._make_turn(0, "Hello", "Hi!")
        payloads = self.converter.convert(trace_or_spans=_spans_to_dicts(spans))
        wf = next(p for p in payloads if p["log_type"] == "workflow")
        agent = next(p for p in payloads if p["log_type"] == "agent")
        gen = next(p for p in payloads if p["log_type"] == "generation")
        assert wf.get("parent_id") is None
        assert agent.get("parent_id") == wf["span_id"] or agent.get("span_parent_id") == wf["span_id"]
        assert gen.get("parent_id") == agent["span_id"] or gen.get("span_parent_id") == agent["span_id"]


class TestStreamingPattern:
    def setup_method(self):
        self.converter = AdkSpanConverter()

    def _make_streaming_spans(self):
        invocation = _make_span(
            name="invocation",
            span_id=0xA000000000000001,
            start_time=1_700_000_000_000_000_000,
            end_time=1_700_000_005_000_000_000,
        )
        agent_run = _make_span(
            name="agent_run",
            span_id=0xA000000000000002,
            parent_span_id=0xA000000000000001,
            attributes={"gen_ai.agent.name": "storyteller"},
            start_time=1_700_000_000_100_000_000,
            end_time=1_700_000_004_900_000_000,
        )
        call_llm = _make_span(
            name="call_llm",
            span_id=0xA000000000000003,
            parent_span_id=0xA000000000000002,
            attributes={
                "gen_ai.request.model": "gemini-2.0-flash",
                "gen_ai.response.model": "gemini-2.0-flash",
                "gen_ai.usage.input_tokens": 25,
                "gen_ai.usage.output_tokens": 200,
                "gen_ai.input.messages": json.dumps(
                    [{"role": "user", "content": "Tell me a short story about a robot learning to paint."}]
                ),
                "gen_ai.output.messages": json.dumps(
                    [{"role": "model", "content": "Once upon a time, in a lab filled with brushes and canvases..."}]
                ),
                "gen_ai.request.temperature": 0.9,
                "gen_ai.request.top_p": 0.95,
            },
            start_time=1_700_000_000_200_000_000,
            end_time=1_700_000_004_800_000_000,
        )
        return [invocation, agent_run, call_llm]

    def test_streaming_produces_three_spans(self):
        from respan_sdk.constants.llm_logging import LOG_TYPE_WORKFLOW, LOG_TYPE_AGENT, LOG_TYPE_GENERATION
        spans = self._make_streaming_spans()
        payloads = self.converter.convert(trace_or_spans=_spans_to_dicts(spans))
        assert len(payloads) == 3
        log_types = {p["log_type"] for p in payloads}
        assert log_types == {LOG_TYPE_WORKFLOW, LOG_TYPE_AGENT, LOG_TYPE_GENERATION}

    def test_streaming_token_counts(self):
        spans = self._make_streaming_spans()
        payloads = self.converter.convert(trace_or_spans=_spans_to_dicts(spans))
        gen = next(p for p in payloads if p["log_type"] == "generation")
        assert gen["prompt_tokens"] == 25
        assert gen["completion_tokens"] == 200
        assert gen["total_request_tokens"] == 225

    def test_streaming_llm_config_preserved(self):
        spans = self._make_streaming_spans()
        payloads = self.converter.convert(trace_or_spans=_spans_to_dicts(spans))
        gen = next(p for p in payloads if p["log_type"] == "generation")
        llm_config = gen["metadata"]["llm_config"]
        assert llm_config["temperature"] == 0.9
        assert llm_config["top_p"] == 0.95

    def test_streaming_latency_reflects_duration(self):
        spans = self._make_streaming_spans()
        payloads = self.converter.convert(trace_or_spans=_spans_to_dicts(spans))
        gen = next(p for p in payloads if p["log_type"] == "generation")
        assert gen["latency"] is not None
        assert gen["latency"] > 4.0


class TestToolCallsPattern:
    def setup_method(self):
        self.converter = AdkSpanConverter()

    def _make_tool_call_spans(self):
        from respan_sdk.constants.llm_logging import LOG_TYPE_WORKFLOW, LOG_TYPE_AGENT, LOG_TYPE_GENERATION, LOG_TYPE_TOOL
        base = 1_700_000_000_000_000_000
        invocation = _make_span(name="invocation", span_id=0xB000000000000001, start_time=base, end_time=base + 4_000_000_000)
        agent_run = _make_span(name="agent_run", span_id=0xB000000000000002, parent_span_id=0xB000000000000001, attributes={"gen_ai.agent.name": "weather_agent"}, start_time=base + 50_000_000, end_time=base + 3_950_000_000)
        llm_1 = _make_span(name="call_llm", span_id=0xB000000000000003, parent_span_id=0xB000000000000002, attributes={"gen_ai.request.model": "gemini-2.0-flash", "gen_ai.response.model": "gemini-2.0-flash", "gen_ai.usage.input_tokens": 30, "gen_ai.usage.output_tokens": 15, "gen_ai.input.messages": json.dumps([{"role": "user", "content": "What's the weather like in Tokyo and London?"}]), "gen_ai.output.messages": json.dumps([{"role": "model", "content": "I'll check the weather for you."}])}, start_time=base + 100_000_000, end_time=base + 1_500_000_000)
        tool_exec = _make_span(name="execute_tool", span_id=0xB000000000000004, parent_span_id=0xB000000000000002, attributes={"gen_ai.tool.name": "get_weather"}, start_time=base + 1_600_000_000, end_time=base + 2_000_000_000)
        llm_2 = _make_span(name="call_llm", span_id=0xB000000000000005, parent_span_id=0xB000000000000002, attributes={"gen_ai.request.model": "gemini-2.0-flash", "gen_ai.response.model": "gemini-2.0-flash", "gen_ai.usage.input_tokens": 60, "gen_ai.usage.output_tokens": 40, "gen_ai.input.messages": json.dumps([{"role": "user", "content": "What's the weather like in Tokyo and London?"}, {"role": "model", "content": "I'll check the weather for you."}, {"role": "tool", "content": json.dumps({"city": "Tokyo", "temp_f": 80})}]), "gen_ai.output.messages": json.dumps([{"role": "model", "content": "Tokyo is 80°F and partly cloudy. London is 59°F and cloudy."}])}, start_time=base + 2_100_000_000, end_time=base + 3_800_000_000)
        return [invocation, agent_run, llm_1, tool_exec, llm_2]

    def test_tool_call_produces_five_payloads(self):
        spans = self._make_tool_call_spans()
        payloads = self.converter.convert(trace_or_spans=_spans_to_dicts(spans))
        assert len(payloads) == 5

    def test_tool_call_log_types(self):
        from respan_sdk.constants.llm_logging import LOG_TYPE_WORKFLOW, LOG_TYPE_AGENT, LOG_TYPE_GENERATION, LOG_TYPE_TOOL
        spans = self._make_tool_call_spans()
        payloads = self.converter.convert(trace_or_spans=_spans_to_dicts(spans))
        type_counts = {}
        for p in payloads:
            lt = p["log_type"]
            type_counts[lt] = type_counts.get(lt, 0) + 1
        assert type_counts[LOG_TYPE_WORKFLOW] == 1
        assert type_counts[LOG_TYPE_AGENT] == 1
        assert type_counts[LOG_TYPE_GENERATION] == 2
        assert type_counts[LOG_TYPE_TOOL] == 1

    def test_tool_span_has_tool_name(self):
        spans = self._make_tool_call_spans()
        payloads = self.converter.convert(trace_or_spans=_spans_to_dicts(spans))
        tool = next(p for p in payloads if p["log_type"] == "tool")
        assert tool["span_tools"] == ["get_weather"]

    def test_generation_spans_have_different_token_counts(self):
        spans = self._make_tool_call_spans()
        payloads = self.converter.convert(trace_or_spans=_spans_to_dicts(spans))
        gens = [p for p in payloads if p["log_type"] == "generation"]
        assert len(gens) == 2
        gens.sort(key=lambda p: p.get("prompt_tokens", 0))
        assert gens[0]["prompt_tokens"] == 30
        assert gens[1]["prompt_tokens"] == 60

    def test_all_spans_share_trace_id(self):
        spans = self._make_tool_call_spans()
        payloads = self.converter.convert(trace_or_spans=_spans_to_dicts(spans))
        trace_ids = {p["trace_id"] for p in payloads}
        assert len(trace_ids) == 1


# ---------------------------------------------------------------------------
# Uninstrument passthrough tests
# ---------------------------------------------------------------------------


class TestUninstrumentPassthrough:
    def test_batch_wrapper_passes_through_when_no_exporter(self):
        import respan_instrumentation_google_adk.instrumentor as inst

        original_exporter = inst._ACTIVE_EXPORTER
        try:
            inst._ACTIVE_EXPORTER = None

            adk_span = _make_span(scope_name="google_adk", name="call_llm")
            non_adk_span = _make_span(scope_name="other_lib", name="other_op", attributes={})
            all_spans = [adk_span, non_adk_span]

            received_args = {}

            def fake_wrapped(*args, **kwargs):
                received_args["args"] = args
                received_args["kwargs"] = kwargs
                return "original_result"

            result = inst._batch_export_wrapper(fake_wrapped, None, (all_spans,), {})
            assert result == "original_result"
            assert received_args["args"] == (all_spans,)
        finally:
            inst._ACTIVE_EXPORTER = original_exporter

    def test_on_end_wrapper_passes_through_when_no_exporter(self):
        import respan_instrumentation_google_adk.instrumentor as inst

        original_exporter = inst._ACTIVE_EXPORTER
        try:
            inst._ACTIVE_EXPORTER = None

            adk_span = _make_span(scope_name="google_adk", name="agent_run")

            received_args = {}

            def fake_wrapped(*args, **kwargs):
                received_args["args"] = args
                received_args["kwargs"] = kwargs
                return "original_result"

            result = inst._on_end_wrapper(fake_wrapped, None, (adk_span,), {})
            assert result == "original_result"
            assert received_args["args"] == (adk_span,)
        finally:
            inst._ACTIVE_EXPORTER = original_exporter
