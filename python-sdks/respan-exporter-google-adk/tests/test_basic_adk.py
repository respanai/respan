"""Unit and integration tests for Respan Google ADK exporter."""

import asyncio
import json
import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from respan_exporter_google_adk.utils import (
    extract_genai_messages,
    is_adk_span,
    otel_span_to_dict,
)
from respan_exporter_google_adk.exporter import RespanGoogleAdkExporter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
        self.exporter = RespanGoogleAdkExporter(api_key="test-key")

    def test_invocation(self):
        assert self.exporter._map_log_type("invocation", None, None) == "workflow"

    def test_agent_run(self):
        assert self.exporter._map_log_type("agent_run", "parent", None) == "agent"

    def test_call_llm(self):
        assert self.exporter._map_log_type("call_llm", "parent", None) == "generation"

    def test_execute_tool(self):
        assert self.exporter._map_log_type("execute_tool", "parent", None) == "tool"

    def test_fallback_model(self):
        assert self.exporter._map_log_type("unknown_span", None, "gemini-2.0") == "generation"

    def test_fallback_no_parent(self):
        assert self.exporter._map_log_type("unknown_span", None, None) == "workflow"

    def test_fallback_with_parent(self):
        assert self.exporter._map_log_type("unknown_span", "some-parent", None) == "task"


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
# build_payload tests
# ---------------------------------------------------------------------------


class TestBuildPayload:
    def setup_method(self):
        self.exporter = RespanGoogleAdkExporter(api_key="test-key")

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
        payloads = self.exporter.build_payload(trace_or_spans=[span_dict])

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
        payloads = self.exporter.build_payload(trace_or_spans=[span_dict])

        assert len(payloads) == 1
        payload = payloads[0]
        assert payload["log_type"] == "agent"
        assert payload["metadata"]["adk_agent_name"] == "weather_agent"
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
        payloads = self.exporter.build_payload(trace_or_spans=[span_dict])

        assert len(payloads) == 1
        payload = payloads[0]
        assert payload["log_type"] == "tool"
        assert payload["span_tools"] == ["get_weather"]

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
        payloads = self.exporter.build_payload(trace_or_spans=[span_dict])

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
        payloads = self.exporter.build_payload(trace_or_spans=[span_dict])

        payload = payloads[0]
        assert payload["session_identifier"] == "conv-abc-123"
        assert payload["metadata"]["conversation_id"] == "conv-abc-123"

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
        payloads = self.exporter.build_payload(trace_or_spans=span_dicts)

        assert len(payloads) == 3
        log_types = {p["log_type"] for p in payloads}
        assert log_types == {"workflow", "agent", "generation"}

        # Output should propagate from generation to workflow/agent
        gen_payload = next(p for p in payloads if p["log_type"] == "generation")
        assert gen_payload["output"] is not None

        wf_payload = next(p for p in payloads if p["log_type"] == "workflow")
        assert wf_payload["output"] is not None


# ---------------------------------------------------------------------------
# Integration test (requires API keys)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.getenv("RESPAN_API_KEY") or not os.getenv("GOOGLE_API_KEY"),
    reason="RESPAN_API_KEY and GOOGLE_API_KEY not set",
)
def test_adk_tracing_exporter_basic():
    """Run an ADK agent and send traces to Respan.

    Requires:
    - RESPAN_API_KEY
    - GOOGLE_API_KEY
    - pip install google-adk
    """
    pytest.importorskip("google.adk")
    from opentelemetry import trace as trace_api
    from opentelemetry.sdk import trace as trace_sdk
    from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

    from respan_exporter_google_adk import RespanGoogleAdkInstrumentor

    tracer_provider = trace_api.get_tracer_provider()
    if not isinstance(tracer_provider, trace_sdk.TracerProvider):
        tracer_provider = trace_sdk.TracerProvider()
        trace_api.set_tracer_provider(tracer_provider)

    tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

    RespanGoogleAdkInstrumentor().instrument(
        api_key=os.getenv("RESPAN_API_KEY"),
        base_url=os.getenv("RESPAN_BASE_URL"),
        passthrough=False,
    )

    import asyncio
    from google.adk import Agent
    from google.adk.runners import InMemoryRunner
    from google.genai import types

    agent = Agent(
        name="test_agent",
        model="gemini-2.0-flash",
        instruction="You are a helpful test assistant. Answer briefly.",
    )

    async def _run():
        runner = InMemoryRunner(agent=agent, app_name="respan_test")
        session = await runner.session_service.create_session(
            app_name="respan_test", user_id="test_user"
        )
        message = types.Content(
            role="user", parts=[types.Part(text="Say hello in one word.")]
        )
        async for event in runner.run_async(
            user_id=session.user_id,
            session_id=session.id,
            new_message=message,
        ):
            pass

    asyncio.run(_run())
    tracer_provider.force_flush()


# ---------------------------------------------------------------------------
# Uninstrument passthrough tests
# ---------------------------------------------------------------------------


class TestUninstrumentPassthrough:
    """Verify wrappers become transparent pass-throughs when _ACTIVE_EXPORTER is None."""

    def test_batch_wrapper_passes_through_when_no_exporter(self):
        """After uninstrument, _batch_export_wrapper should forward all spans unchanged."""
        import respan_exporter_google_adk.instrumentor as inst

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
        """After uninstrument, _on_end_wrapper should forward ADK spans unchanged."""
        import respan_exporter_google_adk.instrumentor as inst

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
