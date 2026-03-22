"""Tests for ADK span enrichment."""
import json
from unittest.mock import Mock

from opentelemetry.semconv_ai import SpanAttributes, TraceloopSpanKindValues

from respan_sdk.constants.llm_logging import (
    LOG_TYPE_CHAT,
    LOG_TYPE_TOOL,
    LOG_TYPE_WORKFLOW,
    LogMethodChoices,
)
from respan_sdk.respan_types.span_types import RespanSpanAttributes
from respan_instrumentation_google_adk.enrichment import (
    _extract_adk_input,
    _extract_adk_output,
    enrich_adk_batch,
)


def _make_adk_span(
    name: str,
    attributes: dict,
    trace_id: int = 0x1234,
    span_id: int = 0xABCD,
    parent_span_id=None,
    scope_name: str = "google_adk",
):
    """Create a mock ReadableSpan mimicking an ADK span."""
    span = Mock()
    span.name = name
    span.attributes = attributes
    scope = Mock()
    scope.name = scope_name
    span.instrumentation_scope = scope
    ctx = Mock()
    ctx.trace_id = trace_id
    ctx.span_id = span_id
    span.get_span_context.return_value = ctx
    if parent_span_id:
        parent = Mock()
        parent.span_id = parent_span_id
        span.parent = parent
    else:
        span.parent = None
    span.start_time = 1000000000
    span.end_time = 2000000000
    span.status = None
    resource = Mock()
    resource.attributes = {}
    span.resource = resource
    return span


# =============================================================================
# _extract_adk_input
# =============================================================================


class TestExtractAdkInput:
    def test_valid_single_content(self):
        raw = json.dumps({
            "contents": [{"role": "user", "parts": [{"text": "hello"}]}]
        })
        result = _extract_adk_input(raw)
        assert result is not None
        parsed = json.loads(result)
        assert parsed == [{"role": "user", "content": "hello"}]

    def test_multiple_contents(self):
        raw = json.dumps({
            "contents": [
                {"role": "user", "parts": [{"text": "What is 2+2?"}]},
                {"role": "model", "parts": [{"text": "4"}]},
            ]
        })
        result = _extract_adk_input(raw)
        parsed = json.loads(result)
        assert len(parsed) == 2

    def test_empty_contents(self):
        raw = json.dumps({"contents": []})
        assert _extract_adk_input(raw) is None

    def test_invalid_json(self):
        assert _extract_adk_input("not json") is None

    def test_empty_string(self):
        assert _extract_adk_input("") is None


class TestExtractAdkOutput:
    def test_valid_output(self):
        raw = json.dumps({
            "content": {
                "role": "model",
                "parts": [{"text": "Hello!"}],
            }
        })
        result = _extract_adk_output(raw)
        assert result is not None
        parsed = json.loads(result)
        assert parsed["role"] == "assistant"
        assert parsed["content"] == "Hello!"

    def test_invalid_json(self):
        assert _extract_adk_output("not json") is None


# =============================================================================
# enrich_adk_batch
# =============================================================================


class TestEnrichAdkBatch:
    def test_non_adk_spans_pass_through(self):
        """Non-ADK spans should be returned unchanged."""
        span = Mock()
        span.name = "http_request"
        span.attributes = {"http.method": "GET"}
        span.instrumentation_scope = None
        span.instrumentation_library = None

        result = enrich_adk_batch([span])
        assert len(result) == 1
        assert result[0] is span

    def test_invocation_span_gets_workflow_kind(self):
        """Root invocation span should get traceloop.span.kind=workflow."""
        span = _make_adk_span("invocation", {
            SpanAttributes.LLM_SYSTEM: "google_genai",
        })

        result = enrich_adk_batch([span])
        assert len(result) == 1
        enriched = result[0]
        assert enriched.attributes.get(SpanAttributes.TRACELOOP_SPAN_KIND) == TraceloopSpanKindValues.WORKFLOW.value
        assert enriched.attributes.get(RespanSpanAttributes.LOG_TYPE.value) == LOG_TYPE_WORKFLOW
        assert enriched.attributes.get(RespanSpanAttributes.LOG_METHOD.value) == LogMethodChoices.TRACING_INTEGRATION.value

    def test_call_llm_gets_entity_path_and_request_type(self):
        """LLM span should get entity path and llm.request.type."""
        span = _make_adk_span("call_llm", {
            SpanAttributes.LLM_SYSTEM: "google_genai",
            SpanAttributes.LLM_REQUEST_MODEL: "gemini-2.5-flash",
        })

        result = enrich_adk_batch([span])
        assert len(result) == 1
        enriched = result[0]
        assert enriched.attributes.get(SpanAttributes.TRACELOOP_ENTITY_PATH) == "adk.call_llm"
        assert enriched.attributes.get(SpanAttributes.LLM_REQUEST_TYPE) == "chat"
        assert enriched.attributes.get(RespanSpanAttributes.LOG_TYPE.value) == LOG_TYPE_CHAT
        assert enriched.attributes.get(RespanSpanAttributes.LOG_METHOD.value) == LogMethodChoices.TRACING_INTEGRATION.value

    def test_token_mapping(self):
        """input_tokens/output_tokens should be mapped to prompt/completion tokens."""
        span = _make_adk_span("call_llm", {
            SpanAttributes.LLM_SYSTEM: "google_genai",
            "gen_ai.usage.input_tokens": 10,
            "gen_ai.usage.output_tokens": 25,
        })

        result = enrich_adk_batch([span])
        enriched = result[0]
        assert enriched.attributes.get(SpanAttributes.LLM_USAGE_PROMPT_TOKENS) == 10
        assert enriched.attributes.get(SpanAttributes.LLM_USAGE_COMPLETION_TOKENS) == 25

    def test_stripped_attrs_not_in_output(self):
        """Internal ADK attributes should be stripped from enriched spans."""
        attrs = {
            SpanAttributes.LLM_SYSTEM: "google_genai",
            "gcp.vertex.agent.event_id": "evt-123",
            "gcp.vertex.agent.invocation_id": "inv-456",
        }
        span = _make_adk_span("call_llm", attrs)

        result = enrich_adk_batch([span])
        enriched = result[0]
        assert "gcp.vertex.agent.event_id" not in enriched.attributes
        assert "gcp.vertex.agent.invocation_id" not in enriched.attributes

    def test_cross_span_agent_name_propagation(self):
        """Agent name from invoke_agent should be propagated to invocation span."""
        invocation = _make_adk_span("invocation", {
            SpanAttributes.LLM_SYSTEM: "google_genai",
        }, trace_id=0xAAAA, span_id=0x1111)

        invoke = _make_adk_span("invoke_agent", {
            "gen_ai.agent.name": "MyAgent",
            SpanAttributes.LLM_SYSTEM: "google_genai",
        }, trace_id=0xAAAA, span_id=0x2222, parent_span_id=0x1111)

        result = enrich_adk_batch([invocation, invoke])
        # Find the invocation span (the one with workflow kind)
        root = [s for s in result if s.attributes.get(SpanAttributes.TRACELOOP_SPAN_KIND) == TraceloopSpanKindValues.WORKFLOW.value]
        assert len(root) == 1
        assert root[0].name == "MyAgent"

    def test_agent_name_from_agent_run_span(self):
        """Agent name on agent_run span should rename invocation (single-agent case)."""
        invocation = _make_adk_span("invocation", {
            SpanAttributes.LLM_SYSTEM: "google_genai",
        }, trace_id=0xBBBB, span_id=0x1111)

        agent_run = _make_adk_span("agent_run", {
            "gen_ai.agent.name": "greeting_agent",
            SpanAttributes.LLM_SYSTEM: "google_genai",
        }, trace_id=0xBBBB, span_id=0x2222, parent_span_id=0x1111)

        result = enrich_adk_batch([invocation, agent_run])
        root = [s for s in result if s.attributes.get(SpanAttributes.TRACELOOP_SPAN_KIND) == TraceloopSpanKindValues.WORKFLOW.value]
        assert len(root) == 1
        assert root[0].name == "greeting_agent"

    def test_mixed_adk_and_non_adk_batch(self):
        """Batch with both ADK and non-ADK spans should handle both correctly."""
        non_adk = Mock()
        non_adk.name = "http_request"
        non_adk.attributes = {"http.method": "GET"}
        non_adk.instrumentation_scope = None
        non_adk.instrumentation_library = None

        adk = _make_adk_span("call_llm", {SpanAttributes.LLM_SYSTEM: "google_genai"})

        result = enrich_adk_batch([non_adk, adk])
        assert len(result) == 2
        # Non-ADK span should be unchanged
        assert result[0] is non_adk
        # ADK span should be enriched
        assert result[1].attributes.get(SpanAttributes.TRACELOOP_ENTITY_PATH) == "adk.call_llm"

    def test_execute_tool_maps_io(self):
        """Tool execution span should map tool args/response to entity I/O."""
        span = _make_adk_span("execute_tool", {
            SpanAttributes.LLM_SYSTEM: "google_genai",
            "gcp.vertex.agent.tool_call_args": '{"query": "test"}',
            "gcp.vertex.agent.tool_response": '{"result": "ok"}',
        })

        result = enrich_adk_batch([span])
        enriched = result[0]
        assert enriched.attributes.get(SpanAttributes.TRACELOOP_ENTITY_INPUT) == '{"query": "test"}'
        assert enriched.attributes.get(SpanAttributes.TRACELOOP_ENTITY_OUTPUT) == '{"result": "ok"}'
        assert enriched.attributes.get(RespanSpanAttributes.LOG_TYPE.value) == LOG_TYPE_TOOL
        assert enriched.attributes.get(RespanSpanAttributes.LOG_METHOD.value) == LogMethodChoices.TRACING_INTEGRATION.value
