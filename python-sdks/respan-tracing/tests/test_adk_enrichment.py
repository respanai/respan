"""
Unit tests for ADK span enrichment functions.

Tests cover:
- _extract_adk_input: parsing ADK llm_request JSON into formatted input messages
- _extract_adk_output: parsing ADK llm_response JSON into formatted output message
- EnrichedSpan: proxy wrapper for name overrides, attribute injection, and stripping
- _enrich_adk_spans: top-level enrichment that separates ADK/non-ADK spans
- _enrich_adk_trace_group: per-trace enrichment (root renaming, I/O propagation,
  token mapping, entity paths, attribute stripping, thread/customer identifiers)
"""

import json
import pytest
from unittest.mock import Mock

from respan_tracing.exporters.respan import EnrichedSpan
from respan_tracing.exporters.adk_enrichment import (
    _ADK_STRIP_ATTRS,
    _extract_adk_input,
    _extract_adk_output,
    _enrich_adk_spans,
    _enrich_adk_trace_group,
)


def _make_adk_span(
    name: str,
    attributes: dict,
    trace_id: int = 0x1234,
    span_id: int = 0xABCD,
    parent_span_id=None,
    scope_name: str = "google_adk",
    resource_attrs=None,
) -> Mock:
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
    resource = Mock()
    resource.attributes = resource_attrs or {}
    span.resource = resource
    return span


# =============================================================================
# _extract_adk_input
# =============================================================================


class TestExtractAdkInput:
    """Tests for _extract_adk_input()."""

    def test_valid_single_content(self):
        """Single content entry with one text part is extracted correctly."""
        raw = json.dumps({
            "contents": [{"role": "user", "parts": [{"text": "hello"}]}]
        })
        result = _extract_adk_input(raw)
        assert result is not None
        parsed = json.loads(result)
        assert parsed == [{"role": "user", "content": "hello"}]

    def test_multiple_contents(self):
        """Multiple content entries with different roles are all extracted."""
        raw = json.dumps({
            "contents": [
                {"role": "user", "parts": [{"text": "What is 2+2?"}]},
                {"role": "model", "parts": [{"text": "4"}]},
            ]
        })
        result = _extract_adk_input(raw)
        assert result is not None
        parsed = json.loads(result)
        assert len(parsed) == 2
        assert parsed[0] == {"role": "user", "content": "What is 2+2?"}
        assert parsed[1] == {"role": "model", "content": "4"}

    def test_multipart_text(self):
        """Content with multiple text parts joins them with newline."""
        raw = json.dumps({
            "contents": [
                {"role": "user", "parts": [{"text": "line1"}, {"text": "line2"}]}
            ]
        })
        result = _extract_adk_input(raw)
        assert result is not None
        parsed = json.loads(result)
        assert parsed == [{"role": "user", "content": "line1\nline2"}]

    def test_empty_contents(self):
        """Empty contents list returns None."""
        raw = json.dumps({"contents": []})
        assert _extract_adk_input(raw) is None

    def test_missing_contents_key(self):
        """Missing 'contents' key returns None (empty list from .get default)."""
        raw = json.dumps({})
        assert _extract_adk_input(raw) is None

    def test_invalid_json(self):
        """Non-JSON string returns None."""
        assert _extract_adk_input("not json") is None

    def test_empty_string(self):
        """Empty string returns None."""
        assert _extract_adk_input("") is None

    def test_non_text_parts_skipped(self):
        """Parts without a 'text' key are excluded from the output."""
        raw = json.dumps({
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": "hello"},
                        {"inlineData": {"mimeType": "image/png", "data": "..."}},
                        {"text": "world"},
                    ],
                }
            ]
        })
        result = _extract_adk_input(raw)
        assert result is not None
        parsed = json.loads(result)
        assert parsed == [{"role": "user", "content": "hello\nworld"}]


# =============================================================================
# _extract_adk_output
# =============================================================================


class TestExtractAdkOutput:
    """Tests for _extract_adk_output()."""

    def test_model_role_mapped_to_assistant(self):
        """The 'model' role is mapped to 'assistant' in output."""
        raw = json.dumps({
            "content": {"role": "model", "parts": [{"text": "hi"}]}
        })
        result = _extract_adk_output(raw)
        assert result is not None
        parsed = json.loads(result)
        assert parsed == {"role": "assistant", "content": "hi"}

    def test_user_role_preserved(self):
        """Non-model roles stay as-is (no mapping)."""
        raw = json.dumps({
            "content": {"role": "user", "parts": [{"text": "echo"}]}
        })
        result = _extract_adk_output(raw)
        assert result is not None
        parsed = json.loads(result)
        assert parsed == {"role": "user", "content": "echo"}

    def test_multipart_text(self):
        """Multiple text parts are joined with newline."""
        raw = json.dumps({
            "content": {
                "role": "model",
                "parts": [{"text": "part1"}, {"text": "part2"}],
            }
        })
        result = _extract_adk_output(raw)
        assert result is not None
        parsed = json.loads(result)
        assert parsed == {"role": "assistant", "content": "part1\npart2"}

    def test_invalid_json(self):
        """Non-JSON string returns None."""
        assert _extract_adk_output("not json") is None

    def test_empty_string(self):
        """Empty string returns None."""
        assert _extract_adk_output("") is None


# =============================================================================
# EnrichedSpan
# =============================================================================


class TestEnrichedSpan:
    """Tests for EnrichedSpan proxy wrapper."""

    def test_name_override(self):
        """When name is provided, .name returns the override."""
        original = _make_adk_span("original_name", {"key": "val"})
        enriched = EnrichedSpan(original, name="overridden")
        assert enriched.name == "overridden"

    def test_name_passthrough(self):
        """When name is None, .name returns the original span name."""
        original = _make_adk_span("original_name", {"key": "val"})
        enriched = EnrichedSpan(original, name=None)
        assert enriched.name == "original_name"

    def test_extra_attributes_merged(self):
        """Extra attributes appear in .attributes."""
        original = _make_adk_span("span", {"orig": "value"})
        enriched = EnrichedSpan(original, extra_attributes={"extra": "new"})
        assert enriched.attributes["extra"] == "new"

    def test_stripped_keys_removed(self):
        """Stripped keys are absent from .attributes."""
        original = _make_adk_span("span", {
            "keep_me": "yes",
            "gcp.vertex.agent.event_id": "evt123",
            "otel.scope.name": "google_adk",
        })
        enriched = EnrichedSpan(
            original,
            stripped_keys={"gcp.vertex.agent.event_id", "otel.scope.name"},
        )
        assert "gcp.vertex.agent.event_id" not in enriched.attributes
        assert "otel.scope.name" not in enriched.attributes

    def test_original_attributes_preserved(self):
        """Non-stripped original attributes remain present."""
        original = _make_adk_span("span", {"keep_me": "yes", "also_keep": 42})
        enriched = EnrichedSpan(original, stripped_keys=set())
        assert enriched.attributes["keep_me"] == "yes"
        assert enriched.attributes["also_keep"] == 42

    def test_extra_overrides_original(self):
        """Extra attributes override same-key originals."""
        original = _make_adk_span("span", {"shared_key": "old"})
        enriched = EnrichedSpan(original, extra_attributes={"shared_key": "new"})
        assert enriched.attributes["shared_key"] == "new"

    def test_getattr_delegation(self):
        """Other properties are forwarded to the original span."""
        original = _make_adk_span("span", {}, trace_id=0xDEAD, span_id=0xBEEF)
        enriched = EnrichedSpan(original)
        ctx = enriched.get_span_context()
        assert ctx.trace_id == 0xDEAD
        assert ctx.span_id == 0xBEEF


# =============================================================================
# _enrich_adk_spans
# =============================================================================


class TestEnrichAdkSpans:
    """Tests for _enrich_adk_spans()."""

    def test_no_adk_spans_returns_unchanged(self):
        """All non-ADK spans are returned as-is."""
        non_adk = _make_adk_span(
            "http_request",
            {"http.method": "GET"},
            scope_name="opentelemetry.instrumentation.requests",
        )
        result = _enrich_adk_spans([non_adk])
        assert len(result) == 1
        assert result[0] is non_adk

    def test_mixed_adk_and_non_adk(self):
        """Non-ADK spans preserved unchanged; ADK spans enriched."""
        non_adk = _make_adk_span(
            "http_request",
            {"http.method": "GET"},
            scope_name="opentelemetry.instrumentation.requests",
        )
        adk_span = _make_adk_span(
            "invocation",
            {"gen_ai.agent.name": "TestAgent"},
            scope_name="google_adk",
        )
        result = _enrich_adk_spans([non_adk, adk_span])
        assert len(result) == 2
        assert non_adk in result
        enriched_items = [s for s in result if isinstance(s, EnrichedSpan)]
        assert len(enriched_items) == 1

    def test_empty_input(self):
        """Empty sequence returns empty list."""
        result = _enrich_adk_spans([])
        assert result == []

    def test_adk_spans_grouped_by_trace_id(self):
        """Spans from different traces are enriched independently."""
        span_trace_a = _make_adk_span(
            "invocation", {}, trace_id=0xAAAA, scope_name="google_adk"
        )
        span_trace_b = _make_adk_span(
            "invocation", {}, trace_id=0xBBBB, scope_name="google_adk"
        )
        result = _enrich_adk_spans([span_trace_a, span_trace_b])
        assert len(result) == 2
        for span in result:
            assert isinstance(span, EnrichedSpan)


# =============================================================================
# _enrich_adk_trace_group
# =============================================================================


def _make_standard_trace_group(
    agent_name="WeatherAgent",
    llm_input_messages='[{"role": "user", "content": "forecast?"}]',
    llm_output_messages='{"role": "assistant", "content": "Sunny."}',
    input_tokens=10,
    output_tokens=20,
    session_id="sess-001",
    resource_attrs=None,
    tool_call_args=None,
    tool_response=None,
):
    """Build a standard ADK trace group: invocation + invoke_agent + call_llm + optional execute_tool."""
    trace_id = 0xF00D
    res_attrs = resource_attrs or {}

    invocation = _make_adk_span(
        "invocation", {}, trace_id=trace_id, span_id=0x0001, resource_attrs=res_attrs,
    )
    invoke_agent = _make_adk_span(
        "invoke_agent",
        {"gen_ai.agent.name": agent_name},
        trace_id=trace_id, span_id=0x0002, parent_span_id=0x0001, resource_attrs=res_attrs,
    )
    call_llm = _make_adk_span(
        "call_llm",
        {
            "gen_ai.input.messages": llm_input_messages,
            "gen_ai.output.messages": llm_output_messages,
            "gen_ai.usage.input_tokens": input_tokens,
            "gen_ai.usage.output_tokens": output_tokens,
            "gen_ai.conversation.id": session_id,
        },
        trace_id=trace_id, span_id=0x0003, parent_span_id=0x0002, resource_attrs=res_attrs,
    )

    spans = [invocation, invoke_agent, call_llm]

    if tool_call_args is not None:
        execute_tool = _make_adk_span(
            "execute_tool",
            {
                "gcp.vertex.agent.tool_call_args": tool_call_args,
                "gcp.vertex.agent.tool_response": tool_response or '{"result": "ok"}',
            },
            trace_id=trace_id, span_id=0x0004, parent_span_id=0x0002, resource_attrs=res_attrs,
        )
        spans.append(execute_tool)

    return spans


class TestEnrichAdkTraceGroup:
    """Tests for _enrich_adk_trace_group() — the core enrichment logic."""

    def test_root_invocation_renamed_to_agent_name(self):
        """The 'invocation' span is renamed to the agent name from 'invoke_agent'."""
        spans = _make_standard_trace_group(agent_name="MyAgent")
        enriched = _enrich_adk_trace_group(spans)
        assert enriched[0].name == "MyAgent"

    def test_root_invocation_gets_workflow_kind(self):
        """The 'invocation' span gets traceloop.span.kind = 'workflow'."""
        spans = _make_standard_trace_group()
        enriched = _enrich_adk_trace_group(spans)
        assert enriched[0].attributes["traceloop.span.kind"] == "workflow"

    def test_input_output_propagated_to_root(self):
        """First LLM input and last LLM output are propagated to the invocation span."""
        spans = _make_standard_trace_group(
            llm_input_messages='[{"role": "user", "content": "hi"}]',
            llm_output_messages='{"role": "assistant", "content": "hello"}',
        )
        enriched = _enrich_adk_trace_group(spans)
        assert enriched[0].attributes["traceloop.entity.input"] == '[{"role": "user", "content": "hi"}]'
        assert enriched[0].attributes["traceloop.entity.output"] == '{"role": "assistant", "content": "hello"}'

    def test_child_spans_get_entity_path(self):
        """Non-invocation spans get traceloop.entity.path = 'adk.<prefix>'."""
        spans = _make_standard_trace_group()
        enriched = _enrich_adk_trace_group(spans)
        assert enriched[1].attributes["traceloop.entity.path"] == "adk.invoke_agent"
        assert enriched[2].attributes["traceloop.entity.path"] == "adk.call_llm"

    def test_llm_span_token_mapping(self):
        """gen_ai.usage.input_tokens mapped to prompt_tokens, output_tokens to completion_tokens."""
        spans = _make_standard_trace_group(input_tokens=100, output_tokens=50)
        enriched = _enrich_adk_trace_group(spans)
        llm_span = enriched[2]
        assert llm_span.attributes["gen_ai.usage.prompt_tokens"] == 100
        assert llm_span.attributes["gen_ai.usage.completion_tokens"] == 50

    def test_llm_span_gets_request_type(self):
        """call_llm/generate_content spans get llm.request.type = 'chat'."""
        spans = _make_standard_trace_group()
        enriched = _enrich_adk_trace_group(spans)
        assert enriched[2].attributes["llm.request.type"] == "chat"

    def test_tool_span_input_output(self):
        """execute_tool span gets tool_call_args as input and tool_response as output."""
        spans = _make_standard_trace_group(
            tool_call_args='{"query": "weather SF"}',
            tool_response='{"temp": "72F"}',
        )
        enriched = _enrich_adk_trace_group(spans)
        tool_span = enriched[3]
        assert tool_span.attributes["traceloop.entity.input"] == '{"query": "weather SF"}'
        assert tool_span.attributes["traceloop.entity.output"] == '{"temp": "72F"}'

    def test_attribute_stripping(self):
        """All attrs in _ADK_STRIP_ATTRS are removed from enriched spans."""
        trace_id = 0xABCD
        call_llm = _make_adk_span(
            "call_llm",
            {
                "gen_ai.input.messages": "[]",
                "gcp.vertex.agent.event_id": "evt-1",
                "gcp.vertex.agent.llm_request": '{"contents": []}',
                "otel.scope.name": "google_adk",
                "service.name": "my-service",
                "gen_ai.operation.name": "generate",
            },
            trace_id=trace_id, span_id=0x0010,
        )
        invocation = _make_adk_span(
            "invocation", {}, trace_id=trace_id, span_id=0x0001
        )
        enriched = _enrich_adk_trace_group([invocation, call_llm])
        llm_attrs = enriched[1].attributes
        assert "gcp.vertex.agent.event_id" not in llm_attrs
        assert "gcp.vertex.agent.llm_request" not in llm_attrs
        assert "otel.scope.name" not in llm_attrs
        assert "service.name" not in llm_attrs
        assert "gen_ai.operation.name" not in llm_attrs

    def test_session_id_as_thread_identifier(self):
        """When no explicit thread_id in resource, session_id used as thread_identifier."""
        spans = _make_standard_trace_group(session_id="session-xyz", resource_attrs={})
        enriched = _enrich_adk_trace_group(spans)
        assert enriched[0].attributes["respan.threads.thread_identifier"] == "session-xyz"

    def test_customer_identifier_propagated(self):
        """Resource-level customer_identifier is propagated to the root span."""
        spans = _make_standard_trace_group(
            resource_attrs={"respan.customer_params.customer_identifier": "cust-42"},
        )
        enriched = _enrich_adk_trace_group(spans)
        assert enriched[0].attributes["respan.customer_params.customer_identifier"] == "cust-42"

    def test_llm_span_input_output_from_adk_fallback(self):
        """When gen_ai.input.messages is missing, falls back to parsing gcp.vertex.agent.llm_request."""
        trace_id = 0xBEEF
        invocation = _make_adk_span("invocation", {}, trace_id=trace_id, span_id=0x0001)
        invoke_agent = _make_adk_span(
            "invoke_agent", {"gen_ai.agent.name": "FallbackAgent"},
            trace_id=trace_id, span_id=0x0002, parent_span_id=0x0001,
        )
        llm_request = json.dumps({
            "contents": [{"role": "user", "parts": [{"text": "fallback input"}]}]
        })
        llm_response = json.dumps({
            "content": {"role": "model", "parts": [{"text": "fallback output"}]}
        })
        call_llm = _make_adk_span(
            "call_llm",
            {
                "gcp.vertex.agent.llm_request": llm_request,
                "gcp.vertex.agent.llm_response": llm_response,
            },
            trace_id=trace_id, span_id=0x0003, parent_span_id=0x0002,
        )
        enriched = _enrich_adk_trace_group([invocation, invoke_agent, call_llm])

        llm_enriched = enriched[2]
        parsed_input = json.loads(llm_enriched.attributes["traceloop.entity.input"])
        assert parsed_input == [{"role": "user", "content": "fallback input"}]

        parsed_output = json.loads(llm_enriched.attributes["traceloop.entity.output"])
        assert parsed_output == {"role": "assistant", "content": "fallback output"}

        root_input = json.loads(enriched[0].attributes["traceloop.entity.input"])
        assert root_input == [{"role": "user", "content": "fallback input"}]

    def test_resource_thread_id_takes_precedence(self):
        """When resource has explicit thread_identifier, it takes precedence over session_id."""
        spans = _make_standard_trace_group(
            session_id="session-from-adk",
            resource_attrs={"respan.threads.thread_identifier": "explicit-thread-id"},
        )
        enriched = _enrich_adk_trace_group(spans)
        assert enriched[0].attributes["respan.threads.thread_identifier"] == "explicit-thread-id"

    def test_generate_content_span_gets_llm_enrichment(self):
        """generate_content prefix (alternative to call_llm) also gets LLM enrichment."""
        trace_id = 0xCAFE
        invocation = _make_adk_span("invocation", {}, trace_id=trace_id, span_id=0x0001)
        gen_content = _make_adk_span(
            "generate_content",
            {
                "gen_ai.input.messages": '[{"role": "user", "content": "test"}]',
                "gen_ai.output.messages": '{"role": "assistant", "content": "reply"}',
                "gen_ai.usage.input_tokens": 5,
                "gen_ai.usage.output_tokens": 8,
            },
            trace_id=trace_id, span_id=0x0002, parent_span_id=0x0001,
        )
        enriched = _enrich_adk_trace_group([invocation, gen_content])
        gen_span = enriched[1]
        assert gen_span.attributes["llm.request.type"] == "chat"
        assert gen_span.attributes["gen_ai.usage.prompt_tokens"] == 5
        assert gen_span.attributes["gen_ai.usage.completion_tokens"] == 8
        assert gen_span.attributes["traceloop.entity.path"] == "adk.generate_content"

    def test_invocation_without_agent_name_keeps_original_name(self):
        """If no invoke_agent span provides an agent name, invocation keeps its original name."""
        trace_id = 0xDEAD
        invocation = _make_adk_span("invocation", {}, trace_id=trace_id, span_id=0x0001)
        invoke_agent = _make_adk_span(
            "invoke_agent", {}, trace_id=trace_id, span_id=0x0002, parent_span_id=0x0001
        )
        enriched = _enrich_adk_trace_group([invocation, invoke_agent])
        assert enriched[0].name == "invocation"
