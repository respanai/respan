"""Tests for ADK span detection."""
from unittest.mock import Mock

from respan_instrumentation_google_adk.adk_detection import (
    ADK_SCOPE_NAMES,
    ADK_SPAN_NAMES,
    is_adk_span,
)


def _make_span(name="test", attributes=None, scope_name=None):
    span = Mock()
    span.name = name
    span.attributes = attributes or {}
    if scope_name is not None:
        scope = Mock()
        scope.name = scope_name
        span.instrumentation_scope = scope
        span.instrumentation_library = scope
    else:
        span.instrumentation_scope = None
        span.instrumentation_library = None
    return span


class TestIsAdkSpan:
    def test_google_adk_scope(self):
        span = _make_span(scope_name="google_adk")
        assert is_adk_span(span) is True

    def test_gcp_vertex_agent_scope(self):
        span = _make_span(scope_name="gcp.vertex.agent")
        assert is_adk_span(span) is True

    def test_google_dash_adk_scope(self):
        span = _make_span(scope_name="google-adk")
        assert is_adk_span(span) is True

    def test_unknown_scope_with_adk_name_and_gen_ai(self):
        span = _make_span(
            name="call_llm",
            attributes={"gen_ai.system": "vertex_ai"},
            scope_name="unknown",
        )
        assert is_adk_span(span) is True

    def test_adk_name_without_gen_ai_returns_false(self):
        span = _make_span(name="call_llm", attributes={"http.method": "POST"}, scope_name="unknown")
        assert is_adk_span(span) is False

    def test_non_adk_span_returns_false(self):
        span = _make_span(name="http_request", attributes={"http.method": "GET"})
        assert is_adk_span(span) is False

    def test_no_scope_no_gen_ai_returns_false(self):
        span = _make_span(name="invocation", attributes={})
        assert is_adk_span(span) is False

    def test_generate_content_with_gen_ai(self):
        span = _make_span(
            name="generate_content",
            attributes={"gen_ai.system": "vertex_ai"},
            scope_name="other",
        )
        assert is_adk_span(span) is True

    def test_all_scope_names_covered(self):
        for scope_name in ADK_SCOPE_NAMES:
            span = _make_span(scope_name=scope_name)
            assert is_adk_span(span) is True

    def test_all_span_names_with_gen_ai(self):
        for span_name in ADK_SPAN_NAMES:
            span = _make_span(
                name=span_name,
                attributes={"gen_ai.request.model": "gemini-2.5-flash"},
                scope_name="unknown",
            )
            assert is_adk_span(span) is True
