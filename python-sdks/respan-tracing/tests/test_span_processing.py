"""
Unit tests for span processing functions: is_processable_span and is_root_span_candidate.

Tests cover:
- User-decorated spans (TRACELOOP_SPAN_KIND)
- Child spans within entity context (TRACELOOP_ENTITY_PATH)
- Standalone auto-instrumented LLM spans (LLM_REQUEST_TYPE)
- Auto-instrumentation noise (HTTP, DB, etc.) — should be filtered
- Root span promotion logic
"""

import pytest
from unittest.mock import Mock
from opentelemetry.semconv_ai import SpanAttributes

from respan_tracing.utils.preprocessing.span_processing import (
    is_processable_span,
    is_root_span_candidate,
)


def _make_span(attributes: dict, name: str = "test_span") -> Mock:
    """Create a mock ReadableSpan with given attributes."""
    span = Mock()
    span.name = name
    span.attributes = attributes
    return span


# =============================================================================
# is_processable_span
# =============================================================================


class TestIsProcessableSpan:
    """Tests for is_processable_span()."""

    def test_user_decorated_span_with_traceloop_span_kind(self):
        """Span with TRACELOOP_SPAN_KIND should be processed."""
        span = _make_span({SpanAttributes.TRACELOOP_SPAN_KIND: "workflow"})
        assert is_processable_span(span) is True

    def test_child_span_with_entity_path(self):
        """Span with TRACELOOP_ENTITY_PATH should be processed."""
        span = _make_span({SpanAttributes.TRACELOOP_ENTITY_PATH: "my_workflow.my_task"})
        assert is_processable_span(span) is True

    def test_standalone_llm_span_chat(self):
        """Standalone auto-instrumented chat span should be processed."""
        span = _make_span({SpanAttributes.LLM_REQUEST_TYPE: "chat"})
        assert is_processable_span(span) is True

    def test_standalone_llm_span_completion(self):
        """Standalone auto-instrumented completion span should be processed."""
        span = _make_span({SpanAttributes.LLM_REQUEST_TYPE: "completion"})
        assert is_processable_span(span) is True

    def test_standalone_llm_span_embedding(self):
        """Standalone auto-instrumented embedding span should be processed."""
        span = _make_span({SpanAttributes.LLM_REQUEST_TYPE: "embedding"})
        assert is_processable_span(span) is True

    def test_pydantic_ai_native_span_operation_name(self):
        """Pydantic AI native span with gen_ai.operation.name should be processed."""
        span = _make_span({"gen_ai.operation.name": "chat"})
        assert is_processable_span(span) is True

    def test_pydantic_ai_native_span_system(self):
        """Pydantic AI native span with gen_ai.system should be processed."""
        span = _make_span({"gen_ai.system": "openai"})
        assert is_processable_span(span) is True

    def test_auto_instrumentation_noise_filtered(self):
        """Span without any recognized attributes should be filtered out."""
        span = _make_span({"http.method": "GET", "http.url": "https://api.openai.com"})
        assert is_processable_span(span) is False

    def test_empty_attributes_filtered(self):
        """Span with empty attributes should be filtered out."""
        span = _make_span({})
        assert is_processable_span(span) is False

    def test_empty_entity_path_filtered(self):
        """Span with empty string entity path should be filtered out (not treated as present)."""
        span = _make_span({SpanAttributes.TRACELOOP_ENTITY_PATH: ""})
        assert is_processable_span(span) is False

    def test_llm_span_within_decorator_context(self):
        """LLM span inside @workflow/@task should be processed via entity_path."""
        span = _make_span({
            SpanAttributes.TRACELOOP_ENTITY_PATH: "my_workflow.chat_step",
            SpanAttributes.LLM_REQUEST_TYPE: "chat",
        })
        assert is_processable_span(span) is True

    def test_priority_traceloop_span_kind_first(self):
        """TRACELOOP_SPAN_KIND takes priority — span is processed even if other attrs present."""
        span = _make_span({
            SpanAttributes.TRACELOOP_SPAN_KIND: "task",
            SpanAttributes.TRACELOOP_ENTITY_PATH: "root.task",
            SpanAttributes.LLM_REQUEST_TYPE: "chat",
        })
        assert is_processable_span(span) is True


# =============================================================================
# is_root_span_candidate
# =============================================================================


class TestIsRootSpanCandidate:
    """Tests for is_root_span_candidate()."""

    def test_user_decorated_span_without_entity_path_is_root(self):
        """User-decorated span without entity_path should become root."""
        span = _make_span({SpanAttributes.TRACELOOP_SPAN_KIND: "workflow"})
        assert is_root_span_candidate(span) is True

    def test_user_decorated_span_with_entity_path_not_root(self):
        """User-decorated span WITH entity_path should NOT become root (it's a child)."""
        span = _make_span({
            SpanAttributes.TRACELOOP_SPAN_KIND: "task",
            SpanAttributes.TRACELOOP_ENTITY_PATH: "parent_workflow.my_task",
        })
        assert is_root_span_candidate(span) is False

    def test_standalone_llm_span_without_entity_path_is_root(self):
        """Standalone LLM span without entity_path should become root."""
        span = _make_span({SpanAttributes.LLM_REQUEST_TYPE: "chat"})
        assert is_root_span_candidate(span) is True

    def test_pydantic_ai_native_span_without_entity_path_is_root(self):
        """Pydantic AI native span without entity_path should become root."""
        span = _make_span({"gen_ai.operation.name": "chat"})
        assert is_root_span_candidate(span) is True

    def test_pydantic_ai_native_span_with_entity_path_not_root(self):
        """Pydantic AI native span inside decorator context should NOT become root."""
        span = _make_span({
            "gen_ai.operation.name": "chat",
            SpanAttributes.TRACELOOP_ENTITY_PATH: "my_workflow.chat_step",
        })
        assert is_root_span_candidate(span) is False

    def test_standalone_llm_span_with_entity_path_not_root(self):
        """LLM span inside decorator context should NOT become root."""
        span = _make_span({
            SpanAttributes.LLM_REQUEST_TYPE: "chat",
            SpanAttributes.TRACELOOP_ENTITY_PATH: "my_workflow.chat_step",
        })
        assert is_root_span_candidate(span) is False

    def test_noise_span_not_root(self):
        """Auto-instrumentation noise should NOT become root."""
        span = _make_span({"http.method": "POST"})
        assert is_root_span_candidate(span) is False

    def test_empty_attributes_not_root(self):
        """Span with empty attributes should NOT become root."""
        span = _make_span({})
        assert is_root_span_candidate(span) is False

    def test_llm_span_with_span_kind_and_no_path_is_root_via_span_kind(self):
        """When both TRACELOOP_SPAN_KIND and LLM_REQUEST_TYPE present, root via span_kind path."""
        span = _make_span({
            SpanAttributes.TRACELOOP_SPAN_KIND: "task",
            SpanAttributes.LLM_REQUEST_TYPE: "chat",
        })
        # Root because span_kind is present and no entity_path
        assert is_root_span_candidate(span) is True

    def test_empty_entity_path_treated_as_absent(self):
        """Empty string entity_path should be treated as absent (span is root candidate)."""
        span = _make_span({
            SpanAttributes.LLM_REQUEST_TYPE: "chat",
            SpanAttributes.TRACELOOP_ENTITY_PATH: "",
        })
        assert is_root_span_candidate(span) is True
