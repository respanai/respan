"""Tests for wrapt-based processor patching."""
from unittest.mock import Mock, patch

from opentelemetry.semconv_ai import SpanAttributes

from respan_instrumentation_google_adk.processor import (
    _respan_processor_on_end_wrapper,
    _exporter_export_wrapper,
    patch_span_processors,
)


def _make_adk_span(name="call_llm"):
    span = Mock()
    span.name = name
    span.attributes = {SpanAttributes.LLM_SYSTEM: "google_genai"}
    scope = Mock()
    scope.name = "google_adk"
    span.instrumentation_scope = scope
    span.instrumentation_library = scope
    ctx = Mock()
    ctx.trace_id = 0x1234
    ctx.span_id = 0xABCD
    span.get_span_context.return_value = ctx
    span.parent = None
    span.start_time = 1000000000
    span.end_time = 2000000000
    span.status = None
    resource = Mock()
    resource.attributes = {}
    span.resource = resource
    return span


def _make_non_adk_span():
    span = Mock()
    span.name = "http_request"
    span.attributes = {"http.method": "GET"}
    span.instrumentation_scope = None
    span.instrumentation_library = None
    return span


class TestRespanProcessorOnEndWrapper:
    def test_adk_span_bypasses_filter(self):
        """ADK spans should be passed directly to the inner processor."""
        wrapped = Mock()
        instance = Mock()
        instance.processor = Mock()

        span = _make_adk_span()
        _respan_processor_on_end_wrapper(wrapped, instance, (span,), {})

        instance.processor.on_end.assert_called_once_with(span)
        wrapped.assert_not_called()

    def test_non_adk_span_passes_through(self):
        """Non-ADK spans should call the original wrapped method."""
        wrapped = Mock()
        instance = Mock()

        span = _make_non_adk_span()
        _respan_processor_on_end_wrapper(wrapped, instance, (span,), {})

        wrapped.assert_called_once_with(span)


class TestExporterExportWrapper:
    def test_non_adk_batch_passes_through(self):
        """Batch with only non-ADK spans should pass through unchanged."""
        wrapped = Mock()
        instance = Mock()

        spans = [_make_non_adk_span(), _make_non_adk_span()]
        _exporter_export_wrapper(wrapped, instance, (spans,), {})

        wrapped.assert_called_once()
        call_args = wrapped.call_args[0][0]
        # Non-ADK spans should pass through unchanged
        assert len(call_args) == 2
        for s in call_args:
            assert s.name == "http_request"

    def test_empty_batch_passes_through(self):
        wrapped = Mock()
        _exporter_export_wrapper(wrapped, Mock(), ([],), {})
        wrapped.assert_called_once()

    def test_adk_batch_gets_enriched(self):
        """Batch with ADK spans should have enriched spans."""
        wrapped = Mock()
        instance = Mock()

        spans = [_make_adk_span("call_llm")]
        _exporter_export_wrapper(wrapped, instance, (spans,), {})

        wrapped.assert_called_once()
        enriched = wrapped.call_args[0][0]
        assert len(enriched) == 1
        # Enriched span should have traceloop attributes
        assert enriched[0].attributes.get(SpanAttributes.TRACELOOP_ENTITY_PATH) == "adk.call_llm"


class TestPatchSpanProcessors:
    def test_idempotent(self):
        """Calling patch_span_processors() multiple times should be safe."""
        import respan_instrumentation_google_adk.processor as proc_mod

        original_patched = proc_mod._PATCHED
        try:
            proc_mod._PATCHED = False
            with patch("respan_instrumentation_google_adk.processor.wrapt") as mock_wrapt:
                patch_span_processors()
                first_call_count = mock_wrapt.wrap_function_wrapper.call_count

                patch_span_processors()
                # Second call should not add more patches
                assert mock_wrapt.wrap_function_wrapper.call_count == first_call_count
        finally:
            proc_mod._PATCHED = original_patched
