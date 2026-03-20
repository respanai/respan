"""Tests for LlamaIndex span handler."""
import inspect
import os
import uuid
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch

import pytest


class TestRespanSpanHandlerInit:
    """Test span handler initialization."""

    def test_creates_with_api_key(self):
        from respan_exporter_llamaindex.span_handler import RespanSpanHandler

        handler = RespanSpanHandler(api_key="test-key")
        assert handler._exporter.api_key == "test-key"

    def test_creates_with_env_api_key(self):
        from respan_exporter_llamaindex.span_handler import RespanSpanHandler

        with patch.dict(os.environ, {"RESPAN_API_KEY": "env-key"}, clear=False):
            handler = RespanSpanHandler()
            assert handler._exporter.api_key == "env-key"

    def test_class_name(self):
        from respan_exporter_llamaindex.span_handler import RespanSpanHandler

        assert RespanSpanHandler.class_name() == "RespanSpanHandler"

    def test_custom_environment(self):
        from respan_exporter_llamaindex.span_handler import RespanSpanHandler

        handler = RespanSpanHandler(api_key="test", environment="staging")
        assert handler._exporter.environment == "staging"


class TestNewSpan:
    """Test span creation."""

    def test_creates_span_record(self):
        from respan_exporter_llamaindex.span_handler import RespanSpanHandler

        handler = RespanSpanHandler(api_key="test")
        span_id = str(uuid.uuid4())

        # Simulate bound args
        def dummy_fn(query: str):
            pass

        sig = inspect.signature(dummy_fn)
        bound = sig.bind(query="What is AI?")

        span = handler.new_span(
            id_=span_id,
            bound_args=bound,
            instance=None,
            parent_span_id=None,
        )

        assert span is not None
        assert span.id_ == span_id
        assert span_id in handler._spans

    def test_creates_child_span(self):
        from respan_exporter_llamaindex.span_handler import RespanSpanHandler

        handler = RespanSpanHandler(api_key="test")
        parent_id = str(uuid.uuid4())
        child_id = str(uuid.uuid4())

        def dummy_fn():
            pass

        sig = inspect.signature(dummy_fn)
        bound = sig.bind()

        handler.new_span(id_=parent_id, bound_args=bound, instance=None)
        handler.new_span(
            id_=child_id,
            bound_args=bound,
            instance=None,
            parent_span_id=parent_id,
        )

        assert child_id in handler._spans
        record = handler._spans[child_id]
        assert record["parent_id"] == parent_id

    def test_captures_instance_class_name(self):
        from respan_exporter_llamaindex.span_handler import RespanSpanHandler

        handler = RespanSpanHandler(api_key="test")
        span_id = str(uuid.uuid4())

        class MockRetriever:
            pass

        def dummy_fn():
            pass

        sig = inspect.signature(dummy_fn)
        bound = sig.bind()

        handler.new_span(
            id_=span_id,
            bound_args=bound,
            instance=MockRetriever(),
        )

        record = handler._spans[span_id]
        assert record["instance_class"] == "MockRetriever"

    def test_tracks_root_spans(self):
        from respan_exporter_llamaindex.span_handler import RespanSpanHandler

        handler = RespanSpanHandler(api_key="test")
        span_id = str(uuid.uuid4())

        def dummy_fn():
            pass

        sig = inspect.signature(dummy_fn)
        bound = sig.bind()

        handler.new_span(id_=span_id, bound_args=bound, instance=None)

        assert span_id in handler._root_span_ids


class TestPrepareToExitSpan:
    """Test span completion."""

    def _setup_handler_with_span(self):
        from respan_exporter_llamaindex.span_handler import RespanSpanHandler

        handler = RespanSpanHandler(api_key="test")
        span_id = str(uuid.uuid4())

        def dummy_fn(query: str = "test"):
            pass

        sig = inspect.signature(dummy_fn)
        bound = sig.bind()

        handler.new_span(id_=span_id, bound_args=bound, instance=None)
        return handler, span_id, bound

    def test_records_result(self):
        """Test that result and end_time are recorded on a child span (which doesn't trigger flush)."""
        from respan_exporter_llamaindex.span_handler import RespanSpanHandler

        handler = RespanSpanHandler(api_key="test")
        parent_id = str(uuid.uuid4())
        child_id = str(uuid.uuid4())

        def dummy_fn():
            pass

        sig = inspect.signature(dummy_fn)
        bound = sig.bind()

        handler.new_span(id_=parent_id, bound_args=bound, instance=None)
        handler.new_span(id_=child_id, bound_args=bound, instance=None, parent_span_id=parent_id)

        handler.prepare_to_exit_span(
            id_=child_id,
            bound_args=bound,
            result="The answer is 42",
        )

        record = handler._spans[child_id]
        assert record["result"] == "The answer is 42"
        assert record["end_time"] is not None

    def test_flushes_root_span_on_exit(self):
        handler, span_id, bound = self._setup_handler_with_span()

        with patch.object(handler, "_flush_trace") as mock_flush:
            handler.prepare_to_exit_span(
                id_=span_id,
                bound_args=bound,
                result="done",
            )
            mock_flush.assert_called_once_with(span_id)

    def test_does_not_flush_child_span(self):
        from respan_exporter_llamaindex.span_handler import RespanSpanHandler

        handler = RespanSpanHandler(api_key="test")
        parent_id = str(uuid.uuid4())
        child_id = str(uuid.uuid4())

        def dummy_fn():
            pass

        sig = inspect.signature(dummy_fn)
        bound = sig.bind()

        handler.new_span(id_=parent_id, bound_args=bound, instance=None)
        handler.new_span(
            id_=child_id,
            bound_args=bound,
            instance=None,
            parent_span_id=parent_id,
        )

        with patch.object(handler, "_flush_trace") as mock_flush:
            handler.prepare_to_exit_span(
                id_=child_id,
                bound_args=bound,
                result="child result",
            )
            mock_flush.assert_not_called()


class TestPrepareToDropSpan:
    """Test span error handling."""

    def test_records_error(self):
        """Test error recording on a child span (doesn't trigger flush)."""
        from respan_exporter_llamaindex.span_handler import RespanSpanHandler

        handler = RespanSpanHandler(api_key="test")
        parent_id = str(uuid.uuid4())
        child_id = str(uuid.uuid4())

        def dummy_fn():
            pass

        sig = inspect.signature(dummy_fn)
        bound = sig.bind()

        handler.new_span(id_=parent_id, bound_args=bound, instance=None)
        handler.new_span(id_=child_id, bound_args=bound, instance=None, parent_span_id=parent_id)

        error = ValueError("Something went wrong")
        handler.prepare_to_drop_span(
            id_=child_id,
            bound_args=bound,
            err=error,
        )

        record = handler._spans[child_id]
        assert record["error"] == "Something went wrong"
        assert record["end_time"] is not None

    def test_flushes_root_span_on_error(self):
        from respan_exporter_llamaindex.span_handler import RespanSpanHandler

        handler = RespanSpanHandler(api_key="test")
        span_id = str(uuid.uuid4())

        def dummy_fn():
            pass

        sig = inspect.signature(dummy_fn)
        bound = sig.bind()

        handler.new_span(id_=span_id, bound_args=bound, instance=None)

        with patch.object(handler, "_flush_trace") as mock_flush:
            handler.prepare_to_drop_span(
                id_=span_id,
                bound_args=bound,
                err=RuntimeError("failed"),
            )
            mock_flush.assert_called_once_with(span_id)


class TestFlushTrace:
    """Test trace flushing to Respan."""

    def test_collects_all_spans_in_trace(self):
        from respan_exporter_llamaindex.span_handler import RespanSpanHandler

        handler = RespanSpanHandler(api_key="test")
        parent_id = str(uuid.uuid4())
        child_id = str(uuid.uuid4())

        def dummy_fn():
            pass

        sig = inspect.signature(dummy_fn)
        bound = sig.bind()

        handler.new_span(id_=parent_id, bound_args=bound, instance=None)
        handler.new_span(
            id_=child_id,
            bound_args=bound,
            instance=None,
            parent_span_id=parent_id,
        )

        handler.prepare_to_exit_span(id_=child_id, bound_args=bound, result="child done")

        with patch.object(handler._exporter, "export") as mock_export:
            handler.prepare_to_exit_span(id_=parent_id, bound_args=bound, result="parent done")
            mock_export.assert_called_once()
            call_kwargs = mock_export.call_args.kwargs
            spans = call_kwargs.get("spans", mock_export.call_args[0][0] if mock_export.call_args[0] else [])
            assert len(spans) == 2

    def test_cleans_up_after_flush(self):
        from respan_exporter_llamaindex.span_handler import RespanSpanHandler

        handler = RespanSpanHandler(api_key="test")
        span_id = str(uuid.uuid4())

        def dummy_fn():
            pass

        sig = inspect.signature(dummy_fn)
        bound = sig.bind()

        handler.new_span(id_=span_id, bound_args=bound, instance=None)

        with patch.object(handler._exporter, "export"):
            handler.prepare_to_exit_span(id_=span_id, bound_args=bound, result="done")

        assert span_id not in handler._spans
        assert span_id not in handler._root_span_ids
