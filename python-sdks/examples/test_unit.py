"""Unit + integration tests for the unified respan stack.

Tests all three packages together:
1. respan-tracing (auto_instrument param, RespanSpanExporterV2)
2. respan (Respan class, explicit instrumentations, decorator re-exports)
3. respan-instrumentation-openai-agents (converter, instrumentor, LocalSpanCollector)
"""

import json
from unittest.mock import MagicMock, patch

import pytest
from conftest import _make_trace


# ---------------------------------------------------------------------------
# 1. respan-tracing: auto_instrument parameter
# ---------------------------------------------------------------------------

class TestAutoInstrument:
    def test_auto_instrument_defaults_true(self):
        import inspect
        from respan_tracing import RespanTelemetry
        sig = inspect.signature(RespanTelemetry.__init__)
        assert sig.parameters["auto_instrument"].default is True

    def test_auto_instrument_false_skips_instrumentations(self):
        from respan_tracing.core.tracer import RespanTracer
        with patch.object(RespanTracer, "_setup_instrumentations") as mock_setup:
            RespanTracer(auto_instrument=False)
            mock_setup.assert_not_called()

    def test_auto_instrument_true_runs_instrumentations(self):
        from respan_tracing.core.tracer import RespanTracer
        with patch.object(RespanTracer, "_setup_instrumentations") as mock_setup:
            RespanTracer(auto_instrument=True)
            mock_setup.assert_called_once()

    def test_tracer_provider_created_regardless(self):
        from respan_tracing.core.tracer import RespanTracer
        t = RespanTracer(auto_instrument=False)
        assert t.tracer_provider is not None

    def test_telemetry_passes_auto_instrument(self):
        from respan_tracing import RespanTelemetry
        from respan_tracing.core.tracer import RespanTracer
        with patch.object(RespanTracer, "_setup_instrumentations") as mock_setup:
            RespanTelemetry(auto_instrument=False)
            mock_setup.assert_not_called()


# ---------------------------------------------------------------------------
# 2. respan-tracing: RespanSpanExporterV2
# ---------------------------------------------------------------------------

class TestRespanSpanExporterV2:
    def test_import(self):
        from respan_tracing.exporters import RespanSpanExporterV2
        assert RespanSpanExporterV2 is not None

    def test_init_defaults(self):
        from respan_tracing.exporters import RespanSpanExporterV2
        exp = RespanSpanExporterV2(api_key="test-key")
        assert exp.endpoint == "https://api.respan.ai/api/v1/traces/ingest"
        assert exp.api_key == "test-key"
        assert exp.max_batch_size == 128
        assert exp.flush_interval == 5.0
        exp.shutdown()

    def test_export_enqueues_data(self):
        from respan_tracing.exporters import RespanSpanExporterV2
        exp = RespanSpanExporterV2(api_key="test-key", flush_interval=999)
        exp.export([{"trace_id": "abc", "span_name": "test"}])
        assert len(exp._queue) == 1
        exp.shutdown()

    def test_export_skips_when_shutdown(self):
        from respan_tracing.exporters import RespanSpanExporterV2
        exp = RespanSpanExporterV2(api_key="test-key", flush_interval=999)
        exp.shutdown()
        exp.export([{"data": "should be dropped"}])
        assert len(exp._queue) == 0

    def test_send_batch_posts_to_endpoint(self):
        from respan_tracing.exporters import RespanSpanExporterV2
        exp = RespanSpanExporterV2(api_key="test-key", flush_interval=999)

        mock_response = MagicMock()
        mock_response.status_code = 200
        with patch.object(exp._session, "post", return_value=mock_response) as mock_post:
            exp.export([{"trace_id": "abc"}])
            exp.flush()
            mock_post.assert_called_once()
            call_kwargs = mock_post.call_args
            payload = json.loads(call_kwargs.kwargs.get("data", call_kwargs[1].get("data", "")))
            assert payload == {"data": [{"trace_id": "abc"}]}
        exp.shutdown()

    def test_no_retry_on_4xx(self):
        from respan_tracing.exporters import RespanSpanExporterV2
        exp = RespanSpanExporterV2(api_key="test-key", flush_interval=999, max_retries=3)

        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = "Bad Request"
        with patch.object(exp._session, "post", return_value=mock_response) as mock_post:
            exp.export([{"trace_id": "abc"}])
            exp.flush()
            assert mock_post.call_count == 1  # No retries on 4xx
        exp.shutdown()

    def test_queue_maxlen_drops_oldest(self):
        from respan_tracing.exporters import RespanSpanExporterV2
        exp = RespanSpanExporterV2(api_key="test-key", max_queue_size=3, flush_interval=999)
        exp.export([{"id": 1}, {"id": 2}, {"id": 3}, {"id": 4}])
        assert len(exp._queue) == 3
        assert exp._queue[0] == {"id": 2}  # oldest dropped
        exp.shutdown()


# ---------------------------------------------------------------------------
# 3. respan package: imports and Respan class
# ---------------------------------------------------------------------------

class TestRespanPackage:
    def test_decorator_reexports(self):
        from respan import workflow, task, agent, tool
        from respan_tracing import workflow as w2, task as t2, agent as a2, tool as tl2
        assert workflow is w2
        assert task is t2
        assert agent is a2
        assert tool is tl2

    def test_client_reexports(self):
        from respan import RespanClient, get_client, respan_span_attributes
        from respan_tracing import RespanClient as RC2, get_client as gc2, respan_span_attributes as rsa2
        assert RespanClient is RC2
        assert get_client is gc2
        assert respan_span_attributes is rsa2

    def test_instrumentation_protocol(self):
        from respan import Instrumentation

        class MyPlugin:
            name = "test"
            def activate(self, exporter): pass
            def deactivate(self): pass

        assert isinstance(MyPlugin(), Instrumentation)

    def test_respan_init_no_instrumentations(self):
        from respan import Respan
        r = Respan(api_key="test-key")
        assert r.telemetry is not None
        assert r.exporter is not None
        assert r.exporter.endpoint == "https://api.respan.ai/api/v1/traces/ingest"
        assert len(r._instrumentations) == 0
        r.shutdown()

    def test_respan_with_explicit_instrumentations(self):
        from respan import Respan

        activated = []

        class FakeInstrumentor:
            name = "fake"
            def activate(self, exporter):
                activated.append(exporter)
            def deactivate(self): pass

        r = Respan(api_key="test-key", instrumentations=[FakeInstrumentor()])
        assert len(activated) == 1
        assert activated[0] is r.exporter
        assert "fake" in r._instrumentations
        r.shutdown()

    def test_respan_flush_and_shutdown(self):
        from respan import Respan
        r = Respan(api_key="test-key")
        r.flush()  # should not raise
        r.shutdown()  # should not raise


# ---------------------------------------------------------------------------
# 4. respan-instrumentation-openai-agents
# ---------------------------------------------------------------------------

class TestInstrumentationPackage:
    def test_import(self):
        from respan_instrumentation_openai_agents import (
            OpenAIAgentsInstrumentor,
            LocalSpanCollector,
            convert_to_respan_log,
        )
        assert OpenAIAgentsInstrumentor is not None
        assert LocalSpanCollector is not None
        assert convert_to_respan_log is not None

    def test_instrumentor_satisfies_protocol(self):
        from respan import Instrumentation
        from respan_instrumentation_openai_agents import OpenAIAgentsInstrumentor
        inst = OpenAIAgentsInstrumentor()
        assert isinstance(inst, Instrumentation)
        assert inst.name == "openai-agents"

    def test_converter_with_trace(self):
        from respan_instrumentation_openai_agents import convert_to_respan_log

        trace = _make_trace("test-trace-123", "test-workflow")
        result = convert_to_respan_log(trace)
        assert result is not None
        assert result["trace_unique_id"] == "test-trace-123"
        assert result["span_name"] == "test-workflow"

    def test_local_span_collector_pop_trace(self):
        from respan_instrumentation_openai_agents import LocalSpanCollector

        collector = LocalSpanCollector()
        trace = _make_trace("t1", "workflow1")

        collector.on_trace_end(trace)
        spans = collector.pop_trace("t1")
        assert len(spans) == 1
        assert spans[0]["trace_unique_id"] == "t1"

        # Second pop returns empty
        assert collector.pop_trace("t1") == []


# ---------------------------------------------------------------------------
# 5. End-to-end: explicit instrumentations
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def test_respan_with_openai_agents_instrumentor(self):
        """Respan(instrumentations=[OpenAIAgentsInstrumentor()]) activates the plugin."""
        from respan import Respan
        from respan_instrumentation_openai_agents import OpenAIAgentsInstrumentor

        r = Respan(api_key="test-key", instrumentations=[OpenAIAgentsInstrumentor()])
        assert "openai-agents" in r._instrumentations
        r.shutdown()

    def test_end_to_end_trace_export(self):
        """Trace → converter → exporter.export() → HTTP POST."""
        from respan import Respan
        from respan_instrumentation_openai_agents import OpenAIAgentsInstrumentor

        r = Respan(api_key="test-key", instrumentations=[OpenAIAgentsInstrumentor()])

        mock_response = MagicMock()
        mock_response.status_code = 200
        with patch.object(r.exporter._session, "post", return_value=mock_response) as mock_post:
            trace = _make_trace("e2e-test", "e2e-workflow")
            processor = r._instrumentations["openai-agents"]._processor
            processor.on_trace_end(trace)

            r.exporter.flush()

            assert mock_post.call_count == 1
            call_kwargs = mock_post.call_args
            payload = json.loads(call_kwargs.kwargs.get("data", call_kwargs[1].get("data", "")))
            assert len(payload["data"]) == 1
            assert payload["data"][0]["trace_unique_id"] == "e2e-test"
            assert payload["data"][0]["span_name"] == "e2e-workflow"

        r.shutdown()

    def test_multiple_instrumentations(self):
        """Multiple instrumentors can be passed and all get activated."""
        from respan import Respan

        activated_names = []

        class FakeInstrumentorA:
            name = "fake-a"
            def activate(self, exporter): activated_names.append("a")
            def deactivate(self): pass

        class FakeInstrumentorB:
            name = "fake-b"
            def activate(self, exporter): activated_names.append("b")
            def deactivate(self): pass

        r = Respan(api_key="test-key", instrumentations=[FakeInstrumentorA(), FakeInstrumentorB()])
        assert activated_names == ["a", "b"]
        assert "fake-a" in r._instrumentations
        assert "fake-b" in r._instrumentations
        r.shutdown()
