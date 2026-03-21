"""Unit + integration tests for the unified respan stack.

Tests all three packages together:
1. respan-tracing (auto_instrument param)
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
# 2. respan package: imports and Respan class
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
            def activate(self): pass
            def deactivate(self): pass

        assert isinstance(MyPlugin(), Instrumentation)

    def test_respan_init_no_instrumentations(self):
        from respan import Respan
        r = Respan(api_key="test-key")
        assert r.telemetry is not None
        assert len(r._instrumentations) == 0
        r.shutdown()

    def test_respan_with_explicit_instrumentations(self):
        from respan import Respan

        activated = []

        class FakeInstrumentor:
            name = "fake"
            def activate(self):
                activated.append(True)
            def deactivate(self): pass

        r = Respan(api_key="test-key", instrumentations=[FakeInstrumentor()])
        assert len(activated) == 1
        assert "fake" in r._instrumentations
        r.shutdown()

    def test_respan_flush_and_shutdown(self):
        from respan import Respan
        r = Respan(api_key="test-key")
        r.flush()  # should not raise
        r.shutdown()  # should not raise


# ---------------------------------------------------------------------------
# 3. respan-instrumentation-openai-agents
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
# 4. End-to-end: explicit instrumentations
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
        """Trace → emitter → inject_span() → OTEL pipeline."""
        from respan import Respan
        from respan_instrumentation_openai_agents import OpenAIAgentsInstrumentor
        from respan_tracing.utils.span_factory import inject_span

        r = Respan(api_key="test-key", instrumentations=[OpenAIAgentsInstrumentor()])

        # Patch inject_span to capture calls
        injected_spans = []
        with patch("respan_instrumentation_openai_agents._otel_emitter.inject_span", side_effect=lambda s: injected_spans.append(s)):
            trace = _make_trace("e2e-test", "e2e-workflow")
            processor = r._instrumentations["openai-agents"]._processor
            processor.on_trace_end(trace)

            assert len(injected_spans) == 1
            span = injected_spans[0]
            assert span.attributes.get("traceloop.entity.name") == "e2e-workflow"

        r.shutdown()

    def test_multiple_instrumentations(self):
        """Multiple instrumentors can be passed and all get activated."""
        from respan import Respan

        activated_names = []

        class FakeInstrumentorA:
            name = "fake-a"
            def activate(self): activated_names.append("a")
            def deactivate(self): pass

        class FakeInstrumentorB:
            name = "fake-b"
            def activate(self): activated_names.append("b")
            def deactivate(self): pass

        r = Respan(api_key="test-key", instrumentations=[FakeInstrumentorA(), FakeInstrumentorB()])
        assert activated_names == ["a", "b"]
        assert "fake-a" in r._instrumentations
        assert "fake-b" in r._instrumentations
        r.shutdown()
