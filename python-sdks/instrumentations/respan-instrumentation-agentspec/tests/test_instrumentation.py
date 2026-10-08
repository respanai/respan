import logging
import sys
from contextvars import ContextVar
from types import ModuleType, SimpleNamespace
from typing import ClassVar

import pytest
from opentelemetry.trace import SpanContext, TraceFlags
from respan_instrumentation_agentspec import AgentSpecInstrumentor, _instrumentation
from respan_instrumentation_agentspec._instrumentation import (
    AGENTSPEC_INSTRUMENTATION_NAME,
    _extract_langchain_usage,
    _patch_agentspec_langgraph_usage,
    _TranslatedProcessorChain,
)
from respan_sdk.constants.span_attributes import (
    RESPAN_CUSTOMER_PARAMS_ID,
    RESPAN_LOG_TYPE,
    RESPAN_METADATA,
    RESPAN_SPAN_CUSTOM_ID,
    RESPAN_SPAN_TOOLS,
    RESPAN_THREADS_ID,
    RESPAN_TRACE_GROUP_ID,
)
from respan_tracing.core.tracer import RespanTracer


class FakeExportProcessor:
    def __init__(self):
        self.started = []
        self.ended = []
        self.did_shutdown = False
        self.did_force_flush = False

    def on_start(self, span, parent_context=None):
        self.started.append((span, parent_context))

    def on_end(self, span):
        self.ended.append(span)

    def shutdown(self):
        self.did_shutdown = True

    def force_flush(self, timeout_millis=30000):
        self.did_force_flush = True
        return True


class FakeTranslator:
    def __init__(self):
        self.ended = []

    def on_end(self, span):
        span._attributes["translated"] = True
        self.ended.append(span)


def _install_fake_modules(monkeypatch):
    active_trace = {"trace": None}

    class FakeOpenInferenceSpanProcessor:
        created: ClassVar[list] = []

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.did_shutdown = False
            self.__class__.created.append(self)

        def shutdown(self):
            self.did_shutdown = True

    class FakeTrace:
        created: ClassVar[list] = []

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.did_start = False
            self.did_end = False
            self.__class__.created.append(self)

        def _start(self):
            self.did_start = True
            active_trace["trace"] = self

        def _end(self):
            self.did_end = True
            active_trace["trace"] = None
            for processor in self.kwargs["span_processors"]:
                shutdown = getattr(processor, "shutdown", None)
                if shutdown is not None:
                    shutdown()

    class FakeRootSpan:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.name = kwargs.get("name")

    def get_trace():
        return active_trace["trace"]

    openinference_module = ModuleType("openinference")
    openinference_instrumentation_module = ModuleType("openinference.instrumentation")
    openinference_agentspec_module = ModuleType(
        "openinference.instrumentation.agentspec"
    )
    openinference_agentspec_module.OpenInferenceSpanProcessor = (
        FakeOpenInferenceSpanProcessor
    )
    openinference_instrumentation_module.agentspec = openinference_agentspec_module

    pyagentspec_module = ModuleType("pyagentspec")
    pyagentspec_tracing_module = ModuleType("pyagentspec.tracing")
    pyagentspec_trace_module = ModuleType("pyagentspec.tracing.trace")
    pyagentspec_spans_module = ModuleType("pyagentspec.tracing.spans")
    pyagentspec_trace_module.Trace = FakeTrace
    pyagentspec_trace_module.get_trace = get_trace
    pyagentspec_spans_module.RootSpan = FakeRootSpan
    native_spans = ModuleType("pyagentspec.tracing.spans.span")
    native_spans._ACTIVE_SPAN_STACK = ContextVar("fake_stack", default=())
    native_spans.get_active_span_stack = native_spans._ACTIVE_SPAN_STACK.get
    pyagentspec_spans_module.span = native_spans
    pyagentspec_tracing_module.trace = pyagentspec_trace_module
    pyagentspec_tracing_module.spans = pyagentspec_spans_module
    pyagentspec_module.tracing = pyagentspec_tracing_module

    monkeypatch.setitem(sys.modules, "openinference", openinference_module)
    monkeypatch.setitem(
        sys.modules,
        "openinference.instrumentation",
        openinference_instrumentation_module,
    )
    monkeypatch.setitem(
        sys.modules,
        "openinference.instrumentation.agentspec",
        openinference_agentspec_module,
    )
    monkeypatch.setitem(sys.modules, "pyagentspec", pyagentspec_module)
    monkeypatch.setitem(sys.modules, "pyagentspec.tracing", pyagentspec_tracing_module)
    monkeypatch.setitem(
        sys.modules,
        "pyagentspec.tracing.trace",
        pyagentspec_trace_module,
    )
    monkeypatch.setitem(
        sys.modules,
        "pyagentspec.tracing.spans",
        pyagentspec_spans_module,
    )

    export_processor = FakeExportProcessor()
    tracer_provider = SimpleNamespace(
        resource="fake-resource",
        _active_span_processor=SimpleNamespace(_span_processors=(export_processor,)),
    )
    monkeypatch.setattr(
        _instrumentation.trace,
        "get_tracer_provider",
        lambda: tracer_provider,
    )
    monkeypatch.setattr(_instrumentation, "OpenInferenceTranslator", FakeTranslator)

    return SimpleNamespace(
        active_trace=active_trace,
        export_processor=export_processor,
        span_processor_class=FakeOpenInferenceSpanProcessor,
        trace_class=FakeTrace,
        tracer_provider=tracer_provider,
    )


@pytest.fixture(autouse=True)
def reset_tracer():
    RespanTracer.reset_instance()
    yield
    if _instrumentation._OWNER is not None:
        _instrumentation._OWNER._is_instrumented = True
        _instrumentation._OWNERS = 1
        _instrumentation._OWNER.deactivate()
    _instrumentation._restore_agentspec_langgraph_usage_patch()
    RespanTracer.reset_instance()


def test_translated_processor_chain_translates_before_export():
    translator = FakeTranslator()
    export_processor = FakeExportProcessor()
    chain = _TranslatedProcessorChain(
        translator=translator,
        processors=(export_processor,),
    )
    span = SimpleNamespace(_attributes={})

    chain.on_end(span)

    assert translator.ended == [span]
    assert export_processor.ended == [span]
    assert export_processor.ended[0]._attributes["translated"] is True


def test_translated_processor_chain_sets_workflow_name_before_export():
    translator = FakeTranslator()
    export_processor = FakeExportProcessor()
    chain = _TranslatedProcessorChain(
        translator=translator,
        processors=(export_processor,),
        workflow_name="agentspec_haiku_agent",
    )
    span = SimpleNamespace(_attributes={})

    chain.on_end(span)

    assert span._attributes["traceloop.workflow.name"] == "agentspec_haiku_agent"
    assert export_processor.ended[0]._attributes["traceloop.workflow.name"] == (
        "agentspec_haiku_agent"
    )


def test_translated_processor_chain_does_not_shutdown_borrowed_processors():
    export_processor = FakeExportProcessor()
    chain = _TranslatedProcessorChain(
        translator=FakeTranslator(),
        processors=(export_processor,),
    )

    chain.shutdown()

    assert export_processor.did_shutdown is False


def test_translated_processor_chain_preserves_full_trace_id_and_enriches_boundaries():
    session_id = "12345678-1234-5678-9abc-def012345678"
    expected_trace_id = int(session_id.replace("-", ""), 16)

    class FakeSpan:
        def __init__(self, *, span_id, parent, attributes):
            self._context = SpanContext(
                trace_id=1,
                span_id=span_id,
                is_remote=False,
                trace_flags=TraceFlags(TraceFlags.SAMPLED),
            )
            self._parent = parent
            self._attributes = {"session.id": session_id, **attributes}

        @property
        def parent(self):
            return self._parent

        def get_span_context(self):
            return self._context

    export_processor = FakeExportProcessor()
    chain = _TranslatedProcessorChain(
        translator=FakeTranslator(),
        processors=(export_processor,),
    )
    root = FakeSpan(span_id=1, parent=None, attributes={RESPAN_LOG_TYPE: "workflow"})
    root_parent = root.get_span_context()
    agent = FakeSpan(
        span_id=2,
        parent=root_parent,
        attributes={
            RESPAN_LOG_TYPE: "agent",
            "traceloop.entity.input": "{}",
            "traceloop.entity.output": "{}",
        },
    )
    chat = FakeSpan(
        span_id=3,
        parent=agent.get_span_context(),
        attributes={
            RESPAN_LOG_TYPE: "chat",
            "gen_ai.prompt.0.role": "user",
            "gen_ai.prompt.0.content": "hello",
            "gen_ai.completion.0.role": "assistant",
            "gen_ai.completion.0.content": "hi",
            RESPAN_CUSTOMER_PARAMS_ID: "agentspec-user",
            RESPAN_THREADS_ID: "agentspec-thread",
            RESPAN_TRACE_GROUP_ID: "agentspec-group",
            RESPAN_SPAN_CUSTOM_ID: "agentspec-custom",
            f"{RESPAN_METADATA}.scenario": "propagated_attributes",
            RESPAN_SPAN_TOOLS: "[]",
            "tools": [],
        },
    )

    for span in (root, agent, chat):
        chain.on_start(span)

    assert root.get_span_context().trace_id == expected_trace_id
    assert agent.get_span_context().trace_id == expected_trace_id
    assert agent.parent.trace_id == expected_trace_id
    assert chat.get_span_context().trace_id == expected_trace_id
    assert chat.parent.trace_id == expected_trace_id
    assert expected_trace_id >> 64

    chain.on_end(chat)
    chain.on_end(agent)
    chain.on_end(root)

    assert chat._attributes["traceloop.entity.input"] == (
        '[{"role":"user","content":"hello"}]'
    )
    assert chat._attributes["traceloop.entity.output"] == (
        '{"role":"assistant","content":"hi"}'
    )
    for boundary in (agent, root):
        assert boundary._attributes["traceloop.entity.input"].endswith('hello"}]')
        assert boundary._attributes["traceloop.entity.output"].endswith('hi"}')
    assert root._attributes[RESPAN_CUSTOMER_PARAMS_ID] == "agentspec-user"
    assert root._attributes[RESPAN_THREADS_ID] == "agentspec-thread"
    assert root._attributes[RESPAN_TRACE_GROUP_ID] == "agentspec-group"
    assert root._attributes[RESPAN_SPAN_CUSTOM_ID] == "agentspec-custom"
    assert root._attributes[f"{RESPAN_METADATA}.scenario"] == ("propagated_attributes")
    assert RESPAN_SPAN_TOOLS not in chat._attributes
    assert "tools" not in chat._attributes
    assert chain._trace_state == {}


def test_extract_langchain_usage_from_message_usage_metadata():
    response = SimpleNamespace(
        generations=[
            [
                SimpleNamespace(
                    message=SimpleNamespace(
                        usage_metadata={
                            "input_tokens": 42,
                            "output_tokens": 9,
                        },
                        response_metadata={},
                    )
                )
            ]
        ],
        llm_output={},
    )

    assert _extract_langchain_usage(response) == (42, 9)


def test_extract_langchain_usage_from_llm_output_token_usage():
    response = SimpleNamespace(
        generations=[],
        llm_output={
            "token_usage": {
                "prompt_tokens": "12",
                "completion_tokens": "5",
            }
        },
    )

    assert _extract_langchain_usage(response) == (12, 5)


def test_callback_wrapper_preserves_native_return_and_usage_scope(monkeypatch):
    import pyagentspec.adapters.langgraph.tracing as module
    from respan_instrumentation_agentspec._callbacks import _CAPTURE

    cls = getattr(
        module, "AgentSpecLlmCallbackHandler", module.AgentSpecCallbackHandler
    )
    seen = []
    sentinel = object()

    def original(self, response, **kwargs):
        seen.append(_CAPTURE.get())
        return sentinel

    monkeypatch.setattr(cls, "on_llm_end", original)
    _patch_agentspec_langgraph_usage()
    response = SimpleNamespace(
        generations=[],
        llm_output={"token_usage": {"prompt_tokens": 0, "completion_tokens": 3}},
    )
    assert cls.on_llm_end(SimpleNamespace(), response, run_id="fixture-run") is sentinel
    assert seen[0]["usage"] == {"input_tokens": 0, "output_tokens": 3}
    assert seen[0]["response_id"] is None
    assert seen[0]["finish_reason"] is None
    assert _CAPTURE.get() is None


def test_activate_starts_agentspec_trace_with_translated_processor_chain(monkeypatch):
    fake = _install_fake_modules(monkeypatch)

    instrumentor = AgentSpecInstrumentor()
    instrumentor.activate()

    assert instrumentor.name == AGENTSPEC_INSTRUMENTATION_NAME
    assert fake.trace_class.created[0].did_start is True
    assert instrumentor._is_instrumented is True

    agentspec_processor = fake.span_processor_class.created[0]
    chain = agentspec_processor.kwargs["otel_span_processor"]
    assert isinstance(chain, _TranslatedProcessorChain)
    assert agentspec_processor.kwargs["resource"] == "fake-resource"
    assert agentspec_processor.kwargs["mask_sensitive_information"] is False

    instrumentor.deactivate()

    assert fake.trace_class.created[0].did_end is True
    assert fake.export_processor.did_shutdown is False
    assert instrumentor._is_instrumented is False


def test_activate_sets_named_root_span_and_workflow_name(monkeypatch):
    fake = _install_fake_modules(monkeypatch)

    instrumentor = AgentSpecInstrumentor(workflow_name="agentspec_haiku_agent")
    instrumentor.activate()

    trace = fake.trace_class.created[0]
    assert trace.kwargs["name"] == "agentspec_haiku_agent"
    assert trace.kwargs["root_span"].name == "agentspec_haiku_agent"

    agentspec_processor = fake.span_processor_class.created[0]
    chain = agentspec_processor.kwargs["otel_span_processor"]
    span = SimpleNamespace(_attributes={})
    chain.on_end(span)

    assert span._attributes["traceloop.workflow.name"] == "agentspec_haiku_agent"


def test_activate_passes_mask_sensitive_information(monkeypatch):
    fake = _install_fake_modules(monkeypatch)

    instrumentor = AgentSpecInstrumentor(mask_sensitive_information=True)
    instrumentor.activate()

    agentspec_processor = fake.span_processor_class.created[0]
    assert agentspec_processor.kwargs["mask_sensitive_information"] is True


def test_activate_is_idempotent(monkeypatch):
    fake = _install_fake_modules(monkeypatch)

    instrumentor = AgentSpecInstrumentor()
    instrumentor.activate()
    instrumentor.activate()

    assert len(fake.trace_class.created) == 1


def test_activate_skips_when_agentspec_trace_already_active(monkeypatch, caplog):
    fake = _install_fake_modules(monkeypatch)
    fake.active_trace["trace"] = object()

    instrumentor = AgentSpecInstrumentor()
    with caplog.at_level(logging.WARNING):
        instrumentor.activate()

    assert fake.trace_class.created == []
    assert instrumentor._is_instrumented is False
    assert "AgentSpec Trace is already active" in caplog.text


def test_activate_skips_when_respan_tracing_is_disabled(monkeypatch, caplog):
    fake = _install_fake_modules(monkeypatch)
    RespanTracer(is_enabled=False)

    instrumentor = AgentSpecInstrumentor()
    with caplog.at_level(logging.INFO):
        instrumentor.activate()

    assert fake.trace_class.created == []
    assert instrumentor._is_instrumented is False
    assert (
        "AgentSpec instrumentation skipped because Respan tracing is disabled"
        in caplog.text
    )


def test_activate_logs_warning_when_dependencies_are_missing(monkeypatch, caplog):
    monkeypatch.setitem(sys.modules, "openinference.instrumentation.agentspec", None)
    instrumentor = AgentSpecInstrumentor()

    with caplog.at_level(logging.WARNING):
        instrumentor.activate()

    assert "Failed to activate AgentSpec instrumentation" in caplog.text
    assert instrumentor._is_instrumented is False
