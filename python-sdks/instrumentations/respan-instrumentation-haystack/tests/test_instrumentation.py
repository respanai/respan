import json
import logging
import sys
from types import ModuleType, SimpleNamespace
from typing import ClassVar

import pytest
from opentelemetry.attributes import BoundedAttributes
from opentelemetry.semconv_ai import SpanAttributes
from respan_instrumentation_haystack import HaystackInstrumentor, _instrumentation
from respan_instrumentation_haystack._constants import (
    HAYSTACK_ASYNC_PIPELINE_CLASS_NAME,
    HAYSTACK_ASYNC_PIPELINE_MODULE,
    HAYSTACK_ASYNC_PIPELINE_RUN_ASYNC_GENERATOR_METHOD_SPAN_NAME,
    HAYSTACK_ASYNC_PIPELINE_RUN_ASYNC_METHOD_SPAN_NAME,
    HAYSTACK_ASYNC_PIPELINE_RUN_SPAN_NAME,
    HAYSTACK_COMPONENT_MODULE,
    HAYSTACK_COMPONENT_RUN_SPAN_NAME,
    HAYSTACK_INSTRUMENTATION_NAME,
    HAYSTACK_PIPELINE_CLASS_NAME,
    HAYSTACK_PIPELINE_MODULE,
    HAYSTACK_PIPELINE_RUN_METHOD_SPAN_NAME,
    HAYSTACK_PIPELINE_RUN_SPAN_NAME,
    OPENINFERENCE_HAYSTACK_MODULE,
    OPENINFERENCE_HAYSTACK_WRAPPERS_MODULE,
)
from respan_instrumentation_haystack._instrumentation import (
    _HaystackParentSpanProcessor,
    _register_haystack_parent_processor,
    _remove_haystack_parent_processor,
)
from respan_sdk.constants.span_attributes import RESPAN_LOG_TYPE
from respan_tracing.core.tracer import RespanTracer
from respan_tracing.utils.preprocessing.span_processing import is_processable_span


def _install_fake_modules(monkeypatch):
    class FakeHaystackInstrumentor:
        pass

    class FakeOpenInferenceInstrumentor:
        created: ClassVar[list[object]] = []

        def __init__(self, instrumentor_class, **kwargs):
            self.instrumentor_class = instrumentor_class
            self.kwargs = kwargs
            self.is_activated = False
            self.is_deactivated = False
            self.__class__.created.append(self)

        def activate(self):
            self.is_activated = True

        def deactivate(self):
            self.is_deactivated = True

    openinference_module = ModuleType("openinference")
    openinference_instrumentation_module = ModuleType("openinference.instrumentation")
    openinference_haystack_module = ModuleType(OPENINFERENCE_HAYSTACK_MODULE)
    openinference_haystack_module.HaystackInstrumentor = FakeHaystackInstrumentor
    openinference_instrumentation_module.haystack = openinference_haystack_module

    monkeypatch.setitem(sys.modules, "openinference", openinference_module)
    monkeypatch.setitem(
        sys.modules,
        "openinference.instrumentation",
        openinference_instrumentation_module,
    )
    monkeypatch.setitem(
        sys.modules,
        OPENINFERENCE_HAYSTACK_MODULE,
        openinference_haystack_module,
    )
    monkeypatch.setattr(
        _instrumentation,
        "OpenInferenceInstrumentor",
        FakeOpenInferenceInstrumentor,
    )

    return SimpleNamespace(
        haystack_instrumentor_class=FakeHaystackInstrumentor,
        openinference_instrumentor_class=FakeOpenInferenceInstrumentor,
    )


@pytest.fixture(autouse=True)
def reset_tracer():
    RespanTracer.reset_instance()
    yield
    RespanTracer.reset_instance()


def test_package_exports_haystack_instrumentor():
    assert HaystackInstrumentor is _instrumentation.HaystackInstrumentor
    assert HaystackInstrumentor.name == HAYSTACK_INSTRUMENTATION_NAME


def test_load_openinference_haystack_class(monkeypatch):
    fake = _install_fake_modules(monkeypatch)

    instrumentor_class = _instrumentation._load_openinference_haystack_class()

    assert instrumentor_class is fake.haystack_instrumentor_class


def test_resolve_registered_main_component_class(monkeypatch):
    class FakeMainComponent:
        __module__ = "__main__"

    component_module = ModuleType(HAYSTACK_COMPONENT_MODULE)
    component_module.component = SimpleNamespace(
        registry={"__main__.FakeMainComponent": FakeMainComponent}
    )
    monkeypatch.setitem(
        sys.modules,
        HAYSTACK_COMPONENT_MODULE,
        component_module,
    )

    component_class = _instrumentation._resolve_registered_component_class(
        module_name="__main__",
        wrapper_path="FakeMainComponent.run",
    )

    assert component_class is FakeMainComponent


def test_patch_main_component_wrapping_falls_back_to_registered_class(monkeypatch):
    class FakeMainComponent:
        __module__ = "__main__"

    calls = []

    def fake_wrap_function_wrapper(target, name, wrapper):
        calls.append((target, name, wrapper))
        if target == "__main__":
            raise AttributeError("module '__main__' has no attribute")
        return "wrapped"

    openinference_haystack_module = ModuleType(OPENINFERENCE_HAYSTACK_MODULE)
    openinference_haystack_module.wrap_function_wrapper = fake_wrap_function_wrapper
    component_module = ModuleType(HAYSTACK_COMPONENT_MODULE)
    component_module.component = SimpleNamespace(
        registry={"__main__.FakeMainComponent": FakeMainComponent}
    )
    monkeypatch.setitem(
        sys.modules,
        OPENINFERENCE_HAYSTACK_MODULE,
        openinference_haystack_module,
    )
    monkeypatch.setitem(
        sys.modules,
        HAYSTACK_COMPONENT_MODULE,
        component_module,
    )

    _instrumentation._patch_main_component_wrapping()
    result = openinference_haystack_module.wrap_function_wrapper(
        module="__main__",
        name="FakeMainComponent.run",
        wrapper="wrapper",
    )

    assert result == "wrapped"
    assert calls == [
        ("__main__", "FakeMainComponent.run", "wrapper"),
        (FakeMainComponent, "run", "wrapper"),
    ]


def test_patch_late_component_registration_wraps_direct_components(monkeypatch):
    class FakeComponentDecorator:
        def _component(self, component_class):
            return component_class

    class FakeSyncWrapper:
        def __init__(self, tracer):
            self.tracer = tracer

    class FakeAsyncWrapper:
        def __init__(self, tracer):
            self.tracer = tracer

    component_decorator = FakeComponentDecorator()
    component_module = ModuleType(HAYSTACK_COMPONENT_MODULE)
    component_module.component = component_decorator
    wrapped_methods = []

    def fake_wrap_function_wrapper(target, name, wrapper):
        wrapped_methods.append((target, name, wrapper))

    haystack_module = ModuleType(OPENINFERENCE_HAYSTACK_MODULE)
    haystack_module.wrap_function_wrapper = fake_wrap_function_wrapper
    wrappers_module = ModuleType(OPENINFERENCE_HAYSTACK_WRAPPERS_MODULE)
    wrappers_module._ComponentRunWrapper = FakeSyncWrapper
    wrappers_module._AsyncComponentRunWrapper = FakeAsyncWrapper
    monkeypatch.setitem(sys.modules, HAYSTACK_COMPONENT_MODULE, component_module)
    monkeypatch.setitem(sys.modules, OPENINFERENCE_HAYSTACK_MODULE, haystack_module)
    monkeypatch.setitem(
        sys.modules,
        OPENINFERENCE_HAYSTACK_WRAPPERS_MODULE,
        wrappers_module,
    )

    tracer = object()
    openinference_instrumentor = SimpleNamespace(
        _tracer=tracer,
        _original_component_run_methods={},
        _original_component_run_async_methods={},
    )
    delegate = SimpleNamespace(_instrumentor=openinference_instrumentor)
    original_component = component_decorator._component

    patch = _instrumentation._patch_late_component_registration(delegate)

    class LateComponent:
        def run(self):
            return {"sync": True}

        async def run_async(self):
            return {"async": True}

    registered = component_decorator._component(LateComponent)

    assert registered is LateComponent
    assert openinference_instrumentor._original_component_run_methods[registered]
    assert openinference_instrumentor._original_component_run_async_methods[registered]
    assert [(target, name) for target, name, _ in wrapped_methods] == [
        (LateComponent, "run"),
        (LateComponent, "run_async"),
    ]
    assert all(wrapper.tracer is tracer for _, _, wrapper in wrapped_methods)

    _instrumentation._restore_late_component_registration(patch)

    assert component_decorator._component == original_component


class _FakeSpanContext:
    def __init__(self, span_id: str):
        self.span_id = int(span_id, 16)


class _FakeSpan:
    def __init__(
        self,
        name: str,
        span_id: str,
        parent: _FakeSpanContext | None = None,
        attributes: dict | None = None,
    ):
        self.name = name
        self._context = _FakeSpanContext(span_id)
        self._parent = parent
        self._attributes = attributes or {}

    @property
    def parent(self):
        return self._parent

    @property
    def attributes(self):
        return self._attributes

    def get_span_context(self):
        return self._context


class _FakeGraph:
    def __init__(self, predecessors: dict[str, tuple[str, ...]]):
        self._predecessors = predecessors

    def predecessors(self, component_name: str):
        return self._predecessors.get(component_name, ())


def _start_span_for_component(
    processor: _HaystackParentSpanProcessor,
    span: _FakeSpan,
    *,
    component_name: str,
    pipeline_context,
):
    token = _instrumentation._CURRENT_COMPONENT_RUN_CONTEXT.set(
        _instrumentation._HaystackComponentRunContext(
            component_name=component_name,
            pipeline_context=pipeline_context,
        )
    )
    try:
        processor.on_start(span)
    finally:
        _instrumentation._CURRENT_COMPONENT_RUN_CONTEXT.reset(token)


def test_haystack_parent_span_processor_skips_native_haystack_spans():
    processor = _HaystackParentSpanProcessor()
    root = _FakeSpan(HAYSTACK_PIPELINE_RUN_METHOD_SPAN_NAME, "1000000000000001")
    native_pipeline = _FakeSpan(
        HAYSTACK_PIPELINE_RUN_SPAN_NAME,
        "2000000000000002",
        root.get_span_context(),
    )
    native_component = _FakeSpan(
        HAYSTACK_COMPONENT_RUN_SPAN_NAME,
        "3000000000000003",
        native_pipeline.get_span_context(),
    )
    child = _FakeSpan(
        "PromptBuilder.run",
        "4000000000000004",
        native_component.get_span_context(),
    )

    for span in (root, native_pipeline, native_component, child):
        processor.on_start(span)

    processor.on_end(child)

    assert child.parent is root.get_span_context()


def test_haystack_parent_span_processor_skips_async_native_span():
    processor = _HaystackParentSpanProcessor()
    root = _FakeSpan(
        HAYSTACK_ASYNC_PIPELINE_RUN_ASYNC_METHOD_SPAN_NAME,
        "1000000000000001",
    )
    generator = _FakeSpan(
        HAYSTACK_ASYNC_PIPELINE_RUN_ASYNC_GENERATOR_METHOD_SPAN_NAME,
        "2000000000000002",
        root.get_span_context(),
    )
    native_async_pipeline = _FakeSpan(
        HAYSTACK_ASYNC_PIPELINE_RUN_SPAN_NAME,
        "3000000000000003",
        generator.get_span_context(),
    )
    child = _FakeSpan(
        "AsyncEcho.run_async",
        "4000000000000004",
        native_async_pipeline.get_span_context(),
    )

    for span in (root, generator, native_async_pipeline, child):
        processor.on_start(span)

    processor.on_end(child)

    assert child.parent is generator.get_span_context()


def test_haystack_parent_span_processor_keeps_exported_parent():
    processor = _HaystackParentSpanProcessor()
    root = _FakeSpan(HAYSTACK_PIPELINE_RUN_METHOD_SPAN_NAME, "1000000000000001")
    child = _FakeSpan(
        "PromptBuilder.run",
        "4000000000000004",
        root.get_span_context(),
    )

    for span in (root, child):
        processor.on_start(span)

    processor.on_end(child)

    assert child.parent is root.get_span_context()


def test_haystack_parent_span_processor_suppresses_native_span_export():
    processor = _HaystackParentSpanProcessor()
    native_component = _FakeSpan(
        HAYSTACK_COMPONENT_RUN_SPAN_NAME,
        "3000000000000003",
        attributes={
            SpanAttributes.TRACELOOP_ENTITY_NAME: HAYSTACK_COMPONENT_RUN_SPAN_NAME,
            SpanAttributes.TRACELOOP_ENTITY_PATH: "haystack-example.Pipeline.run",
            SpanAttributes.TRACELOOP_WORKFLOW_NAME: "haystack-example",
            RESPAN_LOG_TYPE: "span",
            "haystack.component.name": "prompt_builder",
        },
    )

    processor.on_start(native_component)
    processor.on_end(native_component)

    assert native_component.attributes == {
        "haystack.component.name": "prompt_builder",
    }
    assert is_processable_span(native_component) is False


def test_haystack_parent_span_processor_suppresses_immutable_native_attributes():
    processor = _HaystackParentSpanProcessor()
    native_component = _FakeSpan(
        HAYSTACK_COMPONENT_RUN_SPAN_NAME,
        "3000000000000004",
    )
    native_component._attributes = BoundedAttributes(
        maxlen=128,
        attributes={
            SpanAttributes.TRACELOOP_ENTITY_NAME: HAYSTACK_COMPONENT_RUN_SPAN_NAME,
            SpanAttributes.TRACELOOP_ENTITY_PATH: "haystack-example.Pipeline.run",
            SpanAttributes.TRACELOOP_WORKFLOW_NAME: "haystack-example",
            RESPAN_LOG_TYPE: "span",
            "haystack.component.name": "prompt_builder",
        },
        immutable=True,
    )

    processor.on_start(native_component)
    processor.on_end(native_component)

    assert native_component.attributes == {
        "haystack.component.name": "prompt_builder",
    }
    assert is_processable_span(native_component) is False


def test_haystack_parent_span_processor_keeps_openinference_span_exportable():
    processor = _HaystackParentSpanProcessor()
    translated_component = _FakeSpan(
        "PromptBuilder.run",
        "4000000000000004",
        attributes={
            SpanAttributes.TRACELOOP_ENTITY_PATH: "haystack-example.PromptBuilder",
            RESPAN_LOG_TYPE: "task",
        },
    )

    processor.on_start(translated_component)
    processor.on_end(translated_component)

    assert translated_component.attributes == {
        SpanAttributes.TRACELOOP_ENTITY_PATH: "haystack-example.PromptBuilder",
        RESPAN_LOG_TYPE: "task",
    }
    assert is_processable_span(translated_component) is True


def test_haystack_parent_span_processor_promotes_component_chat_io():
    processor = _HaystackParentSpanProcessor()
    span = _FakeSpan(
        "haystack.openai.chat",
        "4000000000000004",
        attributes={
            SpanAttributes.TRACELOOP_ENTITY_INPUT: "{}",
            SpanAttributes.TRACELOOP_ENTITY_OUTPUT: "replies",
            "haystack.component.input": json.dumps(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": [{"text": "Who created Python?"}],
                        }
                    ],
                    "generation_kwargs": {
                        "extra_body": {
                            "prompt": {
                                "prompt_id": "prompt-123",
                                "variables": {"question": "Who created Python?"},
                            }
                        }
                    },
                }
            ),
            "haystack.component.output": json.dumps(
                {
                    "replies": [
                        {
                            "role": "assistant",
                            "content": [
                                {"text": "Python was created by Guido van Rossum."}
                            ],
                        }
                    ]
                }
            ),
        },
    )

    processor.on_start(span)
    processor.on_end(span)

    assert json.loads(span.attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT]) == [
        {"role": "user", "content": "Who created Python?"}
    ]
    assert json.loads(span.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]) == {
        "role": "assistant",
        "content": "Python was created by Guido van Rossum.",
    }
    assert span.attributes["gen_ai.prompt.0.role"] == "user"
    assert span.attributes["gen_ai.prompt.0.content"] == "Who created Python?"
    assert span.attributes["gen_ai.completion.0.role"] == "assistant"
    assert (
        span.attributes["gen_ai.completion.0.content"]
        == "Python was created by Guido van Rossum."
    )


def test_haystack_parent_span_processor_promotes_string_replies():
    processor = _HaystackParentSpanProcessor()
    span = _FakeSpan(
        "OpenAIGenerator.run",
        "4000000000000004",
        attributes={
            "haystack.component.input": json.dumps({"prompt": "Say hi"}),
            "haystack.component.output": json.dumps({"replies": ["Hi"]}),
        },
    )

    processor.on_start(span)
    processor.on_end(span)

    assert json.loads(span.attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT]) == [
        {"role": "user", "content": "Say hi"}
    ]
    assert json.loads(span.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]) == {
        "role": "assistant",
        "content": "Hi",
    }


def test_haystack_parent_span_processor_uses_pipeline_graph_parent():
    processor = _HaystackParentSpanProcessor()
    pipeline_context = _instrumentation._HaystackPipelineRunContext(
        graph=_FakeGraph(
            {
                "prompt_builder": ("ranker", "query_router"),
                "llm": ("prompt_builder",),
            }
        )
    )
    pipeline = _FakeSpan(
        HAYSTACK_PIPELINE_RUN_METHOD_SPAN_NAME,
        "1000000000000001",
    )
    native_ranker = _FakeSpan(
        HAYSTACK_COMPONENT_RUN_SPAN_NAME,
        "2000000000000002",
        pipeline.get_span_context(),
    )
    ranker = _FakeSpan(
        "LostInTheMiddleRanker.run",
        "3000000000000003",
        native_ranker.get_span_context(),
    )
    native_router = _FakeSpan(
        HAYSTACK_COMPONENT_RUN_SPAN_NAME,
        "4000000000000004",
        pipeline.get_span_context(),
    )
    query_router = _FakeSpan(
        "ConditionalRouter.run",
        "5000000000000005",
        native_router.get_span_context(),
    )
    native_prompt_builder = _FakeSpan(
        HAYSTACK_COMPONENT_RUN_SPAN_NAME,
        "6000000000000006",
        pipeline.get_span_context(),
    )
    prompt_builder = _FakeSpan(
        "PromptBuilder.run",
        "7000000000000007",
        native_prompt_builder.get_span_context(),
    )
    native_llm = _FakeSpan(
        HAYSTACK_COMPONENT_RUN_SPAN_NAME,
        "8000000000000008",
        pipeline.get_span_context(),
    )
    llm = _FakeSpan(
        "OfflineProviderGenerator.run",
        "9000000000000009",
        native_llm.get_span_context(),
    )

    processor.on_start(pipeline)
    _start_span_for_component(
        processor,
        native_ranker,
        component_name="ranker",
        pipeline_context=pipeline_context,
    )
    _start_span_for_component(
        processor,
        ranker,
        component_name="ranker",
        pipeline_context=pipeline_context,
    )
    processor.on_end(ranker)
    processor.on_end(native_ranker)
    _start_span_for_component(
        processor,
        native_router,
        component_name="query_router",
        pipeline_context=pipeline_context,
    )
    _start_span_for_component(
        processor,
        query_router,
        component_name="query_router",
        pipeline_context=pipeline_context,
    )
    processor.on_end(query_router)
    processor.on_end(native_router)
    _start_span_for_component(
        processor,
        native_prompt_builder,
        component_name="prompt_builder",
        pipeline_context=pipeline_context,
    )
    _start_span_for_component(
        processor,
        prompt_builder,
        component_name="prompt_builder",
        pipeline_context=pipeline_context,
    )
    processor.on_end(prompt_builder)
    processor.on_end(native_prompt_builder)
    _start_span_for_component(
        processor,
        native_llm,
        component_name="llm",
        pipeline_context=pipeline_context,
    )
    _start_span_for_component(
        processor,
        llm,
        component_name="llm",
        pipeline_context=pipeline_context,
    )
    processor.on_end(llm)
    processor.on_end(native_llm)

    assert ranker.parent is pipeline.get_span_context()
    assert query_router.parent is pipeline.get_span_context()
    assert prompt_builder.parent is query_router.get_span_context()
    assert llm.parent is prompt_builder.get_span_context()


def test_haystack_parent_span_processor_uses_graph_for_direct_pipeline_children():
    processor = _HaystackParentSpanProcessor()
    pipeline_context = _instrumentation._HaystackPipelineRunContext(
        graph=_FakeGraph(
            {
                "prompt_builder": ("query_router",),
                "llm": ("prompt_builder",),
            }
        )
    )
    pipeline = _FakeSpan(
        HAYSTACK_PIPELINE_RUN_METHOD_SPAN_NAME,
        "1000000000000001",
    )
    query_router = _FakeSpan(
        "ConditionalRouter.run",
        "2000000000000002",
        pipeline.get_span_context(),
    )
    prompt_builder = _FakeSpan(
        "PromptBuilder.run",
        "3000000000000003",
        pipeline.get_span_context(),
    )
    llm = _FakeSpan(
        "OfflineProviderGenerator.run",
        "4000000000000004",
        pipeline.get_span_context(),
    )

    pipeline_context.pipeline_span_id = "1000000000000001"
    processor.on_start(pipeline)
    _start_span_for_component(
        processor,
        query_router,
        component_name="query_router",
        pipeline_context=pipeline_context,
    )
    processor.on_end(query_router)
    _start_span_for_component(
        processor,
        prompt_builder,
        component_name="prompt_builder",
        pipeline_context=pipeline_context,
    )
    processor.on_end(prompt_builder)
    _start_span_for_component(
        processor,
        llm,
        component_name="llm",
        pipeline_context=pipeline_context,
    )
    processor.on_end(llm)

    assert query_router.parent is pipeline.get_span_context()
    assert prompt_builder.parent is query_router.get_span_context()
    assert llm.parent is prompt_builder.get_span_context()


def test_register_haystack_parent_processor_before_export_processors(monkeypatch):
    class OpenInferenceTranslator:
        pass

    class FakeExporterProcessor:
        pass

    class FakeActiveSpanProcessor:
        def __init__(self):
            self._span_processors = (
                OpenInferenceTranslator(),
                FakeExporterProcessor(),
            )

    fake_active_span_processor = FakeActiveSpanProcessor()
    fake_tracer_provider = SimpleNamespace(
        _active_span_processor=fake_active_span_processor
    )
    parent_processor = _HaystackParentSpanProcessor()
    monkeypatch.setattr(
        _instrumentation.trace,
        "get_tracer_provider",
        lambda: fake_tracer_provider,
    )

    _register_haystack_parent_processor(parent_processor)

    assert fake_active_span_processor._span_processors[1] is parent_processor

    _remove_haystack_parent_processor(parent_processor)

    assert parent_processor not in fake_active_span_processor._span_processors


def test_activate_uses_openinference_haystack(monkeypatch):
    fake = _install_fake_modules(monkeypatch)

    instrumentor = HaystackInstrumentor()
    instrumentor.activate()

    delegate = fake.openinference_instrumentor_class.created[0]
    assert delegate.instrumentor_class is fake.haystack_instrumentor_class
    assert delegate.kwargs == {}
    assert delegate.is_activated is True
    assert instrumentor._is_instrumented is True
    assert instrumentor.is_instrumented is True

    instrumentor.deactivate()

    assert delegate.is_deactivated is True
    assert instrumentor._is_instrumented is False


def test_activate_passes_custom_openinference_kwargs(monkeypatch):
    fake = _install_fake_modules(monkeypatch)

    instrumentor = HaystackInstrumentor(trace_content=False)
    instrumentor.activate()

    delegate = fake.openinference_instrumentor_class.created[0]
    assert delegate.kwargs == {"trace_content": False}

    instrumentor.deactivate()


def test_reactivate_does_not_duplicate_pipeline_context_wrappers(monkeypatch):
    class FakeAsyncPipeline:
        pass

    class FakePipeline:
        pass

    _install_fake_modules(monkeypatch)
    monkeypatch.setattr(_instrumentation, "_PIPELINE_CONTEXT_PATCH_APPLIED", False)

    wrapped_methods = []

    def fake_wrap_function_wrapper(target, name, wrapper):
        wrapped_methods.append((target, name, wrapper))

    openinference_haystack_module = sys.modules[OPENINFERENCE_HAYSTACK_MODULE]
    openinference_haystack_module.wrap_function_wrapper = fake_wrap_function_wrapper

    async_pipeline_module = ModuleType(HAYSTACK_ASYNC_PIPELINE_MODULE)
    setattr(
        async_pipeline_module,
        HAYSTACK_ASYNC_PIPELINE_CLASS_NAME,
        FakeAsyncPipeline,
    )
    pipeline_module = ModuleType(HAYSTACK_PIPELINE_MODULE)
    setattr(pipeline_module, HAYSTACK_PIPELINE_CLASS_NAME, FakePipeline)
    monkeypatch.setitem(
        sys.modules,
        HAYSTACK_ASYNC_PIPELINE_MODULE,
        async_pipeline_module,
    )
    monkeypatch.setitem(sys.modules, HAYSTACK_PIPELINE_MODULE, pipeline_module)

    instrumentor = HaystackInstrumentor()

    instrumentor.activate()
    instrumentor.deactivate()
    instrumentor.activate()

    assert len(wrapped_methods) == 6

    instrumentor.deactivate()


def test_activate_cleans_up_delegate_when_activation_fails(monkeypatch, caplog):
    fake = _install_fake_modules(monkeypatch)

    def activate_raises(self):
        self.is_activated = True
        raise RuntimeError("boom")

    monkeypatch.setattr(
        fake.openinference_instrumentor_class,
        "activate",
        activate_raises,
    )

    instrumentor = HaystackInstrumentor()
    with caplog.at_level(logging.ERROR):
        instrumentor.activate()

    delegate = fake.openinference_instrumentor_class.created[0]
    assert delegate.is_deactivated is True
    assert instrumentor._delegate is None
    assert instrumentor._is_instrumented is False
    assert "Failed to activate Haystack instrumentation" in caplog.text


def test_activate_skips_when_respan_tracing_is_disabled(monkeypatch, caplog):
    fake = _install_fake_modules(monkeypatch)
    RespanTracer(is_enabled=False)

    instrumentor = HaystackInstrumentor()
    with caplog.at_level(logging.INFO):
        instrumentor.activate()

    assert fake.openinference_instrumentor_class.created == []
    assert instrumentor._is_instrumented is False
    assert (
        "Haystack instrumentation skipped because Respan tracing is disabled"
        in caplog.text
    )


def test_activate_logs_warning_when_dependencies_are_missing(monkeypatch, caplog):
    def import_module_raises(module_name):
        if module_name == OPENINFERENCE_HAYSTACK_MODULE:
            raise ImportError(module_name)
        raise AssertionError(f"unexpected import: {module_name}")

    monkeypatch.setattr(
        _instrumentation.importlib,
        "import_module",
        import_module_raises,
    )
    instrumentor = HaystackInstrumentor()

    with caplog.at_level(logging.WARNING):
        instrumentor.activate()

    assert "Failed to activate Haystack instrumentation" in caplog.text
    assert instrumentor._is_instrumented is False
