"""Released native CPU model execution; no network, fake vendor, or weights download."""

import gc
import inspect
import json
import threading

import pytest
import respan_instrumentation_huggingface._instrumentation as module
from opentelemetry import context, trace
from opentelemetry.context import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.sdk.trace import SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.sampling import ALWAYS_OFF
from opentelemetry.semconv_ai import SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY
from opentelemetry.semconv_ai import SpanAttributes as AI
from respan_instrumentation_huggingface import HuggingFaceInstrumentor
from respan_instrumentation_huggingface._serialization import json_value, redact_text
from respan_sdk.constants.span_attributes import RESPAN_METADATA
from respan_tracing.constants.context_constants import ENABLE_CONTENT_TRACING_KEY
from transformers import GenerationConfig, TextGenerationPipeline, TextIteratorStreamer
from transformers.pipelines.pt_utils import PipelineIterator


@pytest.fixture
def recorded():
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    instrumentor = HuggingFaceInstrumentor(tracer_provider=provider)
    instrumentor.activate()
    assert instrumentor._is_instrumented
    try:
        yield provider, exporter, instrumentor
    finally:
        instrumentor.deactivate()
        provider.shutdown()


def invoke(pipeline, prompt="Tracing Hugging Face", **kwargs):
    return pipeline(prompt, max_new_tokens=1, do_sample=False, **kwargs)


def attrs(exporter):
    return dict(exporter.get_finished_spans()[-1].attributes)


def private(attributes):
    assert AI.TRACELOOP_ENTITY_INPUT not in attributes
    assert AI.TRACELOOP_ENTITY_OUTPUT not in attributes
    assert RESPAN_METADATA + ".huggingface.request" not in attributes
    assert not any(key.endswith(".content") for key in attributes)
    assert "error.message" not in attributes


@pytest.mark.parametrize(
    "prompt,options",
    [
        ("Tracing Hugging Face", {}),
        (["Tracing Hugging", "Face native"], {}),
        ("Tracing Hugging", {"num_return_sequences": 2, "num_beams": 2}),
        (
            [
                {"role": "system", "content": "native"},
                {"role": "user", "content": "Tracing Hugging"},
            ],
            {},
        ),
        (
            [
                [{"role": "user", "content": "Tracing"}],
                [{"role": "user", "content": "Face"}],
            ],
            {},
        ),
        ("Tracing Hugging", {"return_tensors": True}),
        ("", {}),
    ],
)
def test_complete_native_results(pipeline, recorded, prompt, options):
    _, exporter, _ = recorded
    response = invoke(pipeline, prompt, **options)
    value = attrs(exporter)
    assert json.loads(value[AI.TRACELOOP_ENTITY_INPUT]) == prompt
    assert json.loads(value[AI.TRACELOOP_ENTITY_OUTPUT]) == response
    assert value[AI.LLM_SYSTEM] == "huggingface"
    assert not any("tokens" in key and key.startswith("gen_ai.usage") for key in value)


def test_native_response_identity(pipeline, recorded, monkeypatch):
    responses = []
    original = pipeline.postprocess

    def postprocess(*args, **kwargs):
        result = original(*args, **kwargs)
        responses.append(result)
        return result

    monkeypatch.setattr(pipeline, "postprocess", postprocess)
    assert invoke(pipeline) is responses[0]


def test_long_chat_and_run_marker_under_native_attribute_limit(pipeline, recorded):
    provider, exporter, _ = recorded

    class Marker(SpanProcessor):
        def on_start(self, span, parent_context=None):
            span.set_attribute(RESPAN_METADATA, json.dumps({"run_id": "native-75"}))

    provider.add_span_processor(Marker())
    prompt = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": "Tracing " + str(i)}
        for i in range(75)
    ]
    response = invoke(pipeline, prompt)
    value = attrs(exporter)
    assert json.loads(value[AI.TRACELOOP_ENTITY_INPUT]) == prompt
    assert json.loads(value[AI.TRACELOOP_ENTITY_OUTPUT]) == response
    assert json.loads(value[RESPAN_METADATA])["run_id"] == "native-75"


def test_native_5001_token_result(pipeline, recorded):
    response = invoke(pipeline, " ".join(["Tracing"] * 5000), return_tensors=True)
    assert len(response[0]["generated_token_ids"]) == 5001
    assert json.loads(attrs(recorded[1])[AI.TRACELOOP_ENTITY_OUTPUT]) == response


def test_per_call_native_generation_config(pipeline, recorded):
    config = GenerationConfig(
        max_new_tokens=1,
        do_sample=False,
        temperature=0.0,
        top_p=0.75,
        repetition_penalty=1.2,
        pad_token_id=0,
        eos_token_id=2,
    )
    response = pipeline(text_inputs="Tracing", generation_config=config)
    value = attrs(recorded[1])
    native = json.loads(value[RESPAN_METADATA + ".huggingface.request"])
    assert native["call"]["generation_config"]["top_p"] == 0.75
    assert json.loads(value[AI.TRACELOOP_ENTITY_OUTPUT]) == response


def test_lazy_generator_exact_iterator_no_eager_consumption(pipeline, recorded):
    consumed = []

    def prompts():
        for prompt in ("Tracing", "Face"):
            consumed.append(prompt)
            yield prompt

    instrumentor = recorded[2]
    instrumentor.deactivate()
    baseline = invoke(pipeline, prompts())
    native_initial = list(consumed)
    del baseline
    consumed.clear()
    instrumentor.activate()
    iterator = invoke(pipeline, prompts())
    assert type(iterator) is PipelineIterator
    assert consumed == native_initial
    assert recorded[1].get_finished_spans() == ()
    result = list(iterator)
    assert consumed == ["Tracing", "Face"]
    value = attrs(recorded[1])
    assert json.loads(value[AI.TRACELOOP_ENTITY_INPUT]) == consumed
    assert json.loads(value[AI.TRACELOOP_ENTITY_OUTPUT]) == result


def test_two_pending_sibling_iterators(pipeline, recorded):
    provider, exporter, _ = recorded
    with provider.get_tracer("native-test").start_as_current_span("parent"):
        first = invoke(pipeline, (p for p in ["Tracing"]))
        second = invoke(pipeline, (p for p in ["Face"]))
        assert list(first)
        assert list(second)
    children = [s for s in exporter.get_finished_spans() if s.name.endswith(".call")]
    assert len(children) == 2
    assert all(AI.TRACELOOP_ENTITY_OUTPUT in s.attributes for s in children)
    assert len({s.parent.span_id for s in children}) == 1


def test_native_streamer_callback(pipeline, recorded):
    streamer = TextIteratorStreamer(pipeline.tokenizer, skip_prompt=True, timeout=10)
    result = []
    worker = threading.Thread(
        target=lambda: result.append(invoke(pipeline, streamer=streamer))
    )
    worker.start()
    pieces = list(streamer)
    worker.join(10)
    assert not worker.is_alive()
    assert pieces
    assert json.loads(attrs(recorded[1])[AI.TRACELOOP_ENTITY_OUTPUT]) == result[0]


@pytest.mark.parametrize(
    "flag",
    [ENABLE_CONTENT_TRACING_KEY, "trace_content", "override_enable_content_tracing"],
)
def test_initial_canonical_and_legacy_veto(pipeline, recorded, flag):
    token = context.attach(context.set_value(flag, False))
    try:
        assert invoke(pipeline)
    finally:
        context.detach(token)
    private(attrs(recorded[1]))


@pytest.mark.parametrize("name", ["RESPAN_TRACE_CONTENT", "TRACELOOP_TRACE_CONTENT"])
@pytest.mark.parametrize("off", ["false", "0", "NO", "off"])
def test_environment_veto(pipeline, recorded, monkeypatch, name, off):
    monkeypatch.setenv(name, off)
    assert invoke(pipeline)
    private(attrs(recorded[1]))


def test_supplied_context_cannot_widen_ambient(pipeline):
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    instrumentor = HuggingFaceInstrumentor(
        tracer_provider=provider,
        context=context.set_value(ENABLE_CONTENT_TRACING_KEY, True),
    )
    instrumentor.activate()
    token = context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, False))
    try:
        invoke(pipeline)
    finally:
        context.detach(token)
        instrumentor.deactivate()
    private(attrs(exporter))


@pytest.mark.parametrize("phase", ["active", "finished", "detach", "environment"])
def test_late_ancestor_veto(pipeline, recorded, monkeypatch, phase):
    provider, exporter, _ = recorded
    parent = provider.get_tracer("native-test").start_span("parent")
    token = context.attach(trace.set_span_in_context(parent))
    try:
        iterator = invoke(pipeline, (p for p in ["Tracing", "Face"]))
        next(iter(iterator))
        if phase in ("active", "finished"):
            parent.set_attribute(ENABLE_CONTENT_TRACING_KEY, False)
            if phase == "finished":
                parent.end()
        elif phase == "detach":
            veto = context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, False))
            context.detach(veto)
        else:
            monkeypatch.setenv("RESPAN_TRACE_CONTENT", "off")
        list(iterator)
    finally:
        context.detach(token)
        parent.end()
    child = next(s for s in exporter.get_finished_spans() if s.name.endswith(".call"))
    private(child.attributes)


@pytest.mark.parametrize("finished", [False, True])
def test_unknown_local_parent_fails_closed(pipeline, finished):
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    parent = provider.get_tracer("unknown").start_span("unknown")
    if finished:
        parent.end()
    instrumentor = HuggingFaceInstrumentor(tracer_provider=provider)
    instrumentor.activate()
    token = context.attach(trace.set_span_in_context(parent))
    try:
        invoke(pipeline)
    finally:
        context.detach(token)
        parent.end()
        instrumentor.deactivate()
    private(attrs(exporter))


@pytest.mark.parametrize(
    "key", [_SUPPRESS_INSTRUMENTATION_KEY, SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY]
)
def test_native_suppression(pipeline, recorded, key):
    token = context.attach(context.set_value(key, True))
    try:
        assert invoke(pipeline)
    finally:
        context.detach(token)
    assert recorded[1].get_finished_spans() == ()


def test_sampling_before_content_extraction(pipeline, monkeypatch):
    provider = TracerProvider(sampler=ALWAYS_OFF)
    instrumentor = HuggingFaceInstrumentor(tracer_provider=provider)
    instrumentor.activate()

    def forbidden(*args, **kwargs):
        raise AssertionError("No serialization on unsampled native calls")

    monkeypatch.setattr(module, "json_value", forbidden)
    try:
        assert invoke(pipeline)
    finally:
        instrumentor.deactivate()


def test_native_error_diagnostics_and_redaction(pipeline, recorded):
    with pytest.raises(ValueError):
        invoke(pipeline, native_unused_keyword="token=private")
    span = recorded[1].get_finished_spans()[-1]
    assert span.status.status_code.name == "ERROR"
    assert span.attributes["error.type"] == "ValueError"
    assert "private" not in span.attributes.get("error.message", "")
    assert not span.events


@pytest.mark.parametrize("method", ["start", "preprocess", "consume", "finish"])
def test_observer_fault_preserves_native(pipeline, recorded, monkeypatch, method):
    def fault(*args, **kwargs):
        raise RuntimeError("telemetry failure")

    monkeypatch.setattr(module._State, method, fault)
    if method == "consume":
        assert list(invoke(pipeline, (p for p in ["Tracing"])))
    else:
        assert invoke(pipeline)
    assert not module._MANAGER.observer.states


def test_startup_and_end_fault_preserve_native(pipeline, recorded, monkeypatch):
    manager = module._MANAGER
    monkeypatch.setattr(
        manager,
        "native_tracer",
        lambda *args: (_ for _ in ()).throw(RuntimeError("startup")),
    )
    assert invoke(pipeline)
    assert not manager.observer.states


def test_lazy_abandonment_finishes_bodyless(pipeline, recorded):
    iterator = invoke(pipeline, (p for p in ["Tracing"]))
    del iterator
    gc.collect()
    assert len(recorded[1].get_finished_spans()) == 1


def test_shared_owner_conflict_and_identity_restoration(pipeline, recorded):
    provider, exporter, first = recorded
    owned = inspect.getattr_static(TextGenerationPipeline, "__call__")
    second = HuggingFaceInstrumentor(tracer_provider=provider)
    second.activate()
    first.deactivate()
    assert inspect.getattr_static(TextGenerationPipeline, "__call__") is owned
    assert invoke(pipeline)
    assert len(exporter.get_finished_spans()) == 1
    conflict = HuggingFaceInstrumentor(tracer_provider=provider, capture_content=False)
    with pytest.raises(RuntimeError, match="conflict"):
        conflict.activate()
    second.deactivate()
    assert inspect.getattr_static(TextGenerationPipeline, "__call__") is not owned


def test_foreign_wrapper_and_processor_retained(pipeline, recorded, monkeypatch):
    provider, _, instrumentor = recorded
    owned = TextGenerationPipeline.__call__

    def foreign(instance, *args, **kwargs):
        return owned(instance, *args, **kwargs)

    monkeypatch.setattr(TextGenerationPipeline, "__call__", foreign)
    foreign_processor = SpanProcessor()
    provider.add_span_processor(foreign_processor)
    instrumentor.deactivate()
    assert TextGenerationPipeline.__call__ is foreign
    assert foreign_processor in provider._active_span_processor._span_processors
    assert invoke(pipeline)


def test_escaped_credentials_schema_and_unknown_hooks():
    encoded = '{"token":"escaped \\" private", "enabled":false, "zero":0, "empty":[]}'
    value = json.loads(redact_text(encoded))
    assert value == {"token": "[REDACTED]", "enabled": False, "zero": 0, "empty": []}
    assert redact_text(redact_text(encoded)) == redact_text(encoded)
    schema = {
        "properties": {
            "token": {"type": "string", "default": "private"},
            "name": {"default": "native"},
        }
    }
    assert json_value(schema)["properties"]["token"] == {
        "type": "string",
        "default": "[REDACTED]",
    }
    assert "private" not in redact_text(
        "Bearer private Basic private https://u:private@host/x?token=private token=private"
    )

    class Unknown:
        def __iter__(self):
            raise AssertionError("user iterator")

        def __str__(self):
            raise AssertionError("user string")

        def model_dump(self):
            raise AssertionError("user conversion")

    assert json_value({"extra": Unknown()}) == {"extra": None}


def test_empty_native_input_error_is_preserved(pipeline, recorded):
    instrumentor = recorded[2]
    instrumentor.deactivate()
    try:
        invoke(pipeline, [])
    except Exception as native:  # noqa: BLE001
        native_type = type(native)
        native_args = native.args
    else:
        native_type = None
    instrumentor.activate()
    if native_type is None:
        assert invoke(pipeline, []) == []
    else:
        with pytest.raises(native_type) as captured:
            invoke(pipeline, [])
        assert captured.value.args == native_args


def test_native_chat_continuation_floor(pipeline, recorded):
    history = [
        {"role": "user", "content": "Tracing"},
        {"role": "assistant", "content": "Face"},
    ]
    response = invoke(pipeline, history, continue_final_message=True)
    assert len(response[0]["generated_text"]) == 2
    assert json.loads(attrs(recorded[1])[AI.TRACELOOP_ENTITY_OUTPUT]) == response


def test_native_partial_iterator_error_identity(pipeline, recorded):
    expected = ValueError("native generator error token=private")

    def prompts():
        yield "Tracing"
        raise expected

    iterator = invoke(pipeline, prompts())
    first = next(iter(iterator))
    with pytest.raises(ValueError) as caught:
        next(iterator)
    assert caught.value is expected
    value = attrs(recorded[1])
    assert json.loads(value[AI.TRACELOOP_ENTITY_OUTPUT]) == [first]
    assert "private" not in value.get("error.message", "")


def test_late_provider_and_capture_disabled(pipeline, monkeypatch):
    monkeypatch.setattr(trace, "_TRACER_PROVIDER", None)
    instrumentor = HuggingFaceInstrumentor(capture_content=False)
    instrumentor.activate()
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(trace, "_TRACER_PROVIDER", provider)
    try:
        invoke(pipeline)
    finally:
        instrumentor.deactivate()
    private(attrs(exporter))


@pytest.mark.parametrize("method", ["set_attribute", "end"])
def test_native_span_fault_preserves_result(pipeline, recorded, monkeypatch, method):
    from opentelemetry.sdk.trace import _Span

    def fault(*args, **kwargs):
        raise RuntimeError("telemetry failure")

    monkeypatch.setattr(_Span, method, fault)
    assert invoke(pipeline)
    assert not module._MANAGER.observer.states


def test_partial_activation_rollback(pipeline, monkeypatch):
    import opentelemetry.instrumentation.transformers as upstream

    original = inspect.getattr_static(TextGenerationPipeline, "__call__")
    actual = upstream.TransformersInstrumentor._instrument

    def fault(self, **kwargs):
        actual(self, **kwargs)
        raise RuntimeError("after native patch")

    monkeypatch.setattr(upstream.TransformersInstrumentor, "_instrument", fault)
    instrumentor = HuggingFaceInstrumentor(tracer_provider=TracerProvider())
    instrumentor.activate()
    assert not instrumentor._is_instrumented
    assert inspect.getattr_static(TextGenerationPipeline, "__call__") is original
    assert not upstream.TransformersInstrumentor.__new__(
        upstream.TransformersInstrumentor
    ).is_instrumented_by_opentelemetry
    assert invoke(pipeline)


def test_actual_foreign_upstream_instrumentor_retained(pipeline):
    import opentelemetry.instrumentation.transformers as upstream

    foreign = upstream.TransformersInstrumentor()
    foreign.instrument(tracer_provider=TracerProvider())
    wrapped = inspect.getattr_static(TextGenerationPipeline, "__call__")
    own = HuggingFaceInstrumentor()
    try:
        own.activate()
        assert not own._is_instrumented
        own.deactivate()
        assert inspect.getattr_static(TextGenerationPipeline, "__call__") is wrapped
        assert invoke(pipeline)
    finally:
        foreign.uninstrument()


def test_native_current_tools_schema(pipeline, recorded):
    if (
        "tools"
        not in inspect.signature(TextGenerationPipeline._sanitize_parameters).parameters
    ):
        pytest.skip("Native pipeline tools unavailable on exact supported minimum")
    tools = [
        {
            "type": "function",
            "function": {
                "name": "actual_tool",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "token": {"type": "string", "default": "private"},
                        "zero": {"const": 0},
                        "false": {"const": False},
                    },
                    "required": [],
                },
            },
        }
    ]
    response = invoke(pipeline, [{"role": "user", "content": "Tracing"}], tools=tools)
    value = attrs(recorded[1])
    assert json.loads(value[AI.TRACELOOP_ENTITY_OUTPUT]) == response
    retained = json.loads(value[AI.LLM_REQUEST_FUNCTIONS])
    assert retained == json_value(tools)
    assert (
        retained[0]["function"]["parameters"]["properties"]["token"]["default"]
        == "[REDACTED]"
    )


def test_source_text_convenience_projection(pipeline, recorded):
    response = invoke(pipeline, "Tracing")
    value = attrs(recorded[1])
    assert value[AI.LLM_PROMPTS + ".0.content"] == "Tracing"
    assert value[AI.LLM_COMPLETIONS + ".0.content"] == response[0]["generated_text"]
    response = invoke(pipeline, [{"role": "user", "content": "Tracing"}])
    assert (
        attrs(recorded[1])[AI.LLM_COMPLETIONS + ".0.content"]
        == response[0]["generated_text"][-1]["content"]
    )


def test_credential_aliases_and_native_config_only():
    value = json_value(
        {
            "credentials": "private",
            "private_key": "private",
            "nested": {"refresh_token": "private"},
        }
    )
    assert value == {
        "credentials": "[REDACTED]",
        "private_key": "[REDACTED]",
        "nested": {"refresh_token": "[REDACTED]"},
    }
    assert (
        redact_text("credentials=private private_key=private")
        == "credentials=[REDACTED] private_key=[REDACTED]"
    )

    class Spoofed:
        __module__ = "transformers.customer"

        @property
        def __dict__(self):
            raise AssertionError("Unknown storage hook")

    assert json_value(Spoofed()) is None


def test_actual_json_prompt_native_bytes(pipeline, recorded):
    prompt = '{ "enabled" : false, "items" : [0, 1] }'
    response = invoke(pipeline, prompt)
    value = attrs(recorded[1])
    assert json.loads(value[AI.TRACELOOP_ENTITY_INPUT]) == prompt
    assert value[AI.LLM_PROMPTS + ".0.content"] == prompt
    assert json.loads(value[AI.TRACELOOP_ENTITY_OUTPUT]) == response


def test_actual_native_model_reference_credentials(pipeline, recorded, monkeypatch):
    monkeypatch.setattr(
        pipeline.model.config,
        "_name_or_path",
        "https://user:credential@host/model?token=credential",
    )
    assert invoke(pipeline)
    model = attrs(recorded[1])[AI.LLM_REQUEST_MODEL]
    assert "credential" not in model and "user:" not in model
    assert model.startswith("https://host/model")


@pytest.mark.parametrize("mode", ["mutate_raise", "transient_veto", "scrub_fault"])
def test_native_content_setter_fault_and_finishing_veto(
    pipeline, recorded, monkeypatch, mode
):
    from opentelemetry.sdk.trace import _Span

    original = _Span.set_attribute

    def setter(span, key, value):
        if key == AI.TRACELOOP_ENTITY_INPUT:
            if mode == "transient_veto":
                token = context.attach(
                    context.set_value(ENABLE_CONTENT_TRACING_KEY, False)
                )
                context.detach(token)
                return original(span, key, value)
            original(span, key, value)
            raise RuntimeError("native setter observation fault")
        return original(span, key, value)

    if mode == "scrub_fault":

        def scrub_fault(*args):
            raise RuntimeError("scrub fault")

        monkeypatch.setattr(module._State, "scrub", scrub_fault)
    monkeypatch.setattr(_Span, "set_attribute", setter)
    assert invoke(pipeline)
    private(attrs(recorded[1]))
    assert not module._MANAGER.observer.states


def test_native_runtime_descriptor_presence_restored(pipeline):
    runtime = context._RUNTIME_CONTEXT
    present = "detach" in runtime.__dict__
    stored = runtime.__dict__.get("detach")
    original = runtime.detach
    instrumentor = HuggingFaceInstrumentor(tracer_provider=TracerProvider())
    instrumentor.activate()
    instrumentor.deactivate()
    assert ("detach" in runtime.__dict__) is present
    if present:
        assert runtime.__dict__["detach"] is stored
    else:
        assert runtime.detach.__func__ is original.__func__
        assert runtime.detach.__self__ is original.__self__
    assert invoke(pipeline)


def test_native_tracer_kind_links_attributes_and_start_time(recorded):
    from opentelemetry.trace import Link, SpanContext, SpanKind, TraceFlags
    from respan_instrumentation_huggingface._constants import TRANSFORMERS_SCOPE_NAME

    tracer = module._Tracer(
        module._MANAGER, (TRANSFORMERS_SCOPE_NAME, "native-probe"), {}
    )
    link = Link(SpanContext(123, 456, True, TraceFlags(1)))
    with tracer.start_as_current_span(
        "native_options",
        kind=SpanKind.CLIENT,
        attributes={"native_flag": False},
        links=[link],
        start_time=123456789,
    ) as span:
        assert span.kind is SpanKind.CLIENT
        assert span.start_time == 123456789
        assert span.attributes["native_flag"] is False
        assert span.links[0].context is link.context
    assert recorded[1].get_finished_spans()[0].kind is SpanKind.CLIENT


def test_native_logger_equality_and_extra_options_conflict(pipeline, recorded):
    class Logger:
        def __call__(self, error):
            return None

        def __eq__(self, other):
            raise AssertionError("unknown logger equality")

    with pytest.raises(RuntimeError, match="conflict"):
        HuggingFaceInstrumentor(
            tracer_provider=recorded[0], exception_logger=Logger()
        ).activate()
    with pytest.raises(RuntimeError, match="conflict"):
        HuggingFaceInstrumentor(
            tracer_provider=recorded[0], skip_dep_check=True
        ).activate()
    assert invoke(pipeline)


def test_plain_properties_and_sensitive_example_are_redacted():
    assert json_value({"properties": {"api_key": "PRIVATE"}}) == {
        "properties": {"api_key": "[REDACTED]"}
    }
    schema = {
        "type": "object",
        "properties": {
            "api_key": {"type": "string", "example": "PRIVATE", "default": "PRIVATE"},
            "count": {"type": "integer", "default": 0},
        },
    }
    retained = json_value(schema)
    assert retained["properties"]["api_key"] == {
        "type": "string",
        "example": "[REDACTED]",
        "default": "[REDACTED]",
    }
    assert retained["properties"]["count"]["default"] == 0


def test_native_pipeline_subclass_storage_getter_is_not_observed(pipeline, recorded):
    from respan_instrumentation_huggingface._serialization import native_storage
    from transformers.pipelines.base import Pipeline

    class CustomPipeline(TextGenerationPipeline):
        @property
        def __dict__(self):
            raise AssertionError("unknown pipeline storage getter")

    custom = object.__new__(CustomPipeline)
    native_storage(custom, Pipeline).update(native_storage(pipeline, Pipeline))
    response = invoke(custom)
    assert json.loads(attrs(recorded[1])[AI.TRACELOOP_ENTITY_OUTPUT]) == response


def test_native_model_config_storage_does_not_add_getter_calls(pipeline, recorded):
    from respan_instrumentation_huggingface._serialization import native_storage
    from transformers import PretrainedConfig, PreTrainedModel
    from transformers.pipelines.base import Pipeline

    model_base = type(pipeline.model)
    config_base = type(pipeline.model.config)

    class CustomModel(model_base):
        calls = 0

        @property
        def __dict__(self):
            type(self).calls += 1
            return native_storage(self, PreTrainedModel)

    class CustomConfig(config_base):
        calls = 0

        @property
        def __dict__(self):
            type(self).calls += 1
            return native_storage(self, PretrainedConfig)

    model = object.__new__(CustomModel)
    native_storage(model, PreTrainedModel).update(
        native_storage(pipeline.model, PreTrainedModel)
    )
    config = object.__new__(CustomConfig)
    native_storage(config, PretrainedConfig).update(
        native_storage(pipeline.model.config, PretrainedConfig)
    )
    native_storage(model, PreTrainedModel)["config"] = config
    custom = object.__new__(TextGenerationPipeline)
    native_storage(custom, Pipeline).update(native_storage(pipeline, Pipeline))
    native_storage(custom, Pipeline)["model"] = model
    instrumentor = recorded[2]
    instrumentor.deactivate()
    invoke(custom)
    CustomModel.calls = CustomConfig.calls = 0
    baseline = invoke(custom)
    counts = (CustomModel.calls, CustomConfig.calls)
    CustomModel.calls = CustomConfig.calls = 0
    instrumentor.activate()
    response = invoke(custom)
    assert response == baseline
    assert (CustomModel.calls, CustomConfig.calls) == counts
    assert attrs(recorded[1])[AI.LLM_REQUEST_MODEL] == "controlled-local-gpt2"


def test_native_success_and_unconsumed_abandonment_status(pipeline, recorded):
    invoke(pipeline)
    assert recorded[1].get_finished_spans()[-1].status.status_code.name == "OK"
    assert list(invoke(pipeline, (p for p in ["Tracing"])))
    assert recorded[1].get_finished_spans()[-1].status.status_code.name == "OK"
    iterator = invoke(pipeline, (p for p in ["Tracing"]))
    del iterator
    gc.collect()
    span = recorded[1].get_finished_spans()[-1]
    assert span.status.status_code.name == "UNSET"
    assert AI.TRACELOOP_ENTITY_INPUT not in span.attributes
    assert AI.TRACELOOP_ENTITY_OUTPUT not in span.attributes


@pytest.mark.parametrize(
    "authorization",
    [
        'Bearer "P13_CONTROLLED_QUOTED_CREDENTIAL"',
        "Basic 'P13_CONTROLLED_QUOTED_CREDENTIAL'",
    ],
)
def test_native_quoted_authorization_credentials(pipeline, recorded, authorization):
    prompt = json.dumps({"note": "Authorization: " + authorization})
    response = invoke(pipeline, prompt)
    assert response[0]["generated_text"].startswith(prompt)
    value = attrs(recorded[1])
    assert "P13_CONTROLLED_QUOTED_CREDENTIAL" not in json.dumps(value)
    assert redact_text(redact_text(prompt)) == redact_text(prompt)
