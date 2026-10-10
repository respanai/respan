"""Released Ragas APIs and native OTel lifecycles; no vendor replacements."""

import asyncio
import gc
import inspect
import json
from dataclasses import dataclass, field

import numpy as np
import pytest
import ragas
from opentelemetry import context, trace
from opentelemetry.sdk.trace import Span, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.sampling import ALWAYS_OFF
from opentelemetry.semconv_ai import SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY
from opentelemetry.trace import NonRecordingSpan, SpanContext, StatusCode, TraceFlags
from pydantic import BaseModel, ConfigDict
from ragas import EvaluationDataset, MultiTurnSample, SingleTurnSample
from ragas.backends import InMemoryBackend
from ragas.dataset import Dataset
from ragas.executor import Executor
from ragas.messages import AIMessage, HumanMessage
from ragas.metrics import discrete_metric, numeric_metric, ranking_metric
from ragas.metrics.base import MultiTurnMetric
from ragas.metrics.collections import ExactMatch, StringPresence
from ragas.metrics.result import MetricResult
from respan_instrumentation_ragas import RagasInstrumentor
from respan_instrumentation_ragas import _instrumentation as impl
from respan_sdk.constants.span_attributes import RESPAN_LOG_TYPE
from respan_tracing.constants.context_constants import ENABLE_CONTENT_TRACING_KEY

INPUT = "traceloop.entity.input"
OUTPUT = "traceloop.entity.output"


def payload(span, field=OUTPUT):
    return json.loads(span.attributes[field])


def assert_bodyless(spans):
    for span in spans:
        assert INPUT not in span.attributes
        assert OUTPUT not in span.attributes
        assert "error.message" not in span.attributes
        assert span.status.description is None
        assert not span.events


def dataset():
    return EvaluationDataset.from_list(
        [
            {"response": "a", "reference": "a"},
            {"response": "b", "reference": "a"},
        ]
    )


@pytest.mark.parametrize("method", ["score", "ascore", "batch_score", "abatch_score"])
def test_native_collections_complete(runtime, method):
    metric = ExactMatch()
    if "batch" in method:
        result = getattr(metric, method)([{"reference": "a", "response": "b"}] * 75)
    else:
        result = getattr(metric, method)(reference="a", response="b")
    if inspect.isawaitable(result):
        result = asyncio.run(result)
    spans = runtime[1].get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.status.status_code is StatusCode.OK
    assert span.attributes[RESPAN_LOG_TYPE] == "task"
    assert "status_code" not in span.attributes
    assert "gen_ai.request.model" not in span.attributes
    assert "gen_ai.usage.total_tokens" not in span.attributes
    if "batch" in method:
        assert len(result) == len(payload(span)) == 75
        assert len(payload(span, INPUT)["args"][0]) == 75
        assert all(item.value == 0 for item in result)
        assert all(item["value"] == 0 for item in payload(span))
    else:
        assert type(result) is MetricResult and result.value == 0
        assert payload(span)["value"] == 0


@pytest.mark.parametrize("kind", ["discrete", "numeric", "ranking"])
def test_future_decorators_and_large_native_result(runtime, kind):
    if kind == "discrete":

        @discrete_metric(name="new_discrete", allowed_values=["pass", "fail"])
        def metric(response):
            return "pass"

        expected = "pass"
    elif kind == "numeric":

        @numeric_metric(name="new_numeric")
        def metric(response):
            return 0

        expected = 0
    else:
        expected = list(range(5001))

        @ranking_metric(name="new_ranking", allowed_values=5001)
        def metric(response):
            return expected

    result = metric.score(response="native")
    assert result.value == expected
    spans = runtime[1].get_finished_spans()
    assert len(spans) == 1
    assert payload(spans[0])["value"] == expected


def test_native_decorated_result_identity_and_traces(runtime):
    expected = MetricResult(
        0,
        reason="",
        traces={
            "input": {"history": list(range(75))},
            "output": {"vector": np.arange(5001), "flag": False},
        },
    )

    @numeric_metric(name="retained_result")
    def metric(response):
        return expected

    assert metric.score(response="") is expected
    output = payload(runtime[1].get_finished_spans()[0])
    assert output["reason"] == ""
    assert output["traces"]["output"]["flag"] is False
    assert len(output["traces"]["input"]["history"]) == 75
    assert output["traces"]["output"]["vector"] == list(range(5001))


def test_raw_decorator_call_preserves_original_value(runtime):
    @discrete_metric(name="raw", allowed_values=["pass", "fail"])
    def metric(response):
        return "outside_allowed"

    assert metric(response="a") == "outside_allowed"
    assert not runtime[1].get_finished_spans()


@pytest.mark.parametrize("async_call", [False, True])
def test_legacy_native_metric_and_callbacks(runtime, async_call):
    from langchain_core.callbacks import BaseCallbackHandler
    from ragas.metrics import ExactMatch as LegacyExactMatch

    events = []

    class Callback(BaseCallbackHandler):
        def on_chain_end(self, outputs, **kwargs):
            events.append(outputs)

    metric = LegacyExactMatch()
    sample = SingleTurnSample(reference="a", response="a")
    result = (
        metric.single_turn_ascore(sample, callbacks=[Callback()])
        if async_call
        else metric.single_turn_score(sample, callbacks=[Callback()])
    )
    if async_call:
        result = asyncio.run(result)
    assert result == 1.0 and events
    assert len(runtime[1].get_finished_spans()) == 1


def test_future_multi_turn_native_extension(runtime):
    @dataclass
    class Controlled(MultiTurnMetric):
        name: str = "native_conversation"
        _required_columns: dict = field(default_factory=dict)

        def init(self, run_config):
            pass

        async def _multi_turn_ascore(self, sample, callbacks):
            return float(len(sample.user_input))

    sample = MultiTurnSample(
        user_input=[HumanMessage(content="hi"), AIMessage(content="hello")]
    )
    assert asyncio.run(Controlled().multi_turn_ascore(sample)) == 2.0
    body = payload(runtime[1].get_finished_spans()[0], INPUT)
    assert len(body["args"][0]["user_input"]) == 2


@pytest.mark.parametrize("async_call", [False, True])
def test_evaluations_preserve_native_result_and_tree(runtime, async_call):
    from ragas.dataset_schema import EvaluationResult
    from ragas.metrics import ExactMatch as LegacyExactMatch

    args = {
        "dataset": dataset(),
        "metrics": [LegacyExactMatch()],
        "show_progress": False,
    }
    result = (
        asyncio.run(ragas.aevaluate(**args)) if async_call else ragas.evaluate(**args)
    )
    assert type(result) is EvaluationResult
    assert [row["exact_match"] for row in result.scores] == [1.0, 0.0]
    spans = runtime[1].get_finished_spans()
    assert len(spans) == 3
    parent = next(s for s in spans if s.name == "ragas.evaluation")
    assert payload(parent)["scores"] == result.scores
    assert all(
        s.parent.span_id == parent.context.span_id for s in spans if s is not parent
    )


@pytest.mark.parametrize("method", ["results", "aresults"])
def test_deferred_executor_is_lazy_and_preserves_identity(runtime, method):
    from ragas.metrics import ExactMatch as LegacyExactMatch

    result = ragas.evaluate(
        dataset(),
        metrics=[LegacyExactMatch()],
        return_executor=True,
        show_progress=False,
    )
    assert type(result) is Executor
    assert not runtime[1].get_finished_spans()
    assert len(result.jobs) == 2
    values = getattr(result, method)()
    if method == "aresults":
        values = asyncio.run(values)
    assert values == [1.0, 0.0]
    spans = runtime[1].get_finished_spans()
    assert len(spans) == 3
    parent = next(s for s in spans if s.name == "ragas.evaluation")
    assert payload(parent) == values
    assert all(
        s.parent.span_id == parent.context.span_id for s in spans if s is not parent
    )


@pytest.mark.parametrize("action", ["cancel", "abandon", "deactivate"])
def test_unused_executor_finishes_bodyless(runtime, action):
    from ragas.metrics import ExactMatch as LegacyExactMatch

    result = ragas.evaluate(
        dataset(),
        metrics=[LegacyExactMatch()],
        return_executor=True,
        show_progress=False,
    )
    if action == "cancel":
        assert result.cancel() is None and result.is_cancelled()
    elif action == "abandon":
        # Ragas keeps unstarted coroutine jobs on its native Executor. Close these
        # application-owned jobs explicitly; instrumentation never executes them.
        del result
        gc.collect()
    else:
        runtime[2].deactivate()
    spans = runtime[1].get_finished_spans()
    assert len(spans) == 1
    assert_bodyless(spans)


def test_deferred_parent_late_veto_bounds_children(runtime):
    from ragas.metrics import ExactMatch as LegacyExactMatch

    provider, memory, _ = runtime
    parent = provider.get_tracer("app").start_span("parent")
    with trace.use_span(parent):
        result = ragas.evaluate(
            dataset(),
            metrics=[LegacyExactMatch()],
            return_executor=True,
            show_progress=False,
        )
    parent.set_attribute(ENABLE_CONTENT_TRACING_KEY, False)
    parent.end()
    assert result.results() == [1.0, 0.0]
    assert_bodyless([s for s in memory.get_finished_spans() if s.name != "parent"])


@pytest.mark.parametrize("async_callback", [False, True])
def test_experiment_native_rows_and_arun(runtime, async_callback):
    backend = InMemoryBackend()
    rows = Dataset(name="rows", backend=backend)
    rows.append({"value": 0})
    rows.append({"value": 1})
    expected = {"zero": 0, "flag": False, "empty": ""}
    if async_callback:

        @ragas.experiment(backend=backend)
        async def controlled(row):
            return expected
    else:

        @ragas.experiment(backend=backend)
        def controlled(row):
            return expected

    assert asyncio.run(controlled({"value": 0})) is expected
    result = asyncio.run(controlled.arun(rows, name="controlled"))
    assert len(result) == 2
    assert backend.load_experiment("controlled") == [expected, expected]
    spans = runtime[1].get_finished_spans()
    assert len(spans) == 4
    outer = next(s for s in spans if s.name == "ragas.experiment_run")
    assert len(payload(outer)["data"]) == 2
    assert all(s.parent.span_id == outer.context.span_id for s in spans[1:-1])


def test_native_metric_error_has_no_fabricated_output_or_http(runtime):
    with pytest.raises(AssertionError):
        StringPresence().score(reference=1, response="native")
    span = runtime[1].get_finished_spans()[0]
    assert span.status.status_code is StatusCode.ERROR
    assert span.attributes["error.type"] == "AssertionError"
    assert OUTPUT not in span.attributes and "status_code" not in span.attributes


def test_native_decorator_caught_callback_error_stays_result(runtime):
    @numeric_metric(name="caught")
    def metric(response):
        raise RuntimeError("controlled callback")

    result = metric.score(response="a")
    assert type(result) is MetricResult and result.value is None
    span = runtime[1].get_finished_spans()[0]
    assert span.status.status_code is StatusCode.OK
    assert "error.type" not in span.attributes
    assert "controlled callback" in payload(span)["reason"]


@pytest.mark.parametrize(
    "flag,value",
    [
        (ENABLE_CONTENT_TRACING_KEY, False),
        ("trace_content", False),
        ("override_enable_content_tracing", False),
        (context._SUPPRESS_INSTRUMENTATION_KEY, True),
        (SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY, True),
    ],
)
def test_capture_flags_and_suppression(runtime, flag, value):
    token = context.attach(context.set_value(flag, value))
    try:
        assert ExactMatch().score(reference="PRIVATE", response="PRIVATE").value == 1.0
    finally:
        context.detach(token)
    spans = runtime[1].get_finished_spans()
    if value is True:
        assert not spans
    else:
        assert len(spans) == 1
        assert_bodyless(spans)


@pytest.mark.parametrize("env", ["RESPAN_TRACE_CONTENT", "TRACELOOP_TRACE_CONTENT"])
def test_environment_privacy(runtime, monkeypatch, env):
    monkeypatch.setenv(env, "false")
    ExactMatch().score(reference="PRIVATE", response="PRIVATE")
    assert_bodyless(runtime[1].get_finished_spans())


def test_constructor_and_unsampled_do_not_convert_user_rows():
    calls = []

    class Row(BaseModel):
        value: int = 0

        def model_dump(self, *args, **kwargs):
            calls.append("dump")
            raise AssertionError("observer conversion")

        def __str__(self):
            calls.append("str")
            raise AssertionError("observer string")

    @ragas.experiment()
    def controlled(row):
        return 0

    for capture, sampler in [(False, None), (True, ALWAYS_OFF)]:
        provider = TracerProvider(sampler=sampler) if sampler else TracerProvider()
        memory = InMemorySpanExporter()
        provider.add_span_processor(SimpleSpanProcessor(memory))
        owner = RagasInstrumentor(tracer_provider=provider, capture_content=capture)
        owner.activate()
        try:
            assert asyncio.run(controlled(Row())) == 0
            assert not calls
            assert_bodyless(memory.get_finished_spans())
            if sampler:
                assert not memory.get_finished_spans()
        finally:
            owner.deactivate()
            provider.shutdown()


def test_pydantic_storage_avoids_getters_and_preserves_extras(runtime):
    calls = []

    class Row(BaseModel):
        model_config = ConfigDict(extra="allow")
        value: int = 0

        def model_dump(self, *a, **k):
            calls.append("dump")
            raise AssertionError

        def __getattribute__(self, name):
            if name in {"__dict__", "__pydantic_extra__"}:
                calls.append(name)
                raise AssertionError
            return super().__getattribute__(name)

    @ragas.experiment()
    def controlled(row):
        return row

    row = Row(secret="PRIVATE", flag=False)
    assert asyncio.run(controlled(row)) is row
    span = runtime[1].get_finished_spans()[0]
    assert not calls
    assert payload(span) == {"value": 0, "secret": "[REDACTED]", "flag": False}


def test_hostile_unknown_metaclass_and_conversion_not_invoked(runtime):
    calls = []

    class Meta(type):
        def __hash__(cls):
            calls.append("hash")
            raise AssertionError

        def __eq__(cls, other):
            calls.append("eq")
            raise AssertionError

    class Unknown(metaclass=Meta):
        def __str__(self):
            calls.append("str")
            raise AssertionError

        def __repr__(self):
            calls.append("repr")
            raise AssertionError

        def __iter__(self):
            calls.append("iter")
            raise AssertionError

    value = Unknown()

    @ragas.experiment()
    def controlled(row):
        return value

    assert asyncio.run(controlled(value)) is value
    assert not calls
    assert payload(runtime[1].get_finished_spans()[0]) == "<Unknown>"


def test_schema_and_literal_escaped_credentials_complete(runtime):
    expected = {
        "type": "object",
        "properties": {
            "password": {"type": "string", "default": "PRIVATE"},
            "authorization": {"type": "string", "examples": ["PRIVATE"]},
            "value": {"type": "number"},
        },
        "required": ["password"],
        "history": list(range(75)),
        "text": r"Bearer \"PRIVATE SPACE\" Basic \'PRIVATE TWO\'",
        "url": "https://user:PRIVATE@example.invalid/path?access_token=PRIVATE&x=0",
    }

    @ragas.experiment()
    def controlled(row):
        return expected

    assert asyncio.run(controlled({})) is expected
    out = payload(runtime[1].get_finished_spans()[0])
    assert list(out["properties"]) == list(expected["properties"])
    assert out["required"] == ["password"]
    assert out["properties"]["password"]["default"] == "[REDACTED]"
    assert len(out["history"]) == 75
    assert "PRIVATE" not in json.dumps(out)
    from respan_instrumentation_ragas._serialization import json_string

    assert json_string(out) == json_string(json.loads(json_string(out)))


@pytest.mark.parametrize("parent_kind", ["active", "finished", "unknown", "remote"])
def test_native_ancestor_bounds(runtime, parent_kind):
    provider, memory, _ = runtime
    if parent_kind in {"active", "finished"}:
        parent = provider.get_tracer("app").start_span("parent")
        parent.set_attribute(ENABLE_CONTENT_TRACING_KEY, False)
        if parent_kind == "finished":
            parent.end()
    else:
        parent = NonRecordingSpan(
            SpanContext(123, 456, parent_kind == "remote", TraceFlags(1))
        )
    with trace.use_span(parent):
        ExactMatch().score(reference="PRIVATE", response="PRIVATE")
    span = memory.get_finished_spans()[-1]
    if parent_kind == "remote":
        assert INPUT in span.attributes
    else:
        assert_bodyless([span])
    if parent_kind == "active":
        parent.end()


def test_late_end_veto_cleans_actual_readable_snapshot(runtime, monkeypatch):
    original = Span.end

    def end(span, *args, **kwargs):
        if span.name.startswith("ragas."):
            span.add_event("private", {"body": "PRIVATE"})
            token = context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, False))
            try:
                return original(span, *args, **kwargs)
            finally:
                context.detach(token)
        return original(span, *args, **kwargs)

    monkeypatch.setattr(Span, "end", end)
    with pytest.raises(AssertionError):
        StringPresence().score(reference=1, response="PRIVATE")
    spans = runtime[1].get_finished_spans()
    assert len(spans) == 1
    assert_bodyless(spans)
    assert spans[0].attributes["error.type"] == "AssertionError"


@pytest.mark.parametrize("stage", ["entity", "observe", "capture", "error"])
def test_observer_faults_preserve_native_outcomes(runtime, monkeypatch, stage):
    def fault(self, *args, **kwargs):
        if self.span and self.span.is_recording():
            self.span.set_attribute(INPUT, "PRIVATE")
            self.keys.add(INPUT)
            self.span.add_event("private", {"body": "PRIVATE"})
        raise RuntimeError("controlled telemetry failure")

    monkeypatch.setattr(impl._Call, stage, fault)
    if stage == "error":
        with pytest.raises(AssertionError):
            StringPresence().score(reference=1, response="PRIVATE")
    else:
        assert ExactMatch().score(reference="PRIVATE", response="PRIVATE").value == 1
    assert_bodyless(runtime[1].get_finished_spans())


def test_detach_fault_restores_native_context(runtime, monkeypatch):
    parent = runtime[0].get_tracer("app").start_span("parent")
    with trace.use_span(parent):
        original = context.detach

        def fault(token):
            raise RuntimeError("controlled detach")

        monkeypatch.setattr(context, "detach", fault)
        assert ExactMatch().score(reference="a", response="a").value == 1
        assert trace.get_current_span() is parent
        monkeypatch.setattr(context, "detach", original)
    parent.end()


def test_owner_lifecycle_shared_configuration_and_foreign_wrapper(runtime):
    provider, _, owner = runtime
    other = RagasInstrumentor(tracer_provider=provider)
    other.activate()
    with pytest.raises(ValueError):
        RagasInstrumentor(tracer_provider=provider, capture_content=False).activate()
    owner.deactivate()
    ExactMatch().score(reference="a", response="a")
    other.deactivate()
    assert impl._MANAGER is None
    assert len(runtime[1].get_finished_spans()) == 1
    owner.activate()
    wrapped = ragas.evaluate

    def foreign(*args, **kwargs):
        return wrapped(*args, **kwargs)

    ragas.evaluate = foreign
    owner.deactivate()
    assert ragas.evaluate is foreign
    ragas.evaluate = wrapped.__wrapped__


def test_future_class_restore_and_runtime_detach_presence(runtime):
    from ragas.metrics.base import SimpleBaseMetric

    runtime[2].deactivate()
    before = inspect.getattr_static(SimpleBaseMetric, "__init_subclass__")
    rt = context._RUNTIME_CONTEXT
    present = "detach" in rt.__dict__
    stored = rt.__dict__.get("detach")
    runtime[2].activate()

    @numeric_metric(name="temporary")
    def metric(response):
        return 0

    wrapped = type(metric).score
    runtime[2].deactivate()
    assert type(metric).score is wrapped.__wrapped__
    assert inspect.getattr_static(SimpleBaseMetric, "__init_subclass__") is before
    assert ("detach" in rt.__dict__) is present
    assert rt.__dict__.get("detach") is stored


@pytest.mark.parametrize("stage", ["start", "set", "end", "defer"])
def test_native_sdk_fault_injection_preserves_result_and_cleanup(
    runtime, monkeypatch, stage
):
    provider, memory, _ = runtime
    if stage == "start":

        def fault(*args, **kwargs):
            raise RuntimeError("telemetry tracer unavailable")

        monkeypatch.setattr(provider, "get_tracer", fault)
    elif stage == "set":
        original = Span.set_attribute

        def fault(span, name, value):
            original(span, name, value)
            if name == INPUT:
                raise RuntimeError("attribute mutation then failure")

        monkeypatch.setattr(Span, "set_attribute", fault)
    elif stage == "end":
        original = Span.end

        def fault(span, *args, **kwargs):
            original(span, *args, **kwargs)
            raise RuntimeError("end mutation then failure")

        monkeypatch.setattr(Span, "end", fault)
    else:

        def fault(*args):
            raise RuntimeError("defer observation failure")

        monkeypatch.setattr(impl._MANAGER, "defer", fault)
        from ragas.metrics import ExactMatch as LegacyExactMatch

        executor = ragas.evaluate(
            dataset(),
            metrics=[LegacyExactMatch()],
            return_executor=True,
            show_progress=False,
        )
        assert type(executor) is Executor
        assert executor.results() == [1.0, 0.0]
        assert not impl._MANAGER.pending
        assert not impl._MANAGER.states
        return
    assert ExactMatch().score(reference="a", response="a").value == 1
    assert not impl._MANAGER.states
    if stage == "set":
        assert_bodyless(memory.get_finished_spans())


def test_partial_activation_rolls_back_owned_descriptors(runtime, monkeypatch):
    runtime[2].deactivate()
    original = ragas.evaluate
    install = impl._Manager.patch
    calls = []

    def fault(manager, *args, **kwargs):
        install(manager, *args, **kwargs)
        calls.append(1)
        if len(calls) == 3:
            raise RuntimeError("partial install")

    monkeypatch.setattr(impl._Manager, "patch", fault)
    with pytest.raises(RuntimeError, match="partial install"):
        runtime[2].activate()
    assert impl._MANAGER is None
    assert ragas.evaluate is original
    from ragas.metrics import ExactMatch as LegacyExactMatch

    assert (
        ragas.evaluate(
            dataset(), metrics=[LegacyExactMatch()], show_progress=False
        ).scores[0]["exact_match"]
        == 1
    )


def test_finished_private_sibling_cannot_reenable_ancestor(runtime):
    provider, memory, _ = runtime
    parent = provider.get_tracer("app").start_span("parent")
    with trace.use_span(parent):
        child = provider.get_tracer("app").start_span("private_child")
        with trace.use_span(child):
            token = context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, False))
            child.end()
            context.detach(token)
        ExactMatch().score(reference="PRIVATE", response="PRIVATE")
    parent.end()
    assert_bodyless(
        [s for s in memory.get_finished_spans() if s.name == "ragas.metric"]
    )


def test_attach_mutate_then_raise_restores_ambient_context(runtime, monkeypatch):
    original = context.attach
    before = context.get_current()
    calls = []

    def fault(ctx):
        token = original(ctx)
        calls.append(token)
        if len(calls) == 1:
            raise RuntimeError("attach mutation then failure")
        return token

    monkeypatch.setattr(context, "attach", fault)
    assert ExactMatch().score(reference="a", response="a").value == 1
    assert context.get_current() is before
    assert not impl._MANAGER.states


@pytest.mark.parametrize("method", ["score", "ascore", "batch_score", "abatch_score"])
def test_native_llm_metric_released_client_and_http(runtime, method):
    from native_llm import generation_server
    from openai import AsyncOpenAI, OpenAI
    from ragas.llms import llm_factory
    from ragas.metrics import NumericMetric

    with generation_server() as (url, requests):

        async def async_call():
            async with AsyncOpenAI(
                api_key="controlled", base_url=url, max_retries=0
            ) as client:
                llm = llm_factory("controlled-metric", client=client)
                metric = NumericMetric(name="native_llm", prompt="Score {response}")
                if "batch" in method:
                    return await getattr(metric, method)(
                        [{"response": "one"}, {"response": "two"}], llm=llm
                    )
                return await getattr(metric, method)(response="one", llm=llm)

        if method in {"ascore", "abatch_score"}:
            result = asyncio.run(async_call())
        else:
            with OpenAI(api_key="controlled", base_url=url, max_retries=0) as client:
                llm = llm_factory("controlled-metric", client=client)
                metric = NumericMetric(name="native_llm", prompt="Score {response}")
                result = (
                    getattr(metric, method)(
                        [{"response": "one"}, {"response": "two"}], llm=llm
                    )
                    if "batch" in method
                    else metric.score(response="one", llm=llm)
                )
        assert len(requests) == (2 if "batch" in method else 1)
        assert all(request["model"] == "controlled-metric" for request in requests)
        results = result if "batch" in method else [result]
        assert all(
            type(item) is MetricResult and item.value == 0 and item.reason == ""
            for item in results
        )
    spans = runtime[1].get_finished_spans()
    assert len(spans) == 1
    output = payload(spans[0])
    if "batch" in method:
        assert len(output) == 2
    else:
        assert output["value"] == 0
        assert output["traces"]["output"]["reason"] == ""
    assert "gen_ai.request.model" not in spans[0].attributes
    assert "gen_ai.usage.total_tokens" not in spans[0].attributes


def test_native_subclass_metaclass_getters_not_called_by_activation(runtime):
    runtime[2].deactivate()
    calls = []

    class Meta(type(ExactMatch)):
        def __getattribute__(cls, name):
            if name in {"score", "__subclasses__"}:
                calls.append(name)
            return super().__getattribute__(name)

    class NativeMetric(ExactMatch, metaclass=Meta):
        def score(self, **kwargs):
            return super().score(**kwargs)

    metric = NativeMetric()
    runtime[2].activate()
    assert metric.score(reference="a", response="a").value == 1
    assert calls == []
