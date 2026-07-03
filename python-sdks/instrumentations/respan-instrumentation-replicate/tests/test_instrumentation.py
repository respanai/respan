import asyncio
import sys
from types import ModuleType, SimpleNamespace

import pytest
from opentelemetry.semconv_ai import LLMRequestTypeValues, SpanAttributes

from respan_instrumentation_replicate import ReplicateInstrumentor
from respan_instrumentation_replicate import _instrumentation
from respan_instrumentation_replicate._constants import (
    OFF_CONTRACT_ALIASES,
    RESPAN_PARAMS_KEY,
    RESPAN_PARAMS_MODEL_KEY,
)
from respan_instrumentation_replicate._translator import (
    build_model_call_span_data,
    model_from_ref_or_prediction,
    output_to_text,
)
from respan_sdk.constants.llm_logging import LOG_TYPE_TEXT
from respan_sdk.constants.span_attributes import (
    RESPAN_LOG_TYPE,
    RESPAN_METADATA,
    RESPAN_TRACE_GROUP_ID,
)
from respan_tracing.core.tracer import RespanTracer


@pytest.fixture(autouse=True)
def reset_tracer():
    RespanTracer.reset_instance()
    yield
    RespanTracer.reset_instance()


def test_build_model_call_span_data_uses_canonical_attrs_only():
    span_name, attrs = build_model_call_span_data(
        span_name="replicate.run",
        ref="meta/meta-llama-3-8b-instruct",
        input_value={"prompt": "Say hi"},
        output=["hello", " world"],
        kwargs={
            RESPAN_PARAMS_KEY: {
                "workflow_name": "replicate_unit.workflow",
                "metadata": {"example": "unit"},
            }
        },
        stream=True,
    )

    assert span_name == "replicate.run"
    assert attrs[RESPAN_LOG_TYPE] == LOG_TYPE_TEXT
    assert attrs[SpanAttributes.LLM_SYSTEM] == "replicate"
    assert attrs[SpanAttributes.LLM_REQUEST_TYPE] == LLMRequestTypeValues.COMPLETION.value
    assert attrs[SpanAttributes.LLM_REQUEST_MODEL] == "meta/meta-llama-3-8b-instruct"
    assert attrs[f"{SpanAttributes.LLM_PROMPTS}.0.role"] == "user"
    assert attrs[f"{SpanAttributes.LLM_PROMPTS}.0.content"] == "Say hi"
    assert attrs[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"] == "hello world"
    assert attrs[SpanAttributes.LLM_IS_STREAMING] is True
    assert attrs[RESPAN_TRACE_GROUP_ID] == "replicate_unit.workflow"
    assert attrs[f"{RESPAN_METADATA}.example"] == "unit"

    for alias in OFF_CONTRACT_ALIASES:
        assert alias not in attrs


def test_respan_params_model_overrides_reported_model_without_alias():
    _, attrs = build_model_call_span_data(
        span_name="replicate.run",
        ref="owner/replicate-model",
        input_value={"prompt": "Say hi"},
        output=["hello"],
        kwargs={
            RESPAN_PARAMS_KEY: {
                RESPAN_PARAMS_MODEL_KEY: "gpt-4o-mini",
            }
        },
    )

    assert attrs[SpanAttributes.LLM_REQUEST_MODEL] == "gpt-4o-mini"
    assert "model" not in attrs


def test_output_to_text_handles_prediction_and_file_output_like_values():
    prediction = SimpleNamespace(output=["a", "b"], status="succeeded")
    assert output_to_text(prediction) == "ab"

    FileOutput = type("FileOutput", (), {})
    file_output = FileOutput()
    file_output.url = "https://delivery.replicate.com/file"
    assert output_to_text(file_output) == "https://delivery.replicate.com/file"


def test_model_from_prediction_does_not_duplicate_prefixed_version():
    prediction = SimpleNamespace(
        model="owner/model",
        version="owner/model:version-id",
    )
    assert model_from_ref_or_prediction(prediction=prediction) == "owner/model:version-id"


def _install_fake_replicate_modules(monkeypatch):
    emitted_spans = []

    class Prediction:
        def __init__(self, output=None):
            self.id = "pred_unit"
            self.model = "owner/model"
            self.version = "version"
            self.status = "succeeded"
            self.input = {"prompt": "hello"}
            self.output = output or ["done"]
            self.logs = None
            self.error = None
            self.metrics = {"predict_time": 0.1}

        def dict(self):
            return dict(self.__dict__)

        def wait(self):
            return None

        async def async_wait(self):
            return None

        def stream(self):
            return iter(["a", "b"])

        async def async_stream(self):
            for chunk in ["a", "b"]:
                yield chunk

    class Predictions:
        def __init__(self, client=None):
            self._client = client

        def create(self, version=None, input=None, **params):
            return Prediction(output=["created"])

        async def async_create(self, version=None, input=None, **params):
            return Prediction(output=["created"])

        def list(self, cursor=...):
            return [Prediction(output=["listed"])]

        async def async_list(self, cursor=...):
            return [Prediction(output=["listed"])]

        def get(self, id):
            return Prediction(output=["got"])

        async def async_get(self, id):
            return Prediction(output=["got"])

        def cancel(self, id):
            prediction = Prediction(output=[])
            prediction.status = "canceled"
            return prediction

        async def async_cancel(self, id):
            prediction = Prediction(output=[])
            prediction.status = "canceled"
            return prediction

    class Client:
        def __init__(self):
            self.predictions = Predictions(client=self)

        def run(self, ref, input=None, **params):
            return ["hello", " world"]

        async def async_run(self, ref, input=None, **params):
            return ["hello", " async"]

        def stream(self, ref, *, input=None, **params):
            return iter(["s", "t"])

        async def async_stream(self, ref, input=None, **params):
            async def iterator():
                for chunk in ["s", "t"]:
                    yield chunk

            return iterator()

    replicate_module = ModuleType("replicate")
    client_module = ModuleType("replicate.client")
    prediction_module = ModuleType("replicate.prediction")

    default_client = Client()
    replicate_module.default_client = default_client
    replicate_module.run = default_client.run
    replicate_module.async_run = default_client.async_run
    replicate_module.stream = default_client.stream
    replicate_module.async_stream = default_client.async_stream
    client_module.Client = Client
    prediction_module.Prediction = Prediction
    prediction_module.Predictions = Predictions

    monkeypatch.setitem(sys.modules, "replicate", replicate_module)
    monkeypatch.setitem(sys.modules, "replicate.client", client_module)
    monkeypatch.setitem(sys.modules, "replicate.prediction", prediction_module)

    def fake_inject_span(span):
        emitted_spans.append(span)
        return True

    monkeypatch.setattr(_instrumentation, "inject_span", fake_inject_span)
    return SimpleNamespace(
        module=replicate_module,
        client_class=Client,
        prediction_class=Prediction,
        emitted_spans=emitted_spans,
    )


def test_instrumentor_patches_run_and_stream(monkeypatch):
    fake = _install_fake_replicate_modules(monkeypatch)

    instrumentor = ReplicateInstrumentor()
    instrumentor.activate()

    client = fake.client_class()
    assert client.run("owner/model", input={"prompt": "hi"}) == ["hello", " world"]
    assert client.run(
        "owner/model",
        input={"prompt": "hi"},
        respan_params={"model": "gpt-4o-mini"},
    ) == ["hello", " world"]
    assert "".join(client.stream("owner/model", input={"prompt": "hi"})) == "st"

    assert len(fake.emitted_spans) == 3
    assert fake.emitted_spans[0].name == "replicate.run"
    assert fake.emitted_spans[0].attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"] == "hello world"
    assert fake.emitted_spans[1].attributes[SpanAttributes.LLM_REQUEST_MODEL] == "gpt-4o-mini"
    assert fake.emitted_spans[2].name == "replicate.stream"
    assert fake.emitted_spans[2].attributes[SpanAttributes.LLM_IS_STREAMING] is True

    instrumentor.deactivate()
    assert client.run.__func__ is fake.client_class.run


def test_instrumentor_patches_async_run_and_prediction_create(monkeypatch):
    fake = _install_fake_replicate_modules(monkeypatch)

    instrumentor = ReplicateInstrumentor()
    instrumentor.activate()

    async def run_calls():
        client = fake.client_class()
        assert await client.async_run("owner/model", input={"prompt": "hi"}) == [
            "hello",
            " async",
        ]
        prediction = await client.predictions.async_create(
            version="version",
            input={"prompt": "hi"},
            respan_params={"model": "gpt-4o-mini"},
        )
        await prediction.async_wait()

    asyncio.run(run_calls())

    names = [span.name for span in fake.emitted_spans]
    assert names == [
        "async_replicate.run",
        "replicate.predictions.create",
        "replicate.prediction.wait",
    ]
    assert fake.emitted_spans[1].attributes[SpanAttributes.LLM_REQUEST_MODEL] == "gpt-4o-mini"
    assert fake.emitted_spans[2].attributes[SpanAttributes.LLM_REQUEST_MODEL] == "gpt-4o-mini"


def test_activate_skips_when_respan_tracing_is_disabled(monkeypatch, caplog):
    fake = _install_fake_replicate_modules(monkeypatch)
    RespanTracer(is_enabled=False)

    instrumentor = ReplicateInstrumentor()
    with caplog.at_level("INFO"):
        instrumentor.activate()

    assert instrumentor._is_instrumented is False
    assert "Replicate instrumentation skipped because Respan tracing is disabled" in caplog.text
    assert fake.client_class().run("owner/model", input={"prompt": "hi"}) == [
        "hello",
        " world",
    ]
    assert fake.emitted_spans == []
