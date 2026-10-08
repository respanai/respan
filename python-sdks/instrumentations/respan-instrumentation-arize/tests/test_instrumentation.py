"""Released Arize transport/protocol tests; helpers cover ownership fault injection."""

from __future__ import annotations

import asyncio
import contextlib
import importlib.util
import io
import json
import sys
import time
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest
from arize._generated.api_client.exceptions import NotFoundException
from arize.ml.types import Embedding, Environments, ModelTypes
from opentelemetry import context, trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.sampling import ALWAYS_OFF
from opentelemetry.semconv.attributes.http_attributes import HTTP_RESPONSE_STATUS_CODE
from opentelemetry.semconv_ai import (
    SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY,
    SpanAttributes,
)
from opentelemetry.trace import StatusCode
from respan_instrumentation_arize import ArizeInstrumentor, _instrumentation
from respan_instrumentation_arize._constants import (
    ARIZE_CLIENT_SPECS,
    ARIZE_METADATA_OPERATION,
    ARIZE_METADATA_RESOURCE,
    ArizeClientSpec,
)
from respan_instrumentation_arize._serialization import safe_json_dumps
from respan_sdk.constants.span_attributes import RESPAN_LOG_TYPE
from respan_tracing.constants.context_constants import ENABLE_CONTENT_TRACING_KEY
from respan_tracing.core.tracer import RespanTracer

_spec = importlib.util.spec_from_file_location(
    "arize_native_http_fixture", Path(__file__).with_name("_fixture.py")
)
_fixture = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fixture)
FixtureTransport = _fixture.FixtureTransport
SPACE_ID, PROJECT_ID, DATASET_ID = (
    _fixture.SPACE_ID,
    _fixture.PROJECT_ID,
    _fixture.DATASET_ID,
)


@pytest.fixture(autouse=True)
def clean():
    RespanTracer.reset_instance()
    _instrumentation._ACTIVE_INSTANCES = 0
    _instrumentation._restore_arize_clients()
    yield
    _instrumentation._ACTIVE_INSTANCES = 0
    _instrumentation._restore_arize_clients()
    RespanTracer.reset_instance()


@pytest.fixture
def runtime(monkeypatch):
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(trace, "get_tracer_provider", lambda: provider)
    owner = ArizeInstrumentor()
    owner.activate()
    try:
        yield owner, provider, exporter
    finally:
        owner.deactivate()
        provider.shutdown()


@pytest.fixture
def transport():
    fixture = FixtureTransport()
    try:
        yield fixture
    finally:
        fixture.close()


def operation(exporter, resource, method):
    return next(
        span
        for span in exporter.get_finished_spans()
        if span.attributes.get(ARIZE_METADATA_RESOURCE) == resource
        and span.attributes.get(ARIZE_METADATA_OPERATION) == method
    )


def no_content(span):
    assert SpanAttributes.TRACELOOP_ENTITY_INPUT not in span.attributes
    assert SpanAttributes.TRACELOOP_ENTITY_OUTPUT not in span.attributes


def upload(client, vector=True):
    return client.ml.log_stream(
        space_id=SPACE_ID,
        model_name="fixture-regression",
        model_type=ModelTypes.REGRESSION,
        environment=Environments.PRODUCTION,
        prediction_label=1.0,
        embedding_features={
            "controlled": Embedding(vector=[float(i) for i in range(3072)])
        }
        if vector
        else None,
    )


def wait_finished(exporter, count):
    deadline = time.monotonic() + 2
    while len(exporter.get_finished_spans()) < count and time.monotonic() < deadline:
        time.sleep(0.005)
    assert len(exporter.get_finished_spans()) >= count


def test_real_sdk_generated_rest_and_native_context(runtime, transport):
    _, provider, exporter = runtime
    with provider.get_tracer("test").start_as_current_span("real-parent") as parent:
        result = transport.client.datasets.list(space=SPACE_ID, name="controlled-name")
        assert trace.get_current_span() is parent
    assert type(result).__name__ in {"ListDatasetsResponse", "DatasetListResponse"}
    assert result.datasets == []
    span = operation(exporter, "datasets", "list")
    assert span.parent.span_id == parent.get_span_context().span_id
    assert span.attributes[RESPAN_LOG_TYPE] == "task"
    assert (
        json.loads(span.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT])["datasets"]
        == []
    )
    assert "status_code" not in span.attributes
    assert not any(key.startswith("gen_ai.usage.") for key in span.attributes)


def test_real_sdk_error_identity_and_actual_http_status(runtime, transport):
    _, _, exporter = runtime
    transport.fail = True
    with pytest.raises(NotFoundException) as caught:
        transport.client.datasets.list(space=SPACE_ID)
    assert caught.value.status == 404
    span = operation(exporter, "datasets", "list")
    assert span.status.status_code is StatusCode.ERROR
    assert span.attributes[HTTP_RESPONSE_STATUS_CODE] == 404
    assert "status_code" not in span.attributes
    assert SpanAttributes.TRACELOOP_ENTITY_OUTPUT not in span.attributes
    assert span.events[0].attributes["exception.type"] == "NotFoundException"


@pytest.mark.parametrize("policy", ["env", "context", "constructor"])
def test_native_start_bound_never_reenables_capture(
    runtime, transport, monkeypatch, policy
):
    owner, _, exporter = runtime
    token = None
    if policy == "constructor":
        owner.deactivate()
        owner = ArizeInstrumentor(capture_content=False)
        owner.activate()
    elif policy == "env":
        monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")
        transport.before_rest = lambda: monkeypatch.setenv(
            "TRACELOOP_TRACE_CONTENT", "true"
        )
    else:
        token = context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, False))
        transport.before_rest = lambda: context.attach(
            context.set_value(ENABLE_CONTENT_TRACING_KEY, True)
        )
    try:
        result = transport.client.datasets.list(
            space=SPACE_ID, name="private-native-input"
        )
        assert result.datasets == []
    finally:
        if token is not None:
            context.detach(token)
        owner.deactivate()
    no_content(operation(exporter, "datasets", "list"))


@pytest.mark.parametrize("failure", [False, True])
def test_context_veto_observed_before_native_span_detach(runtime, transport, failure):
    _, _, exporter = runtime
    transport.fail = failure
    transport.before_rest = lambda: context.attach(
        context.set_value(ENABLE_CONTENT_TRACING_KEY, False)
    )
    if failure:
        with pytest.raises(NotFoundException):
            transport.client.datasets.list(space=SPACE_ID, name="private-native-input")
    else:
        transport.client.datasets.list(space=SPACE_ID, name="private-native-input")
    span = operation(exporter, "datasets", "list")
    no_content(span)
    assert all("exception.message" not in event.attributes for event in span.events)
    assert context.get_value(ENABLE_CONTENT_TRACING_KEY) is not False


@pytest.mark.parametrize(
    "key",
    [
        context._SUPPRESS_INSTRUMENTATION_KEY,
        SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY,
    ],
)
def test_both_native_suppression_keys(runtime, transport, key):
    _, _, exporter = runtime
    token = context.attach(context.set_value(key, True))
    try:
        assert transport.client.datasets.list(space=SPACE_ID).datasets == []
    finally:
        context.detach(token)
    assert not exporter.get_finished_spans()


def test_sampler_before_payload_serialization(runtime, transport, monkeypatch):
    owner, _, _ = runtime
    owner.deactivate()
    provider = TracerProvider(sampler=ALWAYS_OFF)
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(trace, "get_tracer_provider", lambda: provider)
    owner = ArizeInstrumentor()
    owner.activate()

    def forbidden(*args, **kwargs):
        raise AssertionError("Sampled-off payload was inspected")

    monkeypatch.setattr(_instrumentation, "safe_json_dumps", forbidden)
    try:
        assert transport.client.datasets.list(name="no inspection").datasets == []
    finally:
        owner.deactivate()
        provider.shutdown()
    assert not exporter.get_finished_spans()


def test_actual_future_identity_completion_http_and_full_vector(runtime):
    _, _, exporter = runtime
    fixture = FixtureTransport(delayed=True)
    try:
        future = upload(fixture.client)
        assert fixture.entered.wait(2) and not future.done()
        assert not exporter.get_finished_spans()
        fixture.release.set()
        result = future.result(timeout=2)
        assert result is fixture.responses[0]
        wait_finished(exporter, 1)
        span = operation(exporter, "ml", "log_stream")
        assert span.attributes[HTTP_RESPONSE_STATUS_CODE] == 202
        output = json.loads(span.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT])
        assert output["body"] == {"accepted": True}
        data = json.loads(span.attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT])
        vector = data["kwargs"]["embedding_features"]["controlled"]["vector"]
        assert len(vector) == 3072 and vector[-1] == 3071.0
        assert not _instrumentation._OPEN_OPERATIONS
        assert all(
            state.span is None and state.parent is None
            for state in _instrumentation._ENDED_POLICIES.values()
        )
    finally:
        fixture.close()


def test_future_end_environment_veto(runtime, monkeypatch):
    _, _, exporter = runtime
    fixture = FixtureTransport(delayed=True)
    try:
        future = upload(fixture.client)
        assert fixture.entered.wait(2)
        monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")
        fixture.release.set()
        assert future.result(timeout=2).status_code == 202
        wait_finished(exporter, 1)
        no_content(operation(exporter, "ml", "log_stream"))
    finally:
        fixture.close()


def test_native_future_error_no_output_or_invented_http(runtime):
    _, _, exporter = runtime
    fixture = FixtureTransport(fail=True, delayed=True)
    try:
        future = upload(fixture.client)
        assert fixture.entered.wait(2)
        fixture.release.set()
        import requests

        with pytest.raises(requests.ConnectionError) as caught:
            future.result(timeout=2)
        assert future.exception() is caught.value
        wait_finished(exporter, 1)
        span = operation(exporter, "ml", "log_stream")
        assert span.status.status_code is StatusCode.ERROR
        assert HTTP_RESPONSE_STATUS_CODE not in span.attributes
        assert SpanAttributes.TRACELOOP_ENTITY_OUTPUT not in span.attributes
    finally:
        fixture.close()


def test_native_future_cancel_and_deactivation(runtime):
    owner, _, exporter = runtime
    fixture = FixtureTransport(delayed=True)
    try:
        running = upload(fixture.client)
        assert fixture.entered.wait(2)
        queued = upload(fixture.client)
        assert queued.cancel()
        wait_finished(exporter, 1)
        span = operation(exporter, "ml", "log_stream")
        assert span.status.status_code is StatusCode.UNSET
        assert SpanAttributes.TRACELOOP_ENTITY_OUTPUT not in span.attributes
        owner.deactivate()
        assert not running.cancelled()
        fixture.release.set()
        assert running.result(timeout=2).status_code == 202
        assert len(exporter.get_finished_spans()) == 2
        assert not _instrumentation._OPEN_OPERATIONS
    finally:
        fixture.close()


def test_actual_historical_span_vectors_and_calls_are_complete(runtime, transport):
    _, _, exporter = runtime
    result = transport.client.spans.list(project=PROJECT_ID)
    assert len(result.spans) == 1
    span = operation(exporter, "spans", "list")
    output = json.loads(span.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT])
    values = output["spans"][0]["attributes"]
    vector = next(value for key, value in values.items() if "embedding.vector" in key)
    assert len(vector) == 3072 and vector[-1] == 3071.0
    assert values["tool_calls"][0]["id"] == "historical-call"
    assert not any(
        key.startswith(("gen_ai.prompt.", "gen_ai.completion.", "gen_ai.usage."))
        for key in span.attributes
    )


@pytest.mark.parametrize(
    "resource", ["traces", "audit_logs", "integrations", "webhooks"]
)
def test_new_released_public_resource_clients(runtime, transport, resource):
    _, _, exporter = runtime
    if resource not in transport.client._SUBCLIENTS:
        pytest.skip("This resource is not exposed by the installed Arize minimum")
    client = getattr(transport.client, resource)
    if resource == "traces":
        result = client.list(project=PROJECT_ID)
    elif resource == "webhooks":
        result = client.list(
            organization=__import__("base64")
            .b64encode(b"Organization:fixture")
            .decode()
        )
    else:
        result = client.list()
    assert result is not None
    assert operation(exporter, resource, "list")


def test_actual_experiment_dry_run_retains_native_helper_and_result(
    runtime, transport, monkeypatch
):
    _, _, exporter = runtime
    console = io.StringIO()
    import arize.experiments.client as sdk_experiments

    original_helper = sdk_experiments._get_tracer_resource
    helper_results = []
    native_spans = []

    def native_helper(*args, **kwargs):
        result = original_helper(*args, **kwargs)
        helper_results.append(result)
        return result

    def task(input):
        native_spans.append(trace.get_current_span().get_span_context())
        return "controlled task result"

    monkeypatch.setattr(sdk_experiments, "_get_tracer_resource", native_helper)
    with contextlib.redirect_stdout(console):
        result = transport.client.experiments.run(
            name="controlled-dry-run",
            dataset=DATASET_ID,
            task=task,
            dry_run=True,
            force_http=True,
            concurrency=1,
        )
    assert isinstance(result, tuple) and result[0] is None
    assert type(result[1]) is pd.DataFrame and len(result[1]) == 1
    assert helper_results and native_spans[0].is_valid
    assert sdk_experiments._get_tracer_resource is native_helper
    span = operation(exporter, "experiments", "run")
    output = json.loads(span.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT])
    assert output[0] is None and output[1]["rows"] == 1


def test_shared_foreign_ownership_and_mismatched_settings(runtime, transport):
    first, _, exporter = runtime
    from arize.datasets.client import DatasetsClient

    wrapper = DatasetsClient.list
    second = ArizeInstrumentor()
    second.activate()
    with pytest.raises(ValueError, match="different settings"):
        ArizeInstrumentor(capture_content=False).activate()
    first.deactivate()
    transport.client.datasets.list()
    assert len(exporter.get_finished_spans()) == 1

    def foreign(self, *args, **kwargs):
        return wrapper(self, *args, **kwargs)

    DatasetsClient.list = foreign
    second.deactivate()
    assert DatasetsClient.list is foreign
    transport.client.datasets.list()
    assert len(exporter.get_finished_spans()) == 1
    DatasetsClient.list = wrapper.__wrapped__


def test_partial_registration_rolls_back_owned_fields(runtime, monkeypatch):
    owner, _, _ = runtime
    owner.deactivate()
    from arize.datasets.client import DatasetsClient

    original = DatasetsClient.list
    original_create = DatasetsClient.create
    real_setattr = setattr

    def fail(obj, key, value):
        real_setattr(obj, key, value)
        if obj is DatasetsClient and key == "create" and value is not original_create:
            raise RuntimeError("after owned assignment")

    monkeypatch.setattr(_instrumentation, "setattr", fail, raising=False)
    with pytest.raises(RuntimeError, match="owned assignment"):
        ArizeInstrumentor().activate()
    assert DatasetsClient.list is original
    assert not _instrumentation._ORIGINAL_METHODS
    assert _instrumentation._ACTIVE_INSTANCES == 0


def test_telemetry_start_failure_cannot_change_native_return(
    runtime, transport, monkeypatch
):
    monkeypatch.setattr(
        _instrumentation,
        "_begin_operation",
        lambda *args: (_ for _ in ()).throw(RuntimeError("telemetry only")),
    )
    assert transport.client.datasets.list().datasets == []


def test_safe_known_sdk_serialization_credentials_vectors_and_no_user_hooks():
    class Trap:
        @property
        def model_dump(self):
            raise AssertionError("arbitrary hook invoked")

        def __str__(self):
            raise AssertionError("arbitrary string conversion invoked")

    assert json.loads(safe_json_dumps(Trap())) == {"type": "Trap"}
    vector = {index: float(index) for index in range(256)}
    assert len(json.loads(safe_json_dumps(vector))) == 256
    dataframe = pd.DataFrame(
        [
            {
                "embedding": [float(index) for index in range(3072)],
                "api_key": "fixture-secret",
            }
        ]
    )
    text = safe_json_dumps(dataframe)
    assert "fixture-secret" not in text
    assert len(json.loads(text)["records"][0]["embedding"]) == 3072
    assert "never-export" not in safe_json_dumps(
        {"key": "never-export", "authorization": "Bearer never-export"}
    )


def _custom_owner(runtime, monkeypatch, client_class):
    owner, _, _ = runtime
    owner.deactivate()
    module = ModuleType("arize_fixture_control")
    module.Control = client_class
    monkeypatch.setitem(sys.modules, module.__name__, module)
    spec = ArizeClientSpec(module.__name__, "Control", "control", ("run",))
    owner = ArizeInstrumentor(client_specs=(spec, *ARIZE_CLIENT_SPECS))
    owner.activate()
    return owner


def test_close_pending_child_honors_permanent_parent_veto(runtime, monkeypatch):
    _, _, exporter = runtime
    fixture = FixtureTransport(delayed=True)

    class Control:
        def run(self):
            result = upload(fixture.client)
            assert fixture.entered.wait(2)
            context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, False))
            fixture.client.datasets.list(name="private-child")
            context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, True))
            return result

    owner = _custom_owner(runtime, monkeypatch, Control)
    try:
        result = Control().run()
        owner.deactivate()
        fixture.release.set()
        assert result.result(timeout=2).status_code == 202
        assert len(exporter.get_finished_spans()) == 3
        for span in exporter.get_finished_spans():
            no_content(span)
        spans = {span.context.span_id: span for span in exporter.get_finished_spans()}
        assert all(
            span.parent is None or spans[span.parent.span_id].end_time >= span.end_time
            for span in spans.values()
        )
    finally:
        owner.deactivate()
        fixture.close()


@pytest.mark.parametrize("failure", [False, True])
def test_async_wrapper_context_veto_and_native_identity(runtime, monkeypatch, failure):
    _, _, exporter = runtime
    value = object()
    error = ValueError("controlled async native failure")

    class Control:
        async def run(self):
            assert trace.get_current_span().is_recording()
            await asyncio.sleep(0)
            context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, False))
            if failure:
                raise error
            return value

    owner = _custom_owner(runtime, monkeypatch, Control)
    try:
        if failure:
            with pytest.raises(ValueError) as caught:
                asyncio.run(Control().run())
            assert caught.value is error
        else:
            assert asyncio.run(Control().run()) is value
        no_content(operation(exporter, "control", "run"))
    finally:
        owner.deactivate()


def test_actual_new_dataset_mutations(runtime, transport):
    _, _, exporter = runtime
    if not hasattr(transport.client.datasets, "update_examples"):
        pytest.skip("Minimum SDK has no dataset example mutation methods")
    updated = transport.client.datasets.update_examples(
        dataset=DATASET_ID,
        examples=[
            {
                "id": "example-fixture",
                "input": {"embedding": [float(i) for i in range(3072)]},
            }
        ],
    )
    assert updated.example_ids == ["example-fixture"]
    removed = transport.client.datasets.delete_examples(
        dataset=DATASET_ID,
        dataset_version_id="version-fixture",
        examples=["example-fixture"],
    )
    assert removed.completed
    assert operation(exporter, "datasets", "update_examples")
    assert operation(exporter, "datasets", "delete_examples")
