from __future__ import annotations

import asyncio
import logging
import sys
from types import ModuleType

import pytest

from respan_instrumentation_arize import ArizeInstrumentor
from respan_instrumentation_arize import _instrumentation
from respan_instrumentation_arize._constants import (
    ARIZE_METADATA_OPERATION,
    ARIZE_METADATA_RESOURCE,
    ArizeClientSpec,
)
from respan_sdk.constants.llm_logging import LOG_TYPE_TASK
from respan_sdk.constants.span_attributes import RESPAN_LOG_METHOD, RESPAN_LOG_TYPE
from respan_tracing.core.tracer import RespanTracer


def _install_fake_arize(monkeypatch):
    class FakeDatasetsClient:
        def list(self, *, space=None):
            return {"datasets": [], "space": space}

        def create(self, *, name, examples=None, space=None):
            return {"id": "dataset-1", "name": name, "examples": examples, "space": space}

    class FakeAsyncClient:
        async def run(self, *, value):
            return {"value": value}

    class FakeErrorClient:
        def explode(self):
            raise RuntimeError("boom")

    arize_module = ModuleType("arize")
    datasets_package = ModuleType("arize.datasets")
    datasets_module = ModuleType("arize.datasets.client")
    datasets_module.DatasetsClient = FakeDatasetsClient
    async_package = ModuleType("arize.async_resource")
    async_module = ModuleType("arize.async_resource.client")
    async_module.AsyncClient = FakeAsyncClient
    error_package = ModuleType("arize.error_resource")
    error_module = ModuleType("arize.error_resource.client")
    error_module.ErrorClient = FakeErrorClient

    monkeypatch.setitem(sys.modules, "arize", arize_module)
    monkeypatch.setitem(sys.modules, "arize.datasets", datasets_package)
    monkeypatch.setitem(sys.modules, "arize.datasets.client", datasets_module)
    monkeypatch.setitem(sys.modules, "arize.async_resource", async_package)
    monkeypatch.setitem(sys.modules, "arize.async_resource.client", async_module)
    monkeypatch.setitem(sys.modules, "arize.error_resource", error_package)
    monkeypatch.setitem(sys.modules, "arize.error_resource.client", error_module)
    return FakeDatasetsClient, FakeAsyncClient, FakeErrorClient


@pytest.fixture(autouse=True)
def reset_instrumentation_state():
    RespanTracer.reset_instance()
    _instrumentation._ACTIVE_INSTANCES = 0
    _instrumentation._restore_arize_clients()
    yield
    _instrumentation._ACTIVE_INSTANCES = 0
    _instrumentation._restore_arize_clients()
    RespanTracer.reset_instance()


def test_name_is_arize() -> None:
    assert ArizeInstrumentor.name == "arize"
    assert ArizeInstrumentor().name == "arize"


def test_activate_patches_methods_and_deactivate_restores(monkeypatch) -> None:
    FakeDatasetsClient, _, _ = _install_fake_arize(monkeypatch)
    original_list = FakeDatasetsClient.list
    emitted = []
    monkeypatch.setattr(
        _instrumentation,
        "emit_arize_span",
        lambda **kwargs: emitted.append(kwargs) or True,
    )

    instrumentor = ArizeInstrumentor(
        client_specs=(
            ArizeClientSpec(
                module_name="arize.datasets.client",
                class_name="DatasetsClient",
                resource="datasets",
                methods=("list", "create"),
            ),
        )
    )
    instrumentor.activate()

    assert instrumentor._is_instrumented is True
    assert FakeDatasetsClient.list is not original_list

    result = FakeDatasetsClient().list(space="demo-space")

    assert result == {"datasets": [], "space": "demo-space"}
    assert emitted[0]["resource"] == "datasets"
    assert emitted[0]["method_name"] == "list"
    assert emitted[0]["kwargs"] == {"space": "demo-space"}

    instrumentor.deactivate()

    assert FakeDatasetsClient.list is original_list
    assert instrumentor._is_instrumented is False


def test_multiple_instrumentors_restore_after_last_deactivate(monkeypatch) -> None:
    FakeDatasetsClient, _, _ = _install_fake_arize(monkeypatch)
    original_list = FakeDatasetsClient.list
    monkeypatch.setattr(_instrumentation, "emit_arize_span", lambda **kwargs: True)
    specs = (
        ArizeClientSpec(
            module_name="arize.datasets.client",
            class_name="DatasetsClient",
            resource="datasets",
            methods=("list",),
        ),
    )

    first = ArizeInstrumentor(client_specs=specs)
    second = ArizeInstrumentor(client_specs=specs)

    first.activate()
    second.activate()
    assert FakeDatasetsClient.list is not original_list

    first.deactivate()
    assert FakeDatasetsClient.list is not original_list

    second.deactivate()
    assert FakeDatasetsClient.list is original_list


def test_async_methods_are_wrapped(monkeypatch) -> None:
    _, FakeAsyncClient, _ = _install_fake_arize(monkeypatch)
    emitted = []
    monkeypatch.setattr(
        _instrumentation,
        "emit_arize_span",
        lambda **kwargs: emitted.append(kwargs) or True,
    )
    instrumentor = ArizeInstrumentor(
        client_specs=(
            ArizeClientSpec(
                module_name="arize.async_resource.client",
                class_name="AsyncClient",
                resource="async_resource",
                methods=("run",),
            ),
        )
    )

    instrumentor.activate()
    result = asyncio.run(FakeAsyncClient().run(value=42))

    assert result == {"value": 42}
    assert emitted[0]["resource"] == "async_resource"
    assert emitted[0]["method_name"] == "run"


def test_error_methods_emit_error_span(monkeypatch) -> None:
    _, _, FakeErrorClient = _install_fake_arize(monkeypatch)
    emitted = []
    monkeypatch.setattr(
        _instrumentation,
        "emit_arize_span",
        lambda **kwargs: emitted.append(kwargs) or True,
    )
    instrumentor = ArizeInstrumentor(
        client_specs=(
            ArizeClientSpec(
                module_name="arize.error_resource.client",
                class_name="ErrorClient",
                resource="error_resource",
                methods=("explode",),
            ),
        )
    )
    instrumentor.activate()

    with pytest.raises(RuntimeError, match="boom"):
        FakeErrorClient().explode()

    assert emitted[0]["resource"] == "error_resource"
    assert emitted[0]["method_name"] == "explode"
    assert isinstance(emitted[0]["error"], RuntimeError)


def test_activate_skips_when_respan_tracing_is_disabled(monkeypatch, caplog) -> None:
    FakeDatasetsClient, _, _ = _install_fake_arize(monkeypatch)
    original_list = FakeDatasetsClient.list
    RespanTracer(is_enabled=False)

    instrumentor = ArizeInstrumentor(
        client_specs=(
            ArizeClientSpec(
                module_name="arize.datasets.client",
                class_name="DatasetsClient",
                resource="datasets",
                methods=("list",),
            ),
        )
    )
    with caplog.at_level(logging.INFO):
        instrumentor.activate()

    assert FakeDatasetsClient.list is original_list
    assert instrumentor._is_instrumented is False
    assert "Arize instrumentation skipped because Respan tracing is disabled" in caplog.text


def test_span_attributes_are_canonical() -> None:
    from respan_instrumentation_arize._span_emitter import build_arize_span_attributes

    attrs = build_arize_span_attributes(
        resource="datasets",
        method_name="list",
        args=(),
        kwargs={"space": "demo-space"},
        result={"datasets": []},
    )

    assert attrs[RESPAN_LOG_TYPE] == LOG_TYPE_TASK
    assert attrs[RESPAN_LOG_METHOD] == "tracing_integration"
    assert attrs[ARIZE_METADATA_RESOURCE] == "datasets"
    assert attrs[ARIZE_METADATA_OPERATION] == "list"
    assert "respan.span.tools" not in attrs
    assert "respan.span.tool_calls" not in attrs
    assert "tool_calls" not in attrs
    assert "tools" not in attrs


def test_completed_future_serializes_result() -> None:
    import concurrent.futures

    from respan_instrumentation_arize._serialization import safe_json_dumps

    future: concurrent.futures.Future = concurrent.futures.Future()
    future.set_result({"ok": True})

    assert '"result": {"ok": true}' in safe_json_dumps(future)
