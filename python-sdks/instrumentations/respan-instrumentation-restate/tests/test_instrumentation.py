import asyncio
from contextlib import contextmanager
from types import SimpleNamespace

from respan_instrumentation_restate import RestateInstrumentor
import respan_instrumentation_restate._instrumentation as instrumentation
from respan_sdk.constants.span_attributes import (
    RESPAN_LOG_TYPE,
    RESPAN_THREADS_ID,
    RESPAN_TRACE_GROUP_ID,
)
from opentelemetry.semconv_ai import SpanAttributes


class FakeSpan:
    def __init__(self, attributes: dict):
        self.attributes = dict(attributes)
        self.status = None
        self.exceptions = []

    def set_attribute(self, key, value):
        self.attributes[key] = value

    def set_status(self, status):
        self.status = status

    def record_exception(self, exception):
        self.exceptions.append(exception)


class FakeTracer:
    def __init__(self):
        self.span = None

    @contextmanager
    def start_as_current_span(self, name, *, attributes, **kwargs):
        del name, kwargs
        self.span = FakeSpan(attributes)
        yield self.span


class JsonSerde:
    def deserialize(self, value: bytes):
        return {"name": value.decode()}


def _fake_context():
    return SimpleNamespace(
        handler=SimpleNamespace(
            service_tag=SimpleNamespace(
                kind="workflow",
                name="OrderWorkflow",
                metadata={"team": "payments"},
            ),
            name="run",
            kind="workflow",
            metadata={"purpose": "checkout"},
            handler_io=SimpleNamespace(input_serde=JsonSerde()),
        ),
        invocation=SimpleNamespace(
            invocation_id="inv-123",
            input_buffer=b"Ada",
            key="order-42",
            scope="tenant",
            limit_key="tenant/ada",
            idempotency_key="checkout-1",
        ),
    )


def test_context_manager_maps_restate_invocation_fields(monkeypatch) -> None:
    fake_tracer = FakeTracer()
    context = _fake_context()
    server_context = SimpleNamespace(
        current_context=lambda: context,
        restate_context_is_replaying=SimpleNamespace(get=lambda: True),
    )
    real_import = instrumentation.importlib.import_module

    def import_module(name: str):
        if name == "restate.server_context":
            return server_context
        return real_import(name)

    monkeypatch.setattr(instrumentation.importlib, "import_module", import_module)
    monkeypatch.setattr(instrumentation.trace, "get_tracer", lambda name: fake_tracer)
    monkeypatch.setattr(instrumentation, "_ENABLED", True)
    monkeypatch.setattr(instrumentation, "_CAPTURE_CONTENT", True)

    async def run():
        async with instrumentation._invocation_context():
            pass

    asyncio.run(run())
    attrs = fake_tracer.span.attributes
    assert attrs[RESPAN_LOG_TYPE] == "workflow"
    assert attrs[RESPAN_TRACE_GROUP_ID] == "inv-123"
    assert attrs[RESPAN_THREADS_ID] == "order-42"
    assert '"input": {"name": "Ada"}' in attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT]
    assert '"replaying": true' in attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT]
    assert attrs["status_code"] == 200


def test_context_manager_records_handler_failure(monkeypatch) -> None:
    fake_tracer = FakeTracer()
    server_context = SimpleNamespace(
        current_context=_fake_context,
        restate_context_is_replaying=SimpleNamespace(get=lambda: False),
    )
    real_import = instrumentation.importlib.import_module

    def import_module(name: str):
        if name == "restate.server_context":
            return server_context
        return real_import(name)

    monkeypatch.setattr(instrumentation.importlib, "import_module", import_module)
    monkeypatch.setattr(instrumentation.trace, "get_tracer", lambda name: fake_tracer)
    monkeypatch.setattr(instrumentation, "_ENABLED", True)

    async def run():
        try:
            async with instrumentation._invocation_context():
                raise RuntimeError("deterministic Restate failure")
        except RuntimeError:
            pass

    asyncio.run(run())
    attrs = fake_tracer.span.attributes
    assert attrs["status_code"] == 500
    assert attrs["error.message"] == "deterministic Restate failure"
    assert len(fake_tracer.span.exceptions) == 1


def test_registration_injects_context_only_once() -> None:
    instance = SimpleNamespace(context_managers=None)
    instrumentation._ensure_context_manager(instance)
    instrumentation._ensure_context_manager(instance)
    assert instance.context_managers == [instrumentation._invocation_context]


def test_activate_and_deactivate_patch_all_restate_registration_paths(
    monkeypatch,
) -> None:
    wrapped: list[tuple[str, str]] = []
    unwrapped: list[tuple[str, str]] = []
    real_import = instrumentation.importlib.import_module

    def import_module(name: str):
        if name == "restate":
            return SimpleNamespace()
        return real_import(name)

    monkeypatch.setattr(instrumentation.importlib, "import_module", import_module)
    monkeypatch.setattr(
        instrumentation,
        "wrap_function_wrapper",
        lambda module, target, wrapper: wrapped.append((module, target)),
    )
    monkeypatch.setattr(
        instrumentation,
        "unwrap",
        lambda module, target: unwrapped.append((module, target)),
    )
    monkeypatch.setattr(instrumentation, "_ACTIVATION_COUNT", 0)
    monkeypatch.setattr(instrumentation, "_PATCHED_TARGETS", [])
    monkeypatch.setattr(instrumentation, "_ENABLED", False)

    adapter = RestateInstrumentor()
    adapter.activate()
    assert wrapped == list(instrumentation.RESTATE_REGISTRATION_TARGETS)
    adapter.deactivate()
    assert unwrapped == list(reversed(instrumentation.RESTATE_REGISTRATION_TARGETS))
