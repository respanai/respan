import asyncio
import json
import logging
import sys
from builtins import ExceptionGroup
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import ClassVar

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client import stdio as mcp_stdio
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.trace import StatusCode
from respan_instrumentation_mcp import MCPInstrumentor, _instrumentation
from respan_instrumentation_mcp._instrumentation import (
    _MAX_ATTRIBUTE_CHARS,
    MCP_CLIENT_SESSION_MODULE,
    OPENINFERENCE_MCP_MODULE,
    _json_dumps,
)
from respan_sdk.constants.llm_logging import LOG_TYPE_TASK, LOG_TYPE_TOOL
from respan_sdk.constants.span_attributes import RESPAN_LOG_TYPE
from respan_tracing.core.tracer import RespanTracer


class FakeSpan:
    def __init__(self, name):
        self.name = name
        self.attributes = {}
        self.exceptions = []
        self.status = None

    def set_attribute(self, key, value):
        self.attributes[key] = value

    def record_exception(self, exc):
        self.exceptions.append(exc)

    def set_status(self, status):
        self.status = status


class FakeTracer:
    def __init__(self):
        self.spans = []

    @contextmanager
    def start_as_current_span(self, name):
        span = FakeSpan(name)
        self.spans.append(span)
        yield span


def _install_fake_modules(monkeypatch):
    class FakeOpenInferenceMCPInstrumentor:
        pass

    class FakeOpenInferenceInstrumentor:
        created: ClassVar[list] = []

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

    class FakeClientSession:
        async def list_tools(self):
            return SimpleNamespace(
                tools=[SimpleNamespace(name="summarize_city", description="demo")]
            )

        async def call_tool(self, name, arguments=None):
            return SimpleNamespace(content=[{"type": "text", "text": f"{name}: ok"}])

    openinference_module = ModuleType("openinference")
    openinference_instrumentation_module = ModuleType("openinference.instrumentation")
    openinference_mcp_module = ModuleType(OPENINFERENCE_MCP_MODULE)
    openinference_mcp_module.MCPInstrumentor = FakeOpenInferenceMCPInstrumentor
    openinference_instrumentation_module.mcp = openinference_mcp_module

    mcp_module = ModuleType("mcp")
    mcp_client_module = ModuleType("mcp.client")
    mcp_client_session_module = ModuleType(MCP_CLIENT_SESSION_MODULE)
    mcp_client_session_module.ClientSession = FakeClientSession
    mcp_client_module.session = mcp_client_session_module
    mcp_module.client = mcp_client_module

    monkeypatch.setitem(sys.modules, "openinference", openinference_module)
    monkeypatch.setitem(
        sys.modules,
        "openinference.instrumentation",
        openinference_instrumentation_module,
    )
    monkeypatch.setitem(sys.modules, OPENINFERENCE_MCP_MODULE, openinference_mcp_module)
    monkeypatch.setitem(sys.modules, "mcp", mcp_module)
    monkeypatch.setitem(sys.modules, "mcp.client", mcp_client_module)
    monkeypatch.setitem(
        sys.modules, MCP_CLIENT_SESSION_MODULE, mcp_client_session_module
    )
    monkeypatch.setattr(
        _instrumentation,
        "OpenInferenceInstrumentor",
        FakeOpenInferenceInstrumentor,
    )

    return SimpleNamespace(
        client_session_class=FakeClientSession,
        openinference_mcp_class=FakeOpenInferenceMCPInstrumentor,
        delegate_class=FakeOpenInferenceInstrumentor,
    )


@pytest.fixture(autouse=True)
def reset_state():
    RespanTracer.reset_instance()
    MCPInstrumentor._patches_applied = False
    MCPInstrumentor._activation_count = 0
    MCPInstrumentor._shared_delegate = None
    MCPInstrumentor._shared_patched_methods.clear()
    yield
    RespanTracer.reset_instance()
    MCPInstrumentor._patches_applied = False
    MCPInstrumentor._activation_count = 0
    MCPInstrumentor._shared_delegate = None
    MCPInstrumentor._shared_patched_methods.clear()


def test_activate_delegates_to_openinference_and_traces_client_calls(monkeypatch):
    fake = _install_fake_modules(monkeypatch)
    fake_tracer = FakeTracer()
    monkeypatch.setattr(_instrumentation.trace, "get_tracer", lambda _name: fake_tracer)

    instrumentor = MCPInstrumentor(capture_headers=True)
    instrumentor.activate()

    delegate = fake.delegate_class.created[0]
    assert delegate.instrumentor_class is fake.openinference_mcp_class
    assert delegate.kwargs == {"capture_headers": True}
    assert delegate.is_activated is True
    assert instrumentor._is_instrumented is True

    session = fake.client_session_class()
    result = asyncio.run(session.call_tool("summarize_city", {"city": "Paris"}))

    assert result.content[0]["text"] == "summarize_city: ok"
    span = fake_tracer.spans[0]
    assert span.name == "mcp.tool.summarize_city"
    assert span.attributes[RESPAN_LOG_TYPE] == LOG_TYPE_TOOL
    assert span.attributes[SpanAttributes.TRACELOOP_ENTITY_NAME] == "summarize_city"
    assert span.attributes[SpanAttributes.TRACELOOP_ENTITY_PATH] == "summarize_city"
    assert span.attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT] == (
        '{"arguments": {"city": "Paris"}, "name": "summarize_city"}'
    )
    assert "openinference.span.kind" not in span.attributes
    assert SpanAttributes.TRACELOOP_SPAN_KIND not in span.attributes

    instrumentor.deactivate()

    assert delegate.is_deactivated is True
    assert instrumentor._is_instrumented is False


def test_multiple_instances_share_patch_and_delegate_lifecycle(monkeypatch):
    fake = _install_fake_modules(monkeypatch)
    fake_tracer = FakeTracer()
    monkeypatch.setattr(_instrumentation.trace, "get_tracer", lambda _name: fake_tracer)

    first = MCPInstrumentor(capture_headers=True)
    second = MCPInstrumentor(capture_headers=False)
    first.activate()
    second.activate()

    assert len(fake.delegate_class.created) == 1
    delegate = fake.delegate_class.created[0]
    assert MCPInstrumentor._activation_count == 2
    assert MCPInstrumentor._patches_applied is True

    first.deactivate()
    assert MCPInstrumentor._activation_count == 1
    assert MCPInstrumentor._patches_applied is True
    assert delegate.is_deactivated is False

    session = fake.client_session_class()
    asyncio.run(session.call_tool("summarize_city", {"city": "Paris"}))
    assert [span.name for span in fake_tracer.spans] == ["mcp.tool.summarize_city"]

    second.deactivate()
    assert MCPInstrumentor._activation_count == 0
    assert MCPInstrumentor._patches_applied is False
    assert MCPInstrumentor._shared_patched_methods == []
    assert delegate.is_deactivated is True

    asyncio.run(session.call_tool("summarize_city", {"city": "Rome"}))
    assert [span.name for span in fake_tracer.spans] == ["mcp.tool.summarize_city"]


def test_trace_list_tools_uses_task_log_type(monkeypatch):
    fake_tracer = FakeTracer()
    monkeypatch.setattr(_instrumentation.trace, "get_tracer", lambda _name: fake_tracer)

    async def wrapped():
        return SimpleNamespace(tools=[SimpleNamespace(name="first")])

    instrumentor = MCPInstrumentor()
    result = asyncio.run(
        instrumentor._trace_async_method("list_tools", wrapped, (), {})
    )

    assert result.tools[0].name == "first"
    span = fake_tracer.spans[0]
    assert span.name == "mcp.list_tools"
    assert span.attributes[RESPAN_LOG_TYPE] == LOG_TYPE_TASK
    assert span.attributes[SpanAttributes.TRACELOOP_ENTITY_NAME] == "mcp.list_tools"
    assert span.attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT] == (
        '{"method": "list_tools"}'
    )
    assert (
        '"tools": [{"name": "first"}]'
        in span.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]
    )


def test_json_serializer_preserves_deep_schema_primitives_and_user_quotes():
    value = {
        "tools": [
            {
                "inputSchema": {
                    "properties": {
                        "city": {
                            "title": "City",
                            "type": "string",
                            "description": 'Keep the user\'s "quoted" label',
                        },
                        "rate": {
                            "default": 0.0875,
                            "type": "number",
                        },
                    },
                    "type": "object",
                }
            }
        ]
    }

    serialized = json.loads(_json_dumps(value))
    properties = serialized["tools"][0]["inputSchema"]["properties"]
    assert properties["city"] == {
        "title": "City",
        "type": "string",
        "description": 'Keep the user\'s "quoted" label',
    }
    assert properties["rate"] == {"default": 0.0875, "type": "number"}


def test_json_serializer_bounds_deep_non_schema_containers():
    value = {
        "level": {
            "level": {
                "level": {
                    "level": {"level": {"level": {"level": {"value": "bounded"}}}}
                }
            }
        }
    }

    serialized = json.loads(_json_dumps(value))
    cursor = serialized
    for _ in range(7):
        cursor = cursor["level"]
    assert cursor == {"truncated": True, "type": "builtins.dict"}


def test_json_serializer_redacts_sensitive_keys_and_avoids_repr_leaks():
    serialized = _json_dumps(
        {
            "headers": {
                "Authorization": "Bearer private-auth",
                "x-api-key": "private-api-key",
            },
            "credentials": {"password": "private-password"},
            "usage": {"token_count": 17},
            "quoted_user_text": 'Keep "these" quotes',
            "opaque": object(),
        }
    )
    parsed = json.loads(serialized)

    assert parsed["headers"] == {
        "Authorization": "<redacted>",
        "x-api-key": "<redacted>",
    }
    assert parsed["credentials"] == "<redacted>"
    assert parsed["usage"] == {"token_count": 17}
    assert parsed["quoted_user_text"] == 'Keep "these" quotes'
    assert parsed["opaque"] == {"type": "builtins.object"}
    assert "private-" not in serialized
    assert "0x" not in serialized


def test_json_serializer_oversize_output_is_valid_json_wrapper():
    serialized = _json_dumps({"value": "x" * (_MAX_ATTRIBUTE_CHARS * 2)})
    parsed = json.loads(serialized)

    assert parsed["truncated"] is True
    assert parsed["original_length"] > _MAX_ATTRIBUTE_CHARS
    assert parsed["preview"].startswith('{"value": "')
    assert len(serialized) <= _MAX_ATTRIBUTE_CHARS
    assert len(parsed["preview"]) < _MAX_ATTRIBUTE_CHARS


def test_trace_error_sets_status_and_error_output(monkeypatch):
    fake_tracer = FakeTracer()
    monkeypatch.setattr(_instrumentation.trace, "get_tracer", lambda _name: fake_tracer)

    async def wrapped(*args, **kwargs):
        raise RuntimeError("tool failed")

    instrumentor = MCPInstrumentor()

    with pytest.raises(RuntimeError, match="tool failed"):
        asyncio.run(
            instrumentor._trace_async_method(
                "call_tool",
                wrapped,
                ("broken_tool",),
                {"arguments": {"x": 1}},
            )
        )

    span = fake_tracer.spans[0]
    assert span.attributes[RESPAN_LOG_TYPE] == LOG_TYPE_TOOL
    assert span.exceptions and isinstance(span.exceptions[0], RuntimeError)
    assert span.status.status_code.name == "ERROR"
    assert span.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] == (
        '{"error": "RuntimeError", "message": "tool failed"}'
    )


def test_activate_skips_when_respan_tracing_is_disabled(monkeypatch, caplog):
    fake = _install_fake_modules(monkeypatch)
    RespanTracer(is_enabled=False)

    instrumentor = MCPInstrumentor()
    with caplog.at_level(logging.INFO):
        instrumentor.activate()

    assert fake.delegate_class.created == []
    assert instrumentor._is_instrumented is False
    assert (
        "MCP instrumentation skipped because Respan tracing is disabled" in caplog.text
    )


def test_real_mcp_pydantic_export_and_connection_failure(monkeypatch):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(
        _instrumentation.trace,
        "get_tracer",
        lambda name: provider.get_tracer(name),
    )
    monkeypatch.setattr(
        _instrumentation.trace,
        "get_tracer_provider",
        lambda: provider,
    )

    instrumentor = MCPInstrumentor()
    server_script = Path(__file__).with_name("_real_mcp_server.py")
    success_marker = "mcp-real-success"
    failure_marker = "mcp-real-failure"

    async def run_session(args: list[str]) -> str:
        parameters = StdioServerParameters(
            command=sys.executable,
            args=[str(server_script), *args],
        )
        async with (
            mcp_stdio.stdio_client(parameters) as (read, write),
            ClientSession(read, write) as session,
        ):
            await session.initialize()
            tools = await session.list_tools()
            assert [tool.name for tool in tools.tools] == [
                "summarize_city",
                "current_trace_id",
                "fail",
            ]
            result = await session.call_tool(
                "summarize_city",
                arguments={"city": "Paris"},
            )
            assert result.content[0].text == "Paris: ready"
            trace_result = await session.call_tool("current_trace_id")
            failure = await session.call_tool("fail")
            assert getattr(failure, "is_error", None) or getattr(
                failure, "isError", None
            )
            return trace_result.content[0].text

    async def exercise() -> tuple[str, ExceptionGroup]:
        tracer = provider.get_tracer("respan.mcp.real-test")
        with tracer.start_as_current_span(
            "mcp.real.success",
            attributes={"test.run_id": success_marker},
        ):
            server_trace_id = await run_session([])

        try:
            with tracer.start_as_current_span(
                "mcp.real.failure",
                attributes={"test.run_id": failure_marker},
            ):
                await run_session(["--exit-immediately"])
        except ExceptionGroup as exc:
            return server_trace_id, exc
        raise AssertionError(
            "The deliberate MCP connection failure unexpectedly succeeded"
        )

    instrumentor.activate()
    try:
        assert instrumentor._delegate is not None
        assert instrumentor._delegate._is_instrumented is True
        server_trace_id, failure = asyncio.run(exercise())
        assert "Connection closed" in repr(failure)
        assert provider.force_flush()
        spans = list(exporter.get_finished_spans())
    finally:
        instrumentor.deactivate()
        provider.shutdown()

    # Assert the complete raw export multiset before looking up individual spans.
    assert Counter(span.name for span in spans) == Counter(
        {
            "mcp.real.success": 1,
            "mcp.initialize": 2,
            "mcp.list_tools": 1,
            "mcp.tool.summarize_city": 1,
            "mcp.tool.current_trace_id": 1,
            "mcp.tool.fail": 1,
            "mcp.real.failure": 1,
        }
    )
    assert len({span.context.span_id for span in spans}) == len(spans)

    success_root = next(span for span in spans if span.name == "mcp.real.success")
    failure_root = next(span for span in spans if span.name == "mcp.real.failure")
    assert success_root.attributes["test.run_id"] == success_marker
    assert failure_root.attributes["test.run_id"] == failure_marker
    assert server_trace_id == f"{success_root.context.trace_id:032x}"

    tool_failure = next(span for span in spans if span.name == "mcp.tool.fail")
    assert tool_failure.status.status_code is StatusCode.ERROR
    assert (
        json.loads(tool_failure.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT])[
            "isError"
        ]
        is True
    )
    list_tools_span = next(span for span in spans if span.name == "mcp.list_tools")
    list_tools_output = json.loads(
        list_tools_span.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]
    )
    schema = list_tools_output["tools"][0]["inputSchema"]
    assert schema["properties"]["city"] == {"title": "City", "type": "string"}
    assert schema["type"] == "object"
    assert list_tools_span.parent.span_id == success_root.context.span_id

    tool_span = next(span for span in spans if span.name == "mcp.tool.summarize_city")
    assert json.loads(tool_span.attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT]) == {
        "arguments": {"city": "Paris"},
        "name": "summarize_city",
    }
    assert tool_span.parent.span_id == success_root.context.span_id

    initialize_spans = [span for span in spans if span.name == "mcp.initialize"]
    failed_initialize = next(
        span for span in initialize_spans if span.status.status_code is StatusCode.ERROR
    )
    assert failed_initialize.parent.span_id == failure_root.context.span_id
    failure_output = json.loads(
        failed_initialize.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]
    )
    assert failure_output["error"] in {"McpError", "MCPError"}
    assert failure_output["message"] == "Connection closed"
    assert failure_root.status.status_code is StatusCode.ERROR
    assert all(span.name != "MCP send initialize" for span in spans)
    assert MCPInstrumentor._patches_applied is False
    assert MCPInstrumentor._activation_count == 0
    assert MCPInstrumentor._shared_patched_methods == []


def test_cancellation_is_recorded_and_propagated(monkeypatch):
    fake_tracer = FakeTracer()
    monkeypatch.setattr(_instrumentation.trace, "get_tracer", lambda _name: fake_tracer)

    async def cancelled():
        raise asyncio.CancelledError("cancelled MCP operation")

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(MCPInstrumentor._trace_async_method("call_tool", cancelled, (), {}))
    assert fake_tracer.spans[0].status.status_code is StatusCode.ERROR


def test_disabled_content_hides_resource_uri(monkeypatch):
    fake_tracer = FakeTracer()
    monkeypatch.setattr(_instrumentation.trace, "get_tracer", lambda _name: fake_tracer)
    monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")

    async def read(*args):
        return {"contents": "private resource text"}

    asyncio.run(
        MCPInstrumentor._trace_async_method(
            "read_resource", read, ("private://secret",), {}
        )
    )
    assert "private" not in str(fake_tracer.spans[0].attributes)
    assert SpanAttributes.TRACELOOP_ENTITY_INPUT not in fake_tracer.spans[0].attributes
    assert SpanAttributes.TRACELOOP_ENTITY_OUTPUT not in fake_tracer.spans[0].attributes


def test_partial_activation_restores_patches_and_delegate(monkeypatch):
    fake = _install_fake_modules(monkeypatch)
    original = fake.client_session_class.list_tools
    wrap = _instrumentation.wrap_function_wrapper

    def fail_second(module, target, wrapper):
        if target.endswith("call_tool"):
            raise RuntimeError("patch rejected")
        return wrap(module, target, wrapper)

    monkeypatch.setattr(_instrumentation, "wrap_function_wrapper", fail_second)
    instrumentor = MCPInstrumentor()
    instrumentor.activate()
    assert not instrumentor._is_instrumented
    assert MCPInstrumentor._activation_count == 0
    assert MCPInstrumentor._shared_patched_methods == []
    assert fake.client_session_class.list_tools is original
    assert fake.delegate_class.created[0].is_deactivated
