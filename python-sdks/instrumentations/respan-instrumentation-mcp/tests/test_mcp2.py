"""Released MCP 2.x client/server behavior, including input continuations."""

import asyncio
import json

import pytest

mcpserver = pytest.importorskip("mcp.server.mcpserver")

from mcp import Client
from mcp.server.mcpserver import Context, MCPServer
from mcp.types import InputRequiredResult
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.semconv._incubating.attributes.mcp_attributes import MCP_METHOD_NAME
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.trace import StatusCode
from respan_instrumentation_mcp import MCPInstrumentor, _instrumentation


def test_mcp2_high_level_client_continuation_and_tool_error(monkeypatch):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(_instrumentation.trace, "_TRACER_PROVIDER", provider)
    from mcp.shared import _otel

    monkeypatch.setattr(_otel, "_tracer", provider.get_tracer("mcp-python-sdk"))
    monkeypatch.setattr(
        _instrumentation.trace,
        "get_tracer",
        lambda name, *args, **kwargs: provider.get_tracer(name),
    )
    server = MCPServer("respan-mcp2-features")

    @server.tool()
    def confirm(ctx: Context) -> str | InputRequiredResult:
        if not ctx.request_state:
            return InputRequiredResult(
                input_requests={}, request_state="awaiting-confirmation"
            )
        return "confirmed"

    @server.tool()
    def fail() -> str:
        raise ValueError("controlled MCP tool error")

    async def exercise():
        async with Client(server, mode="2026-07-28") as client:
            pending = await client.session.call_tool(
                "confirm", allow_input_required=True
            )
            assert isinstance(pending, InputRequiredResult)
            completed = await client.session.call_tool(
                "confirm", input_responses={}, request_state=pending.request_state
            )
            assert completed.content[0].text == "confirmed"
            failure = await client.call_tool("fail")
            assert failure.is_error
            return pending.request_state

    instrumentor = MCPInstrumentor()
    instrumentor.activate()
    try:
        state = asyncio.run(exercise())
    finally:
        instrumentor.deactivate()
        provider.shutdown()
    spans = list(exporter.get_finished_spans())
    calls = [span for span in spans if span.name.startswith("mcp.tool.")]
    assert len(calls) == 3
    pending = json.loads(calls[0].attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT])
    assert pending["resultType"] == "input_required"
    assert "requestState" in pending
    resumed = json.loads(calls[1].attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT])
    assert resumed["input_responses"] == {}
    assert resumed["request_state"] == state
    assert calls[2].status.status_code is StatusCode.ERROR
    failure = json.loads(calls[2].attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT])
    assert failure["isError"] is True
    assert calls[1].status.status_code is not StatusCode.ERROR


def test_mcp2_resource_and_prompt_input_continuations(monkeypatch):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(_instrumentation.trace, "_TRACER_PROVIDER", provider)
    from mcp.shared import _otel

    monkeypatch.setattr(_otel, "_tracer", provider.get_tracer("mcp-python-sdk"))
    monkeypatch.setattr(
        _instrumentation.trace,
        "get_tracer",
        lambda name, *args, **kwargs: provider.get_tracer(name),
    )
    server = MCPServer("respan-mcp2-resource-input")

    @server.resource("text://{item}")
    def text(item: str, ctx: Context) -> str | InputRequiredResult:
        if not ctx.request_state:
            return InputRequiredResult(input_requests={}, request_state=item)
        return f"resource {item}"

    @server.prompt()
    def confirmation(ctx: Context) -> str | InputRequiredResult:
        if not ctx.request_state:
            return InputRequiredResult(input_requests={}, request_state="prompt")
        return "confirmed prompt"

    async def exercise():
        async with Client(server, mode="2026-07-28") as client:
            resource = await client.session.read_resource(
                "text://demo", allow_input_required=True
            )
            assert isinstance(resource, InputRequiredResult)
            result = await client.session.read_resource(
                "text://demo", request_state=resource.request_state, input_responses={}
            )
            assert result.contents[0].text == "resource demo"
            prompt = await client.session.get_prompt(
                "confirmation", allow_input_required=True
            )
            assert isinstance(prompt, InputRequiredResult)
            result = await client.session.get_prompt(
                "confirmation", request_state=prompt.request_state, input_responses={}
            )
            assert result.messages[0].content.text == "confirmed prompt"

    instrumentor = MCPInstrumentor()
    instrumentor.activate()
    try:
        asyncio.run(exercise())
    finally:
        instrumentor.deactivate()
        provider.shutdown()
    spans = list(exporter.get_finished_spans())
    assert len(spans) == 4
    for span in (spans[1], spans[3]):
        request = json.loads(span.attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT])
        assert request["request_state"]
        assert request["input_responses"] == {}
    for span in (spans[0], spans[2]):
        assert (
            json.loads(span.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT])[
                "resultType"
            ]
            == "input_required"
        )


@pytest.mark.parametrize("override", [False, True])
def test_mcp2_content_switch_and_owner_lifecycle(monkeypatch, override):
    from mcp import ClientSession
    from openinference.instrumentation.mcp import MCPInstrumentor as Upstream
    from opentelemetry import context

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(_instrumentation.trace, "_TRACER_PROVIDER", provider)
    from mcp.shared import _otel

    monkeypatch.setattr(_otel, "_tracer", provider.get_tracer("mcp-python-sdk"))
    monkeypatch.setattr(
        _instrumentation.trace,
        "get_tracer",
        lambda name, *args, **kwargs: provider.get_tracer(name),
    )
    monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")
    server = MCPServer("respan-mcp2-private")

    @server.tool()
    def echo(text: str) -> str:
        return text

    upstream = Upstream()
    upstream.instrument()
    original = ClientSession.call_tool
    first, second = MCPInstrumentor(), MCPInstrumentor()
    first.activate()
    second.activate()
    first.deactivate()

    async def exercise():
        token = context.attach(
            context.set_value("override_enable_content_tracing", override)
        )
        try:
            async with Client(server, mode="2026-07-28") as client:
                result = await client.call_tool("echo", {"text": "private payload"})
                assert result.content[0].text == "private payload"
        finally:
            context.detach(token)

    try:
        asyncio.run(exercise())
        spans = exporter.get_finished_spans()
        tool_spans = [span for span in spans if span.name == "mcp.tool.echo"]
        assert len(tool_spans) == 1
        assert {span.name for span in spans} <= {"mcp.tool.echo", "mcp.list_tools"}
        attrs = tool_spans[0].attributes
        assert (SpanAttributes.TRACELOOP_ENTITY_INPUT in attrs) is override
        assert (SpanAttributes.TRACELOOP_ENTITY_OUTPUT in attrs) is override
        assert ("private payload" in str(attrs)) is override
    finally:
        second.deactivate()
        assert upstream.is_instrumented_by_opentelemetry
        upstream.uninstrument()
        provider.shutdown()
    assert ClientSession.call_tool is original


def test_native_sdk_protocol_spans_are_suppressed_only_for_the_owned_operation(
    monkeypatch,
):
    from mcp.shared import _otel
    from opentelemetry.trace import SpanKind
    from respan_instrumentation_mcp._native import owned_operation

    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(_instrumentation.trace, "_TRACER_PROVIDER", provider)
    monkeypatch.setattr(_otel, "_tracer", provider.get_tracer("mcp-python-sdk"))
    tracer = provider.get_tracer("scope-test")
    inst = MCPInstrumentor()
    inst.activate()
    try:
        with (
            tracer.start_as_current_span("owned") as owner,
            owned_operation("call_tool", "example"),
        ):
            with _otel.otel_span(
                "MCP send tools/call example",
                kind=SpanKind.CLIENT,
                attributes={MCP_METHOD_NAME: "tools/call"},
            ) as duplicate:
                assert not duplicate.is_recording()
                assert duplicate.get_span_context() == owner.get_span_context()
            with _otel.otel_span(
                "MCP send ping",
                kind=SpanKind.CLIENT,
                attributes={MCP_METHOD_NAME: "ping"},
            ):
                pass
            with _otel.otel_span(
                "custom tool work",
                kind=SpanKind.INTERNAL,
                attributes={MCP_METHOD_NAME: "tools/call"},
            ):
                pass
            with (
                tracer.start_as_current_span("independent"),
                _otel.otel_span(
                    "tools/call example",
                    kind=SpanKind.SERVER,
                    attributes={MCP_METHOD_NAME: "tools/call"},
                ),
            ):
                pass
        # A separately served SDK request has no in-process owner marker.
        with _otel.otel_span(
            "tools/call standalone",
            kind=SpanKind.SERVER,
            attributes={MCP_METHOD_NAME: "tools/call"},
        ):
            pass
    finally:
        inst.deactivate()
        provider.shutdown()
    spans = exporter.get_finished_spans()
    assert {span.name for span in spans} == {
        "owned",
        "MCP send ping",
        "custom tool work",
        "independent",
        "tools/call example",
        "tools/call standalone",
    }
    assert len(spans) == 6


def test_native_sdk_helpers_restore_after_repeated_activation():
    from mcp.server import _otel as server_otel
    from mcp.shared import _otel, jsonrpc_dispatcher

    modules = (_otel, jsonrpc_dispatcher, server_otel)
    originals = [module.otel_span for module in modules]
    for _ in range(2):
        inst = MCPInstrumentor()
        inst.activate()
        assert all(
            module.otel_span is not original
            for module, original in zip(modules, originals)
        )
        inst.deactivate()
        assert all(
            module.otel_span is original for module, original in zip(modules, originals)
        )
