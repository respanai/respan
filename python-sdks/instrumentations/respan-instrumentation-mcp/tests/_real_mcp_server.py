"""Real MCP 1.x/2.x stdio server for instrumentation integration tests."""

from __future__ import annotations

import sys

if "--exit-immediately" in sys.argv:
    raise SystemExit(3)

if "--without-openinference" not in sys.argv:
    try:
        from openinference.instrumentation.mcp import (
            MCPInstrumentor as OpenInferenceMCPInstrumentor,
        )
    except ImportError:
        OpenInferenceMCPInstrumentor = None
    else:
        OpenInferenceMCPInstrumentor().instrument()

try:
    from mcp.server.mcpserver import MCPServer
except ImportError:
    from mcp.server.fastmcp import FastMCP as MCPServer
    from mcp.server.fastmcp.server import Settings

    Settings.model_rebuild(force=True)

server = MCPServer("respan-mcp-instrumentation-test")


@server.tool()
def summarize_city(city: str) -> str:
    return f"{city}: ready"


@server.tool()
def current_trace_id() -> str:
    try:
        from opentelemetry import trace
    except ImportError:
        return ""

    span_context = trace.get_current_span().get_span_context()
    if not span_context.is_valid:
        return ""
    return f"{span_context.trace_id:032x}"


@server.tool()
def fail() -> str:
    raise ValueError("controlled MCP server error")


if __name__ == "__main__":
    server.run(transport="stdio")
