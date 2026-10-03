# respan-instrumentation-mcp

Respan instrumentation plugin for [Model Context Protocol](https://modelcontextprotocol.io/) Python applications.

The package enables upstream OpenInference MCP transport context propagation and adds Respan spans for common `mcp.ClientSession` operations such as tool listing, tool calls, resource reads, and prompt fetches.

## Configuration

### 1. Install

```bash
pip install respan-ai respan-instrumentation-mcp "mcp>=1.27,<3"
```

### 2. Set Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `RESPAN_API_KEY` | Yes | Your Respan API key. Authenticates tracing export. |
| `RESPAN_BASE_URL` | No | Defaults to `https://api.respan.ai/api`. |

## Quickstart

```python
import asyncio
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from respan import Respan, workflow
from respan_instrumentation_mcp import MCPInstrumentor

respan = Respan(instrumentations=[MCPInstrumentor()])


@workflow(name="mcp_tool_call_workflow")
async def run_mcp_client() -> None:
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["server.py"],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            print([tool.name for tool in tools.tools])
            result = await session.call_tool(
                "summarize_city",
                arguments={"city": "Paris"},
            )
            print(result)


asyncio.run(run_mcp_client())
respan.flush()
respan.shutdown()
```

Supports MCP Python SDK 1.27 and the 2.x line, validated with released 1.27.0
and 2.3.0 plus OpenInference MCP 2.0.12 and the Respan OpenInference bridge
1.2.5. MCP protocol attribute constants require OpenTelemetry semantic
conventions 0.65b0 or newer. The SDK 2 `Client` delegates its operations
to `ClientSession`, so the same instrumentation captures those calls without
adding an extra wrapper span. SDK 2 servers use `mcp.server.mcpserver.MCPServer`;
SDK 1 servers use `mcp.server.fastmcp.FastMCP`.

Tool calls, resource reads, and prompt fetches preserve MCP 2 input-required
responses and subsequent `input_responses` / `request_state` arguments. Result
models serialize with MCP wire aliases, including `inputSchema`, `resultType`,
and `isError`. A tool result whose error flag is true produces an error span
while returning the original result to the caller. SDK exceptions and task
cancellation are recorded and propagated unchanged.

Set `TRACELOOP_TRACE_CONTENT=false` to omit request and response content. The
standard `override_enable_content_tracing` context flag can enable content for
a scope. Captured payloads retain the existing sensitive-key redaction and size
bounds. Resource URIs remain in captured inputs instead of span names. Multiple
instrumentor instances share patches until the final owner deactivates, and a
pre-existing upstream instrumentor remains active.

## Further Reading

See the [Respan example projects](https://github.com/respanai/respan-example-projects) for runnable MCP scripts.

MCP 2's native protocol spans are suppressed only while an already-traced
client operation owns the same protocol method, native span name, and trace/span
context. Context injection retains that operation's identity. Unrelated
protocol methods, work beneath a different parent, and independently served
server requests retain their SDK spans.
