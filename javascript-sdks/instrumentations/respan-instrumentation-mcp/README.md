# @respan/instrumentation-mcp

Trace real Model Context Protocol client and server operations through Respan.
The adapter supports `@modelcontextprotocol/sdk` v1 (`>=1.0.0 <2`) and the
stable split `@modelcontextprotocol/client` / `@modelcontextprotocol/server`
v2 packages (`>=2.0.0 <3`). Install the SDK generation your application uses.
Node.js 18 or later is required for v1; the split v2 SDK requires Node.js 20.

```bash
npm install @respan/respan @respan/instrumentation-mcp @modelcontextprotocol/client @modelcontextprotocol/server
```

```typescript
import { Client, InMemoryTransport } from "@modelcontextprotocol/client";
import { McpServer } from "@modelcontextprotocol/server";
import { Respan } from "@respan/respan";
import { MCPInstrumentor } from "@respan/instrumentation-mcp";

const respan = new Respan({ instrumentations: [new MCPInstrumentor()] });
await respan.initialize();
const server = new McpServer({ name: "example-server", version: "1.0.0" });
server.registerTool(
  "hello",
  { description: "Return a greeting" },
  async () => ({
    content: [{ type: "text", text: "Hello from MCP" }],
  }),
);
const client = new Client({ name: "example-client", version: "1.0.0" });
const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair();
await server.connect(serverTransport);
await client.connect(clientTransport);
try {
  await respan.withWorkflow({ name: "mcp-example" }, async () => {
    await client.listTools();
    console.log(await client.callTool({ name: "hello", arguments: {} }));
  });
} finally {
  await client.close();
  await server.close();
  await respan.shutdown();
}
```

For v1, install `@modelcontextprotocol/sdk` and import its client/server
subpaths. The adapter detects installed v1 and v2 modules. Pass `clientModule`
and `serverModule`, or `clientModules` and `serverModules`, when using a
bundler or independently loaded SDK copies. For v1 high-level servers, include
both `server/index.js` and `server/mcp.js` exports in `serverModule` to capture
all protocol handlers. A high-level-only module retains legacy tool callback
support.

The adapter captures native `request()` and `notification()` operations and
registered handlers. This includes tools, resource and prompt pagination,
resource subscriptions, completion, roots, logging, sampling, elicitation,
progress and cancellation. On v1 versions exposing experimental tasks, native
task creation, status, result, list and cancel requests are captured without
wrapping or replacing the returned async iterator. The v1.0.0 minimum has no
high-level `McpServer`, elicitation or experimental tasks API. The stable v2
SDK does not expose v1 experimental tasks. In SDK1.0.0, aborting a client request
rejects its promise, but the SDK sends a legacy cancellation method that its
server does not handle; the server callback continues until its actual result.
The adapter preserves that native boundary.

`tools/call` emits a tool execution span with `{name, arguments}` and the
actual result. Other protocol operations emit task spans with their actual
method, parameters and result. Sampling is protocol work: the adapter does
not infer a provider, model, token usage, or an LLM child span. Native JSON-RPC
IDs are retained as `mcp.request.id`; they are not model tool-call IDs.
Complete tool definitions, multimodal blocks, structured results, `_meta`,
pagination cursors and falsy values are retained in canonical JSON input and
output without adapter length, depth or collection caps.

Content capture occurs after the actual OpenTelemetry sampler admits the
span. Set `traceContent: false`, `TRACELOOP_TRACE_CONTENT=false`, or the
canonical Traceloop content context to disable it. Tracing suppression calls
through to the SDK. Constructor settings, observed context vetoes and denied
ancestors remain ceilings for the operation. `recordInputs` and
`recordOutputs` independently disable their canonical payload lane. Denied
spans also reject late payload attributes, events, exceptions and status
messages on the original SDK span. Application getters, `toJSON` hooks and
proxy objects are not evaluated by serialization.

`captureClientOperations` and `captureServerTools` control client and server
capture respectively; the latter includes server protocol callbacks as well
as tools. `activate()` coalesces concurrent activation. `deactivate()` restores
owned patches, keeps foreign wrappers, stops registered callback tracing and
allows spans already in flight to finish. `isActive()` reports plugin state.

The existing public serialization and client-name utility exports remain
available. Serialization now retains complete enumerable data properties,
including `_meta`, rather than truncating at 16,000 characters or six levels.
