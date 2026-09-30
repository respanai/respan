# Respan Dify instrumentation

Trace the official `dify-client` Node.js SDK with Respan.

This package supports Dify `3.1.x` on Node.js 18 or newer. It patches the
SDK's exported `HttpClient`, so Chat, Completion, Workflow, Knowledge Base,
Workspace, file, feedback, conversation, and generic Service API operations
are covered without maintaining a fragile list of high-level methods.
Blocking calls emit when their promise resolves. SSE and binary spans remain
open until the returned stream is exhausted, closed early, or fails.

## Install

```bash
npm install @respan/respan @respan/instrumentation-dify dify-client
```

## Usage

```ts
import { ChatClient } from "dify-client";
import { Respan } from "@respan/respan";
import { DifyInstrumentor } from "@respan/instrumentation-dify";

const respan = new Respan({
  apiKey: process.env.RESPAN_API_KEY,
  instrumentations: [new DifyInstrumentor()],
});
await respan.initialize();

const client = new ChatClient({
  apiKey: process.env.DIFY_CHAT_API_KEY!,
  baseUrl: process.env.DIFY_BASE_URL,
});
const response = await client.createChatMessage({
  inputs: {},
  query: "Reply with one concise sentence about tracing.",
  user: "respan-example-user",
  response_mode: "blocking",
});
console.log(response.data.answer);

await respan.shutdown();
```

For streaming calls, consume the documented `AsyncIterable` (or call
`toText()`). The instrumentation preserves `data`, `status`, `headers`,
`requestId`, `toReadable()`, and `toText()` on the original Dify stream.

Pass `includeContent: false` to omit request/response bodies and
prompt/completion content while retaining operation, status, correlation, and
token-usage attributes:

```ts
new DifyInstrumentor({ includeContent: false })
```

Credential-like fields are always recursively redacted from captured requests
and echoed responses.

When multiple instrumentor instances overlap, the first active instance's
content policy remains in effect until the last instance is deactivated;
conflicting instances log a warning instead of changing it. Deactivation
restores only wrappers still owned by this instrumentor, so a later patch from
another library is left intact.

The integration is explicit-only because Dify is an application/workflow
service rather than a leaf model provider.
