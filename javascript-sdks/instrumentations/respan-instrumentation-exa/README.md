# Respan Exa instrumentation (TypeScript)

Native OpenTelemetry instrumentation for the official [`exa-js`](https://www.npmjs.com/package/exa-js) SDK.

Requires Node.js 18 or newer. The package build and seven-test suite are
validated on Node.js 18 in addition to the current development runtime.

The package traces the npm-stable 2.19 API surface that carries AI work:

- search, contents, answer, and streaming calls
- deprecated search/similarity compatibility methods still shipped by Exa 2.x
- Agent run create, stream, wait, poll, stop, and lifecycle operations
- the legacy Research client for compatibility (Exa recommends deep search for new research flows)
- executable provider-neutral, OpenAI Chat/Responses, and Anthropic `webSearch` helpers through `Exa.search()`

The npm 2.19 artifact does **not** ship a `getContents` tool helper. Core `getContents()` is supported. Exa tagged 2.20 upstream with that helper on September 1, 2026, but it was not npm's `latest` dist-tag when this package was authored. The prototype-based patch is tolerant of that additive 2.20 API, and the helper will route through the already-instrumented core method once published.

Websets, Search Monitor CRUD, and beta Agent Monitor operations are
intentionally not patched in the first release. They are long-lived
control-plane APIs rather than in-process SDK operations.

Entity names are provider-neutral and use the same snake_case vocabulary as the Python package: tools use names such as `search` and `get_contents`, answers use `answer`, Agent runs use `run`, and legacy Research uses `research`. Native JavaScript operation spelling such as `getContents`, stream state, result counts, request IDs, cost, citations, and legacy markers are stored only in the canonical `respan.metadata` JSON object. Answer spans set `gen_ai.request.model` only when the SDK request or response supplies a model; otherwise semantic naming resolves to bare `llm`.

## Install

```bash
npm install @respan/respan @respan/instrumentation-exa exa-js
```

## Use

```ts
import { Exa } from "exa-js";
import { Respan } from "@respan/respan";
import { ExaInstrumentor } from "@respan/instrumentation-exa";

const respan = new Respan({
  apiKey: process.env.RESPAN_API_KEY,
  instrumentations: [new ExaInstrumentor()],
});
await respan.initialize();

const exa = new Exa(process.env.EXA_API_KEY);
await exa.search("recent advances in retrieval", {
  type: "auto",
  contents: { highlights: true },
});
await respan.shutdown();
```

The integration is explicit-only. Exa is a search/tool/agent SDK and can overlap with OpenAI or Anthropic instrumentation when its tool adapters are used, so it is not direct-LLM auto-instrumentation.

Set `captureContent: false`, or set `TRACELOOP_TRACE_CONTENT=false`, to omit query, result, prompt, completion, and stream payloads while retaining operation/status metadata. API keys and authorization-like fields are always redacted.

Streaming spans end when the async iterator is exhausted, fails, or is closed by `return()`/an early `for await` exit.
