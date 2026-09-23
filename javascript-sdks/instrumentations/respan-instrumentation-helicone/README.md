# @respan/instrumentation-helicone

Respan instrumentation for `@helicone/helpers`. It captures calls made through
`HeliconeManualLogger` as canonical Respan spans while leaving Helicone's own
logging behavior unchanged.

The tested compatibility range is `@helicone/helpers >=1.8.3 <1.9.0` because
the adapter patches the 1.8.x `HeliconeManualLogger` call surface directly.

The integration covers `logRequest`, `logStream`, `logSingleStream`,
`logSingleRequest`, `HeliconeLogBuilder`, direct `sendLog` calls, and Helicone's
custom `tool`, `vector_db`, and `data` events. Successful calls are captured at
the shared `sendLog` chokepoint. Operations that fail before `sendLog` still
produce one error span, without a duplicate success/error pair.

## Installation

```bash
npm install @respan/respan @respan/instrumentation-helicone @helicone/helpers@~1.8.3
```

## Usage

```ts
import { HeliconeManualLogger } from "@helicone/helpers";
import { HeliconeInstrumentor } from "@respan/instrumentation-helicone";
import { Respan } from "@respan/respan";

const respan = new Respan({
  apiKey: process.env.RESPAN_API_KEY,
  instrumentations: [new HeliconeInstrumentor()],
});
await respan.initialize();

const helicone = new HeliconeManualLogger({
  apiKey: process.env.HELICONE_API_KEY!,
});

await helicone.logRequest(
  {
    model: "gpt-4o-mini",
    messages: [{ role: "user", content: "Say hello." }],
  },
  async (recorder) => {
    const response = {
      choices: [{ message: { role: "assistant", content: "Hello!" } }],
      usage: { prompt_tokens: 4, completion_tokens: 2, total_tokens: 6 },
    };
    recorder.appendResults(response);
    return response;
  },
  { "Helicone-Session-Id": "session-123" },
  "openai",
);

await respan.shutdown();
```

For manual activation, `await instrumentor.activate()` before the first
Helicone call. The `instrumentHelicone(options)` convenience function is also
async and resolves only after its patches are installed.

`traceContent` defaults to `true`. Set it to `false` on the instrumentor to
omit request and response bodies while retaining operation, model, provider,
usage, status, and safe correlation attributes:

```ts
new HeliconeInstrumentor({ traceContent: false });
```

The instrumentation does not record the Helicone logger API key or copy
authorization/unknown headers into spans. It maps only Helicone
user/session/property headers with explicit semantics, and redacts structured
sensitive fields and error strings. With `traceContent: true`, ordinary user
prompt and response text is intentionally captured.

This integration is explicit-only because Helicone is an observability bridge;
automatic activation could duplicate provider or framework instrumentation.
