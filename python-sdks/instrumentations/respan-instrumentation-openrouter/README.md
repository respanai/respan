# Respan OpenRouter instrumentation

Trace OpenRouter's native Python SDK and OpenAI-compatible calls through the normal OTel pipeline.

## Install

```bash
pip install 'respan-instrumentation-openrouter[native]'
```

The native extra installs `openrouter>=1.3.22,<2`. Existing OpenAI-compatible installations can omit the extra. Both paths use released Respan packages; the adapter does not require source checkouts of its dependencies.

| Dependency | Supported lower bound | Current validation |
| --- | --- | --- |
| OpenRouter native SDK | 1.3.22 | 1.3.22 |
| OpenAI-compatible SDK | 3.0.0 | 3.24.0 |
| Respan OpenAI delegate | 1.2.1 | 1.2.3 |
| Respan tracing / SDK | 2.17.0 / 2.6.26 | 2.20.1 / 2.7.6 |
| OTel semantic conventions | 0.66b0 | 0.66b0 |
| AI semantic conventions | 0.5.1 | 0.5.1 |

The explicit OTel semantic-conventions floor supplies the modern streaming and reasoning keys. OTel API and SDK resolve together with that floor. Python 3.11–3.13 is supported.

## Native SDK usage

```python
import os
from openrouter import OpenRouter
from respan_instrumentation_openrouter import OpenRouterInstrumentor
from respan_tracing import RespanTelemetry, workflow

telemetry = RespanTelemetry(
    api_key=os.environ["RESPAN_API_KEY"],
    is_auto_instrument=False,
)
instrumentor = OpenRouterInstrumentor()
instrumentor.activate()

with OpenRouter(api_key=os.environ["OPENROUTER_API_KEY"]) as client:

    @workflow(name="openrouter_chat")
    def run(prompt: str) -> str:
        result = client.chat.send(
            model="openai/gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
        )
        return result.choices[0].message.content or ""

    print(run("Explain one benefit of tracing."))

telemetry.flush()
instrumentor.deactivate()
telemetry.tracer.tracer_provider.shutdown()
```

## Coverage

- Native `chat.send` / `send_async`, stable and beta `responses.send` / `send_async`, and `embeddings.generate` / `generate_async`.
- OpenAI-compatible chat and Responses `create` / `parse`, text completions, embeddings, and synchronous/asynchronous streaming.
- Native request DTOs, routing options and server-tool schemas continue through the original SDK methods. Complete request tool definitions and current-turn function calls are captured; historical tool calls remain in input context.
- Embedding float arrays and encoded values are retained in full. Input/output, source integer usage, cache and reasoning counters, response IDs, and streaming flags use canonical fields. SDK-coerced boolean counts and missing totals are not invented.
- Recording spans begin at the SDK call. Stream exhaustion, explicit close, exceptions and context-manager exit finish once under the call-time parent. Iterator `send` / `throw` and `asend` / `athrow` delegate to native implementations. Always close an abandoned stream. Cancellation keeps the native exception and records OTel error/type without an invented HTTP status.
- Shared instances, independent OpenAI owners, partial activation rollback and foreign wrapper restoration are supported. No backend or generic delegate source changes are needed.

`normalize_all_openai_spans=True` preserves the existing default for applications dedicated to OpenRouter. Use `False` in mixed-provider processes: only the exact `openrouter.ai` host and its subdomains are observed as OpenRouter, while the existing OpenAI delegate handles other hosts.

`capture_content=False`, `TRACELOOP_TRACE_CONTENT=0/false/no/off`, and Respan's content context opt-out remove payload and diagnostic content. Start-time and ancestor restrictions remain in effect if content is later enabled; a final veto is checked before context detachment. Standard OTel and LLM suppression and sampling remain effective. Complete entity input/output retains every message and choice; indexed prompt/completion fields cover the first eight of each to preserve common attributes within the OTel field limit. Known DTOs and built-in structures are serialized; arbitrary clients, iterables and object representations are not inspected.

HTTP error status is emitted only from a real provider response/exception. Application errors and cancellation carry OTel error status and type without a fabricated HTTP status or error-as-completion payload. Secrets are redacted in captured content and diagnostics.

Native audio/image/video/rerank/admin endpoints and the separate `openrouter-agent-sdk` are outside this coverage. Controlled examples validate instrumentation and native SDK parsing; they do not claim live model quality, native server-tool execution or Gateway routing support. Backend storage/projection acceptance remains a separate validation gate.
