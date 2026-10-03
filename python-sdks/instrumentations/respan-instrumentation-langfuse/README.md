# Respan instrumentation for Langfuse

Send Langfuse Python 3.12–4.x observations through the active Respan tracing
runtime. Requires Python 3.11–3.13.

```bash
pip install respan-ai respan-instrumentation-langfuse
export RESPAN_API_KEY=your-respan-api-key
```

```python
from langfuse import Langfuse, propagate_attributes
from respan import Respan
from respan_instrumentation_langfuse import LangfuseInstrumentor

respan = Respan(instrumentations=[])
instrumentor = LangfuseInstrumentor()
instrumentor.instrument()
client = Langfuse(public_key="pk-lf-local", secret_key="sk-lf-local")

try:
    with propagate_attributes(session_id="example-session"):
        with client.start_as_current_observation(
            name="answer", as_type="generation"
        ) as generation:
            generation.update(
                model="example-model",
                input=[{"role": "user", "content": "Hello"}],
                output=[{"role": "assistant", "content": "Hi"}],
            )
    client.flush()
    respan.flush()
finally:
    client.shutdown()
    instrumentor.uninstrument()
    respan.shutdown()
```

The example credentials enable local Langfuse span creation. Initialize Respan
before exporting; the instrumentor uses that runtime's API key and endpoint.
Enable instrumentation before Langfuse flushes any observations.

The adapter intercepts Langfuse's default OTLP/HTTP exporter, translates its
spans, and injects them into Respan. Unrelated OTLP exporters pass through.
Custom Langfuse exporters with an endpoint that does not identify Langfuse are
not intercepted. Langfuse API operations such as prompt fetching and scores
continue to use their configured Langfuse client and credentials.

Supported observations include generations, embeddings, agents, tools,
guardrails, and generic spans. Generations retain messages, tool calls, model,
and reported token usage. Embeddings retain model, reported usage, input, and
full vectors. Agent and tool observations retain their own span types;
guardrails use `guardrail`. Other observations use `workflow` for roots and
`task` for children. Trace and parent IDs, timestamps, errors, user/session
attribution, and metadata are preserved.

Generation input/output can be a single message object or a message list.
Historical prompt tool calls stay separate from current output calls, and
`model_parameters["tools"]` becomes the canonical request tool definitions.
Absent input/output remains absent, preserving the SDK's content capture choices.

Langfuse 4 replaces `update_current_trace()` with `propagate_attributes()` for
trace name, user, session, and metadata. Place that context around the root
observation so descendants inherit the attributes. Update observation input
and output with `update_current_span()` or the yielded observation's `update()`.

Use `instrumentor.exported_span_count` to check how many observations were
injected during the active session. A successful local injection does not
confirm backend ingestion. Runnable deterministic scenarios are in the
[Langfuse examples](https://github.com/respanai/respan-example-projects/tree/main/python/tracing/langfuse).
