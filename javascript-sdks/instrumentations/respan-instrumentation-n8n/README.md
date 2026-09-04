# @respan/instrumentation-n8n

Respan instrumentation for n8n's native OpenTelemetry workflow, node, agent,
LLM, tool, and memory spans.

This package does not start a second OpenTelemetry provider. Its preload hook
intercepts construction of n8n's own `NodeSDK`, only when the resource contains
both `n8n.instance.id` and `n8n.instance.role`. It inserts `N8nSpanProcessor`
before n8n's batch processor and wraps n8n's original OTLP/protobuf exporter
with `N8nTransformingExporter`.

The older package at `javascript-sdks/legacy/respan-exporter-n8n` is an archived
Gateway and prompt-management community node, not a trace exporter. Its
published `1.0.0` build hard-codes the retired `https://api.respan.co/api`
host, so it is not a current maintained Gateway path. Use n8n's HTTP Request
node with `POST https://api.respan.ai/api/chat/completions`, or a future
maintained community-node release, for Gateway calls. A maintained Gateway node
can coexist with this instrumentation because native execution tracing is a
separate concern.

## Supported versions

- n8n `2.19.0` through the tested/current `2.37.7` for `workflow.execute` and
  `node.execute`
- n8n `2.33.0` through `2.37.7` for current Agents roots, inner AI SDK LLM
  calls, tools, and memory spans
- tested target: n8n `2.37.7`
- instrumentation runtime: Node.js `>=22.16`; for n8n `2.37.7`, use Node.js
  24 LTS (the tested runtime is `24.19.0`)

n8n marks native OpenTelemetry tracing as Preview. The module hook is therefore
version-gated to the OpenTelemetry `@opentelemetry/sdk-node` range used by n8n
2.19 through the tested/current 2.37.7 (`>=0.213.0 <0.222.0`). The npm peer
range is correspondingly capped below n8n `2.38.0`; n8n versions after 2.37.7
and n8n 3.x must be revalidated before the peer range and support statement are
expanded.

There is no Python n8n instrumentation. n8n's Python task runner executes user
Code-node snippets in an isolated runner; it is not a Python workflow SDK or a
stable instrumentation surface.

## Install

The first npm release of `@respan/instrumentation-n8n` is pending. Build it and
its Respan dependencies from a monorepo checkout:

```bash
cd /path/to/respan/javascript-sdks
npx --yes @yarnpkg/cli-dist@4.9.2 install
npx --yes @yarnpkg/cli-dist@4.9.2 workspace @respan/respan-sdk build
npx --yes @yarnpkg/cli-dist@4.9.2 workspace @respan/tracing build
npx --yes @yarnpkg/cli-dist@4.9.2 workspace @respan/instrumentation-n8n build

cd /path/to/your-n8n-project
npm install --install-links=false \
  n8n@2.37.7 \
  file:/path/to/respan/javascript-sdks/respan-sdk \
  file:/path/to/respan/javascript-sdks/respan-tracing \
  file:/path/to/respan/javascript-sdks/instrumentations/respan-instrumentation-n8n
```

Install all packages in the same Node.js project so the preload entrypoint is
resolvable by the n8n process.

## Configure n8n 2.37.7

```bash
export RESPAN_API_KEY="your-respan-api-key"
export NODE_OPTIONS="--import=@respan/instrumentation-n8n/register"

export N8N_OTEL_ENABLED=true
export N8N_OTEL_EXPORTER_OTLP_ENDPOINT="https://api.respan.ai/api"
export N8N_OTEL_EXPORTER_OTLP_TRACING_PATH="/v2/traces"
export N8N_OTEL_EXPORTER_OTLP_HEADERS="Authorization=Bearer ${RESPAN_API_KEY}"
export N8N_OTEL_TRACES_INCLUDE_NODE_SPANS=true
export N8N_AGENTS_TRACING_ENABLED=true

npx n8n start
```

n8n's native exporter uses OTLP/HTTP Protobuf. Respan's `/api/v2/traces`
endpoint accepts both OTLP JSON and Protobuf.

`N8N_OTEL_TRACES_PRODUCTION_ONLY` defaults to `true`. For a local manual-run
audit only, set it to `false`; production deployments normally should keep the
default.

In queue mode, install/preload this package and set the same `N8N_OTEL_*`
configuration on main, worker, and webhook processes. This preserves n8n's
native cross-process trace propagation.

## Privacy

The translator never reads workflow execution records or reconstructs node
payloads. It can only promote content already present on n8n's native spans.
n8n Agent tracing records prompts, tool arguments, responses, and tool results
by default. Disable either side at the n8n source when required:

```bash
export N8N_AGENTS_TRACING_RECORD_INPUTS=false
export N8N_AGENTS_TRACING_RECORD_OUTPUTS=false
```

When those attributes are absent, this package does not synthesize them.
These two recording flags are not a strict Agent-data off switch in n8n
2.37.7: the root Agent tool catalog and memory correlation fields such as
owners, IDs, types, and stores can still be emitted independently. Disable
`N8N_AGENTS_TRACING_ENABLED` at the n8n source when no Agent content or
correlation metadata may be captured.

## Runtime behavior

`N8nInstrumentor.activate()` and `.deactivate()` are reference-counted and
idempotent. Deactivation removes the constructor hook for future n8n SDK
instances; it does not mutate a provider that is already running. Restart n8n
after removing the preload.

n8n can reload its OpenTelemetry SDK when settings change. While the preload is
active, every newly constructed n8n provider is instrumented. Prefer environment
variables for the endpoint, path, and headers so UI changes cannot drift worker
processes from the Respan configuration.

The hook leaves non-n8n `NodeSDK` instances untouched and fails open when n8n
supplies an unknown custom processor layout. Translation errors never affect
workflow execution. The exporter delegates a privacy-sanitized batch with raw
n8n Agent `ai.*` attributes/events removed; if even that fail-safe cleanup
fails for one malformed span, it drops only that affected span.

## Canonical mapping

| n8n span | Respan log type | Semantic name | Content |
| --- | --- | --- | --- |
| `workflow.execute` | `workflow` | `workflow` | workflow/execution IDs and status moved into `respan.metadata.n8n` inside canonical `respan.metadata` |
| `node.execute` | `task` | `task` | node identity, type, item counts, and termination reason moved into metadata |
| `<agent>.generate` / `<agent>.stream` | `agent` | `agent.<name>` | thread correlation and `gen_ai.prompt` promoted when n8n records them |
| `ai.generateText.doGenerate` / `ai.streamText.doStream` | `text` | `llm.<model>` (or bare `llm`) | model, prompt/completion messages, tool definitions/calls, provider token/cache usage, and stream timing promoted to the canonical LLM contract/metadata |
| `execute_tool <name>` | `tool` | `tool.<name>` | arguments and result promoted to canonical entity input/output |
| `query_memory` / `save_memory` | `task` | `task` | memory operation input/output promoted to entity content and memory identity/details moved into `respan.metadata.n8n.memory` |

Current `@n8n/agents` uses `@ai-sdk/otel`'s `LegacyOpenTelemetry`. Its outer
`ai.generateText` / `ai.streamText` span is structural, so semantic export
drops it and reparents the detailed LLM span to the n8n Agent root. Legacy span
name mode preserves the emitted wrapper tree. Likewise, the AI SDK emits an
`ai.toolCall` wrapper around n8n's authoritative `execute_tool` span. The
processor suppresses that wrapper only when the matching n8n owner appears and
reparents the owner to the detailed LLM span; an unmatched `ai.toolCall`
remains a canonical tool span instead of being lost.

The translator does not set `traceloop.span.kind` on these auto-emitted spans.
Raw `n8n.*`, n8n Agent metadata fields, `gen_ai.memory.*`, and all AI SDK
`ai.*` attributes are stripped from the export-only clone after canonical
promotion. This includes `ai.request.headers.*`; request headers and credentials
are never copied into `respan.metadata`. Arbitrary non-sensitive
`ai.telemetry.metadata.*` values are retained under
`respan.metadata.n8n.telemetry`; secret-shaped keys are retained only with a
`[REDACTED]` value. Raw AI SDK stream events are removed after their timing
diagnostics are promoted into `respan.metadata.n8n.ai_sdk`. Parent context,
status, exception events, links, timing, and resource attributes are preserved.
Existing `status_code`, `http.response.status_code`, `http.status_code`, or
`gen_ai.response.status_code` values are projected into backend `status_code`;
otherwise OTel error spans use `500` and OK/UNSET spans use `200`.
For ERROR spans, a standard exception event's `exception.message` is also
promoted to canonical `error.message` when that attribute is absent; the
original exception event remains on the exported span.
