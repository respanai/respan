# Respan Span Contract

This document defines the canonical span attribute shape every Respan
instrumentation (Python or JavaScript, first-party or OI-delegated) must
emit. The Respan ingest API at
[`/v2/traces`](https://www.respan.ai/docs/apis/traces/create-trace)
accepts standard OTLP/HTTP and reads the fields below directly. **Do not
add new `respan.*` aliases or top-level shortcut fields to "make
ingestion convenient" — the SDK conforms to the documented API; the
backend is not modified to accommodate translator drift.**

## Source rules

Each constant has exactly one home. Import it from there. **Do not
re-export, re-declare, or shadow it elsewhere** — duplicates drift.

- **Traceloop / GenAI attributes** (`gen_ai.*`, `llm.*`, `traceloop.*`):
  import directly from the Traceloop semantic-conventions package.
  - Python: `from opentelemetry.semconv_ai import SpanAttributes`
  - JavaScript: `from "@traceloop/ai-semantic-conventions"`
- **OpenInference attributes** (`openinference.*`,
  `llm.input_messages.*`, `llm.output_messages.*`, and other OI
  conventions): import directly from the OpenInference constants
  package.
  - Python: `from openinference.semconv.trace import SpanAttributes`
  - JavaScript: `from "@arizeai/openinference-semantic-conventions"`
- **Respan-specific attributes** (`respan.entity.log_type`,
  `respan.customer_params.*`, `respan.threads.*`, `respan.trace.*`,
  `respan.metadata`): import directly from `respan-sdk` (Python:
  `respan_sdk.constants`; JS: `@respan/respan-sdk`).
- **SDK-specific keys** (LangChain callback fields, Vercel AI SDK
  `ai.*`, n8n event keys, etc.): keep them **inside the instrumentation
  package that owns that SDK**. Do not promote them into `respan-sdk` —
  they are a translator-internal detail, not part of the public span
  contract.

**Never invent a duplicate.** If a key already exists in Traceloop,
OpenInference, or `respan-sdk`, use that constant. Don't create a
parallel local constant, a re-export, or a renamed alias just because
the import path is long.

- **OTel attribute spec compliance**: span attribute values are
  primitives or homogeneous primitive arrays. Structured data
  (objects, lists of objects) → JSON-stringify the value. No object
  arrays as attribute values, even though they may survive an in-process
  pipeline.

## Common to all spans

| Field | Type | Notes |
|---|---|---|
| `respan.entity.log_type` | string | `workflow` \| `agent` \| `task` \| `tool` \| `chat` \| `text` \| `embedding` \| `guardrail` |
| `traceloop.span.kind` | string | `workflow` \| `agent` \| `task` \| `tool`. **Set only on user-decorated spans** (`@workflow`, `@task`, `@agent`, `@tool` / `withWorkflow`, …). Auto-emitted instrumentation spans must NOT set this — `respan.entity.log_type` carries the type for ingestion. |
| `traceloop.entity.name` | string | display name |
| `traceloop.entity.path` | string | `""` for root candidates |
| `traceloop.entity.input` | string (JSON) | structured input, JSON-stringified |
| `traceloop.entity.output` | string (JSON) | structured output, JSON-stringified |
| `respan.customer_params.*` | string | customer attribution; bridge via `RespanSpanProcessor.on_start` |
| `respan.threads.thread_identifier` | string | thread correlation |
| `respan.trace.trace_group_identifier` | string | trace group correlation |
| `respan.metadata` | string (JSON) | free-form metadata; merged into ingest metadata bucket |

## LLM spans (`log_type=chat` or `text`)

| Field | Type | Notes |
|---|---|---|
| `gen_ai.system` | string | `openai` \| `anthropic` \| `google` \| `bedrock` \| … (lowercase) |
| `gen_ai.request.model` | string | model id, e.g. `gpt-4o-mini` |
| `llm.request.type` | string | `chat` (required for backend prompt/completion parsing) |
| `gen_ai.prompt.{N}.role` | string | `user` \| `assistant` \| `tool` \| `system` |
| `gen_ai.prompt.{N}.content` | string | message content; JSON-stringify if structured |
| `gen_ai.prompt.{N}.tool_calls` | string (JSON) | historical tool calls in input context |
| `gen_ai.completion.0.role` | string | `assistant` |
| `gen_ai.completion.0.content` | string | response text |
| `gen_ai.completion.0.tool_calls` | string (JSON) | **only** tool calls emitted by THIS turn — do NOT merge from prompt history |
| `llm.request.functions` | string (JSON) | tool definitions available for this request |
| `gen_ai.usage.input_tokens` | int | prompt tokens (modern semconv) |
| `gen_ai.usage.output_tokens` | int | completion tokens (modern semconv) |
| `gen_ai.usage.prompt_tokens` | int | legacy alias for input_tokens (publish both) |
| `gen_ai.usage.completion_tokens` | int | legacy alias for output_tokens (publish both) |
| `llm.usage.total_tokens` | int | total tokens |
| `llm.usage.cache_read_input_tokens` | int | cache hits, if available |

### Anti-pattern

**Do not** set the following — they are duplicates of canonical fields and create cross-translator drift:

- `respan.span.tools`, `respan.span.tool_calls`, `respan.span.handoffs` (use `llm.request.functions`, `gen_ai.completion.0.tool_calls`)
- top-level `tools`, `tool_calls`, `model`, `prompt_tokens`, `completion_tokens`, `total_request_tokens`, `span_tools`, `has_tool_calls`, `parallel_tool_calls`

If you find these in existing code, that's debt to remove, not a pattern to copy.

## Tool execution spans (`log_type=tool`)

| Field | Type | Notes |
|---|---|---|
| `traceloop.entity.name` | string | tool name |
| `traceloop.entity.input` | string (JSON) | `{name, arguments}` |
| `traceloop.entity.output` | string (JSON) | tool result |

The span's existence + `respan.entity.log_type=tool` IS the tool call.
Do NOT additionally set `tool_calls`, `respan.span.tool_calls`, or
`gen_ai.tool.*` on tool execution spans.

"Was a tool used in this trace?" is answered by *"any
`log_type=tool` child span exists?"* — not by summing tool_call fields
across spans.

## Embedding spans (`log_type=embedding`)

| Field | Type | Notes |
|---|---|---|
| `gen_ai.request.model` | string | embedding model |
| `llm.request.type` | string | `embedding` |
| `gen_ai.usage.input_tokens` | int | input token count |

**Strip** before export:

- The embedding vectors themselves (`ai.embedding`, `ai.embeddings`,
  vendor-specific arrays) — they bloat span size and aren't useful at
  trace level.
- Vendor-specific synthetic token fields (e.g., Vercel's
  `ai.usage.tokens`) once the canonical fields above are populated.

## Workflow / Agent / Task spans

Just the common fields. No tool-specific or LLM-specific attributes.

## Translator responsibilities

A translator's job is to **bridge a vendor-specific span shape into the
canonical contract**. It is not to invent new Respan-prefixed fields.

Vendor-specific shapes a translator must handle today:

- Vercel AI SDK: `ai.*` → canonical
- OpenInference (CrewAI, Haystack, Google ADK, LangChain, …):
  `openinference.*`, indexed `llm.input_messages.N.*` /
  `llm.output_messages.N.*` → canonical

After translation, the resulting span goes through OTLP unchanged. The
backend reads canonical fields directly.

## Test pattern

Each translator package must include unit tests asserting:

1. The canonical fields above are populated correctly for each
   `log_type`.
2. **No off-contract aliases** are present (`respan.span.tools`,
   `tools`, `tool_calls`, `model`, `prompt_tokens`, `span_tools`,
   `has_tool_calls`, etc., must all be `undefined` / `not in attrs`).
3. Vendor-specific raw attributes are stripped before export.

## Migration path for existing translators

1. **Step 1 — emit both** canonical and current aliases (today's state).
   Backend continues to read whatever it reads today.
2. **Step 2 — verify the canonical field is present** in every span via
   a contract test. Assert it. Tests fail if a translator regresses.
3. **Step 3 — stop emitting aliases.** Translator emits canonical only.
   Backend already reads canonical (per public API docs); aliases were
   parsed as a courtesy, not a documented contract.

Each translator gets one PR per step. No coordinated big-bang.
