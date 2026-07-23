# Auto-Instrumentation Expansion Plan

This document records the implementation plan for broadening Respan auto-instrumentation to more direct LLM SDK integrations while keeping agent and framework instrumentations explicit.

## Goal

Broaden zero-config tracing for direct LLM provider SDKs without creating duplicate logs or duplicate trace spans on the Respan platform.

Auto-instrumentation should cover leaf provider calls such as OpenAI, Anthropic, Cohere, Together AI, Writer, Bedrock, Vertex AI, Google GenAI, OpenRouter, Mistral, Groq, and Ollama where the instrumentation maps one SDK call to one LLM span.

Auto-instrumentation should not cover agent, orchestration, workflow, callback, or framework integrations such as OpenAI Agents, Claude Agent SDK, Vercel AI SDK, LangChain, LlamaIndex, CrewAI, Pydantic AI, Haystack, Google ADK, BeeAI, Mastra, MCP, or similar packages. Those should remain explicit because they create higher-level spans and often overlap with direct provider SDK calls.

## Classification Rules

Use this category split everywhere: runtime registry, CLI setup, docs, and tests.

| Category | Auto-enabled by default | Examples |
| --- | --- | --- |
| `direct-llm` | Yes, when installed and not disabled | OpenAI, Anthropic, Azure OpenAI, Cohere, Together AI, Writer, Bedrock, Vertex AI, Google GenAI, OpenRouter, Mistral, Groq, Ollama |
| `agent-framework` | No | OpenAI Agents, Claude Agent SDK, Pydantic AI, Google ADK, CrewAI, BeeAI |
| `app-framework` | No | LangChain, LlamaIndex, Haystack, Vercel AI SDK, Mastra |
| `protocol-or-tooling` | No | MCP, Codex CLI, Gemini CLI, OpenCode |
| `vector-db` | No for this project phase | Chroma, Pinecone, Qdrant, Milvus, Weaviate, LanceDB |

## Duplicate-Prevention Policy

The runtime should resolve conflicts before activating instrumentors.

Priority order:

1. Explicit user-provided instrumentations
2. Explicit user-enabled auto instrumentations
3. First-party direct LLM auto instrumentations
4. Generic OTEL or Traceloop auto instrumentations
5. HTTP or gateway fallback spans

Rules:

- If an explicit agent or framework instrumentation is active, disable direct LLM auto-instrumentation for providers that framework wraps by default.
- If a first-party direct LLM instrumentation exists, prefer it over a generic OTEL or Traceloop instrumentation for the same provider.
- If the user passes an explicit `instrumentations` list, keep the current exclusive behavior unless they opt into combining it with auto-instrumentation.
- If a package cannot be loaded because its SDK or instrumentation package is not installed, skip it silently by default but include it in the status report.
- Add a future exporter or processor backstop that can warn or drop near-duplicate LLM spans when activation-time conflict detection misses a case.

## Shared Registry Shape

Create one registry per language with the same conceptual fields.

```ts
type AutoInstrumentationCategory =
  | "direct-llm"
  | "agent-framework"
  | "app-framework"
  | "protocol-or-tooling"
  | "vector-db";

type AutoInstrumentationEntry = {
  id: string;
  category: AutoInstrumentationCategory;
  provider?: string;
  sdkPackage: string;
  instrumentationPackage: string;
  instrumentorClass: string;
  enabledByDefault: boolean;
  priority: number;
  conflictsWith: string[];
  docsUrl?: string;
};
```

```python
@dataclass(frozen=True)
class AutoInstrumentationEntry:
    id: str
    category: str
    provider: str | None
    sdk_package: str
    instrumentation_package: str
    instrumentor_class: str
    enabled_by_default: bool
    priority: int
    conflicts_with: tuple[str, ...]
    docs_url: str | None = None
```

The registry should be the source of truth for:

- runtime auto-discovery
- duplicate conflict decisions
- CLI setup recommendations
- docs support tables
- tests that assert auto-enabled vs explicit-only behavior

## TypeScript Implementation Steps

### 1. Add a first-party direct LLM registry

Add a registry under `javascript-sdks/respan/src/`, for example:

- `src/_auto_instrumentation_registry.ts`

Initial `direct-llm` candidates:

| ID | Package | Instrumentor |
| --- | --- | --- |
| `openai` | `@respan/instrumentation-openai` | `OpenAIInstrumentor` |
| `anthropic` | `@respan/instrumentation-anthropic` | `AnthropicInstrumentor` |
| `azure-openai` | `@respan/instrumentation-azure-openai` | `AzureOpenAIInstrumentor` |
| `cohere` | `@respan/instrumentation-cohere` | `CohereInstrumentor` |
| `together-ai` | `@respan/instrumentation-together-ai` | `TogetherAIInstrumentor` |
| `aws-bedrock` | `@respan/instrumentation-aws-bedrock` | `AWSBedrockInstrumentor` |
| `writer` | `@respan/instrumentation-writer` | `WriterInstrumentor` |
| `google-genai` | `@respan/instrumentation-google-genai` | `GoogleGenAIInstrumentor` |
| `openrouter` | `@respan/instrumentation-openrouter` | `OpenRouterInstrumentor` |
| `vertexai` | `@respan/instrumentation-vertexai` | `VertexAIInstrumentor` |

Verify exact package names and exported class names before enabling each entry.

### 2. Replace hard-coded facade discovery

Update `javascript-sdks/respan/src/_core.ts` so `_autoDiscoverInstrumentations()` reads from the registry instead of the current hard-coded OpenAI, Anthropic, and Azure OpenAI list.

Only entries with `category === "direct-llm"` and `enabledByDefault === true` should be loaded by default.

### 3. Preserve explicit-only framework behavior

Keep agent and framework packages out of the auto-enabled direct LLM registry. If they are listed in the registry for status or docs, mark `enabledByDefault: false`.

Examples that must remain explicit:

- `@respan/instrumentation-openai-agents`
- `@respan/instrumentation-claude-agent-sdk`
- `@respan/instrumentation-vercel`
- `@respan/instrumentation-langchain`
- `@respan/instrumentation-llama-index`
- `@respan/instrumentation-google-adk`
- `@respan/instrumentation-beeai`
- `@respan/instrumentation-mcp`

### 4. Add conflict resolution before activation

Add a resolver that builds an activation plan before calling `activate()`.

The plan should include:

- `enabled`: instrumentations that will be activated
- `disabled`: instrumentations skipped by user config
- `conflicted`: instrumentations skipped because a higher-priority instrumentation owns the same provider
- `missing`: instrumentation package or SDK package not installed
- `failed`: activation attempted but failed

### 5. Add user-facing status

Add a method such as:

```ts
respan.getInstrumentationStatus()
```

Return structured status instead of only console logs. Console output should be concise and only visible at `info` or `debug`.

### 6. Keep lower-level Traceloop discovery controlled

The `@respan/respan` facade should continue to disable lower-level `@respan/tracing` Traceloop instrumentors that conflict with first-party direct LLM instrumentations.

If a first-party direct LLM instrumentation is added, update the facade's disabled Traceloop list so the generic instrumentor does not duplicate it.

### 7. Add tests

Add focused tests for:

- no explicit `instrumentations` means direct LLM auto entries are considered
- `instrumentations: []` disables facade auto-discovery
- `disabledInstrumentations` blocks matching IDs, package names, or class names
- agent/framework registry entries are not auto-enabled
- first-party direct LLM entries win over lower-priority generic instrumentors
- status reports missing packages and conflict skips

## Python Implementation Steps

### 1. Add first-party direct LLM plugin discovery

Python currently auto-discovers OTEL instrumentors from the `opentelemetry_instrumentor` entry point group. Add a separate first-party registry for Respan direct LLM plugins.

Candidate location:

- `python-sdks/respan/src/respan/_auto_instrumentation_registry.py`

Initial `direct-llm` candidates:

| ID | Distribution | Import / class |
| --- | --- | --- |
| `openai` | `respan-instrumentation-openai` | `respan_instrumentation_openai:OpenAIInstrumentor` |
| `anthropic` | `respan-instrumentation-anthropic` | `respan_instrumentation_anthropic:AnthropicInstrumentor` |
| `cohere` | `respan-instrumentation-cohere` | `respan_instrumentation_cohere:CohereInstrumentor` |
| `together` | `respan-instrumentation-together` | `respan_instrumentation_together:TogetherInstrumentor` |
| `aws-bedrock` | `respan-instrumentation-aws-bedrock` | `respan_instrumentation_aws_bedrock:AWSBedrockInstrumentor` |
| `google-genai` | `respan-instrumentation-google-genai` | `respan_instrumentation_google_genai:GoogleGenAIInstrumentor` |
| `openrouter` | `respan-instrumentation-openrouter` | `respan_instrumentation_openrouter:OpenRouterInstrumentor` |
| `writer` | `respan-instrumentation-writer` | `respan_instrumentation_writer:WriterInstrumentor` |
| `mistralai` | `respan-instrumentation-mistralai` | `respan_instrumentation_mistralai:MistralAIInstrumentor` |
| `groq` | `respan-instrumentation-groq` | `respan_instrumentation_groq:GroqInstrumentor` |
| `ollama` | `respan-instrumentation-ollama` | `respan_instrumentation_ollama:OllamaInstrumentor` |
| `aleph-alpha` | `respan-instrumentation-aleph-alpha` | `respan_instrumentation_aleph_alpha:AlephAlphaInstrumentor` |

Verify exact import paths and class names before enabling each entry.

### 2. Keep OTEL auto-instrumentation separate

Do not remove the existing `respan-tracing` OTEL entry-point discovery. Treat it as the generic OTEL path.

Add a higher-priority first-party plugin path in `respan`:

1. create `RespanTelemetry`
2. resolve first-party direct LLM plugin activation plan
3. activate selected direct LLM plugins
4. keep existing explicit plugins activated according to priority

### 3. Fix naming consistency

The current code uses `is_auto_instrument`. Some docs use `auto_instrument`.

Decide whether to support both:

- keep `is_auto_instrument` as the canonical internal name
- optionally accept `auto_instrument` as a backwards-compatible alias in `Respan.__init__`
- update docs to use the canonical public parameter after the decision

### 4. Add a direct LLM auto flag

Avoid overloading the existing OTEL auto flag.

Candidate API:

```python
Respan(
    is_auto_instrument=True,
    auto_instrument_direct_llms=True,
)
```

Or, if keeping one user-facing flag:

```python
Respan(is_auto_instrument=True)
```

Then internally split it into:

- first-party direct LLM plugin auto-discovery
- generic OTEL entry-point auto-discovery

The important behavior is that direct LLM first-party plugins should be able to win over generic OTEL instrumentors for the same provider.

### 5. Add conflict resolution

Before activating plugins, create a plan that accounts for:

- explicit `instrumentations`
- first-party direct LLM registry entries
- generic OTEL instrumentors
- user-provided `block_instruments`
- user-provided direct LLM disables

If explicit agent or framework plugins are active, skip direct provider plugins that would duplicate their internal LLM calls.

### 6. Add user-facing status

Add a method or property such as:

```python
respan.instrumentation_status()
```

The status should include:

- enabled direct LLM plugins
- enabled generic OTEL instrumentors
- explicit plugins
- skipped packages
- conflict skips
- activation failures

### 7. Add tests

Add focused tests for:

- `Respan()` auto-selects installed first-party direct LLM plugins
- explicit framework plugins prevent duplicate direct LLM plugins
- explicit `instrumentations=[...]` keeps current duplicate-safe behavior
- `is_auto_instrument=True` with explicit plugins intentionally combines both
- missing first-party plugins are skipped and reported
- generic OTEL instrumentors are lower priority than first-party direct LLM plugins

## Documentation Updates

Update these docs after implementation:

- `contribution/architecture.md`
- `contribution/writing-instrumentations.md`
- tracing quickstart
- setup SDK page
- Python SDK initialize page
- TypeScript SDK initialize page
- CLI skill reference generated source

Docs should show:

- direct LLM SDKs auto-instrument by default
- agent/framework integrations require explicit instrumentors
- how to disable one provider
- how to view instrumentation status
- how duplicate prevention works

## Rollout Plan

1. Implement TypeScript registry and facade auto-discovery expansion.
2. Add TypeScript conflict-resolution and status reporting.
3. Add TypeScript tests.
4. Implement Python first-party direct LLM registry.
5. Add Python conflict-resolution and status reporting.
6. Add Python tests.
7. Update docs and CLI setup references.
8. Verify with one direct provider example per language and one explicit framework example per language to confirm duplicate prevention.

## Acceptance Criteria

- Direct LLM SDK integrations can be auto-enabled without user imports when installed.
- Agent and framework instrumentations remain explicit-only.
- Explicit framework instrumentation disables overlapping direct LLM auto-instrumentation by default.
- The runtime exposes structured instrumentation status.
- Duplicate spans are prevented by activation policy, with processor/exporter dedupe available as a future safety net.
- Docs and runtime behavior agree on which integrations are auto-enabled.
