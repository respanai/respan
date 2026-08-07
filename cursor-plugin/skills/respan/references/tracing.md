# Tracing

Set up and configure Respan SDK tracing: instrument an app to capture LLM calls as
structured traces, then refine with decorators, context propagation, and processors.

This file covers **both** the initial setup steps and the advanced reference. Tracing
is a separate setup from the gateway — do one at a time, never both in the same pass.

Docs: `https://respan.ai/docs/documentation/features/tracing/traces/tracing-sdk.md`
Full docs index: `https://www.respan.ai/docs/llms.txt`

---

## Setup

Use this section when the user asks to set up Respan tracing in their project.

### Hard Rules

- **Interactive mode:** Ask the user questions when you need input. Do not assume.
- **Only add Respan code.** Do not refactor or modify unrelated code.
- **Pin exact versions.** Never use `latest` or unpinned ranges.
- **Do not guess APIs.** Use only the patterns from the integration docs linked below.
- **If Respan is already installed/configured, do not duplicate work.** Check for existing `respan` imports first.
- **Read the code before proposing changes.** Understand the actual workflow, not just the dependencies.

The API key is stored in `.env` as `RESPAN_API_KEY`.

### 1. Ask first: Auto or Full?

**Before analyzing anything, ask the user which setup they want.** This is the first question — the two branches diverge immediately.

1. **Auto** — the fastest path. Install the core SDK and add `Respan()` (one line). Every LLM call from a supported direct SDK — OpenAI, Anthropic, Azure OpenAI, Bedrock, Vertex, Cohere, Together, Gemini, LiteLLM — is automatically captured as a flat span. **No decorators and no framework instrumentor, even if a framework is detected.** Best for a quick start or a live demo: traces flowing in under two minutes.
2. **Full** — structured setup. Adds framework-specific instrumentation and/or workflow structure on top:
   - **Explicit instrumentor** — for agent frameworks (LangChain, CrewAI, OpenAI Agents, Claude Agent SDK, LlamaIndex, Haystack, …) so the framework's agent / tool / chain structure is captured.
   - **Decorators** — wrap the user's own functions with `@workflow` / `@task` for nested spans.

Follow only the branch the user picks; skip the other one.

> **Frameworks under Auto:** Auto never installs a framework instrumentor. If the app uses a framework (LangChain, CrewAI, an agents SDK) and the user picks Auto, they get only the flat LLM spans the underlying direct SDK emits — no agent / tool / chain structure. Capturing framework structure requires Full.

---

### Auto path

**A1. Detect language and package manager** — `package.json` (JS/TS) or `pyproject.toml` / `requirements.txt` (Python).

**A2. Install the core SDK** (pin the exact current version):
```bash
# Python
pip install respan-ai

# TypeScript
npm install @respan/respan
```

**A3. Add initialization code** at the top of the entrypoint, before any LLM client is created:
```python
# Python
from respan import Respan
Respan()
```
```typescript
// TypeScript
import { Respan } from "@respan/respan";
const respan = new Respan();
await respan.initialize();
```

Do **not** add decorators or a framework instrumentor. Go straight to **Verify**.

---

### Full path

The rest of this section is the **Full** path — follow it only if the user chose Full. It adds an explicit framework instrumentor and/or decorators on top of the core SDK.

#### 1. Analyze the Project

**1a. Detect language and package manager:**
- Check `package.json` (JS/TS) or `pyproject.toml` / `requirements.txt` (Python)
- Detect package manager from lock files

**1b. Detect libraries in priority order:**

Check higher-priority categories first. If a match is found, use that instrumentation — do NOT also add lower-level SDK instrumentation.

**Priority 1 — Agent Frameworks & High-Level SDKs:**

| Library | Python package | JS/TS package | Respan instrumentation (Python) | Respan instrumentation (JS/TS) | Docs |
|---------|---------------|---------------|--------------------------------|-------------------------------|------|
| Vercel AI SDK | — | `ai` | — | `@respan/instrumentation-vercel` | [docs](https://respan.ai/docs/integrations/vercel-ai-sdk.md) |
| OpenAI Agents SDK | `openai-agents` | `@openai/agents` | `respan-instrumentation-openai-agents` | `@respan/instrumentation-openai-agents` | [docs](https://respan.ai/docs/integrations/openai-agents-sdk.md) |
| Claude Agent SDK | `claude-agent-sdk` | — | `respan-instrumentation-claude-agent-sdk` | — | [docs](https://respan.ai/docs/integrations/claude-agents-sdk.md) |
| Pydantic AI | `pydantic-ai` | — | `respan-instrumentation-pydantic-ai` | — | [docs](https://respan.ai/docs/integrations/pydantic-ai.md) |
| LangChain | `langchain` | `langchain` | `respan-instrumentation-langchain` | `@respan/instrumentation-langchain` | [docs](https://respan.ai/docs/integrations/langchain.md) |
| LangGraph | `langgraph` | `@langchain/langgraph` | `respan-instrumentation-langchain` | `@respan/instrumentation-langchain` | [docs](https://respan.ai/docs/integrations/langgraph.md) |
| CrewAI | `crewai` | — | `respan-instrumentation-crewai` | — | [docs](https://respan.ai/docs/integrations/crewai.md) |
| LlamaIndex | `llama-index` | — | `respan-instrumentation-llama-index` | — | [docs](https://respan.ai/docs/integrations/llama-index.md) |
| Haystack | `haystack-ai` | — | `respan-instrumentation-haystack` | — | [docs](https://respan.ai/docs/integrations/haystack.md) |
| Mastra | — | `mastra` | — | via OTEL | [docs](https://respan.ai/docs/integrations/mastra.md) |
| Google ADK | `google-adk` | — | `respan-instrumentation-google-adk` | — | [docs](https://respan.ai/docs/integrations/google-adk.md) |

If a Priority 1 framework is found, use its instrumentation. Do NOT also add Priority 2 instrumentation for the same provider.

**LangChain / LangGraph setup differs by language** (one package covers both frameworks: `respan-instrumentation-langchain` / `@respan/instrumentation-langchain`). In **Python**, pass `LangChainInstrumentor()` to `Respan(instrumentations=[...])` — it patches the LangChain and LangGraph callback managers globally on init, so every chain, graph, node, LLM, tool, and retriever run is traced **automatically with no per-call setup**. `add_respan_callback(config=...)` is optional and only labels a run with a name, tags, or metadata. In **TypeScript**, create a `LangChainInstrumentor`, pass it to `new Respan({ instrumentations: [...] })`, **and** attach `instrumentor.addCallback(...)` to the **outermost** chain or graph invocation (`.invoke()` / `.stream()`, not inside a node — LangChain/LangGraph propagate it to nested runs); the TypeScript instrumentor does **not** patch globally, so without the callback no spans are emitted. (The other Priority-1 frameworks use the standard `Respan(instrumentations=[XInstrumentor()])` plugin pattern.)

**Priority 2 — Direct LLM SDKs** (only if no P1 framework covers this provider):

These are **auto-instrumented** — just `Respan()` / `new Respan()`, no extra packages needed:

| Library | Python package | JS/TS package | Docs |
|---------|---------------|---------------|------|
| OpenAI SDK | `openai` | `openai` | [docs](https://respan.ai/docs/integrations/openai-sdk.md) |
| Anthropic SDK | `anthropic` | `@anthropic-ai/sdk` | [docs](https://respan.ai/docs/integrations/anthropic.md) |
| Azure OpenAI | `openai` (azure config) | `openai` | [docs](https://respan.ai/docs/integrations/providers/azure.md) |
| Google Vertex AI | `google-cloud-aiplatform` | — | [docs](https://respan.ai/docs/integrations/vertex-ai.md) |
| AWS Bedrock | `boto3` | — | [docs](https://respan.ai/docs/integrations/aws-bedrock.md) |
| Cohere | `cohere` | — | [docs](https://respan.ai/docs/integrations/providers/cohere.md) |
| Together AI | `together` | — | [docs](https://respan.ai/docs/integrations/together-ai.md) |

**Note:** LiteLLM in JS uses the OpenAI-compatible API, so the OpenAI auto-instrument covers it. For Python LiteLLM, see [LiteLLM guide](https://respan.ai/docs/integrations/litellm.md). For Google GenAI (`@google/genai`), see [Google GenAI guide](https://respan.ai/docs/integrations/google-genai.md).

**1c. Read the actual code and understand the workflow:**

This is the most important step. Read the entrypoint and all files that make LLM calls. Map out:

- What is the **overall workflow**? (e.g. "user sends question → retrieve context → generate answer → format response")
- What are the **individual steps/tasks**? (e.g. "embed query", "search DB", "call GPT", "parse output")
- Are there **agent loops**? (e.g. a loop that calls tools until done)
- Are there **tool calls**? (e.g. functions the LLM invokes)

#### 2. Propose an Implementation Plan

Present the user with a concrete plan before making any changes. The plan should include:

**a) Packages to install** — core SDK + instrumentation package (with exact versions)

**b) Initialization code** — where to add it (which file, which line)

**c) Workflow structure** — how to wrap the existing code:

For **agent frameworks** (Priority 1): The framework instrumentation auto-captures the workflow structure. Usually just need init code, no manual wrapping needed. Fetch and follow the integration doc.

For **direct LLM SDKs** (Priority 2): Individual LLM calls will be auto-traced as flat spans. Propose wrapping the logical workflow with Respan decorators/wrappers to get structured nested traces:

TypeScript example:
```typescript
// Before: flat traces — each LLM call is an isolated span
const outline = await openai.chat.completions.create({...});
const draft = await openai.chat.completions.create({...});

// After: structured traces — nested spans showing the workflow
const result = await withWorkflow({ name: "write_article" }, async () => {
  const outline = await withTask({ name: "generate_outline" }, async () => {
    return await openai.chat.completions.create({...});
  });
  const draft = await withTask({ name: "write_draft" }, async () => {
    return await openai.chat.completions.create({...});
  });
  return draft;
});
```

Python example:
```python
# Before: flat traces
outline = client.chat.completions.create(...)
draft = client.chat.completions.create(...)

# After: structured traces
@workflow(name="write_article")
def write_article(topic):
    outline = generate_outline(topic)
    return write_draft(outline)

@task(name="generate_outline")
def generate_outline(topic):
    return client.chat.completions.create(...)

@task(name="write_draft")
def write_draft(outline):
    return client.chat.completions.create(...)
```

Within Full, pick what the app needs (they can combine):
- **Framework detected** → add its **explicit instrumentor** (or the LangChain callback). The framework auto-captures its agent / tool / chain structure — usually no manual wrapping needed.
- **Direct LLM SDK, or you want structure over your own orchestration code** → add **decorators** (`@workflow` / `@task`) to wrap the logical workflow into nested spans.

(If the user only wanted flat spans with zero code changes, that's the **Auto** path — go back to it.)

When adding decorators:
- **If multiple independent workflows are detected** (e.g. `writeArticle()`, `summarizeDoc()`, `classifyEmail()`), list them and ask which ones to instrument. Don't assume all of them.
- **Show the user what the trace will look like** — describe the span hierarchy:
```
workflow: write_article
  ├── task: generate_outline
  │     └── llm: openai.chat (auto-captured)
  └── task: write_draft
        └── llm: openai.chat (auto-captured)
```

Wait for user confirmation before proceeding.

#### 3. Implement

**a) Install packages:**

For direct LLM SDKs (Priority 2) — just the core SDK:
```bash
# Python
pip install respan-ai

# TypeScript
npm install @respan/respan
```

For agent frameworks (Priority 1) — also install the instrumentor. Check the docs link in the table above for the exact packages.

**b) Add initialization code** — at the top of the entrypoint, before any LLM client is created:

For **direct LLM SDKs** (auto-instrumented):
```python
# Python
from respan import Respan
Respan()
```
```typescript
// TypeScript
import { Respan } from "@respan/respan";
const respan = new Respan();
await respan.initialize();
```

For **agent frameworks** (explicit instrumentor — fetch the docs URL from the table for the exact pattern):
```python
# Python example (OpenAI Agents)
from respan import Respan
from respan_instrumentation_openai_agents import OpenAIAgentsInstrumentor
Respan(instrumentations=[OpenAIAgentsInstrumentor()])
```
```typescript
// TypeScript example (OpenAI Agents)
import { Respan } from "@respan/respan";
import { OpenAIAgentsInstrumentor } from "@respan/instrumentation-openai-agents";
const respan = new Respan({ instrumentations: [new OpenAIAgentsInstrumentor()] });
await respan.initialize();
```

**c) Add workflow wrappers** — if the user chose structured traces in the plan.

### Verify (final test — always run this, both paths)

As the final step — for **both** Auto and Full — **run the user's program once with a small request** and confirm tracing works end to end:

- The app runs without errors
- A trace appears at https://platform.respan.ai or via `respan traces list --limit 5`
- If wrappers were added, the trace shows the expected nested span hierarchy

If no trace appears, the instrumentation isn't taking effect — debug (init placement, missing `flush()`, wrong entrypoint) and re-run before declaring setup done.

---

## Reference

Advanced tracing configuration: decorators, context propagation, processors, and span attributes.

### Decorators

Wrap functions to create structured span hierarchies. All decorators share the same signature.

| Decorator | Purpose |
|-----------|---------|
| `@workflow` / `withWorkflow` | Root span — top-level pipeline |
| `@task` / `withTask` | Step within a workflow |
| `@agent` / `withAgent` | Agent loop span |
| `@tool` / `withTool` | Tool/function call span |

#### Python

```python
from respan import Respan, workflow, task, agent, tool

Respan()

@workflow(name="write_article")
def write_article(topic: str):
    outline = generate_outline(topic)
    return write_draft(outline)

@task(name="generate_outline")
def generate_outline(topic: str):
    return client.chat.completions.create(...)

@task(name="write_draft")
def write_draft(outline: str):
    return client.chat.completions.create(...)
```

#### TypeScript

```typescript
import { Respan, withWorkflow, withTask } from "@respan/respan";

const respan = new Respan();
await respan.initialize();

const result = await withWorkflow({ name: "write_article" }, async () => {
  const outline = await withTask({ name: "generate_outline" }, async () => {
    return await client.chat.completions.create({...});
  });
  return await withTask({ name: "write_draft" }, async () => {
    return await client.chat.completions.create({...});
  });
});
```

#### Decorator Parameters

```python
@workflow(
    name="my_workflow",               # Display name (default: function name)
    version=1,                        # Version number
    processors="my_processor",        # Route to specific processor(s)
    sample_rate=0.5,                  # 0.0-1.0, fraction of spans exported
    export_filter={                   # Conditional export
        "metadata.env": {"operator": "", "value": "production"}
    },
)
```

All parameters are optional. Without any, the function name is used as the span name.

#### Async Support

Decorators work with `async def`, generators, and async generators automatically.

### Context Propagation

Attach attributes to **all spans** within a scope, including auto-instrumented LLM calls.

#### propagate_attributes

```python
from respan import Respan, propagate_attributes

Respan()

with propagate_attributes(
    customer_identifier="user_123",
    thread_identifier="conv_abc",
    environment="production",
    metadata={"plan": "pro", "team": "search"},
):
    # All spans here (including auto-instrumented OpenAI calls) get these attributes
    result = run_pipeline()
```

##### Available attributes

| Attribute | Description |
|-----------|-------------|
| `customer_identifier` | User/customer ID — enables per-user analytics |
| `customer_email` | Customer email |
| `customer_name` | Customer display name |
| `thread_identifier` | Conversation thread ID — groups related calls |
| `custom_identifier` | Indexed custom identifier for fast queries |
| `group_identifier` | Group related traces together |
| `environment` | Environment name (e.g. `"production"`, `"staging"`) |
| `metadata` | Dict of custom key-value pairs (merged in nested contexts) |
| `prompt` | Dict with `prompt_id` and `variables` for prompt logging |

##### TypeScript

```typescript
const result = await respan.propagateAttributes(
  {
    customerIdentifier: "user_123",
    threadIdentifier: "conv_abc",
    metadata: { plan: "pro" },
  },
  async () => {
    return await runPipeline();
  }
);
```

##### Nesting behavior

- Nested `propagate_attributes` calls merge with the outer context
- `metadata` dicts are merged (not replaced)
- For duplicate keys, inner values override outer
- Async-safe via `contextvars`

#### respan_span_attributes

Attaches attributes to the **current active span only** (not auto-instrumented child spans):

```python
from respan import respan_span_attributes

with respan_span_attributes({
    "customer_identifier": "user-123",
    "metadata": {"priority": "high"},
}):
    pass
```

Use `propagate_attributes` when you need attributes on all nested spans (most common). Use `respan_span_attributes` when you only want to tag the current span.

### Imperative Span Creation

For runtime-determined span names, use the client API instead of decorators:

```python
from respan import get_client

client = get_client()

with client.start_span(name="dynamic_step", kind="task") as span:
    result = do_work()
    # span is automatically closed and exported
```

### Updating the Current Span

Add attributes or change status on the active span at runtime:

```python
from respan import get_client

client = get_client()

# Inside a decorated function or start_span context:
client.update_current_span(
    respan_params={
        "customer_identifier": "user_123",
        "metadata": {"step": "final"},
    },
    name="renamed_span",
)

# Add events
client.add_event("checkpoint_reached", attributes={"items": 42})

# Record exceptions (span continues)
try:
    risky_operation()
except Exception as e:
    client.record_exception(e)
```

### Custom Processors

Route spans to multiple destinations with filtering:

```python
from respan_tracing import RespanTelemetry

telemetry = RespanTelemetry(api_key="...", base_url="...")

# Add a second exporter for debug logging
from opentelemetry.sdk.trace.export import ConsoleSpanExporter

telemetry.add_processor(
    exporter=ConsoleSpanExporter(),
    name="debug",
    filter_fn=lambda span: span.attributes.get("environment") == "debug",
)
```

Route specific spans to a processor using the `processors` parameter on decorators:

```python
@task(name="sensitive_step", processors="debug")
def sensitive_step():
    ...
```

### Flush

Always call `flush()` before process exit to ensure all spans are exported:

```python
respan = Respan()
# ... your code ...
respan.flush()
```

In serverless/Lambda: call `flush()` at the end of every handler invocation.

### Trace Hierarchy Example

```
workflow: handle_request
  +-- task: classify_intent
  |     +-- llm: openai.chat (auto-captured)
  +-- agent: support_agent
  |     +-- tool: lookup_order
  |     +-- llm: openai.chat (auto-captured)
  |     +-- tool: process_refund
  +-- task: generate_response
        +-- llm: openai.chat (auto-captured)
```

LLM calls within decorated functions are automatically nested as child spans.
