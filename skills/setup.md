# Respan Setup (Agent Instructions)

## Hard Rules

- **Interactive mode:** Ask the user questions when you need input. Do not assume.
- **Only add Respan code.** Do not refactor or modify unrelated code.
- **Pin exact versions.** Never use `latest` or unpinned ranges.
- **Do not guess APIs.** Use only the patterns documented in this file.
- **If Respan is already installed/configured, do not duplicate work.** Check for existing `respan` imports first.

---

## Execution Requirements

Before writing any code:

1. Create a **checklist** from the steps below.
2. Execute each step in order.
3. Do not skip steps.

---

## Steps

### 1. Ask What to Set Up

Ask the user:

> What would you like to set up?
> 1. **Tracing** — instrument your app to capture LLM calls as structured traces
> 2. **Gateway** — route LLM requests through the Respan proxy for logging, caching, and key management
> 3. **Both** — gateway for requests + tracing for full observability

Wait for the user's response before proceeding.

---

### 2. Detect Frameworks and Libraries

Scan the project to identify:

- **Language:** Check for `package.json` (TypeScript/JavaScript) or `pyproject.toml` / `requirements.txt` (Python)
- **LLM libraries in use:** Check dependency files for these known libraries:

| Library | Python package | JS/TS package |
|---------|---------------|---------------|
| OpenAI SDK | `openai` | `openai` |
| Anthropic SDK | `anthropic` | `@anthropic-ai/sdk` |
| OpenAI Agents SDK | `openai-agents` / `openai[agents]` | `@openai/agents` |
| Vercel AI SDK | — | `ai` |
| Pydantic AI | `pydantic-ai` | — |
| Claude Agent SDK | `claude-agent-sdk` | — |

- **Package manager:** Detect from lock files (`pnpm-lock.yaml`, `yarn.lock`, `package-lock.json`, `poetry.lock`, `uv.lock`)
- **Entrypoint:** Identify the main application file (e.g. `main.py`, `app.py`, `index.ts`, `src/index.ts`)

Present what you found to the user:

> Detected:
> - Language: [TypeScript/Python]
> - LLM libraries: [openai, anthropic, ...]
> - Package manager: [npm/yarn/pnpm/pip/poetry/uv]
> - Entrypoint: [path/to/main.ts]
>
> Does this look right?

Wait for confirmation before proceeding.

---

### 3. Install the Respan SDK and Instrumentations

Based on the detected language and libraries, install the appropriate packages.

#### TypeScript / JavaScript

**Core package:** `@respan/respan`

Look up latest version:
```bash
npm view @respan/respan version
```

Install with exact version using the detected package manager:

| Package manager | Command |
|----------------|---------|
| npm | `npm install --save-exact @respan/respan@<VERSION> --no-audit --no-fund` |
| yarn | `yarn add --exact @respan/respan@<VERSION>` |
| pnpm | `pnpm add --save-exact @respan/respan@<VERSION>` |

**Instrumentation packages** — install only for libraries detected in step 2:

| LLM library | Instrumentation package |
|-------------|------------------------|
| `openai` | `@respan/instrumentation-openai` |
| `@anthropic-ai/sdk` | `@respan/instrumentation-anthropic` |
| `@openai/agents` | `@respan/instrumentation-openai-agents` |
| `ai` (Vercel AI SDK) | `@respan/instrumentation-vercel` |

#### Python

**Core package:** `respan-ai`

Look up latest version:
```bash
pip index versions respan-ai
```

Install with exact version using the detected package manager:

| Package manager | Command |
|----------------|---------|
| pip | `pip install respan-ai==<VERSION>` |
| poetry | `poetry add respan-ai==<VERSION>` |
| uv | `uv add respan-ai==<VERSION>` |

**Instrumentation packages** — install only for libraries detected in step 2:

| LLM library | Instrumentation package |
|-------------|------------------------|
| `openai` | `respan-instrumentation-openai` |
| `anthropic` | `respan-instrumentation-anthropic` |
| `openai-agents` | `respan-instrumentation-openai-agents` |
| `pydantic-ai` | `respan-instrumentation-pydantic-ai` |
| `claude-agent-sdk` | `respan-instrumentation-claude-agent-sdk` |
| `crewai` | `respan-instrumentation-crewai` |
| `haystack-ai` | `respan-instrumentation-haystack` |

---

### 4. Ask About Function Wrapping

Ask the user:

> Would you like to wrap your functions with Respan decorators for structured traces?
>
> This lets you group LLM calls into logical steps:
> - `@workflow` / `withWorkflow` — top-level pipeline (e.g. "handle_request")
> - `@task` / `withTask` — individual steps (e.g. "generate_outline", "summarize")
> - `@agent` / `withAgent` — agent loops
> - `@tool` / `withTool` — tool calls
>
> Without decorators, you still get auto-traced LLM calls — but they appear as flat spans.
> With decorators, you get nested traces showing how your app flows.
>
> Would you like to add decorators? (yes/no)

If yes, ask the user which functions to wrap — or suggest candidates based on the code structure.

---

### 5. Set Up Instrumentation

Add initialization code to the project entrypoint, **before** any LLM client is created.

#### Tracing Only

##### TypeScript

```typescript
import { Respan } from "@respan/respan";
// Import detected instrumentations:
import { OpenAIInstrumentor } from "@respan/instrumentation-openai";

const respan = new Respan({
  appName: "<project-name>",
  instrumentations: [new OpenAIInstrumentor()],
});
await respan.initialize();
```

##### Python

```python
from respan import Respan
# Import detected instrumentations:
from respan_instrumentation_openai import OpenAIInstrumentor

respan = Respan(
    app_name="<project-name>",
    instrumentations=[OpenAIInstrumentor()],
)
```

#### Gateway Only

Point the LLM client's base URL at the Respan proxy. The Respan API key authenticates both the proxy and tracing.

##### TypeScript

```typescript
import OpenAI from "openai";

const client = new OpenAI({
  apiKey: process.env.RESPAN_API_KEY,
  baseURL: process.env.RESPAN_BASE_URL || "https://api.respan.ai/api",
});
```

##### Python

```python
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["RESPAN_API_KEY"],
    base_url=os.getenv("RESPAN_BASE_URL", "https://api.respan.ai/api"),
)
```

#### Both (Gateway + Tracing)

Combine both: initialize the Respan SDK for tracing AND point the LLM client at the gateway.

##### TypeScript

```typescript
import OpenAI from "openai";
import { Respan } from "@respan/respan";
import { OpenAIInstrumentor } from "@respan/instrumentation-openai";

const respan = new Respan({
  appName: "<project-name>",
  instrumentations: [new OpenAIInstrumentor()],
});
await respan.initialize();

const client = new OpenAI({
  apiKey: process.env.RESPAN_API_KEY,
  baseURL: process.env.RESPAN_BASE_URL || "https://api.respan.ai/api",
});
```

##### Python

```python
import os
from openai import OpenAI
from respan import Respan
from respan_instrumentation_openai import OpenAIInstrumentor

respan = Respan(
    app_name="<project-name>",
    instrumentations=[OpenAIInstrumentor()],
)

client = OpenAI(
    api_key=os.environ["RESPAN_API_KEY"],
    base_url=os.getenv("RESPAN_BASE_URL", "https://api.respan.ai/api"),
)
```

#### Adding Decorators (if user said yes in step 4)

##### TypeScript

```typescript
import { withWorkflow, withTask } from "@respan/respan";

const result = await withWorkflow({ name: "my_pipeline" }, async () => {
  const outline = await withTask({ name: "generate_outline" }, async () => {
    return await client.chat.completions.create({ ... });
  });
  const draft = await withTask({ name: "write_draft" }, async () => {
    return await client.chat.completions.create({ ... });
  });
  return draft;
});
```

##### Python

```python
from respan import workflow, task

@workflow(name="my_pipeline")
def run_pipeline(topic: str):
    outline = generate_outline(topic)
    return write_draft(outline)

@task(name="generate_outline")
def generate_outline(topic: str):
    return client.chat.completions.create(...)

@task(name="write_draft")
def write_draft(outline: str):
    return client.chat.completions.create(...)
```

---

### 6. Verify Traces

Run the application and verify traces appear in Respan.

If the `respan` CLI is available, use it to check:

```bash
respan traces list --limit 5
```

If the CLI is not available, instruct the user to check the Respan platform at https://platform.respan.ai.

Confirm:
- The app runs without errors
- At least one trace appears
- If decorators were added, verify the trace shows nested spans

---

### 7. Final Summary

Summarize:

- What mode was set up (tracing / gateway / both)
- What SDK and instrumentation packages were installed (with versions)
- Where code was modified (list files)
- Whether decorators were added
- Whether traces were verified
