# respan-instrumentation-agno

Respan instrumentation plugin for [Agno](https://docs.agno.com/).

This package patches Agno's native `Agent` and `Team` run methods and emits
Respan-compatible OpenTelemetry spans directly. It does not use Agno's
OpenInference integration.

## Configuration

### 1. Install

```bash
pip install respan-instrumentation-agno
```

### 2. Set Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `RESPAN_API_KEY` | Yes | Your Respan API key. Authenticates both proxy and tracing. |
| `RESPAN_BASE_URL` | No | Defaults to `https://api.respan.ai/api`. |

## Quickstart

### 3. Run Script

```python
import os

from dotenv import load_dotenv

load_dotenv()

respan_api_key = os.environ["RESPAN_API_KEY"]
respan_base_url = os.getenv("RESPAN_BASE_URL", "https://api.respan.ai/api")
os.environ["OPENAI_API_KEY"] = respan_api_key
os.environ["OPENAI_BASE_URL"] = respan_base_url

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from respan import Respan
from respan_instrumentation_agno import AgnoInstrumentor

respan = Respan(
    api_key=respan_api_key,
    base_url=respan_base_url,
    instrumentations=[AgnoInstrumentor()],
)

agent = Agent(
    name="Haiku Agent",
    model=OpenAIChat(id="gpt-4o-mini"),
)

result = agent.run("Write a one-line haiku about tracing.")
print(result.content)

respan.flush()
```

### 4. View Dashboard

After running the script, traces appear on your [Respan dashboard](https://platform.respan.ai).

## Further Reading

- [Agno examples in respan-example-projects](https://github.com/respanai/respan-example-projects/tree/main/python/tracing/agno)
- [Respan example projects](https://github.com/respanai/respan-example-projects)
- [Agno documentation](https://docs.agno.com/)

## SDK compatibility and emitted spans

The adapter is tested with Agno 2.6.5 and 3.1.1. It supports sync/async
`Agent` and `Team` runs, content streams, tools, team delegation, and
`continue_run` / `acontinue_run` after confirmation. Continuations accept a
RunOutput or a persisted run ID; the adapter reads that exact prior run through
the SDK to avoid reporting historical assistant turns again.

Every current assistant message with a recorded model response becomes a chat
span, with that message's usage and tool calls. A tool waiting for confirmation
has no execution span until it runs. Team member runs remain children of their
team. Outputs without message records retain the aggregate fallback behavior.
Spans are reconstructed when a run or consumed stream finishes, so individual
model/tool timing uses the enclosing run's interval.

Streaming captures the SDK's final RunOutput internally without exposing an
extra item unless the caller requested `yield_run_output=True`. Early close
emits available partial content; cancellation and failures finalize error spans.
No usage is invented when an interrupted stream has not supplied metrics.

Multiple instrumentors share patch ownership. Explicit Agent/Team instances
remain independently instrumented after a global instrumentor is deactivated.
Inputs and outputs use bounded JSON serialization, and credential-shaped keys
are redacted. The adapter retains at most 64 stream events plus the final output.

The compatibility tests use released Agno and OpenAI clients with deterministic
HTTP responses. Install `openai` to run those tests; the live gateway test remains
opt-in. The [Agno examples](https://github.com/respanai/respan-example-projects/tree/main/python/tracing/agno)
include fixture mode for the complete sync/async, tool, team, and continuation suite.
