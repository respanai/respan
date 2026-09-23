# respan-instrumentation-claude-agent-sdk

Respan instrumentation plugin for the [Claude Agent SDK](https://code.claude.com/docs/en/agent-sdk/overview).

This package enables the Claude Agent SDK's native OpenTelemetry emission and
normalizes those spans into the Respan/Traceloop conventions used by the OTLP
pipeline.

Tool execution spans preserve `gen_ai.tool.call.id` so their inputs and results
can be matched to the agent's canonical tool calls. Repeated observations of the
same invocation ID produce one canonical call, with JSON argument formatting and
object-key order ignored during comparison. If the same ID has conflicting names
or arguments, the processor logs a warning and keeps the first call. Missing
fields are filled from later observations of the same ID. Calls without an ID
are compared by name and normalized arguments.

Repeated activation on the same tracer provider shares one normalization
processor. Deactivating one instrumentor keeps that processor active until its
last owner deactivates. Normalization also preserves already-normalized tool
types, names, inputs, and results.
The upstream SDK patches are global: the first active instance's provider,
`agent_name`, and `capture_content` settings remain in effect until all instances
deactivate.

Failed-tool hooks retain their error result when content capture is enabled.
For both `query()` and `ClaudeSDKClient`, terminal SDK tool results also close
matching spans when a post-tool hook is missing. Explicit permission denials
are recorded as SDK denial outcomes, not fabricated tool stderr. Repeated
results and late post-tool hooks cannot close the same span twice. Unmatched
tool or subagent spans still report genuine cleanup errors.
This compatibility handling is provided by Respan; it does not require a fork
or changes to the installed upstream instrumentation package.

## Configuration

### 1. Install

```bash
pip install respan-instrumentation-claude-agent-sdk
```

### 2. Set Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `RESPAN_API_KEY` | Yes | Your Respan API key. Authenticates both proxy and tracing. |
| `RESPAN_BASE_URL` | No | Defaults to `https://api.respan.ai/api`. |

All vendor-specific variables (for example `ANTHROPIC_API_KEY`) are derived
from these in your application code.

## Quickstart

### 3. Run Script

```python
import asyncio
import os

import claude_agent_sdk
from claude_agent_sdk import ClaudeAgentOptions, ResultMessage
from respan import Respan
from respan_instrumentation_claude_agent_sdk import ClaudeAgentSDKInstrumentor

respan_api_key = os.environ["RESPAN_API_KEY"]
respan_base_url = os.getenv("RESPAN_BASE_URL", "https://api.respan.ai/api")

os.environ["ANTHROPIC_API_KEY"] = respan_api_key
os.environ["ANTHROPIC_AUTH_TOKEN"] = respan_api_key
os.environ["ANTHROPIC_BASE_URL"] = f"{respan_base_url}/anthropic"

respan = Respan(
    api_key=respan_api_key,
    base_url=respan_base_url,
    instrumentations=[ClaudeAgentSDKInstrumentor(capture_content=True)],
)


async def main() -> None:
    options = ClaudeAgentOptions(
        model="sonnet",
        max_turns=1,
        permission_mode="bypassPermissions",
        cwd=os.getcwd(),
        env={
            "ANTHROPIC_API_KEY": respan_api_key,
            "ANTHROPIC_AUTH_TOKEN": respan_api_key,
            "ANTHROPIC_BASE_URL": f"{respan_base_url}/anthropic",
        },
    )

    async for message in claude_agent_sdk.query(
        prompt="Reply with exactly hello_from_claude_sdk.",
        options=options,
    ):
        if isinstance(message, ResultMessage):
            print(message.result)


asyncio.run(main())
respan.flush()
```

### 4. View Dashboard

After running the script, traces appear on your [Respan dashboard](https://platform.respan.ai).

## Further Reading

See the [python/tracing/claude-agent-sdk](https://github.com/respanai/respan-example-projects/tree/main/python/tracing/claude-agent-sdk)
example for a runnable end-to-end workflow that covers tool use, multi-turn
sessions, and edge cases.
