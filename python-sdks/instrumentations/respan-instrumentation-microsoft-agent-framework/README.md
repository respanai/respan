# respan-instrumentation-microsoft-agent-framework

Normalize [Microsoft Agent Framework](https://github.com/microsoft/agent-framework) native OpenTelemetry spans into the Respan contract. Supports `agent-framework-core>=1.8.1,<2`, validated against 1.8.1 and 1.20.0.

```bash
pip install respan-ai respan-instrumentation-microsoft-agent-framework agent-framework-openai
```

Initialize Respan before running agents, workflows, or embedding clients:

```python
import asyncio
import os
from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient
from respan import Respan
from respan_instrumentation_microsoft_agent_framework import (
    MicrosoftAgentFrameworkInstrumentor,
)


async def main():
    respan = Respan(
        api_key=os.environ["RESPAN_API_KEY"],
        instrumentations=[MicrosoftAgentFrameworkInstrumentor()],
        is_auto_instrument=False,
    )
    try:
        client = OpenAIChatClient()
        agent = Agent(client=client, name="assistant", instructions="Answer concisely.")
        print(await agent.run("Say hello."))
    finally:
        respan.shutdown()


asyncio.run(main())
```

Configure the OpenAI client with its provider credentials separately. The [paired examples](https://github.com/respanai/respan-example-projects/tree/main/python/tracing/microsoft-agent-framework) include deterministic local clients and an explicit live Gateway example.

The adapter captures agents, workflows, tasks, tools, chat calls and embeddings. Chat requests retain tool definitions and historical tool calls; completion fields contain only the current response's calls. Tool execution spans preserve `gen_ai.tool.call.id`, including the distinction between false/zero results and absent results. Embeddings keep full vectors, original response identity, and only actual reported input usage. Invalid negative, fractional, or boolean usage is omitted.

`capture_content=False`, `TRACELOOP_TRACE_CONTENT=false`, and Respan's scoped content setting omit content. Per-span policy is retained through deferred stream completion. OpenTelemetry suppression applies to native framework spans. Multiple active plugins share telemetry hooks and one processor; their `capture_content` policies must agree. Final deactivation restores owned settings and hooks while preserving later replacements and explicit native disablement. Unrelated instrumentation spans are not normalized.

Native `ResponseStream` APIs, results, cleanup, links and errors remain intact. SDK 1.20 supports completion, early close, failure, and cancellation. SDK 1.8.1 has no `ResponseStream.close()` and does not finish a cancelled native chat span; the same limitation is reproduced without Respan. Upgrade to current Agent Framework for those lifecycle behaviors.

The default examples export synthetic content to `RESPAN_BASE_URL` (default `https://api.respan.ai/api`) using `RESPAN_API_KEY`. Local checks and successful export are separate from stored-trace semantic acceptance.
