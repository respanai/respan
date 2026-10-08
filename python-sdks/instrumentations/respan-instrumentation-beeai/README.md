# Respan BeeAI Instrumentation

Trace [BeeAI Framework](https://framework.beeai.dev/) chat and streamed calls,
embeddings, tool execution, agents, and workflows through Respan's OpenTelemetry
pipeline. The instrumentor observes BeeAI's public Emitter lifecycle events and
connects nested runs using BeeAI's run IDs. It leaves SDK methods and results intact.

```bash
pip install respan-ai respan-instrumentation-beeai
```

Initialize Respan before running BeeAI components:

```python
import asyncio

from beeai_framework.agents.requirement import RequirementAgent
from beeai_framework.backend import ChatModel
from respan import Respan
from respan_instrumentation_beeai import BeeAIInstrumentor


async def main():
    respan = Respan(instrumentations=[BeeAIInstrumentor()])
    try:
        agent = RequirementAgent(llm=ChatModel.from_name("openai:gpt-4.1-nano"))
        response = await agent.run("Explain observability for agents.")
        print(response.last_message.text)
    finally:
        respan.shutdown()


asyncio.run(main())
```

BeeAI 0.1.51 and 0.1.85 are validated with released tracing dependencies. The
instrumentor uses AI semantic conventions 0.5.1 for streaming and cache attributes.
The Respan contract package floor is 2.7.6; older declared releases do not
provide the canonical span constants module. It no longer requires OpenInference's BeeAI instrumentation or patches its
processor classes. Avoid activating two BeeAI instrumentors from different
libraries for the same application, since each library creates its own spans.

`BeeAIInstrumentor(trace_content=False)` or `TRACELOOP_TRACE_CONTENT=false`
disables payload capture. Respan's `ENABLE_CONTENT_TRACING_KEY=False` context setting also disables capture. This decision is fixed at run start; disabling capture
before run finish also vetoes earlier buffered content. A configuration object
with supported `hide_*` payload flags (inputs, outputs, messages, text, images,
embedding vectors, tools, or invocation parameters) enabled suppresses all payload
content. Hidden invocation parameters also suppress request parameter attributes. Model identifiers,
source usage, and error types remain available. Generic content is redacted and bounded;
large values use a valid JSON envelope with `truncated: true`. Tool definitions,
current calls, arguments, execution inputs/results, correlation IDs, and vectors
preserve complete known payloads with redaction. Arbitrary object
serialization hooks are not called.

Multiple instances with matching settings share one listener. An instance with
different settings is rejected so a private configuration cannot silently share
an existing listener that captures content.
The last owner removes only that listener. OpenTelemetry suppression and sampling
apply when a run starts. Exceptions set OpenTelemetry ERROR and diagnostic events;
they do not create synthetic completions or HTTP status codes.

Usage is taken from explicitly populated BeeAI usage fields. Stream chunks are
read before BeeAI merges default zeros into its aggregate result. Cache fields
are available only in SDK versions that expose them. Embedding vectors come from
the result, and embedding input usage comes from its usage model. Tool execution
IDs are emitted only when BeeAI's context contains the actual `tool_call_msg`;
SDK paths that omit it cannot be correlated by inventing an ID.

The paired examples cover the supported surfaces with controlled model
boundaries and the released BeeAI run, agent, workflow, tool, and stream APIs.
Provider-specific behavior, serving/ACP/A2A transports, evaluators, and external
MCP execution require separate live integrations; this package observes their
BeeAI runs rather than adding spans for their transport internals.
