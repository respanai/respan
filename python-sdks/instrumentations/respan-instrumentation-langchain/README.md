# respan-instrumentation-langchain

Native LangChain callbacks produce workflow, task, chat, text, tool, and agent
spans through the active OpenTelemetry provider. Runnable/LCEL, models, tools,
retrievers, and LangGraph execution use the same callback managers. The adapter
keeps the SDK's return values, exceptions, and iterators.

```bash
pip install respan-ai respan-instrumentation-langchain
```

```python
from langchain_core.runnables import RunnableLambda
from respan import Respan
from respan_instrumentation_langchain import LangChainInstrumentor

respan = Respan(instrumentations=[LangChainInstrumentor()])
chain = RunnableLambda(lambda value: {"answer": value["question"]})
print(chain.invoke({"question": "hello"}))
respan.flush()
```

Set `RESPAN_API_KEY` for export. `RESPAN_BASE_URL` defaults to
`https://api.respan.ai/api`. For per-run callbacks, initialize a tracing provider
and use `chain.invoke(value, config=add_respan_callback(config))`. The helper
copies the config and callback manager; it preserves foreign handlers. Automatic
instrumentation preserves an existing explicit Respan handler, including its
privacy settings. Passing a new handler explicitly replaces Respan handlers in
the returned config only.

## Supported surfaces

The tested minimums are `langchain-core==0.3.0` and optional
`langgraph==0.2.20`. Current validation uses LangChain 1.4.3, core 1.6.6,
OpenAI integration 1.6.7, and LangGraph 1.2.12. Core callback support does not
require the main `langchain` package. Supported model methods include invoke,
stream, batch and async variants, and events. Actual model/tool callbacks capture
current tool calls, tool schemas, actual call IDs, source usage details, and
structured output. Known tool/vector payloads remain complete; other large
content is bounded. Retrievers export actual documents. Exceptions set OTel
ERROR and contain no fabricated output or HTTP status.

LangGraph state graphs work through native callbacks. Dynamic interrupt/resume
and `langchain.agents.create_agent` require releases that expose those APIs;
they are not available at the minimums. Graph interrupts remain normal control
flow. `on_agent_action`/`on_agent_finish` callbacks describe legacy agent
activity; they do not manufacture tool executions. Provider-internal HTTP spans
are owned by provider instrumentation, and the adapter does not replace SDK
iterators or install callback spans as the caller's current context.

Optional `langflow>=1.1.0` supports components that emit these callbacks. The
Langflow application itself is not exercised by the controlled fixtures. Set
`metadata={"framework": "langflow"}` to identify such runs. Root grouping uses a
Respan trace group identifier; each independent root keeps its own OTel trace.

The dependency floors are released `respan-tracing>=2.16.6,<3`,
`respan-sdk>=2.7.6`, AI semantic conventions `>=0.5.1,<0.6`, and OTel semantic
conventions `>=0.59b0`. They provide the canonical streaming, usage detail, tool
ID, and provider constants used directly by the adapter.

## Privacy and lifecycle

`LangChainInstrumentor(include_content=False)` or an explicit
`RespanCallbackHandler(include_content=False)` excludes prompts, outputs,
schemas, and raw callback metadata. `TRACELOOP_TRACE_CONTENT=false` and Respan's
content context opt-out apply at start and completion. An observed opt-out
permanently clears buffered content for that run and its ancestors; reenabling
capture cannot expand a private start. Sampling and instrumentation suppression
are honored before payload serialization. Actual usage and structural identifiers
can remain when content is private.

Compatible instrumentor instances share owned callback hooks until the last
owner deactivates. Incompatible settings/providers raise an error. Registration
failure restores owned hooks, and deactivation preserves later foreign patches.
`include_metadata=False` excludes raw callback tags/serialized metadata;
`metadata.respan_params` carries explicit Respan structural attributes. Callback
parent IDs establish the trace tree without leaking context across stream yields.

See the [paired examples](https://github.com/respanai/respan-example-projects/tree/main/python/tracing/langchain)
for controlled provider fixtures, privacy, vectors, graphs, and opt-in live/export
modes. Local tests, accepted exports, and stored trace semantics are separate
validation gates.
