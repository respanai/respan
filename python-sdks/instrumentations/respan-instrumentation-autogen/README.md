# respan-instrumentation-autogen

Trace AutoGen AgentChat assistants, teams, tools and OpenAI model clients with
Respan. Select `api="legacy"` for the separately supported `autogen` namespace.
The adapter uses OpenInference's serializers and content configuration, and owns
reversible SDK method wrappers that preserve iterator context and lifecycle.

## Install and use

```bash
pip install respan-ai respan-instrumentation-autogen
```

```python
import asyncio
import os
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from respan import Respan
from respan_instrumentation_autogen import AutoGenInstrumentor

runtime = Respan(
    api_key=os.environ["RESPAN_API_KEY"],
    is_auto_instrument=False,
    instrumentations=[AutoGenInstrumentor()],
)


async def main():
    model = OpenAIChatCompletionClient(
        model="gpt-4o-mini", api_key=os.environ["OPENAI_API_KEY"]
    )
    try:
        result = await AssistantAgent("assistant", model_client=model).run(
            task="Explain tracing in one sentence."
        )
        print(result.messages[-1].content)
    finally:
        await model.close()
        runtime.shutdown()


asyncio.run(main())
```

Provider credentials and trace-export credentials are configured separately.
Configure `base_url` on the model client when using a provider-compatible endpoint.

## Compatibility and coverage

- Microsoft AutoGen AgentChat/Core/Ext **0.5.1 through 0.7.5**, with matching
  AgentChat/Ext versions and OpenInference AgentChat **0.1.21** or compatible 0.1.x.
- Declared AutoGen 0.4.0 support was not resolvable: every published OI AgentChat
  release requires AutoGen >=0.5.0, and its current runtime check requires >=0.5.1.
  The package now declares that verified floor instead of an untestable range.
- Assistants retain native `TaskResult`/`Response` and structured output types.
  Completed agent spans end before `run()` returns. Teams retain their logical
  agent/model/tool hierarchy while duplicate Core message-bus spans are filtered.
- OpenAI/Azure OpenAI `create` and `create_stream` preserve arguments, result
  identity and streaming chunks. Streams expose `__anext__`, `asend`, `athrow`
  and `aclose`, bind their creation context, close underlying SDK generators and
  do not leave the consumer inside an instrumentation span.
- Tools retain real call IDs, complete schemas, zero/false results, failure state
  and tool-result history. Latest `AgentTool` nested calls are covered; that API
  is absent from the minimum SDK.
- Actual usage is retained; missing/invalid counts are omitted. Error responses
  are never invented. Errors use OTel status plus canonical error fields; actual
  OpenAI HTTP response status is preserved without top-level shortcut aliases.
- AgentChat has no embedding client boundary. Configure the provider or vector
  database's instrumentor for embedding/memory operations. Other provider model
  clients require their own provider instrumentation for model-call spans.

`TRACELOOP_TRACE_CONTENT=false`, the runtime content context, and OpenInference
`TraceConfig` are respected. Stream privacy is captured when the iterator is
created, even when consumed later. OTel suppression is scoped. Multiple owners
share patches and processors until the last deactivation; conflicting owner
settings raise `ValueError`. Foreign wrappers are preserved and old owned
wrappers become inert. Activation failures restore partial patches.

## Legacy autogen

The extras intentionally preserve the old APIs; they do not select the latest
packages named `pyautogen` or `autogen`. Microsoft no longer publishes the newer
`pyautogen` namespace. These distributions share `import autogen`; install only
one family per environment.

```bash
# Exact pyautogen 0.2.2 minimum requires Python 3.11.
pip install respan-ai 'respan-instrumentation-autogen[legacy-pyautogen]' \
  'pyautogen==0.2.2' 'openai==1.109.1'

# Separate environment: autogen 0.7 is a matching pyautogen metapackage.
pip install respan-ai 'respan-instrumentation-autogen[legacy-autogen]' \
  'autogen==0.7.6' 'openai==1.109.1'
```

Use `AutoGenInstrumentor(api="legacy")`. The supported families remain
`pyautogen>=0.2.2,<0.3` and `autogen>=0.7,<0.8`. Sync/async chats, replies,
GroupChat and function execution use the upstream AG2 wrappers with legacy
argument/result adaptation. Native return values, including `None` from
pyautogen 0.2.2 chats, are unchanged. Function results are tool content even when
containing an `assistant` role. Compose a provider instrumentor for actual
legacy model-call spans.

Run the requirements files in `tests/` in separate environments. Real SDK
fixtures cover both legacy families with OpenAI 1.109.1 and OI AG2 0.1.11;
provider roundtrips use deterministic HTTP transports. Full application-specific
dependency pins must still resolve independently.

The paired [AutoGen examples](https://github.com/respanai/respan-example-projects/tree/main/python/tracing/autogen)
cover the modern suite and both legacy environments without provider network calls.
