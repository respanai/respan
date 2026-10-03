# respan-instrumentation-pydantic-ai

Respan instrumentation plugin for [PydanticAI](https://ai.pydantic.dev/).

This package enables PydanticAI's native OpenTelemetry emission and maps the
resulting PydanticAI attributes directly into the Respan/Traceloop conventions
used by the OTLP pipeline.

## Install

```bash
pip install respan-instrumentation-pydantic-ai
```

## Quickstart

```python
import os

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from respan import Respan
from respan_instrumentation_pydantic_ai import PydanticAIInstrumentor

respan = Respan(
    api_key=os.environ["RESPAN_API_KEY"],
    instrumentations=[PydanticAIInstrumentor()],
)

agent = Agent(
    OpenAIChatModel(
        "gpt-4o-mini", provider=OpenAIProvider(api_key=os.environ["OPENAI_API_KEY"])
    )
)

result = agent.run_sync("Write a one-line haiku about tracing.")
print(result.output)

respan.flush()
```

## Notes

- By default the instrumentor enables global PydanticAI instrumentation via `Agent.instrument_all(...)`.
- Pass `agent=...` to `PydanticAIInstrumentor(...)` if you only want to instrument one agent instance.
- The plugin uses explicit `InstrumentationSettings(version=5)` to keep emitted spans on the current Pydantic AI v2 GenAI semantic conventions.
- This package does not depend on OpenInference at runtime; it consumes native PydanticAI telemetry directly.

## Current SDK coverage

Compatibility is tested with PydanticAI 2.0.0 and 2.54.0 using its released
`TestModel`, `FunctionModel`, `TestEmbeddingModel`, and native telemetry.
The plugin maps version 5 messages and supports the current SDK’s optional
version 6 format with tool-role results. It preserves request-level input,
output, cache, and reasoning token counts without counting agent usage again.

Global activation enables both `Agent` and `Embedder` telemetry. Embedding
queries and document batches preserve their inputs, full returned vectors,
reported usage, provider, model, parentage, and native errors. To scope
instrumentation to one embedder, pass `embedder=my_embedder`; `agent=...`
and `embedder=...` are mutually exclusive. Shared owners restore the previous
settings only when the last owner deactivates. `include_content=False` omits
embedding inputs/vectors and excludes content-free native message placeholders.

The full PydanticAI distribution currently installs Logfire 5.1.1, which requires
OpenTelemetry SDK `>=1.39,<1.45`. The tested released runtime uses
`respan-tracing==2.20.1`, `respan-sdk==2.7.6`, OpenTelemetry 1.44.0, and AI semantic
conventions 0.5.1. Use a separate environment for integrations requiring OTel
1.45 or later.

Native PydanticAI 2.54 leaves the child model span unset when iteration of a
`FunctionModel` stream raises, while its enclosing agent span records the error.
The adapter preserves that upstream behavior.

See the companion [Python examples](https://github.com/respanai/respan-example-projects/tree/main/python/tracing/pydantic-ai)
for deterministic models, tools, structured output, streams, embeddings,
controlled failures, and content opt-out. Before publication, install the
paired adapter checkout and example requirements in one pip command.
