# respan-instrumentation-together

Respan instrumentation plugin for the official
[Together AI Python SDK](https://github.com/togethercomputer/together-py).

The package patches the generated Together SDK resources and emits canonical
Respan spans through the active OTEL pipeline. It captures sync calls, async
calls, streaming chat and text completions, chat tool definitions, model tool
calls, text completions, embeddings, rerank requests, image generations, and
token usage when Together returns it.

## Installation

```bash
pip install respan-ai respan-instrumentation-together together
```

## Usage

```python
from together import Together
from respan import Respan
from respan_instrumentation_together import TogetherInstrumentor

respan = Respan(instrumentations=[TogetherInstrumentor()])
client = Together()

response = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[{"role": "user", "content": "Say hello in one sentence."}],
)
print(response.choices[0].message.content)
respan.flush()
```
