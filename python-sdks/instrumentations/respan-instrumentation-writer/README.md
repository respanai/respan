# respan-instrumentation-writer

Respan instrumentation plugin for the
[Writer Python SDK](https://github.com/writer/writer-python).

This package patches Writer client resource methods and emits spans using the
Respan/Traceloop GenAI attribute shape used across this repository.

## Install

```bash
pip install respan-instrumentation-writer
```

## Quickstart

```python
import os

from respan import Respan
from respan_instrumentation_writer import WriterInstrumentor
from writerai import Writer

respan = Respan(
    api_key=os.environ["RESPAN_API_KEY"],
    instrumentations=[WriterInstrumentor()],
)

client = Writer(api_key=os.environ["WRITER_API_KEY"])

completion = client.chat.chat(
    model="palmyra-x5",
    messages=[{"role": "user", "content": "Write one line about tracing."}],
    max_tokens=64,
)

print(completion.choices[0].message.content)
respan.flush()
```

## Notes

- The instrumentor patches both `Writer` and `AsyncWriter` resources.
- `chat.chat()`, streamed chat, text completions, Knowledge Graph questions,
  application content generation, vision, translation, and direct Writer tool
  calls are traced.
- Writer-specific SDK field names stay local to this package; emitted spans use
  canonical GenAI/LLM and Respan-owned attribute constants.
