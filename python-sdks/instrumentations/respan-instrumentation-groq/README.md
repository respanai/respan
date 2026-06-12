# Respan Groq Instrumentation

Trace the official Groq Python SDK with Respan.

## Installation

```bash
pip install respan-ai respan-instrumentation-groq
```

## Usage

```python
from groq import Groq
from respan import Respan
from respan_instrumentation_groq import GroqInstrumentor

respan = Respan(instrumentations=[GroqInstrumentor()])
client = Groq()

response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[{"role": "user", "content": "Say hello in one short sentence."}],
)
print(response.choices[0].message.content)

respan.flush()
respan.shutdown()
```

The package delegates SDK patching to `openinference-instrumentation-groq`
and uses Respan's OpenInference translator to normalize emitted spans into
the Respan tracing pipeline.
