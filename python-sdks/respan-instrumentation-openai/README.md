# respan-instrumentation-openai

Respan instrumentation plugin for direct OpenAI SDK usage.

## Installation

```bash
pip install respan-instrumentation-openai
```

## Usage

```python
from respan import Respan
from respan_instrumentation_openai import OpenAIInstrumentor

respan = Respan(instrumentations=[OpenAIInstrumentor()])
```

This wraps `opentelemetry-instrumentation-openai` and adds Respan-specific prompt capture support.
