# respan-instrumentation-huggingface

Respan instrumentation plugin for Hugging Face Transformers.

This package wraps the upstream `opentelemetry-instrumentation-transformers`
instrumentor and adds a small Respan contract processor so
`TextGenerationPipeline.__call__` spans export with canonical Respan fields.

## Installation

```bash
pip install respan-ai respan-instrumentation-huggingface
```

## Usage

```python
from respan import Respan, workflow
from respan_instrumentation_huggingface import HuggingFaceInstrumentor
from transformers import pipeline

respan = Respan(
    app_name="huggingface-text-generation",
    instrumentations=[HuggingFaceInstrumentor()],
)


@workflow(name="huggingface_text_generation")
def run_generation():
    generator = pipeline(
        "text-generation",
        model="sshleifer/tiny-gpt2",
    )
    return generator(
        "Tracing helps teams",
        max_length=30,
        temperature=0.7,
        top_p=0.9,
    )


try:
    print(run_generation())
finally:
    respan.flush()
    respan.shutdown()
```

Set `TRACELOOP_TRACE_CONTENT=false` to suppress prompt and completion text in
span attributes while still exporting model and request metadata.
