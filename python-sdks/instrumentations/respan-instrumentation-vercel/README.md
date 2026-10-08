# respan-instrumentation-vercel

Trace Vercel's Python AI SDK with Respan. Supports `ai >=0.7.0,<0.9.0` on Python 3.12 and 3.13.

Initialize the Respan runtime, then activate `VercelInstrumentor` before importing operation functions:

```python
from respan import Respan
from respan_instrumentation_vercel import VercelInstrumentor

respan = Respan(instrumentations=[VercelInstrumentor()])

import ai
```

The instrumentor captures generation, streaming, agent and tool execution, embeddings, image/video/audio operations, transcription, reranking, and evaluation. Non-chat operations keep their native results without being treated as chat completions. Deferred `DictSink` traces retain their inputs, outputs, and parent relationships when replayed.

AI SDK 0.8 exposes evaluation as `ai.ops.experimental.evaluate`. Both mapped questions and Pydantic question/output models are supported, including Noul answers. A typed evaluation's `output_type` is captured as its JSON schema. AI SDK 0.7's `ai.ops.experimental_evaluate` remains supported.

Set `capture_content=False` on the instrumentor, `TRACELOOP_TRACE_CONTENT=false`, or the runtime content-tracing context to suppress captured inputs and outputs. Deactivate only after in-flight calls finish.

Feature-compatible examples live in `respan-example-projects/python/tracing/vercel-ai-sdk`.
