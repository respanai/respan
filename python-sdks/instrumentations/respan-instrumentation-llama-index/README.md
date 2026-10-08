# respan-instrumentation-llama-index

Respan instrumentation plugin for LlamaIndex Core and standalone Workflows. It registers native LlamaIndex
instrumentation handlers and emits Respan-compatible OpenTelemetry spans through
the Respan tracing pipeline.

## Configuration

### 1. Install

```bash
pip install respan-instrumentation-llama-index
```

### 2. Set Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `RESPAN_API_KEY` | Yes | Your Respan API key. Authenticates both proxy and tracing. |
| `RESPAN_BASE_URL` | No | Defaults to `https://api.respan.ai/api`. |

All vendor-specific variables are derived from these in your application code.

## Quickstart

### 3. Run Script

```python
import os

from dotenv import load_dotenv
from llama_index.core import Document, SummaryIndex, Settings
from llama_index.llms.openai import OpenAI
from respan import Respan
from respan_instrumentation_llama_index import LlamaIndexInstrumentor

load_dotenv()

respan_api_key = os.environ["RESPAN_API_KEY"]
respan_base_url = os.getenv("RESPAN_BASE_URL", "https://api.respan.ai/api")

os.environ["OPENAI_API_KEY"] = respan_api_key
os.environ["OPENAI_BASE_URL"] = respan_base_url
os.environ["OPENAI_API_BASE"] = respan_base_url

respan = Respan(
    api_key=respan_api_key,
    base_url=respan_base_url,
    app_name="llama-index-quickstart",
    instrumentations=[LlamaIndexInstrumentor()],
)

Settings.llm = OpenAI(
    model="gpt-4o-mini",
    api_key=respan_api_key,
    api_base=respan_base_url,
)
index = SummaryIndex.from_documents(
    [Document(text="Respan captures traces from LlamaIndex applications.")]
)
query_engine = index.as_query_engine()

try:
    response = query_engine.query("What does Respan capture?")
    print(response)
finally:
    respan.shutdown()
```

### 4. View Dashboard

After running the script, traces appear on your [Respan dashboard](https://platform.respan.ai).

## Further Reading

See the [python/tracing/llama-index](https://github.com/respanai/respan-example-projects/tree/main/python/tracing/llama-index)
examples for runnable scripts covering Core LLM, embedding, retrieval, and agent APIs, plus standalone workflow execution and provider failure paths.

## Compatibility and capture

Validated against released core `0.14.23` and `0.14.25`, Workflows `2.14.0`
and `2.25.0`, and native instrumentation `0.4.3` and `0.6.0`. The adapter uses
native dispatcher events for LLMs, dense and sparse embeddings, tools, agents,
query engines, structured predictions, and standalone workflows. Application
return values, exceptions, workflow handles, and streaming iterators remain
unchanged.

Content capture respects `capture_content=False`, `TRACELOOP_TRACE_CONTENT=false`,
and the Respan context content policy. The initial opt-out remains a bound for
that call; a later veto removes already captured content. OpenTelemetry
suppression and sampling apply to native spans. Compatible instrumentor instances
share their registration, and deactivation removes only the adapter's own hooks.

Ordinary payloads are bounded and secrets redacted. Known tool schemas, tool
calls, dense vectors, and sparse vectors remain complete while content capture is
allowed. Provider usage comes from actual response counters, including cache and
reasoning details. OpenAI embedding usage is observed at the provider response
boundary because native embedding events omit that information. A cached or
custom embedding without provider usage gets no invented token count.

Native error status is preserved without synthetic HTTP status codes or error
outputs. Older workflows provide less detailed start tags; the adapter captures
the available native arguments instead. Remote stores, hosted workflows, and
billed live model calls require separate validation. The paired
`python/tracing/llama-index` examples run controlled fixtures by default and
include explicit export and live options.
