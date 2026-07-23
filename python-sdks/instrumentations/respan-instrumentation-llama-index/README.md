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

response = query_engine.query("What does Respan capture?")
print(response)

respan.flush()
```

### 4. View Dashboard

After running the script, traces appear on your [Respan dashboard](https://platform.respan.ai).

## Further Reading

See the [python/tracing/llama-index](https://github.com/respanai/respan-example-projects/tree/main/python/tracing/llama-index)
examples for runnable scripts covering Core LLM, embedding, retrieval, and agent APIs, plus standalone Workflows success and failure paths.
