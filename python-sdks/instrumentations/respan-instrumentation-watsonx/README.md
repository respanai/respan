# Respan IBM watsonx.ai Instrumentation

Native Respan instrumentation for the official `ibm-watsonx-ai` Python SDK.

## Installation

```bash
pip install respan-ai respan-instrumentation-watsonx ibm-watsonx-ai
```

## Usage

```python
from respan import Respan
from respan_instrumentation_watsonx import WatsonxInstrumentor

respan = Respan(
    api_key="RESPAN_API_KEY",
    instrumentations=[WatsonxInstrumentor()],
)
```

The instrumentor traces:

- `ModelInference.generate()`
- `ModelInference.generate_text()`
- `ModelInference.generate_text_stream()`
- `ModelInference.chat()`
- `ModelInference.chat_stream()`
- `ModelInference.agenerate()`
- `ModelInference.agenerate_stream()`
- `ModelInference.achat()`
- `ModelInference.achat_stream()`
- `Embeddings.generate()`
- `Embeddings.embed_query()`
- `Embeddings.embed_documents()`
- `Embeddings.agenerate()`
- `Embeddings.aembed_query()`
- `Embeddings.aembed_documents()`

Chat and text-generation spans use canonical GenAI fields such as
`gen_ai.prompt.*`, `gen_ai.completion.*`, `llm.request.functions`, and token
usage attributes. Embedding spans record the input and usage metadata but do not
export embedding vectors.
