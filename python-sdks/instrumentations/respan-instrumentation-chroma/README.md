# respan-instrumentation-chroma

Respan instrumentation plugin for [Chroma](https://www.trychroma.com/) Python applications.

The package adds Respan task spans for Chroma client and collection operations, including collection lifecycle calls, writes, reads, vector queries, updates, deletes, and counts. Embeddings and image payloads are summarized by count and dimension instead of being written as raw span attributes.

## Configuration

### 1. Install

```bash
pip install respan-ai respan-instrumentation-chroma chromadb
```

### 2. Set Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `RESPAN_API_KEY` | Yes | Your Respan API key. Authenticates tracing export. |
| `RESPAN_BASE_URL` | No | Defaults to `https://api.respan.ai/api`. |

## Quickstart

```python
import chromadb

from respan import Respan, workflow
from respan_instrumentation_chroma import ChromaInstrumentor

respan = Respan(instrumentations=[ChromaInstrumentor()])


@workflow(name="chroma_query_workflow")
def run_chroma_query() -> dict:
    client = chromadb.Client()
    collection = client.get_or_create_collection("respan_docs")
    collection.upsert(
        ids=["doc-1"],
        embeddings=[[0.1, 0.2, 0.3]],
        documents=["Respan traces Chroma operations."],
        metadatas=[{"topic": "observability"}],
    )
    return collection.query(
        query_embeddings=[[0.1, 0.2, 0.3]],
        n_results=1,
        include=["documents", "metadatas", "distances"],
    )


print(run_chroma_query())
respan.flush()
respan.shutdown()
```

## Further Reading

See the [Respan example projects](https://github.com/respanai/respan-example-projects) for runnable Chroma scripts.
