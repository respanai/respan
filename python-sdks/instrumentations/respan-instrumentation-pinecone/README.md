# respan-instrumentation-pinecone

Respan instrumentation for Pinecone data, control-plane, and inference operations,
including the synchronous, asynchronous, and gRPC clients.

```bash
pip install respan-ai respan-instrumentation-pinecone pinecone
```

```python
from pinecone import Pinecone
from respan import Respan
from respan_instrumentation_pinecone import PineconeInstrumentor

respan = Respan(instrumentations=[PineconeInstrumentor()])
index = Pinecone().index("documents")
print(index.query(vector=[0.1, 0.2], top_k=3))
respan.flush()
```
