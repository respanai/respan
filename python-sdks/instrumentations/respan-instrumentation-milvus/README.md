# respan-instrumentation-milvus

Respan instrumentation for synchronous and asynchronous PyMilvus operations.

```bash
pip install respan-ai respan-instrumentation-milvus pymilvus
```

```python
from pymilvus import MilvusClient
from respan import Respan
from respan_instrumentation_milvus import MilvusInstrumentor

respan = Respan(instrumentations=[MilvusInstrumentor()])
client = MilvusClient(uri="./milvus.db")
print(client.list_collections())
respan.flush()
```
