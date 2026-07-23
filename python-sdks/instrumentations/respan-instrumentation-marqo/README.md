# respan-instrumentation-marqo

Respan instrumentation for Marqo client, index, document, search, and embedding operations.

```bash
pip install respan-ai respan-instrumentation-marqo marqo
```

```python
import marqo
from respan import Respan
from respan_instrumentation_marqo import MarqoInstrumentor

respan = Respan(instrumentations=[MarqoInstrumentor()])
index = marqo.Client(url="http://localhost:8882").index("documents")
print(index.search("AI observability"))
respan.flush()
```
