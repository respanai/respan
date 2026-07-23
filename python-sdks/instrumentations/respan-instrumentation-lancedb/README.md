# respan-instrumentation-lancedb

Respan instrumentation for synchronous and asynchronous LanceDB connections,
tables, and query execution.

```bash
pip install respan-ai respan-instrumentation-lancedb lancedb
```

```python
import lancedb
from respan import Respan
from respan_instrumentation_lancedb import LanceDBInstrumentor

respan = Respan(instrumentations=[LanceDBInstrumentor()])
db = lancedb.connect("/tmp/lancedb")
table = db.create_table("documents", [{"vector": [0.1, 0.2], "text": "hello"}], mode="overwrite")
print(table.search([0.1, 0.2]).limit(1).to_list())
respan.flush()
```
