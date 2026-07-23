# respan-instrumentation-elasticsearch

Respan instrumentation for the official Elasticsearch Python client. It traces the shared transport boundary, so synchronous and asynchronous clients receive the same coverage for index, get, search, bulk, update, delete, cluster, and other API operations.

Every request emits a canonical Respan `task` span. Request and response bodies are JSON-serialized into `traceloop.entity.input` and `traceloop.entity.output`; transport failures and HTTP error responses record OpenTelemetry error status plus backend-visible `status_code` and `error.message` attributes.

## Install

```bash
pip install respan-ai respan-instrumentation-elasticsearch "elasticsearch[async]>=8.13,<10"
```

## Usage

```python
from elasticsearch import Elasticsearch
from respan import Respan, workflow
from respan_instrumentation_elasticsearch import ElasticsearchInstrumentor

respan = Respan(instrumentations=[ElasticsearchInstrumentor()])
client = Elasticsearch("http://localhost:9200")


@workflow(name="elasticsearch_search_workflow")
def search():
    return client.search(index="articles", query={"match": {"title": "tracing"}})


print(search())
client.close()
respan.shutdown()
```

`AsyncElasticsearch` is instrumented automatically as well.

## Content capture

Request and response content is captured by default. Disable it when payloads may contain sensitive data:

```python
ElasticsearchInstrumentor(capture_content=False)
```

With capture disabled, spans retain operation, sanitized target, status, and timing while omitting request and response bodies. Headers are never captured. `activate()` and `deactivate()` are idempotent.

See the Respan example projects for an offline, runnable sync/async suite backed by a local mock Elasticsearch HTTP server.
