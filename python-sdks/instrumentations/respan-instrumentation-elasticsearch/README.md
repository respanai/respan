# Elasticsearch instrumentation

Add Respan TASK input/output to the official Elasticsearch client's native
OpenTelemetry spans. Elasticsearch 8.13 through 9.x sync and async clients use
their real transport and typed response objects. This adapter retains native
span names and helper tracing, and observes the API response before its native
span ends. It creates a span for direct `elastic_transport` requests.

```bash
pip install 'elasticsearch[async]>=8.13,<10' respan-instrumentation-elasticsearch
```

For review before the paired change is published, install this package checkout
with `python -m pip install .` from its directory, or install its validated wheel.
The older published plugin does not contain these repairs. The checkout version
is managed by the repository's release workflow.

Configure your initialized native client and exporter in the application, then
pass the same OTel provider explicitly:

```python
from elasticsearch import Elasticsearch
from opentelemetry.sdk.trace import TracerProvider

from respan_instrumentation_elasticsearch import ElasticsearchInstrumentor


def search_with_tracing(client: Elasticsearch, provider: TracerProvider):
    instrumentor = ElasticsearchInstrumentor(tracer_provider=provider)
    instrumentor.activate()
    try:
        return client.search(index="documents", query={"match_all": {}})
    finally:
        instrumentor.deactivate()
```

Without an explicit provider, the adapter resolves the application's global
provider when calls begin. Native SDK tracing must be enabled; the SDK's
`OTEL_PYTHON_INSTRUMENTATION_ELASTICSEARCH_ENABLED=false` opt-out is honored.
The community OTel instrumentor already defers to these native spans and only
wraps sync transport; activating it does not provide canonical Respan capture.
The SDK's mature tracing remains the delegate here.

Automatic DB operations emit `respan.entity.log_type=task`, upstream database,
server and HTTP attributes sourced from native spans/response metadata, and
canonical JSON input/output. They do not emit embedding/model/token usage or
invent a status. Requests retain method, target, params, headers, native path
configuration, body and effective transport settings. Output preserves the
native body, including full vectors, history, schema, false, zero and empty
values. Native response metadata is retained separately at
`respan.metadata.elasticsearch.response`. Raised native errors retain their
actual type/status and sanitized native message; no error output is fabricated.
Ignored HTTP errors preserve the actual returned body and remain successful
native API calls. `HeadApiResponse` preserves its actual boolean, including
false. Object/list/text/binary responses and native streaming_bulk generators
retain their concrete type, identity, bytes, iteration and resource behavior.

Content capture defaults on with no adapter size/item cap. Set
`capture_content=False`, `RESPAN_TRACE_CONTENT=false`, or
`TRACELOOP_TRACE_CONTENT=false` for bodyless spans. The canonical Respan context
flag and Traceloop override/attribute flag are honored before extraction, at
callbacks, and before detach/end. Initial and observed ancestor vetoes cannot
widen later. General and LLM suppression and sampling prevent extraction;
unknown local span carriers fail closed, while genuine remote parents retain
propagation. Credential values are redacted in native JSON, headers, text and
URLs, including sensitive schema defaults. Unknown customer object conversion
and formatting hooks are omitted from telemetry. Approved structural run
markers survive private spans; other diagnostic events/messages/descriptions
are cleared, including the SDK's finished-span snapshot on late vetoes.

`max_attribute_chars=None` is the default. An explicitly supplied positive
integer opts into a preview envelope for oversized canonical JSON; native data
is never modified. The OTel SDK's own attribute-count limit can bound indexed
or convenience fields; full canonical input/output are written last.

Optional `request_hook(span, method, target, kwargs)` and
`response_hook(span, native_body)` run only during eligible capture. Their
failures fail closed for telemetry and preserve the native operation.
Activation/deactivation is idempotent and shared for matching configuration;
conflicting active configuration raises `ValueError`. Owned patches/processors
are removed by identity, retaining foreign wrappers. Deactivation during an
active call vetoes its retained content; its native call still completes.

Elasticsearch 8.13 lacks newer `helpers_span`/`use_span` methods, so bulk helper
wrapper spans depend on the native version; request spans are captured in both.
The adapter does not drain a caller generator or aggregate a helper wrapper's
unconsumed results. Fixture validation uses genuine localhost HTTP transports
and parsers; it does not validate a running Elasticsearch search engine.

Complete local-first examples are in `python/tracing/elasticsearch` of the
examples repository. Live-cluster calls and synthetic trace export have
separate explicit opt-ins. HTTP export success and stored-trace semantic
acceptance are separate checks.
