# Respan instrumentation for Marqo

Native OpenTelemetry instrumentation for the official Marqo Python client,
validated against Marqo 3.18.2 and the declared minimum 3.5.1.

```python
import os

import marqo
from opentelemetry.sdk.trace import TracerProvider

from respan_instrumentation_marqo import MarqoInstrumentor

provider = TracerProvider()  # Add your application's processors/exporters.
instrumentor = MarqoInstrumentor(tracer_provider=provider)
instrumentor.activate()
try:
    client = marqo.Client(url=os.environ["MARQO_URL"])
    result = client.index("documents").search(q="native tracing")
    print(result)
finally:
    instrumentor.deactivate()
    provider.shutdown()
```

The paired examples use actual released Marqo clients against a local controlled
HTTP server, with no Marqo credentials. Respan trace export requires an explicit
opt-in. These controlled examples do not establish hosted-service acceptance.

## Coverage

Client operations include `create_index`, `delete_index`, `get_indexes`, and
`bulk_search`. Index operations include create/delete, document add/update/get/
get-many/delete, search, recommend, embed, settings/stats/status/health, and model
ejection where available in the installed SDK. Native convenience aliases
produce one operation span. `index` and `get_index` retain their native handle
factory behavior and do not invent database-operation spans. Native minimum
model ejection requires its original `model_device` argument; the adapter does
not rewrite SDK signatures or parameters.

Database operations are task spans with standard database attributes and full
sanitized public inputs, observed serialized requests, and native results.
Actual `Index.embed` calls are embedding spans: complete returned vectors appear
in canonical entity output; nonvector response fields remain separately in
capture-gated metadata. Model and HTTP status fields are emitted only when
observed. No token usage, totals, response, error output, or HTTP status is
invented. A final observed HTTP status describes an actual request and does not
claim an aggregate batch status. Native worker requests may execute outside the
call context, leaving transport metadata unobserved while public batch inputs
and aggregate native results remain complete.

The released upstream instrumentor wraps only add/search/delete-documents and
does not provide this broader native API coverage or canonical payload/privacy
behavior. This adapter retains its existing native boundary. Native results,
exceptions, warnings, request serialization, callbacks, and SDK session ownership
are preserved. Instrumentation does not drain responses, repeat provider calls,
or close Marqo's module-owned HTTP session.

## Content and lifecycle

Sampling and general/LLM suppression are checked before telemetry content
inspection. Constructor `capture_content=False`, canonical Respan content flags,
legacy `trace_content`/`override_enable_content_tracing`, and both content
environment switches disable bodies. Initial and observed active/finished
ancestor denial cannot widen; unknown local parents fail closed. Later vetoes
clear owned input/output, native request buffers, metadata, diagnostics, SDK
events, and readable span snapshots before export.

Known native JSON and typed storage retain full records, 5001-value vectors,
schemas and false/zero/empty fields. Credentials in fields, quoted authentication
text and URLs are redacted; schema property names remain intact while sensitive
defaults/examples are redacted. Unknown user conversion, getter, equality, hash,
iterator, string and representation hooks are not invoked solely for telemetry.

Compatible owners share patches until the last deactivation. Configuration
conflicts raise `ValueError`, and restoration removes only owned descriptors and
processors. Observer startup, serialization, setter, attach/detach and end faults
preserve native outcomes and caller context. The supported runtime floors are
validated separately from current companions and a clean wheel built from the
sdist. Stored Respan trace fidelity remains a separate acceptance gate.
