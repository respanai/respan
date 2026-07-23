# Respan OpenLIT instrumentation

`respan-instrumentation-openlit` sends OpenLIT's native OpenTelemetry spans
through the active Respan tracer provider and normalizes them to the Respan span
contract. It does not wrap provider calls a second time, so enabling this plugin
does not create a duplicate Respan span around each OpenLIT span.

## Install

```bash
pip install respan-ai respan-instrumentation-openlit
```

## Use

```python
from respan import Respan
from respan_instrumentation_openlit import OpenLITInstrumentor

respan = Respan(
    api_key="...",
    instrumentations=[OpenLITInstrumentor(capture_content=True)],
)
```

Activation calls `openlit.init()` with the existing OpenTelemetry provider,
metrics and events disabled, and an offline empty pricing file. OpenLIT therefore
owns provider/framework patching while Respan owns export. Pass a custom
`pricing_json` if OpenLIT cost calculation is required.

`capture_content=False` disables OpenLIT message capture and strips any content
attributes from pre-existing OpenLIT spans before export. `disabled_instrumentors`
is forwarded to OpenLIT and can be used to exclude selected libraries.

For OpenAI sync and async embedding calls, the adapter enriches OpenLIT native
spans with every returned vector element in canonical `traceloop.entity.output`.
The enrichment does not wrap provider methods or create another span, and it is
disabled together with all other payload capture by `capture_content=False`.

Do not also enable a Respan provider instrumentation for the same client in the
same process unless two nested provider spans are intentional. Deactivation only
uninstruments OpenLIT instrumentors that this plugin activated; pre-existing
OpenLIT instrumentation is preserved.
