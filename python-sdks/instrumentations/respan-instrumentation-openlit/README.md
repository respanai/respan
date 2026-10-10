# Respan OpenLIT instrumentation

This plugin routes OpenLIT's native OpenTelemetry spans through the active Respan provider and normalizes them to the Respan span contract. OpenLIT retains provider/framework instrumentation; bounded observers enrich its existing OpenAI spans without creating duplicate provider spans.

```bash
pip install respan-ai respan-instrumentation-openlit
```

```python
from respan import Respan
from respan_instrumentation_openlit import OpenLITInstrumentor

respan = Respan(
    api_key="...",
    instrumentations=[OpenLITInstrumentor(capture_content=True)],
)
```

Supported Python versions are 3.11–3.13. The declared OpenLIT range is 1.44–<2; tested combinations are OpenLIT 1.45.0/OpenAI 2.54.0 and OpenLIT 1.44.0/OpenAI 1.92.0. OpenLIT itself requires OpenAI <3. The paired OpenTelemetry floor is API/SDK 1.39 and semantic conventions 0.60b0: released OpenLIT fails to import its log exporter at its own declared 1.38 floor. AI conventions 0.5.1 and Respan SDK 2.7.6 supply the canonical keys used here; Respan tracing 2.17 is tested at the floor.

Activation supplies the active provider to `openlit.init()`, disables metrics/events and generic HTTP transport instrumentation by default, and uses a packaged empty offline pricing file. `pricing_json`, `disabled_instrumentors`, and `capture_transport_spans` retain their native configuration roles. Multiple owners share one activation and must agree on configuration and provider. Deactivation restores only still-owned wrappers and native configuration fields; foreign replacements remain intact. In-flight native spans retain their privacy processor until they finish.

`capture_content=False`, `TRACELOOP_TRACE_CONTENT=false`, `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=false`, and the Respan content context opt-out prevent adapter payload capture. Capture is bounded at span start, including supplied parent context, and a later opt-out permanently vetoes the owned span and descendants. A checkpoint before native context detach retains the bound for delayed streams after a parent has finished. OTel/model suppression and non-recording spans bypass adapter content serialization. OpenLIT events are removed during normalization. Credential values are redacted, including quoted text and URL credentials; known tool schema property names/types remain structural while sensitive defaults/examples redact.

Real released-SDK tests cover OpenAI sync/async Chat Completions and Responses, native Chat parse, Responses.parse under OpenLIT 1.45, text streaming, early close/cancellation, tools, errors and embeddings. Request/history and current tool calls remain separate; actual IDs, single-encoded argument strings, complete schemas, dense/sparse tool results and embedding vectors are retained. `max_content_length` bounds ordinary request text; recognized tool/vector payloads retain their full data. Unknown objects are represented without arbitrary stringification. Source response token counts are validated before OpenAI coercion; absent, invalid or boolean counts are omitted, and actual zero is retained. Error spans preserve native error status and actual HTTP status without invented outputs or shortcut status attributes.

Coverage is bounded to these OpenAI surfaces and a controlled Anthropic text fixture. Other OpenLIT integrations continue to delegate to native instrumentation; their usage provenance and feature coverage are not established by this audit. A bare OpenLIT 1.44 fixture marks a successful OpenAI result ERROR when optional nested usage details are absent; the adapter preserves that native status. Minimum-compatible positive fixtures supply those actual details. Native stream/context behavior remains unchanged where it is an upstream SDK property. Do not enable another provider instrumentation for the same client unless nested spans are intentional. Backend storage/projection fidelity requires separate exact-run trace inspection after export.
