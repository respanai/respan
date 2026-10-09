# respan-instrumentation-sagemaker

Observe the three native SageMaker Runtime boto3 operations: InvokeEndpoint, InvokeEndpointWithResponseStream and InvokeEndpointAsync. Current boto3/botocore 1.43.108 and the declared 1.34.0 floor are verified against released companions. Install and activate with an existing Respan or OTel provider:

```python
from respan_instrumentation_sagemaker import SageMakerInstrumentor

instrumentor = SageMakerInstrumentor()
instrumentor.activate()
# Call the normal boto3 SageMaker Runtime client and read/close its returned body.
instrumentor.deactivate()
```

Native StreamingBody and EventStream instances, response dictionaries, bytes, iteration, context-manager return values, close behavior and exceptions are preserved. The adapter reads no response body in advance; it observes caller consumption through owned instance hooks. Normal bodies finish at EOF/full read or close; streams retain consumed fragments and native partial data when an error follows. Unconsumed bodies do not claim a completed result.

Known native JSON retains complete histories, schemas, tool IDs/arguments, false/zero and dense/sparse vectors. Fragmented JSON/NDJSON/SSE application data is assembled from actual consumed PayloadPart bytes. Native embedding records map to embedding spans; unknown ML payloads remain tasks. Async submission preserves its actual response fields without inventing an inference result or submission state. Current session/routing response fields and native request selectors are retained in scoped metadata. Endpoint names are not fabricated model names; source model, usage, cache/reasoning and explicit totals are mapped only when present. Native configured OTel attribute limits can bound convenience indexed attributes; complete canonical bodies are written last.

`capture_content=False`, canonical content-disabled context, supported legacy flags, `TRACELOOP_TRACE_CONTENT=false` or `RESPAN_TRACE_CONTENT=false` omit owned content and error messages. Ambient/supplied, initial, active/finished ancestor, unknown local parent and runtime context exits are bounded irreversibly. Sampling and instrumentation suppression precede content inspection. Unknown conversion hooks/iterators and opaque request streams are not read solely for tracing. Credential values are redacted while schemas and valid encoded JSON retain their structure.

Observer faults preserve native outcomes and release owned hooks/contexts. Shared activation/configuration and foreign wrappers/processors retain their ownership. The released SDK/constants and AI semantic-convention floors are verified by installed native tests and a wheel built from the sdist. The companion examples default to controlled local SDK transports; hosted permissions and arbitrary custom model protocols are separate checks. Stored backend projections require exact same-run inspection and are not repaired with ingestion aliases.
