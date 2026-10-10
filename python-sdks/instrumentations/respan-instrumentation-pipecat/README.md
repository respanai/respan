# respan-instrumentation-pipecat

Wraps the released OpenInference Pipecat instrumentor and translates its native
turn, LLM, STT, TTS and function spans into Respan attributes. Pipecat continues
to execute its original pipeline, handlers, frames and provider streams.

```bash
pip install respan-ai respan-instrumentation-pipecat
```

```python
import os
from respan import Respan
from respan_instrumentation_pipecat import PipecatInstrumentor

sdk = Respan(
    api_key=os.environ["RESPAN_API_KEY"],
    base_url=os.getenv("RESPAN_BASE_URL", "https://api.respan.ai/api"),
    instrumentations=[PipecatInstrumentor()],
)
# Construct the native PipelineWorker (or supported older PipelineTask) here.
# Run it with Pipecat's WorkerRunner / PipelineRunner.
try:
    ...
finally:
    sdk.flush()
    sdk.shutdown()
```

Use the upstream-compatible dependency pair:

| Pipecat | Native OpenInference delegate | Pipeline API |
| --- | --- | --- |
| >=1.1,<1.3 | >=1.0.2,<2 | PipelineTask / PipelineRunner |
| >=1.3,<2 | >=2,<3 | PipelineWorker / WorkerRunner |

The package accepts both families; install a compatible pair when pinning or
upgrading. Validation covers Pipecat1.12.0 with native2.0.8 and Pipecat1.1.0
with native1.0.2. Python3.11–3.13 is declared; validation uses Python3.12.
Respan SDK2.7.6, tracing2.17.0, AI semantic conventions0.4.13 and
OpenInference semantic conventions0.1.28 are the tested direct floors.

Actual native message history, current tool definitions/call IDs/arguments, and
dense or sparse tool-result vectors are retained when content capture is
enabled. Indexed message attributes are bounded to protect the default OTel
attribute budget; canonical entity JSON retains the complete history. Sensitive
values are redacted while JSON-schema property names remain structural data.
Native token metrics retain supplied input/output/total, cache and reasoning
counts. The Pipecat OpenAI service also observes the original HTTP/SSE usage
before OpenAI converts it to DTOs; invalid or absent counts are omitted. Other
providers use their native typed metrics and are not independently HTTP-verified.

Content capture has an initial bound, ancestor privacy and an irreversible veto
observed before context detach and span end. Ambient and explicitly supplied
content contexts must both permit capture at span start. Set `capture_content=False`,
`TRACELOOP_TRACE_CONTENT=false`, or the Respan content context to disable it.
Upstream `TraceConfig(hide_inputs=True, hide_outputs=True)` remains supported.
Suppression and the provider's sampler run before owned content extraction.
Errors retain their actual type and genuine provider HTTP status; the adapter
does not manufacture error output, HTTP500 or usage.

Initialize an OTel SDK `TracerProvider` (or Respan) before activation. An
uninitialized proxy provider is rejected before any native hook is changed.
Content under a recording parent created before activation is conservatively
disabled because its initial privacy bound was not observed.
Activation is shared by owners using the same provider and configuration;
conflicting settings raise `ValueError`. The last owner removes its hooks and
processor. Foreign replacements survive, partial activation rolls back, and
ongoing native pipeline execution remains usable after telemetry deactivation.

The [paired examples](https://github.com/respanai/respan-example-projects/tree/main/python/tracing/pipecat)
run controlled native pipelines without credentials by default. They cover
HTTP/SSE, real registered tool handlers, long history/schema/vector payloads,
private capture, genuine provider errors, native cancellation, partial output
and STT/TTS text frames. Voice fixtures
verify frame observation, not live audio-model, transport, codec or latency
acceptance. Live provider calls and Respan export are explicit opt-ins. Native
worker bus, distributed execution, realtime audio providers and provider routing
are outside this controlled validation.
