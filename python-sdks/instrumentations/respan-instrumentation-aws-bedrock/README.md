# respan-instrumentation-aws-bedrock

Trace native `boto3` Bedrock Runtime inference calls with Respan.

```bash
pip install 'respan-instrumentation-aws-bedrock[instruments]'
```

```python
import boto3
from respan import Respan
from respan_instrumentation_aws_bedrock import AWSBedrockInstrumentor

respan = Respan(instrumentations=[AWSBedrockInstrumentor()])
client = boto3.client("bedrock-runtime", region_name="us-east-1")
response = client.converse(
    modelId="YOUR_BEDROCK_MODEL_ID",
    messages=[{"role": "user", "content": [{"text": "Say hello."}]}],
    inferenceConfig={"maxTokens": 32},
)
respan.flush()
```

The adapter supports `InvokeModel`, `InvokeModelWithResponseStream`, `Converse`,
and `ConverseStream` when the installed boto3 service model exposes them.
Boto3 1.34.0 exposes the InvokeModel operations; Converse requires a newer SDK.
It observes the original `StreamingBody` and `EventStream` as the application
reads or closes them. It preserves native object types, bytes, errors, and
context-manager return values. An unread body emits its span when it is closed
or instrumentation is deactivated. A partial body has no synthetic completion.

Spans retain full JSON request histories, tool definitions, tool IDs and
fragmented streamed arguments, reasoning and multimodal blocks, source usage
including zero/cache counts, and full embedding vectors. Usage totals are
captured only when returned by the provider. No content length limit is imposed
by this adapter; boto3's native deserializer determines which fields it exposes.

`AWSBedrockInstrumentor(capture_content=False)` disables payload and diagnostic
capture. `TRACELOOP_TRACE_CONTENT=false`, `RESPAN_TRACE_CONTENT=false`, context
content vetoes, both OpenTelemetry suppression keys, and native sampling are
respected. Policy can tighten while a stream is consumed; it cannot widen a
capture that began disabled. Observed parent policies remain effective after the
parent ends. Unknown local span carriers conservatively disable content.
Credential values are redacted from telemetry; native requests and responses
retain their original data.

Activation is shared across matching instrumentor instances. Repeated activation
is idempotent; a different active content/provider configuration raises
`ValueError`. Deactivation restores only owned patches and processors, retaining
foreign wrappers. An optional `tracer_provider` accepts an existing OTel provider.

This package keeps the repository's native botocore adapter. The upstream
Traceloop Bedrock instrumentor also wraps native bodies and event streams, so
changing delegates would not preserve the native identity contract addressed
here. Aiobotocore, guardrail/token-count/async administration operations, and
bidirectional transport are outside this adapter's scope. Boto3 1.43.108 includes
an `InvokeModelWithBidirectionalStream` service shape but does not expose the
operation as a client method; bidirectional clients use a separate SDK.
