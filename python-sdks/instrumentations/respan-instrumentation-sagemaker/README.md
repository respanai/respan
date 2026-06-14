# respan-instrumentation-sagemaker

Respan instrumentation plugin for AWS SageMaker Runtime calls made through
`boto3`.

## Installation

```bash
pip install respan-ai respan-instrumentation-sagemaker boto3
```

## Usage

```python
import json

import boto3
from respan import Respan
from respan_instrumentation_sagemaker import SageMakerInstrumentor

respan = Respan(instrumentations=[SageMakerInstrumentor()])
client = boto3.client("sagemaker-runtime", region_name="us-east-1")

response = client.invoke_endpoint(
    EndpointName="my-llm-endpoint",
    Body=json.dumps({"inputs": "Reply with one short sentence."}).encode("utf-8"),
    ContentType="application/json",
    Accept="application/json",
)

body = json.loads(response["Body"].read())
print(body[0]["generated_text"])
respan.flush()
```

The instrumentor patches `botocore.client.BaseClient._make_api_call` and only
emits spans for SageMaker Runtime operations. It currently normalizes
`InvokeEndpoint`, `InvokeEndpointWithResponseStream`, and `InvokeEndpointAsync`
into canonical Respan LLM spans.
