# respan-instrumentation-aws-bedrock

Respan instrumentation plugin for AWS Bedrock Runtime calls made through
`boto3`.

## Installation

```bash
pip install respan-ai respan-instrumentation-aws-bedrock boto3
```

## Usage

```python
import json

import boto3
from respan import Respan
from respan_instrumentation_aws_bedrock import AWSBedrockInstrumentor

respan = Respan(instrumentations=[AWSBedrockInstrumentor()])
client = boto3.client("bedrock-runtime", region_name="us-east-1")

response = client.invoke_model(
    modelId="anthropic.claude-3-5-haiku-20241022-v1:0",
    body=json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 256,
        "messages": [{"role": "user", "content": "Say hello."}],
    }),
    contentType="application/json",
    accept="application/json",
)

body = json.loads(response["body"].read())
print(body["content"][0]["text"])
respan.flush()
```

The instrumentor patches `botocore.client.BaseClient._make_api_call` and only
emits spans for Bedrock Runtime operations. It currently normalizes
`InvokeModel`, `InvokeModelWithResponseStream`, `Converse`, and
`ConverseStream` into canonical Respan chat spans.
