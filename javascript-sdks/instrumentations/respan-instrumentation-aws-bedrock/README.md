# @respan/instrumentation-aws-bedrock

Respan instrumentation plugin for AWS Bedrock Runtime calls made through
`@aws-sdk/client-bedrock-runtime`.

```bash
npm install @respan/respan @respan/instrumentation-aws-bedrock @aws-sdk/client-bedrock-runtime
```

```typescript
import {
  BedrockRuntimeClient,
  ConverseCommand,
} from "@aws-sdk/client-bedrock-runtime";
import { Respan } from "@respan/respan";
import { AWSBedrockInstrumentor } from "@respan/instrumentation-aws-bedrock";

const respan = new Respan({
  instrumentations: [new AWSBedrockInstrumentor()],
});
await respan.initialize();

const client = new BedrockRuntimeClient({ region: "us-east-1" });
const response = await client.send(
  new ConverseCommand({
    modelId: "anthropic.claude-3-haiku-20240307-v1:0",
    messages: [
      {
        role: "user",
        content: [{ text: "Write one sentence about observability." }],
      },
    ],
  }),
);
```

The plugin patches `BedrockRuntimeClient.prototype.send` and emits canonical
Respan chat spans for:

- `InvokeModel`
- `InvokeModelWithResponseStream`
- `Converse`
- `ConverseStream`

Streaming spans are emitted when the returned async stream is consumed.
