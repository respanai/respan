# Respan Exporter for Anthropic Agent SDK

**[respan.ai](https://respan.ai)** | **[Documentation](https://docs.respan.ai)**

Exporter for Anthropic Agent SDK telemetry to Respan.

## Installation

```bash
npm install @respan/exporter-anthropic-agents
```

## Quickstart

```typescript
import { RespanAnthropicAgentsExporter } from "@respan/exporter-anthropic-agents";

const exporter = new RespanAnthropicAgentsExporter();

for await (const message of exporter.query({
  prompt: "Review this repository and summarize architecture.",
  options: {
    allowedTools: ["Read", "Glob", "Grep"],
    permissionMode: "acceptEdits",
  },
})) {
  console.log(message);
}
```

## Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `RESPAN_API_KEY` | Yes | Respan API key used for telemetry export. |
| `RESPAN_BASE_URL` | No | Respan base URL for telemetry export. Defaults to `https://api.respan.ai`. |
| `ANTHROPIC_BASE_URL` | No | Inference/proxy base URL used by the Anthropic SDK. |
| `ANTHROPIC_API_KEY` | Usually | Key used by the Anthropic SDK for inference calls. |
| `ANTHROPIC_AUTH_TOKEN` | Optional | Alternate auth token used by some Anthropic client flows. |

Set both groups together when needed. `RESPAN_*` controls tracing export, while `ANTHROPIC_*` controls where model requests are sent.

```bash
# Tracing export (Respan telemetry)
RESPAN_API_KEY=your_respan_key
RESPAN_BASE_URL=https://api.respan.ai/api

# Inference/proxy routing (Anthropic SDK)
ANTHROPIC_BASE_URL=http://localhost:8000/api
ANTHROPIC_API_KEY=your_inference_key
ANTHROPIC_AUTH_TOKEN=your_inference_key
```

`RESPAN_BASE_URL` controls telemetry export only. To use it with the TypeScript exporter, pass it as the `endpoint` constructor param:

```typescript
const baseUrl = process.env.RESPAN_BASE_URL || "https://api.respan.ai";

const exporter = new RespanAnthropicAgentsExporter({
  endpoint: `${baseUrl}/api/v1/traces/ingest`,
});
```

### Tracing vs Inference URLs (Important)

There are two independent URL planes:

- **Tracing export URL** (Respan telemetry ingest): controlled by exporter `endpoint` (typically derived from `RESPAN_BASE_URL`).
- **Inference/proxy URL** (where Claude requests are sent): controlled by Anthropic SDK env/options such as `ANTHROPIC_BASE_URL`.

Using **Respan tracing + Respan gateway** together:

```typescript
import { RespanAnthropicAgentsExporter } from "@respan/exporter-anthropic-agents";

const apiKey = process.env.RESPAN_API_KEY!;
const respanBaseUrl = (process.env.RESPAN_BASE_URL ?? "https://api.respan.ai/api").replace(/\/+$/, "");

const exporter = new RespanAnthropicAgentsExporter({
  apiKey,
  endpoint: `${respanBaseUrl}/v1/traces/ingest`, // tracing export
});

const options = exporter.withOptions({
  env: {
    ANTHROPIC_BASE_URL: `${respanBaseUrl}/anthropic`, // inference proxy
    ANTHROPIC_API_KEY: apiKey,
    ANTHROPIC_AUTH_TOKEN: apiKey,
  },
});
```

### Constructor Parameters

Recommended pattern (matches the runnable examples):

```typescript
const exporter = new RespanAnthropicAgentsExporter({
  apiKey: "your_respan_key", // Optional; falls back to RESPAN_API_KEY
  endpoint: `${baseUrl}/api/v1/traces/ingest`, // Build from RESPAN_BASE_URL
});
```

Advanced overrides:

```typescript
const exporter = new RespanAnthropicAgentsExporter({
  endpoint: "https://custom-host/api/v1/traces/ingest", // Full ingest endpoint URL
  timeoutMs: 15000,
  maxRetries: 3,
  baseDelaySeconds: 1,
  maxDelaySeconds: 30,
});
```

## Examples

Runnable examples with full setup instructions:

- **TypeScript examples root:** [typescript/tracing/anthropic-agents-sdk](https://github.com/respanai/respan-example-projects/tree/main/typescript/tracing/anthropic-agents-sdk)
- **TypeScript basic scripts:**
  - [hello_world_test.ts](https://github.com/respanai/respan-example-projects/blob/main/typescript/tracing/anthropic-agents-sdk/hello_world_test.ts)
  - [wrapped_query_test.ts](https://github.com/respanai/respan-example-projects/blob/main/typescript/tracing/anthropic-agents-sdk/wrapped_query_test.ts)
  - [tool_use_test.ts](https://github.com/respanai/respan-example-projects/blob/main/typescript/tracing/anthropic-agents-sdk/tool_use_test.ts)
  - [gateway_test.ts](https://github.com/respanai/respan-example-projects/blob/main/typescript/tracing/anthropic-agents-sdk/gateway_test.ts)
- **Python examples root:** [python/tracing/anthropic-agents-sdk](https://github.com/respanai/respan-example-projects/tree/main/python/tracing/anthropic-agents-sdk)

## Dev Guide

### Running Tests

```bash
npm test
```
