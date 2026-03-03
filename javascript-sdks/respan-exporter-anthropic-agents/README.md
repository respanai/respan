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
| `RESPAN_API_KEY` | Yes | Your Respan API key. Falls back to `RESPAN_API_KEY`. |
| `RESPAN_BASE_URL` | No | Base URL for all Respan services. Defaults to `https://api.respan.ai`. Pass as `endpoint` constructor param with `/api/v1/traces/ingest` appended. |

`RESPAN_BASE_URL` is the single base URL for all Respan services. To use it with the TypeScript exporter, pass it as the `endpoint` constructor param:

```typescript
const baseUrl = process.env.RESPAN_BASE_URL || "https://api.respan.ai";

const exporter = new RespanAnthropicAgentsExporter({
  endpoint: `${baseUrl}/api/v1/traces/ingest`,
});
```

### Constructor Parameters

```typescript
const exporter = new RespanAnthropicAgentsExporter({
  apiKey: "your_respan_key",                         // Overrides RESPAN_API_KEY
  endpoint: "https://api.respan.ai/api/v1/traces/ingest", // Full ingest endpoint URL
  timeoutMs: 15000,
  maxRetries: 3,
  baseDelaySeconds: 1,
  maxDelaySeconds: 30,
});
```

## Examples

Runnable examples with full setup instructions:

- **Python:** [anthropic_agents_python_example](https://github.com/Keywords-AI/keywordsai-example-projects/tree/main/anthropic_agents_python_example)
- **TypeScript:** [anthropic_agents_typescript_example](https://github.com/Keywords-AI/keywordsai-example-projects/tree/main/anthropic_agents_typescript_example)

## Dev Guide

### Running Tests

```bash
npm test
```
