---
name: respan
description: >-
  Respan for LLM observability. Use to set up tracing or the gateway, manage
  prompts (create/version/deploy), run evals, experiments, and datasets, or
  work with traces and logs. Trigger for any Respan platform task or when
  instrumenting an app to send telemetry to Respan.
---

# Respan

Respan is an AI gateway with built-in observability. This skill points you to the Respan docs page for each task. The docs are the source of truth: fetch the page and follow it, and don't rely on memory for Respan package names, APIs, or URLs. Every page below is markdown.

## Set up Respan in a project

To set up tracing or the gateway, fetch and follow `https://respan.ai/docs/agent-setup.md`. It covers the API key, finds the right integration page, and verifies the setup with the Respan CLI.

## Docs by task

| Task | Page |
|------|------|
| Set up tracing or the gateway | `https://respan.ai/docs/agent-setup.md` |
| Trace an app with the Respan SDK | `https://respan.ai/docs/documentation/features/tracing/quickstart.md` |
| Trace or route a specific framework or LLM SDK | Find its page in `https://respan.ai/docs/llms.txt`. Integration pages are labeled **(tracing)** or **(gateway)**. |
| Route LLM calls through the gateway | `https://respan.ai/docs/documentation/features/gateway/gateway-quickstart.md` |
| Gateway retries and fallbacks | `https://respan.ai/docs/documentation/features/gateway/retries.md` |
| Gateway caching | `https://respan.ai/docs/documentation/features/gateway/caching.md` |
| Gateway limits | `https://respan.ai/docs/documentation/features/gateway/limits.md` |
| Models the gateway serves | `https://respan.ai/docs/documentation/features/gateway/models-catalog.md` |
| Prompt management | `https://respan.ai/docs/documentation/features/prompt-management/prompt-management-quickstart.md` |
| Monitors and notifications | `https://respan.ai/docs/documentation/features/monitoring/monitors.md` |
| Inspect spans and traces | `https://respan.ai/docs/documentation/features/observe/view-debug-traces.md` |
| Custom IDs and attributes | `https://respan.ai/docs/documentation/features/observe/custom-ids.md` |
| API keys | `https://respan.ai/docs/documentation/admin/respan-api-keys.md` |
| Provider keys | `https://respan.ai/docs/documentation/admin/llm-provider-keys.md` |
| Respan CLI | `https://respan.ai/docs/documentation/cli.md` |
| Respan MCP server | `https://respan.ai/docs/documentation/mcp.md` |
| Anything else, including the API reference | `https://respan.ai/docs/llms.txt` |

## Work with Respan data

To read traces and logs or manage prompts, use the Respan MCP server's tools when they're available. Otherwise, use the Respan CLI, such as `respan traces list` or `respan logs list`. The CLI page lists every command.
