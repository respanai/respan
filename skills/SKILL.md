---
name: respan
description: >-
  Respan AI gateway and LLM observability. Use for any Respan task: setting up
  tracing or the gateway in an app, querying traces and logs, managing prompts,
  or setting up monitors. Points to the Respan docs, the source of truth for
  packages, APIs, and setup steps.
user-invocable: true
---

# Respan

Respan is an AI gateway with built-in observability. Use this skill for any Respan task, such as setting up tracing or the gateway, working with traces and logs, managing prompts, or setting up monitors.

## Use the docs

The Respan docs are the source of truth. Read the relevant page before you write Respan code or answer a question about Respan, and use only the packages, APIs, and URLs it shows. Respan was formerly Keywords AI, so don't use `keywordsai` packages or APIs from memory.

- Every docs page is available as markdown. Add `.md` to its URL.
- The index at `https://www.respan.ai/docs/llms.txt` lists every page with a one-line summary, including the API reference. Integration pages are labeled **(tracing)** or **(gateway)**.
- If a page below doesn't load, look it up in the index. Don't guess URLs.

## Set up Respan in a project

Fetch and follow `https://www.respan.ai/docs/documentation/agent-setup.md`. It sets up tracing or the gateway end to end: the API key, the integration for the app's framework, and a check with the Respan CLI that data arrived.

## Common pages

| Task | Page |
|------|------|
| Trace an app with the Respan SDK | `https://www.respan.ai/docs/documentation/features/tracing/quickstart.md` |
| Route LLM calls through the gateway | `https://www.respan.ai/docs/documentation/features/gateway/gateway-quickstart.md` |
| Trace or route a specific framework or LLM SDK | Its **(tracing)** or **(gateway)** page in the index |
| Gateway retries and fallbacks | `https://www.respan.ai/docs/documentation/features/gateway/retries.md` |
| Gateway caching | `https://www.respan.ai/docs/documentation/features/gateway/caching.md` |
| Gateway limits | `https://www.respan.ai/docs/documentation/features/gateway/limits.md` |
| Models the gateway serves | `https://www.respan.ai/docs/documentation/features/gateway/models-catalog.md` |
| Prompt management | `https://www.respan.ai/docs/documentation/features/prompt-management/prompt-management-quickstart.md` |
| Monitors and notifications | `https://www.respan.ai/docs/documentation/features/monitoring/monitors.md` |
| Inspect spans and traces | `https://www.respan.ai/docs/documentation/features/observe/view-debug-traces.md` |
| Custom IDs and attributes | `https://www.respan.ai/docs/documentation/features/observe/custom-ids.md` |
| API keys | `https://www.respan.ai/docs/documentation/admin/respan-api-keys.md` |
| Provider keys | `https://www.respan.ai/docs/documentation/admin/llm-provider-keys.md` |
| Respan CLI | `https://www.respan.ai/docs/documentation/cli.md` |
| Respan MCP server | `https://www.respan.ai/docs/documentation/mcp.md` |

## Work with Respan data

- **Respan MCP server:** if its tools are available, use them to query traces and logs and to manage prompts, datasets, and other platform resources.
- **Respan CLI:** otherwise, use the CLI. Run `respan --help` to see its commands, and add `--help` to any command for its flags. It reads `RESPAN_API_KEY` from the environment or the project's `.env`.
