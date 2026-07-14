---
name: respan
description: >-
  Respan for LLM observability. Use to set up tracing or the gateway, manage
  prompts (create/version/deploy), run evals, experiments, and datasets, or
  work with traces and logs. Trigger for any Respan platform task or when
  instrumenting an app to send telemetry to Respan.
---

# Respan

Use the Respan CLI and SDK for LLM observability — tracing, evals, prompts, datasets, and gateway routing.

## When To Use

- **Set up tracing** (instrument your app to capture LLM calls — install + decorators, propagation, processors) → read [references/tracing.md](references/tracing.md)
- **Set up gateway** (route LLM calls through the Respan proxy — wiring + model switching, fallbacks, caching) → read [references/gateway.md](references/gateway.md)
- **Prompt management** (create, version, deploy) → read [references/prompts.md](references/prompts.md)
- **Evals** (datasets, evaluators, experiments) → read [references/evals.md](references/evals.md)
- **Monitors & automation** (alerts, online evals, webhooks) → read [references/monitors.md](references/monitors.md)

Tracing and gateway are separate setups — configure one at a time, never both in the same pass.

## Core Principles

1. **Read the reference first.** Each reference file has the exact setup steps, API patterns, MCP tools, and CLI commands.
2. **Use MCP tools** for platform operations (prompts, datasets, evaluators, experiments, traces, logs).
3. **Use CLI** when MCP is not available: `respan traces list`, `respan prompts list`, etc.
4. **Fetch docs** for integration-specific details not covered in references.

## Quick Reference

| Task | Reference / Command |
|------|--------------------|
| Set up tracing (steps + decorators, propagation) | [references/tracing.md](references/tracing.md) |
| Set up gateway (steps + features) | [references/gateway.md](references/gateway.md) |
| Prompt management | [references/prompts.md](references/prompts.md) |
| Evals & experiments | [references/evals.md](references/evals.md) |
| Monitors & automation | [references/monitors.md](references/monitors.md) |
| List traces | `respan traces list --limit 10` |
| View a trace | `respan traces get <id>` |
| Check auth | `respan auth status` |

## Documentation Access

Any doc page can be fetched as markdown:
`https://respan.ai/docs/integrations/openai-sdk.md`
`https://respan.ai/docs/sdks/typescript-sdk/overview.md`

Full docs index: `https://www.respan.ai/docs/llms.txt`

Platform: `https://platform.respan.ai`
