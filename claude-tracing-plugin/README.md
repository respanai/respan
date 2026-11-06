# Respan Session Traces

Sends your Claude Code sessions to Respan: one trace per turn, with a span for each
request to the model and each tool call, and subagents nested under the tool call that
started them. Each request reports its own tokens, including cache reads and writes, and
its time to first token.

**Installing this plugin turns tracing on.** It sends your prompts, responses, and tool
inputs and outputs to Respan. To browse traces or use the Respan skill from Claude Code,
install the separate `respan` plugin. It doesn't send your sessions anywhere.

## Install

```bash
claude plugin marketplace add respanai/respan
claude plugin install session-traces@respan
```

Then run `/plugin configure session-traces@respan` in Claude Code to set your API
key, which you can create at https://platform.respan.ai. Traces start with your next
session. Writing traces needs an API key: an OAuth sign-in to the Respan MCP server
doesn't cover it.

## Options

| Option | Default | What it does |
| :- | :- | :- |
| Respan API Key | Required | Stored in your OS keychain |
| Respan deployment | `platform` | `platform` or `enterprise`: the Respan your API key belongs to |

To stop tracing, uninstall or disable the plugin. To stop it for one project or run, set
`RESPAN_TRACING=false`. The environment can only turn tracing off. The API key and the
destination come from your own settings, so a repository can't send your sessions
anywhere.

If you have a `.claude/respan.json` from `respan integrate claude-code`, its `customer_id`,
`span_name`, and `workflow_name` still apply, and other keys become metadata. Its `base_url`
is ignored. While this plugin is active, the old hook stops sending, so turns aren't
reported twice.

## Performance

Tracing records in the background and never waits for Respan during a turn. Measured on
Claude Code 2.1.318, it adds about 0.2 ms to each streamed chunk and no measurable time to a
turn. If Respan stops answering, interactive sessions are unaffected and exit normally.

A `claude -p` run waits up to a second at exit to send its last batch. If Respan hangs, the
process can stay alive for up to 30 seconds after its result, because a request in flight
can't be cancelled.

## Requirements and limits

Tracing needs Claude Code v2.1.287 or later, or the desktop app with v2.1.286 or later. It
doesn't run:

- In WSL sessions in the desktop app
- In cloud sessions, unless the plugin reaches them
- When your organization sets `allowManagedModsOnly` or `allowManagedHooksOnly`, or turns off
  `WebFetch` by policy

On a Team or Enterprise login, the system prompt and the context Claude Code adds to requests
aren't included.

## Develop

```bash
node --test claude-tracing-plugin/tests/*.test.mjs
claude plugin validate ./claude-tracing-plugin --strict
```

`claude --plugin-dir ./claude-tracing-plugin` doesn't prompt for the API key, and without one
the mod doesn't load. To test the real flow, run `claude plugin marketplace add ./` from the
repo root, then follow [Install](#install).
