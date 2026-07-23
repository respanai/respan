# Respan — Claude Code Plugin

Packages Respan's `/respan` skill and the hosted Respan MCP server into a single
installable Claude Code plugin. One install gives a user the Respan know-how
(tracing, gateway, prompts, evals, datasets) **and** live access to the Respan
platform via MCP tools — no separate `claude mcp add` step.

## Install

From inside Claude Code:

```
/plugin marketplace add respanai/respan
/plugin install respan@respan
```

You'll be prompted for a Respan API key at install time (create one at
https://platform.respan.ai); it's stored in your OS keychain, never in a settings
file. Once installed, the `/respan` skill auto-activates by description and the
hosted MCP tools (`mcp__respan__*`) are live in the session.

> The plugin skill is namespaced by Claude Code as `/respan:respan` if you invoke
> it explicitly — but you rarely need to type it, since it fires automatically
> when a task matches its description.

## What's inside

The marketplace manifest lives at the **repo root** and points at this plugin
directory, so `respanai/respan` resolves as a marketplace on its own:

```
<repo root>/
├── .claude-plugin/
│   └── marketplace.json    # marketplace entry → source: "./plugin"
└── plugin/
    ├── .claude-plugin/
    │   └── plugin.json     # plugin manifest: name, version, api_key user config
    ├── .mcp.json           # connects the hosted MCP server at mcp.respan.ai
    ├── scripts/
    │   └── build-plugin.mjs  # copies the shared skill into skills/ (see below)
    └── skills/
        └── respan/         # GENERATED — do not hand-edit
            ├── SKILL.md
            └── references/*.md
```

## Single source of truth

The skill lives in exactly one place: `respan/skills/` at the monorepo root.
`scripts/build-plugin.mjs` copies it into `plugin/skills/respan/` at build time
and prepends the YAML frontmatter a plugin skill needs. This mirrors the CLI's
`generate:skill-refs` step — one skill, assembled into each distribution (CLI
bundle and this plugin). **Never edit `plugin/skills/` by hand**; edit
`respan/skills/` and re-run the build. The generated files are committed so the
published plugin is self-contained (marketplaces copy the plugin directory).

```bash
node plugin/scripts/build-plugin.mjs
```

Wire this into the release process so the plugin skill can never drift from the
CLI's copy.

## Test locally

From the monorepo root, load the plugin directly without a marketplace:

```bash
claude --plugin-dir ./plugin
```

You'll be prompted for a Respan API key (create one at https://platform.respan.ai).
Then invoke the skill and confirm the MCP tools are live (`mcp__respan__*`).

To exercise the full marketplace path (add + install) exactly as end users do,
point Claude Code at the repo root:

```bash
claude plugin marketplace add ./
claude plugin install respan@respan
```

Validate the marketplace + plugin from the repo root before publishing:

```bash
claude plugin validate . --strict
```

## Authentication

The manifest declares a `userConfig.api_key` (marked `sensitive`), so Claude Code
prompts for it at install time and stores it in the OS keychain. `.mcp.json`
injects it as `Authorization: Bearer ${user_config.api_key}` against the hosted
server `https://mcp.respan.ai/mcp`. The server also supports OAuth browser
sign-in; API-key config is what this plugin ships with first.

## Distribution

The repo root ships a `.claude-plugin/marketplace.json` that references this
plugin via `source: "./plugin"`, so `respanai/respan` resolves as a marketplace
directly — that's the self-hosted install path shown under [Install](#install)
above, and it's the canonical way to get the plugin. It needs no Anthropic review
and updates the moment you push.

To also list it in Anthropic's community catalog for discoverability at
https://claude.com/plugins:

1. `node plugin/scripts/build-plugin.mjs` and commit the result.
2. `claude plugin validate . --strict` (from the repo root).
3. Submit at https://platform.claude.com/plugins/submit, pointing at
   `respanai/respan`. Anthropic runs automated validation + safety screening,
   pins your commit SHA in the catalog, and syncs nightly. **Don't** open a PR
   against `anthropics/claude-plugins-community` — it's read-only and PRs are
   auto-closed.
