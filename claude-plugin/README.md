# Respan — Claude Code Plugin

Packages Respan's `/respan` skill and the hosted Respan MCP server into a single
installable Claude Code plugin. One install gives a user the Respan know-how
(tracing, gateway, prompts, evals, datasets) **and** live access to the Respan
platform via MCP tools — no separate `claude mcp add` step.

## Install

From inside Claude Code:

```
/plugin marketplace add respanai/claude-plugins
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

The marketplace lives in its own repository,
[respanai/claude-plugins](https://github.com/respanai/claude-plugins), and points at
this directory with a `git-subdir` source, so installing the plugin downloads this
directory only:

```
<repo root>/
├── scripts/
│   └── build-plugins.mjs   # copies the shared skill into each plugin (see below)
└── claude-plugin/
    ├── .claude-plugin/
    │   └── plugin.json     # plugin manifest: name, version, api_key user config
    ├── .mcp.json           # connects the hosted MCP server at mcp.respan.ai
    └── skills/
        └── respan/         # GENERATED — do not hand-edit
            ├── SKILL.md
            └── references/*.md
```

> The sibling `cursor-plugin/` packages the same skill for Cursor. Both are
> generated from `respan/skills/` by the same build script.

## Single source of truth

The skill lives in exactly one place: `respan/skills/` at the monorepo root.
`scripts/build-plugins.mjs` copies it into each plugin's `skills/respan/` at
build time and prepends the YAML frontmatter a plugin skill needs. This mirrors
the CLI's `generate:skill-refs` step — one skill, assembled into each
distribution (CLI bundle, this plugin, and the Cursor plugin in
`../cursor-plugin/`, which uses the identical skill format). **Never edit
`claude-plugin/skills/` by hand**; edit `respan/skills/` and re-run the build.
The generated files are committed so the published plugin is self-contained
(marketplaces copy the plugin directory).

```bash
node scripts/build-plugins.mjs
```

Wire this into the release process so the plugin skill can never drift from the
CLI's copy.

## Test locally

From the monorepo root, load the plugin directly without a marketplace:

```bash
claude --plugin-dir ./claude-plugin
```

You'll be prompted for a Respan API key (create one at https://platform.respan.ai).
Then invoke the skill and confirm the MCP tools are live (`mcp__respan__*`).

The full marketplace path (add + install) installs from `main`, exactly as end
users do:

```bash
claude plugin marketplace add respanai/claude-plugins
claude plugin install respan@respan
```

Validate the plugin from the repo root before publishing:

```bash
claude plugin validate ./claude-plugin --strict
```

## Authentication

The manifest declares a `userConfig.api_key` (marked `sensitive`), so Claude Code
prompts for it at install time and stores it in the OS keychain. `.mcp.json`
injects it as `Authorization: Bearer ${user_config.api_key}` against the hosted
server `https://mcp.respan.ai/mcp`. The server also supports OAuth browser
sign-in; API-key config is what this plugin ships with first.

## Distribution

[respanai/claude-plugins](https://github.com/respanai/claude-plugins) is the
marketplace: it lists this plugin and `session-traces` from `../claude-tracing-plugin/`,
each pointing at its directory on `main` here. That's the self-hosted install path
shown under [Install](#install) above, and it's the canonical way to get the plugin.
It needs no Anthropic review. To release a change, merge it here with a higher
`version` in `.claude-plugin/plugin.json`; users get it with
`claude plugin marketplace update respan`.

To also list it in Anthropic's community catalog for discoverability at
https://claude.com/plugins:

1. `node scripts/build-plugins.mjs` and commit the result.
2. `claude plugin validate ./claude-plugin --strict` (from the repo root).
3. Submit at https://platform.claude.com/plugins/submit. The plugin's code is in
   `claude-plugin/` here, and the marketplace that lists it is
   `respanai/claude-plugins`. Anthropic runs automated validation + safety screening,
   pins your commit SHA in the catalog, and syncs nightly. **Don't** open a PR
   against `anthropics/claude-plugins-community` — it's read-only and PRs are
   auto-closed.
