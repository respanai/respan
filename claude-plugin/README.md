# Respan — Claude Code Plugin

Packages Respan's `/respan` skill and the hosted Respan MCP server into a single
installable Claude Code plugin. One install gives a user the Respan know-how
(tracing, gateway, prompts, evals, datasets) **and** live access to the Respan
platform via MCP tools — no separate `claude mcp add` step, and no API key to paste.

## Install

From inside Claude Code:

```
/plugin marketplace add respanai/claude-plugins
/plugin install respan@respan
```

The `/respan` skill auto-activates by description as soon as the plugin is
installed. To use the hosted MCP tools (`mcp__plugin_respan_respan__*`), sign in
once: run `/mcp`, select `plugin:respan:respan`, and finish the sign-in in your
browser. If you can't use a browser, as in CI, see
[Using an API key](#using-an-api-key).

### Respan Enterprise

The plugin connects to Respan's platform server, `https://mcp.respan.ai/mcp`, by
default. If your team signs in to Respan Enterprise, switch it before you sign in:

1. Run `/plugin configure respan@respan` and set **Respan server** to
   `https://mcp.respan.ai/mcp/enterprise`.
2. Restart Claude Code, then run `/mcp`, select `plugin:respan:respan`, and sign in
   with your Enterprise account.

The option also appears in `/config`. Each server has its own sign-in, so if you
change it later, sign in again with `/mcp`. The option needs Claude Code v2.1.271
or later.

> The plugin skill is namespaced by Claude Code as `/respan:respan` if you invoke
> it explicitly — but you rarely need to type it, since it fires automatically
> when a task matches its description.

### Updating from 0.1.0

Version 0.1.0 came from the old `respanai/respan` marketplace and asked for an API
key. That marketplace is gone, so `claude plugin update` can't reach the plugin.
Remove the old marketplace, add the new one, and reinstall:

```bash
claude plugin marketplace remove respan
claude plugin marketplace add respanai/claude-plugins
claude plugin install respan@respan
```

Removing the old marketplace uninstalls the plugin and deletes the API key it
saved, where it can. Then restart Claude Code and sign in with `/mcp`.

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
    │   └── plugin.json     # plugin manifest: name, version, server option
    ├── .mcp.json           # connects the hosted MCP server the option picks
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

Run `/mcp`, select `plugin:respan:respan`, and sign in. Then invoke the skill and
confirm the MCP tools are live (`mcp__plugin_respan_respan__*`).

The full marketplace path (add + install) installs from `main`, exactly as end
users do:

```bash
claude plugin marketplace add respanai/claude-plugins
claude plugin install respan@respan
claude mcp list
```

`claude mcp list` should show `plugin:respan:respan` as `! Needs authentication`
until you sign in. These commands don't need a sign-in, so you can keep them out
of your own config by running `export CLAUDE_CONFIG_DIR="$(mktemp -d)"` first.

Validate the plugin from the repo root before publishing:

```bash
claude plugin validate ./claude-plugin --strict
```

## Authentication

Sign-in is OAuth. There is no API key to configure, and no secret in this repo.

`.mcp.json` declares only the server's URL, `${user_config.server}`, which Claude
Code fills in from the **Respan server** option: `https://mcp.respan.ai/mcp` by
default, or `https://mcp.respan.ai/mcp/enterprise`. An unauthenticated request gets
a `401` with a `WWW-Authenticate` header pointing at that server's OAuth metadata,
so Claude Code reports `! Needs authentication` until you sign in with `/mcp`.
Claude Code registers itself with the server and stores the token. The two servers
use separate OAuth authorization servers, so an Enterprise account signs in only
through the Enterprise one.

### Using an API key

Use a key where nobody can finish a browser sign-in: CI, containers, or `claude -p`
on a machine that hasn't signed in. Create one at https://platform.respan.ai, set
it as `RESPAN_API_KEY`, and add the server yourself:

```bash
claude mcp add --transport http --scope user respan https://mcp.respan.ai/mcp \
  --header 'Authorization: Bearer ${RESPAN_API_KEY}'
```

Keep the single quotes. Claude Code then reads the key from the environment when
it starts, and saves only the `${RESPAN_API_KEY}` reference in its config. For
Respan Enterprise, use `https://mcp.respan.ai/mcp/enterprise` and an Enterprise key.

- If the plugin is installed too and its **Respan server** option matches this
  URL, `claude mcp list` shows only this `respan` server. Its tools are named
  `mcp__respan__*`.
- `✔ Connected` doesn't prove the key works. The server accepts any key when
  connecting and rejects a bad one only when a tool runs, so run a tool to check.

> **Why doesn't the plugin offer a key option as well?** A plugin's server can't
> make its header optional. If a required `${user_config.*}` option is unset, the
> server doesn't load at all. If an optional one is unset, Claude Code still sends
> the `Authorization` header, which turns off the OAuth fallback.

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
