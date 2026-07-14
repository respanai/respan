# Respan — Cursor Plugin

Packages Respan's skill and the hosted Respan MCP server into a single
installable Cursor plugin. One install gives you the Respan know-how (tracing,
gateway, prompts, evals, datasets) **and** live access to the Respan platform
via MCP tools — no separate MCP setup step.

## Install

From the [Cursor marketplace](https://cursor.com/marketplace), search for
**Respan** and install.

On first use, Cursor opens a browser window to sign in to Respan. There is no
API key to paste — the hosted MCP server speaks OAuth, and Cursor registers
itself dynamically (see [Authentication](#authentication)).

Once installed, the `respan` skill activates by description whenever a task
matches it, and the Respan MCP tools are live in the session.

## What's inside

The marketplace manifest lives at the **repo root** and points at this plugin
directory, so `respanai/respan` resolves as a marketplace on its own:

```
<repo root>/
├── .cursor-plugin/
│   └── marketplace.json    # marketplace entry → source: "./cursor-plugin"
├── scripts/
│   └── build-plugins.mjs   # copies the shared skill into skills/ (see below)
└── cursor-plugin/
    ├── .cursor-plugin/
    │   └── plugin.json     # plugin manifest: name, version, logo
    ├── mcp.json            # connects the hosted MCP server at mcp.respan.ai
    ├── assets/
    │   └── logo.svg        # marketplace logo (128×128 tile)
    └── skills/
        └── respan/         # GENERATED — do not hand-edit
            ├── SKILL.md
            └── references/*.md
```

`plugin.json` deliberately declares **no** component paths. Cursor auto-discovers
`skills/` and `mcp.json` by convention, and naming an explicit path in the
manifest *disables* the default folder scan for that component — so leaving them
out is what keeps discovery working.

## Single source of truth

The skill lives in exactly one place: `respan/skills/` at the monorepo root.
`scripts/build-plugins.mjs` copies it into `cursor-plugin/skills/respan/` at
build time and prepends the YAML frontmatter a plugin skill needs. Cursor and
Claude Code agree on the skill format (`skills/<name>/SKILL.md` with `name` +
`description` frontmatter), so both plugins consume the same source unchanged.

**Never edit `cursor-plugin/skills/` by hand**; edit `respan/skills/` and re-run
the build. The generated files are committed, because Cursor clones the repo and
reads committed files — it never runs a build step on your behalf.

```bash
node scripts/build-plugins.mjs
```

## Authentication

Sign-in is OAuth. There is no API key to paste, and no secret in this repo.

`mcp.respan.ai` is a spec-compliant OAuth 2.1 resource server: an unauthenticated
request returns `401` with a `WWW-Authenticate` header pointing at
`/.well-known/oauth-protected-resource`, and the authorization-server metadata
advertises `/authorize`, `/token`, and `/register` with PKCE (`S256`) against
public clients (`token_endpoint_auth_methods_supported: ["none"]`).

`mcp.json` therefore declares an `auth` block with a pre-registered `CLIENT_ID`
and nothing else — no `CLIENT_SECRET` (Respan is a public client; the flow is
PKCE-protected) and no `scopes` (the server advertises no `scopes_supported`, so
there is nothing for Cursor to request).

> **Why the `auth` block, given the server supports Dynamic Client Registration?**
> Cursor's docs imply DCR is automatic and an `auth` block is only needed when the
> provider lacks it. In practice that is not true for **plugin-supplied** MCP
> servers: with a bare `{ "url": ... }` config, Cursor never initiates OAuth off
> the `401` + `WWW-Authenticate` challenge, exposes no login affordance in MCP
> settings, and every tool call 401s. Adding the `auth` block is what surfaces the
> login flow. Don't "simplify" this back down to a bare URL — it silently breaks
> authentication.

The `CLIENT_ID` is a public-client identifier, not a credential, so committing it
is safe. It was minted once against the server's `/register` endpoint with
Cursor's two redirect URIs baked in
(`https://www.cursor.com/agents/mcp/oauth/callback` for web,
`http://localhost:8787/callback` for desktop). The server validates redirect URIs
against the ones sealed into the ID and rejects anything else.

The Claude Code plugin still ships the API-key path (`userConfig.api_key`)
because Claude Code prompts for secrets at install time; Cursor has no
equivalent, and OAuth is the better flow anyway.

## Test locally

Cursor reads plugins straight from disk, so point it at this directory and
exercise the real install path before publishing:

1. Install the plugin from this local directory in Cursor.
2. Trigger the skill and confirm the Respan MCP tools appear.
3. Confirm the OAuth browser flow completes and the tools return live data.

Step 3 is the one that matters: it's the only way to prove `/authorize` accepts
the redirect URIs Cursor uses (`https://www.cursor.com/agents/mcp/oauth/callback`
for web, `http://localhost:8787/callback` for desktop).

## Distribution

Push to a public Git repository, then submit the repository link at
https://cursor.com/marketplace/publish.

Before submitting:

1. `node scripts/build-plugins.mjs` and commit the result.
2. Confirm the logo is committed and referenced by a relative path.
3. Confirm every manifest path is relative and valid (no `..`, no absolute
   paths).
4. Test locally (above).
