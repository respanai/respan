# CI/CD

This repository treats package publishing as an explicit, checked-in inventory instead of ad hoc per-package logic.
Only the selected active SDK packages are included in automation. Packages under `*/legacy/` stay in the repo but are intentionally outside the main CI/CD path.

## What The Pipeline Does

- `ci.yml` runs on pull requests and pushes to `main`
- it validates the package inventory in `.github/release-packages.json`
- it validates `.release-intents/*.json`
- it fails if a release-managed package changed without a matching release intent
- it runs only affected release-managed packages, plus internal dependents that are impacted by those changes
- it builds every affected publishable Python package
- it installs the JavaScript workspace per job and then builds or tests each affected publishable JavaScript package through Yarn workspaces
- CI retries the Yarn install up to three total attempts only for its intermittent `onCancel` promise-settlement crash. Other install errors fail immediately; builds and tests are not retried, and Yarn's hardened checks remain enabled.
- it runs lightweight packaging smoke checks:
  - JavaScript: `npm pack --dry-run`
  - Python: install the built wheel into a clean venv and import the package

## What Publishes On `main`

- `publish.yml` runs after the `CI` workflow completes successfully for `main`, or by manual dispatch
- a package is published only when the merged change introduced a release intent for it
- JavaScript packages publish to npm from the Yarn workspace
- Python packages build first, then publish from built artifacts with PyPI Trusted Publishing
- after publish, each package is smoke-tested from the registry with retries to handle index propagation delay
- after successful publish, the workflow syncs the exact released versions back into the package manifests on `main`

This means merges to `main` can publish automatically without forcing every PR to hand-edit final manifest versions.

## Included Packages

Current release automation covers:

- Python: `respan-ai`, `respan-sdk`, `respan-tracing`, and all packages under `python-sdks/instrumentations/`
- JavaScript: `@respan/respan`, `@respan/cli`, `@respan/respan-sdk`, `@respan/tracing`, and all packages under `javascript-sdks/instrumentations/`

Legacy packages under `python-sdks/legacy/` and `javascript-sdks/legacy/` are intentionally excluded from release automation.

## Release Metadata Contract

Every publishable package must have:

- one entry in `.github/release-packages.json`
- a manifest name that matches that inventory entry
- a single authoritative manifest version
- public publish metadata for npm packages

Every PR that changes a release-managed package must also add one `.release-intents/*.json` file covering that package with one of:

- `none`
- `new`
- `patch`
- `minor`
- `major`

For JavaScript packages, the root workspace in `javascript-sdks/package.json` must include each release-managed package path. Legacy packages must stay out of that workspace list.

## Required Registry Setup

Before the publish workflow can succeed, configure Trusted Publishing for each registry package:

- npm: add this repository and the `publish.yml` workflow as a Trusted Publisher for each npm package
- PyPI: add this repository and the `publish.yml` workflow as a Trusted Publisher for each PyPI project

Recommended GitHub environments:

- `npm`
- `pypi`

Use those environments for approval gates if you want a manual checkpoint before packages are released.

## Local Validation

Run:

```bash
python3 scripts/release_inventory.py --validate
python3 scripts/release_intents.py validate
```

List publishable packages:

```bash
python3 scripts/release_inventory.py --ecosystem all
```

List packages that would publish between two refs:

```bash
python3 scripts/release_intents.py plan --ecosystem all --changed-from <base> --changed-to <head>
```
