# Contributor Docs

This directory is the source of truth for repository-level contributor documentation.

Start here:

- [architecture.md](architecture.md)
  - system layers, package responsibilities, and dependency direction
- [writing-instrumentations.md](writing-instrumentations.md)
  - how to add a new active instrumentation package
- [span-contract.md](span-contract.md)
  - canonical span attribute shape every instrumentation must emit (Python or JS, first-party or OI-delegated)
- [cicd.md](cicd.md)
  - what CI and publish do, and what the pipeline expects
- [publish.md](publish.md)
  - contributor-facing release workflow and release intents

Use these docs as the maintained path.
Do not add duplicate contributor docs under individual package trees.
