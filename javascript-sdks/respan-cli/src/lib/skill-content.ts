// Skill content assembly.
//
// The skill is a single SKILL.md that links to the Respan docs for each task.
// It's authored as markdown under skills/ and embedded into
// skill-refs.generated.ts by scripts/embed-skill-refs.mjs. This module only wraps the routing index in
// the skill frontmatter that coding agents expect.

import { SKILL_MD } from './skill-refs.generated.js';

export function getSkillMd(): string {
  return `---
name: respan
description: Use Respan for tracing, evals, prompts, gateway, and SDK setup. Covers CLI commands, SDK instrumentation, and platform features.
user-invocable: true
---

${SKILL_MD}`;
}
