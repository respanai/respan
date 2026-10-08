// Skill content assembly.
//
// The skill is a single SKILL.md that links to the Respan docs for each task.
// It's authored under skills/, frontmatter included, and embedded into
// skill-refs.generated.ts by scripts/embed-skill-refs.mjs.

import { SKILL_MD } from './skill-refs.generated.js';

export function getSkillMd(): string {
  return SKILL_MD;
}
