/**
 * Prompts seeder for `respan demo`.
 *
 * Recreates the three demo prompts (name + single deployed live version) in the
 * authenticated account, and tears them down again for `respan demo clear`.
 * Deduplicates by prompt name so re-runs never create duplicates. All
 * prompt-specific logic lives here; the orchestrator stays service-agnostic.
 */

import { DEMO_PROMPTS, DemoPromptFixture } from './prompt-fixtures.js';
import {
  Seeder,
  SeederContext,
  SeederResult,
  SeedItemResult,
  ExistingResource,
} from './types.js';

/** Prompts in the account whose name exactly matches a demo fixture name.
 *
 * Queried with the `name` `iexact` filter, one query per fixture name. NOTE: the
 * `icontains`/`iexact` filters do NOT OR-match across a multi-value array — a
 * single `value: [a, b, c]` matches nothing — so each name must be queried on
 * its own. Server matching is case-insensitive, so we re-confirm with an exact,
 * case-sensitive match client-side. Returns every match (so duplicates are all
 * found, for clear). */
async function findExistingPrompts(ctx: SeederContext): Promise<ExistingResource[]> {
  const found: ExistingResource[] = [];
  for (const fixture of DEMO_PROMPTS) {
    const response = await ctx.client.prompts.listPrompts({
      Authorization: ctx.authHeader,
      page_size: 100,
      filters: { name: { operator: 'iexact', value: [fixture.name] } },
    });
    for (const row of response.results ?? []) {
      if (row.id && row.name === fixture.name) {
        found.push({ name: row.name, id: row.id });
      }
    }
  }
  return found;
}

/** Create one prompt, then commit and deploy its version so it is the live
 * version immediately.
 *
 * The management API treats a freshly created version as an uncommitted draft:
 * `deploy: true` on `createPromptVersion` is a no-op for a draft, and deploying
 * one directly fails ("Cannot deploy a draft version. Commit it first."). The
 * working sequence is create → commit → deploy. (Committing opens a fresh draft
 * on top of the committed version — the same editing model the platform uses
 * for hand-edited prompts.) */
async function createFromFixture(
  ctx: SeederContext,
  fixture: DemoPromptFixture,
): Promise<string> {
  const prompt = await ctx.client.prompts.createPrompt({
    Authorization: ctx.authHeader,
    name: fixture.name,
    ...(fixture.description ? { description: fixture.description } : {}),
  });

  const promptId = prompt.id;
  if (!promptId) {
    throw new Error(`createPrompt did not return an id for "${fixture.name}".`);
  }

  const version = await ctx.client.prompts.createPromptVersion({
    Authorization: ctx.authHeader,
    prompt_id: promptId,
    ...fixture.version,
  });

  const versionNumber = version.version;
  if (versionNumber == null) {
    throw new Error(`createPromptVersion did not return a version number for "${fixture.name}".`);
  }

  await ctx.client.prompts.commitPromptVersion({
    Authorization: ctx.authHeader,
    prompt_id: promptId,
  });

  await ctx.client.prompts.deployPromptVersion({
    Authorization: ctx.authHeader,
    prompt_id: promptId,
    version: versionNumber,
  });

  return promptId;
}

export const promptsSeeder: Seeder = {
  service: 'prompts',
  label: 'Prompts',

  findExisting(ctx: SeederContext): Promise<ExistingResource[]> {
    return findExistingPrompts(ctx);
  },

  async run(ctx: SeederContext): Promise<SeederResult> {
    const existingNames = new Set((await findExistingPrompts(ctx)).map((p) => p.name));
    const items: SeedItemResult[] = [];

    for (const fixture of DEMO_PROMPTS) {
      if (existingNames.has(fixture.name)) {
        items.push({ name: fixture.name, status: 'skipped', reason: 'already exists' });
        continue;
      }
      const id = await createFromFixture(ctx, fixture);
      items.push({ name: fixture.name, status: 'created', id });
    }

    return { service: this.service, label: this.label, items };
  },

  async clear(ctx: SeederContext): Promise<SeederResult> {
    const existing = await findExistingPrompts(ctx);
    const items: SeedItemResult[] = [];

    for (const fixture of DEMO_PROMPTS) {
      const matches = existing.filter((p) => p.name === fixture.name);
      if (matches.length === 0) {
        items.push({ name: fixture.name, status: 'skipped', reason: 'not found' });
        continue;
      }
      for (const match of matches) {
        await ctx.client.prompts.deletePrompt({
          Authorization: ctx.authHeader,
          prompt_id: match.id,
        });
        items.push({ name: match.name, status: 'deleted', id: match.id });
      }
    }

    return { service: this.service, label: this.label, items };
  },
};
