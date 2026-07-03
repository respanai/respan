/**
 * Datasets seeder for `respan demo`.
 *
 * Creates the demo datasets (see dataset-fixtures.ts) and bulk-loads their rows,
 * then tears them down for `respan demo clear`. Deduplicates by dataset name so
 * re-runs never create duplicates. All dataset-specific logic lives here; the
 * orchestrator stays service-agnostic.
 *
 * Rows are written via `bulkCreateDatasetLogs` (POST .../logs/bulk/), which is
 * synchronous and returns success/error counts. Each row's `input` object becomes
 * the row's mapped variables; `expected_output` is JSON-stringified so the grader
 * sees valid JSON (a raw object is coerced to single-quoted python-repr).
 */

import { DEMO_DATASETS, DemoDatasetFixture } from './dataset-fixtures.js';
import {
  Seeder,
  SeederContext,
  SeederResult,
  SeedItemResult,
  ExistingResource,
} from './types.js';

/** Datasets in the account whose name exactly matches a demo fixture name.
 *
 * Listed in full and matched client-side (case-sensitive, exact) rather than via
 * server filters, which keeps dedup independent of filter-operator semantics.
 * Returns every match so duplicates are all found (for clear). */
async function findExistingDatasets(ctx: SeederContext): Promise<ExistingResource[]> {
  const wanted = new Set(DEMO_DATASETS.map((d) => d.name));
  const response = await ctx.client.datasets.listDatasets({
    Authorization: ctx.authHeader,
    page_size: 100,
  });
  const found: ExistingResource[] = [];
  for (const row of (response as { results?: Array<Record<string, unknown>> }).results ?? []) {
    const name = typeof row.name === 'string' ? row.name : '';
    const id = String(row.id ?? row.dataset_id ?? '');
    if (id && wanted.has(name)) found.push({ name, id });
  }
  return found;
}

/** Create one dataset and bulk-load its rows. Returns the new dataset id. */
async function createFromFixture(
  ctx: SeederContext,
  fixture: DemoDatasetFixture,
): Promise<string> {
  const created = (await ctx.client.datasets.createDataset({
    Authorization: ctx.authHeader,
    name: fixture.name,
    is_empty: true,
    ...(fixture.description ? { description: fixture.description } : {}),
  })) as Record<string, unknown>;

  const datasetId = String(created.id ?? created.dataset_id ?? '');
  if (!datasetId) {
    throw new Error(`createDataset did not return an id for "${fixture.name}".`);
  }

  const logs = fixture.rows.map((row) => ({
    input: row.input,
    ...(row.expected_output !== undefined
      ? {
          expected_output:
            typeof row.expected_output === 'string'
              ? row.expected_output
              : JSON.stringify(row.expected_output),
        }
      : {}),
  }));

  const result = (await ctx.client.datasets.bulkCreateDatasetLogs({
    Authorization: ctx.authHeader,
    dataset_id: datasetId,
    logs,
  })) as unknown as Record<string, unknown>;

  const errorCount = typeof result.error_count === 'number' ? result.error_count : 0;
  if (errorCount > 0) {
    const detail = Array.isArray(result.errors) ? `: ${JSON.stringify(result.errors).slice(0, 200)}` : '';
    throw new Error(`bulkCreateDatasetLogs reported ${errorCount} error(s) for "${fixture.name}"${detail}`);
  }

  return datasetId;
}

export const datasetsSeeder: Seeder = {
  service: 'datasets',
  label: 'Datasets',

  findExisting(ctx: SeederContext): Promise<ExistingResource[]> {
    return findExistingDatasets(ctx);
  },

  async run(ctx: SeederContext): Promise<SeederResult> {
    const existingNames = new Set((await findExistingDatasets(ctx)).map((d) => d.name));
    const items: SeedItemResult[] = [];

    for (const fixture of DEMO_DATASETS) {
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
    const existing = await findExistingDatasets(ctx);
    const items: SeedItemResult[] = [];

    for (const fixture of DEMO_DATASETS) {
      const matches = existing.filter((d) => d.name === fixture.name);
      if (matches.length === 0) {
        items.push({ name: fixture.name, status: 'skipped', reason: 'not found' });
        continue;
      }
      for (const match of matches) {
        await ctx.client.datasets.deleteDataset({
          Authorization: ctx.authHeader,
          dataset_id: match.id,
        });
        items.push({ name: match.name, status: 'deleted', id: match.id });
      }
    }

    return { service: this.service, label: this.label, items };
  },
};
