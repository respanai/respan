/**
 * Evaluators seeder for `respan demo`.
 *
 * Recreates the demo evaluators (see evaluator-fixtures.ts) in the authenticated
 * account and tears them down for `respan demo clear`. Each evaluator is two kinds
 * of resource: grader record(s) and a Blockly workflow that references them.
 *
 * Recreate recipe (the engine — LLM vs human — lives in the workflow node, not the
 * grader record, so the captured graph is replayed verbatim):
 *   1. create each grader (or reuse one with the same name)
 *   2. map each grader's captured `originalId` to its new id
 *   3. deep-clone the task graph and rewrite every eval task's `config.evaluator_id`
 *      via that map (task ids are kept verbatim so the compute task's
 *      `state.<task_id>` input references stay valid)
 *   4. createWorkflow -> commit -> deploy v1
 *
 * Dedup is by workflow name (the user-facing evaluator name). The SDK has no commit
 * method, so commit is a raw POST to `/api/workflows/{id}/commits/` (as in
 * lib/eval/runner.ts). All evaluator-specific logic lives here.
 */

import { DEMO_EVALUATORS, DemoEvaluatorFixture } from './evaluator-fixtures.js';
import {
  Seeder,
  SeederContext,
  SeederResult,
  SeedItemResult,
  ExistingResource,
} from './types.js';

const WORKFLOW_NAMES = new Set(DEMO_EVALUATORS.map((e) => e.workflowName));
const GRADER_NAMES = new Set(DEMO_EVALUATORS.flatMap((e) => e.graders.map((g) => g.name)));

/** Demo evaluator workflows present in the account, matched by exact name. */
async function findExistingWorkflows(ctx: SeederContext): Promise<ExistingResource[]> {
  const response = await ctx.client.workflows.listWorkflows({
    Authorization: ctx.authHeader,
    page_size: 100,
  });
  const found: ExistingResource[] = [];
  for (const row of (response as { results?: Array<Record<string, unknown>> }).results ?? []) {
    const name = typeof row.name === 'string' ? row.name : '';
    const id = String(row.workflow_id ?? row.id ?? '');
    if (id && WORKFLOW_NAMES.has(name)) found.push({ name, id });
  }
  return found;
}

/** Demo grader records present in the account, matched by exact name. */
async function findExistingGraders(ctx: SeederContext): Promise<ExistingResource[]> {
  const response = await ctx.client.evaluators.listEvaluators({ Authorization: ctx.authHeader });
  const found: ExistingResource[] = [];
  for (const row of (response as { results?: Array<Record<string, unknown>> }).results ?? []) {
    const name = typeof row.name === 'string' ? row.name : '';
    const id = String(row.id ?? '');
    if (id && GRADER_NAMES.has(name)) found.push({ name, id });
  }
  return found;
}

/** Create each grader the evaluator needs (reusing one of the same name if it
 * already exists) and return a map from captured `originalId` to the live id. */
async function ensureGraders(
  ctx: SeederContext,
  fixture: DemoEvaluatorFixture,
  existingByName: Map<string, string>,
): Promise<Map<string, string>> {
  const idMap = new Map<string, string>();
  for (const grader of fixture.graders) {
    let id = existingByName.get(grader.name);
    if (!id) {
      const created = (await ctx.client.evaluators.createEvaluator({
        Authorization: ctx.authHeader,
        name: grader.name,
        type: grader.type as any,
        score_value_type: grader.score_value_type as any,
        score_config: grader.score_config as any,
        passing_conditions: grader.passing_conditions,
        llm_config: grader.llm_config as any,
      })) as unknown as Record<string, unknown>;
      id = String(created.id ?? created.evaluator_id ?? '');
      if (!id) throw new Error(`createEvaluator did not return an id for grader "${grader.name}".`);
      existingByName.set(grader.name, id);
    }
    idMap.set(grader.originalId, id);
  }
  return idMap;
}

/** Deep-clone the captured task graph and point every eval task at the live grader
 * id. Tasks without an `evaluator_id` (transform/compute) pass through unchanged. */
function remapTasks(
  tasks: Record<string, unknown>[],
  idMap: Map<string, string>,
): Record<string, unknown>[] {
  const clone = JSON.parse(JSON.stringify(tasks)) as Record<string, unknown>[];
  for (const task of clone) {
    const config = task.config as Record<string, unknown> | undefined;
    const original = config?.evaluator_id;
    if (typeof original === 'string') {
      const live = idMap.get(original);
      if (!live) throw new Error(`task references unknown grader id ${original}`);
      config!.evaluator_id = live;
    }
  }
  return clone;
}

function normalizeBaseUrl(baseUrl: string): string {
  return baseUrl.replace(/\/+$/, '').replace(/\/api$/, '');
}

/** Commit the workflow's draft (no SDK method — raw POST, as in runner.ts). */
async function commitWorkflow(ctx: SeederContext, workflowId: string): Promise<void> {
  const url = `${normalizeBaseUrl(ctx.baseUrl)}/api/workflows/${encodeURIComponent(workflowId)}/commits/`;
  const response = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Authorization: ctx.authHeader },
    body: JSON.stringify({}),
  });
  if (!response.ok) {
    const text = await response.text().catch(() => '');
    throw new Error(`commitWorkflow ${response.status}: ${text || response.statusText}`);
  }
}

/** Create the grader(s) and the deployed workflow for one evaluator. */
async function createFromFixture(
  ctx: SeederContext,
  fixture: DemoEvaluatorFixture,
  existingGradersByName: Map<string, string>,
): Promise<string> {
  const idMap = await ensureGraders(ctx, fixture, existingGradersByName);
  const tasks = remapTasks(fixture.tasks, idMap);

  const created = (await ctx.client.workflows.createWorkflow({
    Authorization: ctx.authHeader,
    type: 'evaluators',
    trigger_event_type: 'eval_only',
    name: fixture.workflowName,
    tasks: tasks as any,
  } as any)) as unknown as Record<string, unknown>;

  const workflowId = String(created.workflow_id ?? created.id ?? '');
  if (!workflowId) throw new Error(`createWorkflow did not return an id for "${fixture.workflowName}".`);

  await commitWorkflow(ctx, workflowId);
  await ctx.client.workflows.deployWorkflow({
    Authorization: ctx.authHeader,
    workflow_id: workflowId,
    version: 1,
  });

  return workflowId;
}

export const evaluatorsSeeder: Seeder = {
  service: 'evaluators',
  label: 'Evaluators',

  /** Both workflows and graders, so the clear preview/teardown covers every
   * resource the seeder owns. */
  async findExisting(ctx: SeederContext): Promise<ExistingResource[]> {
    const [workflows, graders] = await Promise.all([
      findExistingWorkflows(ctx),
      findExistingGraders(ctx),
    ]);
    return [...workflows, ...graders];
  },

  async run(ctx: SeederContext): Promise<SeederResult> {
    const existingWorkflows = new Set((await findExistingWorkflows(ctx)).map((w) => w.name));
    const existingGradersByName = new Map(
      (await findExistingGraders(ctx)).map((g) => [g.name, g.id] as const),
    );
    const items: SeedItemResult[] = [];

    for (const fixture of DEMO_EVALUATORS) {
      if (existingWorkflows.has(fixture.workflowName)) {
        items.push({ name: fixture.workflowName, status: 'skipped', reason: 'already exists' });
        continue;
      }
      const id = await createFromFixture(ctx, fixture, existingGradersByName);
      items.push({ name: fixture.workflowName, status: 'created', id });
    }

    return { service: this.service, label: this.label, items };
  },

  async clear(ctx: SeederContext): Promise<SeederResult> {
    const items: SeedItemResult[] = [];

    // Delete workflows first, then the grader records they referenced.
    for (const match of await findExistingWorkflows(ctx)) {
      await ctx.client.workflows.deleteWorkflow({
        Authorization: ctx.authHeader,
        workflow_id: match.id,
      });
      items.push({ name: match.name, status: 'deleted', id: match.id });
    }
    for (const match of await findExistingGraders(ctx)) {
      await ctx.client.evaluators.deleteEvaluator({
        Authorization: ctx.authHeader,
        evaluator_id: match.id,
      });
      items.push({ name: match.name, status: 'deleted', id: match.id });
    }

    if (items.length === 0) {
      for (const fixture of DEMO_EVALUATORS) {
        items.push({ name: fixture.workflowName, status: 'skipped', reason: 'not found' });
      }
    }

    return { service: this.service, label: this.label, items };
  },
};
