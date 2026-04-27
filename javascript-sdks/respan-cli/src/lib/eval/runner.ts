import { readFileSync, writeFileSync } from 'node:fs';
import { RespanClient } from '@respan/respan-api';
import { EvalFile, parseEvalFile } from './schema.js';

export interface RunEvalResult {
  prompt_id: string;
  prompt_version_id?: string;
  prompt_version_deployed?: number;
  dataset_id: string;
  evaluator_workflow_ids: string[];
  experiment_id: string;
  created_prompt: boolean;
  created_dataset: boolean;
  created_experiment: boolean;
  evaluators_created: number;
  rows_added: number;
}

export async function runEval(
  filePath: string,
  client: RespanClient,
  authHeader: string,
  baseUrl: string,
  log: (msg: string) => void = () => {},
): Promise<RunEvalResult> {
  const raw = readFileSync(filePath, 'utf8');
  const evalFile = parseEvalFile(raw);

  const promptResult = await ensurePrompt(evalFile, client, authHeader, log);
  persist(filePath, evalFile);

  const datasetResult = await ensureDataset(evalFile, client, authHeader, log);
  persist(filePath, evalFile);

  const evaluatorResult = await ensureEvaluatorWorkflows(evalFile, client, authHeader, baseUrl, log);
  persist(filePath, evalFile);

  const experimentResult = await ensureExperiment(
    evalFile,
    promptResult.id,
    datasetResult.id,
    evaluatorResult.workflow_ids,
    client,
    authHeader,
    log,
  );
  persist(filePath, evalFile);

  return {
    prompt_id: promptResult.id,
    prompt_version_id: promptResult.version_id,
    prompt_version_deployed: promptResult.deployed_version,
    dataset_id: datasetResult.id,
    evaluator_workflow_ids: evaluatorResult.workflow_ids,
    experiment_id: experimentResult.id,
    created_prompt: promptResult.created,
    created_dataset: datasetResult.created,
    created_experiment: experimentResult.created,
    evaluators_created: evaluatorResult.created_count,
    rows_added: datasetResult.rows_added,
  };
}

async function ensurePrompt(
  evalFile: EvalFile,
  client: RespanClient,
  Authorization: string,
  log: (msg: string) => void,
): Promise<{ id: string; version_id?: string; deployed_version?: number; created: boolean }> {
  const { prompt } = evalFile;
  if (prompt.id) {
    log(`prompt: reusing ${prompt.id}`);
    return { id: prompt.id, version_id: prompt.version_id, created: false };
  }

  log(`prompt: creating "${prompt.name}"`);
  const created = (await client.prompts.createPrompt({
    Authorization,
    name: prompt.name,
    ...(prompt.description ? { description: prompt.description } : {}),
  })) as Record<string, unknown>;

  const promptId = String(created.id ?? created.prompt_id ?? '');
  if (!promptId) throw new Error('createPrompt: response missing id');
  prompt.id = promptId;

  log(`prompt: creating version on ${promptId}`);
  const version = (await client.prompts.createPromptVersion({
    Authorization,
    prompt_id: promptId,
    messages: prompt.messages as unknown as Record<string, unknown>[],
    model: prompt.model,
    ...(prompt.temperature !== undefined ? { temperature: prompt.temperature } : {}),
    ...(prompt.max_tokens !== undefined ? { max_tokens: prompt.max_tokens } : {}),
    ...(prompt.top_p !== undefined ? { top_p: prompt.top_p } : {}),
  })) as Record<string, unknown>;

  const versionId = String(version.id ?? version.version_id ?? '');
  if (versionId) prompt.version_id = versionId;

  const versionNumber = typeof version.version === 'number'
    ? version.version
    : Number.parseInt(String(version.version ?? '1'), 10);

  log(`prompt: committing draft version ${versionNumber}`);
  await client.prompts.commitPromptVersion({ Authorization, prompt_id: promptId });

  log(`prompt: deploying version ${versionNumber}`);
  await client.prompts.deployPromptVersion({ Authorization, prompt_id: promptId, version: versionNumber });

  return {
    id: promptId,
    version_id: versionId || undefined,
    deployed_version: versionNumber,
    created: true,
  };
}

async function ensureDataset(
  evalFile: EvalFile,
  client: RespanClient,
  Authorization: string,
  log: (msg: string) => void,
): Promise<{ id: string; created: boolean; rows_added: number }> {
  const { dataset } = evalFile;
  if (dataset.id) {
    log(`dataset: reusing ${dataset.id}`);
    return { id: dataset.id, created: false, rows_added: 0 };
  }

  log(`dataset: creating "${dataset.name}"`);
  const created = (await client.datasets.createDataset({
    Authorization,
    name: dataset.name,
    is_empty: true,
    ...(dataset.description ? { description: dataset.description } : {}),
  })) as Record<string, unknown>;

  const datasetId = String(created.id ?? created.dataset_id ?? '');
  if (!datasetId) throw new Error('createDataset: response missing id');
  dataset.id = datasetId;

  const logs = dataset.rows.map((row) => {
    const { expected_output, ...variables } = row as Record<string, unknown> & { expected_output?: unknown };
    return {
      input: variables,
      ...(expected_output !== undefined ? { expected_output } : {}),
    };
  });
  const result = (await client.datasets.bulkCreateDatasetLogs({
    Authorization,
    dataset_id: datasetId,
    logs,
  })) as unknown as Record<string, unknown>;
  const successCount = typeof result.success_count === 'number' ? result.success_count : 0;
  const errorCount = typeof result.error_count === 'number' ? result.error_count : 0;
  log(`dataset: added ${successCount}/${logs.length} row(s)` + (errorCount ? ` (${errorCount} errors)` : ''));

  return { id: datasetId, created: true, rows_added: successCount };
}

async function ensureEvaluatorWorkflows(
  evalFile: EvalFile,
  client: RespanClient,
  Authorization: string,
  baseUrl: string,
  log: (msg: string) => void,
): Promise<{ workflow_ids: string[]; created_count: number }> {
  const experiment = evalFile.experiment ?? (evalFile.experiment = {});
  const evaluators = experiment.evaluators ?? [];
  const explicitWorkflowIds = experiment.evaluator_workflow_ids ?? [];

  let createdCount = 0;
  const workflowIds: string[] = [...explicitWorkflowIds];

  for (const ev of evaluators) {
    if (!ev.evaluator_id) {
      log(`evaluator: creating "${ev.name}" (${ev.type})`);
      const created = (await client.evaluators.createEvaluator({
        Authorization,
        name: ev.name,
        type: ev.type,
        score_value_type: ev.score_value_type,
        ...(ev.description ? { description: ev.description } : {}),
        ...(ev.score_config ? { score_config: ev.score_config as any } : {}),
        ...(ev.passing_conditions ? { passing_conditions: ev.passing_conditions } : {}),
        ...(ev.llm_config ? { llm_config: ev.llm_config as any } : {}),
        ...(ev.code_config ? { code_config: ev.code_config as any } : {}),
      })) as unknown as Record<string, unknown>;
      const evaluatorId = String(created.id ?? created.evaluator_id ?? '');
      if (!evaluatorId) throw new Error('createEvaluator: response missing id');
      ev.evaluator_id = evaluatorId;
      createdCount += 1;
    } else {
      log(`evaluator: reusing ${ev.evaluator_id}`);
    }

    if (!ev.workflow_id) {
      const workflowName = `${ev.name} workflow`;
      log(`workflow: creating "${workflowName}" wrapping evaluator ${ev.evaluator_id}`);
      const nodeIdSeed = ev.evaluator_id!.slice(0, 8);
      const blocklyNodeId = safeNodeId(nodeIdSeed);
      const taskId = `blockly_hidden_eval_task_${blocklyNodeId}`;
      const taskLabel = `blockly_hidden_eval_${blocklyNodeId}`;
      const outputField = outputFieldForScoreType(ev.score_value_type);
      const taskConfig: Record<string, unknown> = {
        evaluator_id: ev.evaluator_id,
        score_value_type: ev.score_value_type,
        score_config: ev.score_config ?? {},
        is_auto_persist_enabled: true,
        _blockly_hidden_eval: true,
        _blockly_is_result: true,
        _blockly_node_id: blocklyNodeId,
        ...(outputField ? { _blockly_output_field: outputField } : {}),
        _blockly_evaluator_kind: ev.type,
        ...(ev.llm_config ? { llm_config: ev.llm_config } : {}),
        ...(ev.code_config ? { code_config: ev.code_config } : {}),
      };
      const createdWf = (await client.workflows.createWorkflow({
        Authorization,
        type: 'evaluators',
        trigger_event_type: 'eval_only',
        name: workflowName,
        tasks: [{
          id: taskId,
          type: 'eval',
          generation_method: ev.type,
          label: taskLabel,
          config: taskConfig,
          next: null,
        } as any],
      } as any)) as unknown as Record<string, unknown>;
      const workflowId = String(createdWf.workflow_id ?? createdWf.id ?? '');
      if (!workflowId) throw new Error('createWorkflow: response missing workflow_id');
      ev.workflow_id = workflowId;

      log(`workflow: committing draft v1`);
      await commitWorkflow(Authorization, baseUrl, workflowId);

      log(`workflow: deploying v1`);
      const deployed = (await client.workflows.deployWorkflow({
        Authorization,
        workflow_id: workflowId,
        version: 1,
      })) as unknown as Record<string, unknown>;
      const deployedVersion = typeof deployed.version === 'number' ? deployed.version : 1;
      ev.workflow_version_deployed = deployedVersion;
      const versionRowId = String(deployed.id ?? '');
      if (versionRowId) ev.workflow_version_id = versionRowId;
    } else {
      log(`workflow: reusing ${ev.workflow_id}`);
    }

    if (!ev.workflow_version_id) {
      log(`workflow: looking up deployed version row id for ${ev.workflow_id}`);
      const versionRowId = await getDeployedWorkflowVersionId(Authorization, baseUrl, ev.workflow_id);
      if (!versionRowId) {
        throw new Error(
          `workflow ${ev.workflow_id} has no deployed version; rerun after committing/deploying.`,
        );
      }
      ev.workflow_version_id = versionRowId;
    }
    workflowIds.push(ev.workflow_version_id);
  }

  return { workflow_ids: workflowIds, created_count: createdCount };
}

async function ensureExperiment(
  evalFile: EvalFile,
  promptId: string,
  datasetId: string,
  workflowIds: string[],
  client: RespanClient,
  Authorization: string,
  log: (msg: string) => void,
): Promise<{ id: string; created: boolean }> {
  const experiment = evalFile.experiment ?? (evalFile.experiment = {});
  if (experiment.id) {
    log(`experiment: reusing ${experiment.id}`);
    return { id: experiment.id, created: false };
  }

  const name = experiment.name ?? evalFile.experiment_name ?? `${evalFile.name} run`;
  const evaluatorSlugs = experiment.evaluator_slugs ?? [];
  if (workflowIds.length === 0 && evaluatorSlugs.length === 0) {
    throw new Error(
      'experiment requires at least one evaluator. Add `experiment.evaluators[]` to your eval file ' +
      '(or `experiment.evaluator_workflow_ids` / `experiment.evaluator_slugs` for existing ones).',
    );
  }
  const evaluatorSummary = workflowIds.length > 0
    ? `evaluator_workflow_ids: ${workflowIds.join(', ')}`
    : `evaluator_slugs: ${evaluatorSlugs.join(', ')}`;
  log(`experiment: creating "${name}" with ${evaluatorSummary}`);
  const created = (await client.experiments.createExperiment({
    Authorization,
    name,
    dataset_id: datasetId,
    ...(experiment.description ? { description: experiment.description } : {}),
    workflow: [{ type: 'prompt', config: { prompt_id: promptId } }],
    ...(workflowIds.length > 0 ? { evaluator_workflow_ids: workflowIds } : {}),
    ...(evaluatorSlugs.length > 0 ? { evaluator_slugs: evaluatorSlugs } : {}),
  } as any)) as unknown as Record<string, unknown>;

  const experimentId = String(created.id ?? created.experiment_id ?? '');
  if (!experimentId) throw new Error('createExperiment: response missing id');
  experiment.id = experimentId;
  if (!experiment.name) experiment.name = name;

  return { id: experimentId, created: true };
}

function safeNodeId(raw: string): string {
  const sanitized = raw.replace(/[^a-zA-Z0-9_]/g, '_').slice(0, 30);
  return sanitized || 'node';
}

function outputFieldForScoreType(scoreValueType: string): string | null {
  switch (scoreValueType) {
    case 'boolean':
      return 'boolean_value';
    case 'single_select':
    case 'text':
    case 'comment':
      return 'string_value';
    case 'multi_select':
    case 'json':
      return null;
    default:
      return 'primary_score';
  }
}

async function getDeployedWorkflowVersionId(
  Authorization: string,
  baseUrl: string,
  workflowId: string,
): Promise<string | null> {
  const url = `${normalizeBaseUrl(baseUrl)}/api/workflows/${encodeURIComponent(workflowId)}/versions/`;
  const response = await fetch(url, { headers: { Authorization } });
  if (!response.ok) return null;
  const body = (await response.json()) as { results?: Array<Record<string, unknown>> };
  const results = body.results ?? [];
  const deployed = results.find((v) => v.is_enabled === true && v.is_read_only === true)
    ?? results.find((v) => v.is_read_only === true);
  return deployed ? String(deployed.id ?? '') : null;
}

function normalizeBaseUrl(baseUrl: string): string {
  return baseUrl.replace(/\/+$/, '').replace(/\/api$/, '');
}

async function commitWorkflow(Authorization: string, baseUrl: string, workflowId: string): Promise<void> {
  const url = `${normalizeBaseUrl(baseUrl)}/api/workflows/${encodeURIComponent(workflowId)}/commits/`;
  const response = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Authorization },
    body: JSON.stringify({}),
  });
  if (!response.ok) {
    const text = await response.text().catch(() => '');
    throw new Error(`commitWorkflow ${response.status}: ${text || response.statusText}`);
  }
}

function persist(filePath: string, evalFile: EvalFile): void {
  writeFileSync(filePath, JSON.stringify(evalFile, null, 2) + '\n', 'utf8');
}
