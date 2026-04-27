export interface EvalPromptMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

export interface EvalPromptBlock {
  id?: string;
  version_id?: string;
  name: string;
  description?: string;
  model: string;
  messages: EvalPromptMessage[];
  temperature?: number;
  max_tokens?: number;
  top_p?: number;
}

export interface EvalDatasetBlock {
  id?: string;
  name: string;
  description?: string;
  rows: Record<string, unknown>[];
}

export interface EvalEvaluatorBlock {
  evaluator_id?: string;
  workflow_id?: string;
  workflow_version_id?: string;
  workflow_version_deployed?: number;

  name: string;
  type: 'llm' | 'code' | 'human';
  score_value_type: 'numerical' | 'boolean' | 'percentage' | 'single_select' | 'multi_select' | 'json' | 'text' | 'categorical' | 'comment';
  description?: string;
  score_config?: {
    min_score?: number;
    max_score?: number;
    choices?: { name?: string; value?: string }[];
  };
  passing_conditions?: Record<string, unknown>;
  llm_config?: {
    model: string;
    evaluator_definition: string;
    scoring_rubric?: string;
    temperature?: number;
    max_tokens?: number;
    top_p?: number;
  };
  code_config?: {
    eval_code_snippet: string;
  };
}

export interface EvalExperimentBlock {
  id?: string;
  name?: string;
  description?: string;
  evaluators?: EvalEvaluatorBlock[];
  evaluator_slugs?: string[];
  evaluator_workflow_ids?: string[];
}

export interface EvalFile {
  name: string;
  experiment_name?: string;
  prompt: EvalPromptBlock;
  dataset: EvalDatasetBlock;
  experiment?: EvalExperimentBlock;
}

export function parseEvalFile(raw: string): EvalFile {
  const data = JSON.parse(raw) as Partial<EvalFile>;
  if (!data.name) throw new Error('eval file: `name` is required');
  if (!data.prompt) throw new Error('eval file: `prompt` block is required');
  if (!data.prompt.name) throw new Error('eval file: `prompt.name` is required');
  if (!data.prompt.model) throw new Error('eval file: `prompt.model` is required');
  if (!Array.isArray(data.prompt.messages) || data.prompt.messages.length === 0) {
    throw new Error('eval file: `prompt.messages` must be a non-empty array');
  }
  if (!data.dataset) throw new Error('eval file: `dataset` block is required');
  if (!data.dataset.name) throw new Error('eval file: `dataset.name` is required');
  if (!Array.isArray(data.dataset.rows)) {
    throw new Error('eval file: `dataset.rows` must be an array');
  }
  return data as EvalFile;
}
