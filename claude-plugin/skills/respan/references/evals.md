# Evals Reference

Build evaluation pipelines: datasets, evaluators, and experiments.

Docs: `https://respan.ai/docs/documentation/features/evals/concepts.md`

---

## Overview

Two evaluation modes:

- **Offline** — build datasets, run experiments across prompt/model variants, compare scores
- **Online** — automatically score live production traffic (see [monitors.md](monitors.md))

Offline eval workflow:

```
Create dataset -> Create evaluator -> Create experiment -> Run & compare
```

---

## Datasets

A dataset is a collection of test rows (input/output pairs) for experiments.

### Create from scratch

```
Use create_dataset:
  name: "intent_classification_testset"
  description: "Test cases for intent classifier"

Then add rows with create_dataset_span:
  dataset_id: "<id>"
  input: '{"user_message": "I want a refund"}'
  output: '{"intent": "refund_request"}'
```

### Create from production traces

Sample real spans from traced data:

```
1. Use list_traces or list_logs to find relevant spans
2. Use add_spans_to_dataset to add span IDs to a dataset:
     dataset_id: "<id>"
     span_ids: ["span_1", "span_2", "span_3"]
```

Or use the platform UI: Datasets -> Insert by sampling (filter, set sampling %, preview, confirm).

### CSV import

Upload CSV via the platform UI. Columns map to prompt variables. Optional `ideal_output` column for expected outputs. Max 500 rows per import.

### MCP tools

| Action | Tool |
|--------|------|
| Create dataset | `create_dataset(name, description)` |
| List datasets | `list_datasets` |
| Get dataset | `get_dataset(dataset_id)` |
| Update dataset | `update_dataset(dataset_id, name, description)` |
| Add a row manually | `create_dataset_span(dataset_id, input, output)` |
| Add existing spans | `add_spans_to_dataset(dataset_id, span_ids)` |
| List rows | `list_dataset_spans(dataset_id)` |
| Get a row | `retrieve_dataset_span(dataset_id, log_id)` |
| Update a row | `update_dataset_span(dataset_id, log_id, ...)` |

---

## Evaluators

An evaluator scores LLM outputs. It is a workflow built from **graders** connected with conditions.

### Grader types

| Type | How it works |
|------|-------------|
| **LLM grader** | A language model judges the output. Uses variables: `{{output}}`, `{{input}}`, `{{expected_output}}`, `{{metadata}}`, `{{metrics}}` |
| **Code grader** | A Python function returns a score. Signature: `main(eval_inputs)` where `eval_inputs["output"]` is required |
| **Human grader** | Team members review and score outputs manually |

### Score value types

| Type | Use case |
|------|----------|
| `numerical` | Rating scales (e.g. 0-5) |
| `boolean` | Pass/fail checks |
| `categorical` | Multi-choice classifications |
| `comment` | Qualitative text feedback |

### Create an evaluator

```
Use create_evaluator:
  name: "response_quality"
  score_value_type: "numerical"
  type: "llm"
  description: "Scores response quality 1-5"
  evaluator_slug: "response-quality"
```

For complex evaluators with grader workflows, conditions, and compute blocks — use the platform UI (Evaluators -> Create).

### Evaluator workflow blocks

- **Markers:** `Original input`, `Final result` (entry/exit)
- **Graders:** LLM, Code, or Human scoring
- **Conditions:** `If...Then...Else` branching based on scores
- **Compute:** Average/weighted average of multiple grader scores
- **Metrics:** Built-in: completion tokens, cost, latency, model, prompt tokens, total tokens
- **Constants:** Fixed threshold values

### MCP tools

| Action | Tool |
|--------|------|
| List evaluators | `list_evaluators` |
| Get evaluator | `get_evaluator(evaluator_id)` |
| Create evaluator | `create_evaluator(name, score_value_type, ...)` |
| Update evaluator | `update_evaluator(evaluator_id, ...)` |
| Run evaluator | `run_evaluator(evaluator_id, dataset_id or log_ids)` |

---

## Experiments

An experiment runs evaluators across a dataset to compare prompt/model configurations.

### Workflow

1. Create a dataset with test rows
2. Create evaluator(s) to score outputs
3. Create an experiment linking dataset + evaluators
4. Run the experiment — it generates outputs for each row and scores them
5. Compare results across configurations

### Create an experiment

```
Use create_experiment:
  name: "v2_vs_v3_comparison"
  dataset_id: "<dataset-id>"
  evaluator_slugs: ["response-quality", "hallucination-check"]
  workflows:
    - type: "prompt"
      config: { prompt_id: "<prompt-v2-id>" }
    - type: "prompt"
      config: { prompt_id: "<prompt-v3-id>" }
```

### Workflow types

| Type | Description |
|------|-------------|
| `prompt` | Run a saved prompt template against dataset rows |
| `completion` | Run a model completion with config |
| `custom` | User-defined logic |

### MCP tools

| Action | Tool |
|--------|------|
| List experiments | `list_experiments` |
| Get experiment | `get_experiment(experiment_id)` |
| Create experiment | `create_experiment(name, dataset_id, ...)` |
| List experiment spans | `list_experiment_spans(experiment_id)` |
| Get experiment span | `get_experiment_span(experiment_id, log_id)` |
| Update experiment span | `update_experiment_span(experiment_id, log_id, ...)` |
| Get summary stats | `get_experiment_spans_summary(experiment_id, start_time, end_time)` |

---

## End-to-End Example

```
Step 1: Create dataset
  create_dataset(name="qa_testset")
  create_dataset_span(dataset_id="...", input="What is Python?", output="A programming language")
  create_dataset_span(dataset_id="...", input="Who made Rust?", output="Mozilla Research")

Step 2: Create evaluator
  create_evaluator(
    name="accuracy_check",
    score_value_type="boolean",
    type="llm",
    evaluator_slug="accuracy-check"
  )

Step 3: Create experiment
  create_experiment(
    name="gpt4o_vs_claude",
    dataset_id="...",
    evaluator_slugs=["accuracy-check"]
  )

Step 4: Review results
  list_experiment_spans(experiment_id="...")
  get_experiment_spans_summary(experiment_id="...", start_time="...", end_time="...")
```

---

## CLI Commands

```bash
respan datasets list                   # List datasets
respan datasets create --name <name>   # Create a dataset
respan datasets get <id>               # Get dataset details
respan datasets spans <id>             # List dataset spans
respan evaluators list                 # List evaluators
respan evaluators get <id>             # Get evaluator details
respan evaluators create               # Create an evaluator
respan evaluators run <id>             # Run an evaluator
respan experiments list                # List experiments
respan experiments get <id>            # Get experiment details
respan experiments create              # Create an experiment
respan eval <file>                     # Run a one-shot eval defined in a JSON file
```

## One-Shot Evals via JSON

`respan eval <file.json>` runs a complete eval pipeline from a single JSON spec —
creates the prompt, dataset, evaluators, and experiment in one command.
IDs are written back into the file so reruns reuse existing resources.

### Minimal example

```json
{
  "name": "Movie matcher",
  "prompt": {
    "name": "movie-matcher",
    "model": "gpt-5-mini",
    "messages": [
      { "role": "system", "content": "Identify the movie. Reply with only the title." },
      { "role": "user", "content": "{{description}}" }
    ]
  },
  "dataset": {
    "name": "movie-descriptions",
    "rows": [
      { "description": "A detective investigates seven deadly sins.", "expected_output": "Se7en" },
      { "description": "A hacker learns reality is a simulation.", "expected_output": "The Matrix" }
    ]
  },
  "experiment": {
    "evaluators": [
      {
        "name": "Title match",
        "type": "llm",
        "score_value_type": "boolean",
        "llm_config": {
          "model": "openai/gpt-5-mini",
          "evaluator_definition": "Expected: {{expected_output}}. Got: {{output}}. Correct? Reply true or false."
        }
      }
    ]
  }
}
```

Run:

```bash
respan eval movie-matcher.eval.json
```

The runner creates the prompt (and deploys version 1), creates the dataset (and bulk-loads rows),
creates each evaluator + its workflow (and deploys), then creates the experiment linking them.
Each step writes the resulting IDs back to the JSON file so subsequent runs are idempotent —
re-running with the same file reuses the existing resources.
