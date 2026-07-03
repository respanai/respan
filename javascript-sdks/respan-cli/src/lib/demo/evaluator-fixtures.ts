/**
 * Frozen snapshot of the demo evaluators, captured byte-for-byte from the
 * owner's deployed evaluators (graders + Blockly workflow task graphs).
 *
 * Source of truth for `respan demo`. Each entry is one user-facing evaluator: the
 * deployed workflow's task graph plus the grader record(s) it references. The
 * grader `type` is captured verbatim ("human" is the stored default) — the actual
 * LLM engine lives in each eval task's `generation_method`/`_blockly_evaluator_kind`,
 * so replaying the graph reproduces the verified scoring behavior exactly.
 *
 * This is a frozen snapshot: if the source evaluators change, this file does NOT
 * update until someone re-snapshots and ships a new build.
 *
 * Recreate recipe (see seed-evaluators.ts): create each grader -> map its
 * `originalId` to the new id -> rewrite every eval task's `config.evaluator_id`
 * via that map -> createWorkflow -> commit -> deploy. Task ids are kept verbatim
 * so the compute task's `state.<task_id>` input references stay valid.
 */

/** A grader record referenced by an evaluator's workflow. */
export interface DemoEvaluatorGrader {
  name: string;
  /** The grader's id inside the captured task graph. Remapped to the freshly
   * created grader id at seed time so the eval tasks point at the new record. */
  originalId: string;
  type: string;
  score_value_type: string;
  score_config: Record<string, unknown>;
  passing_conditions: Record<string, unknown>;
  llm_config: Record<string, unknown>;
}

/** One demo evaluator: the deployed workflow graph plus its grader(s). */
export interface DemoEvaluatorFixture {
  /** Workflow name (the user-facing evaluator name); used for dedup. */
  workflowName: string;
  description?: string;
  graders: DemoEvaluatorGrader[];
  /** Verbatim deployed workflow task graph. Only each eval task's
   * `config.evaluator_id` is rewritten at seed time. */
  tasks: Record<string, unknown>[];
}

export const DEMO_EVALUATORS: DemoEvaluatorFixture[] =
[
  {
    "workflowName": "Demo evaluator 1: support-policy-compliance",
    "description": "Scores a Lumen support reply against refund/escalation policy (1-5).",
    "graders": [
      {
        "name": "support-policy-compliance",
        "originalId": "86d100fb91c547f4b35a2d0a3db1d095",
        "type": "human",
        "score_value_type": "numerical",
        "score_config": {
          "max_score": 5.0,
          "min_score": 0.0
        },
        "passing_conditions": {
          "primary_score": {
            "value": 3,
            "operator": "gte"
          }
        },
        "llm_config": {
          "model": "openai/gpt-5.1",
          "top_p": 1.0,
          "max_tokens": 200,
          "temperature": 0.1,
          "scoring_rubric": "",
          "presence_penalty": 0.0,
          "frequency_penalty": 0.0,
          "evaluator_definition": "You are checking a Lumen Electronics support reply against policy.\n\nCustomer message: {{input}}\nAgent reply: {{output}}\n\nPolicy: refunds only within 30 days; final-sale items never refundable;\nrefunds over $200 need a stated reason; never refund a delayed or\nundelivered order; if \"delivered but never arrived\" or \"lost in mail\",\nescalate instead of refunding.\n\nScore 1 to 5:\n5 = follows every rule\n3 = right outcome but skipped a step\n1 = broke a refund or escalation rule\n\nReply with only the number."
        }
      }
    ],
    "tasks": [
      {
        "id": "d970b254-ea77-493e-a722-c9ca0c4132e9",
        "next": "90112acd-bb05-46ed-a1c5-ad9fb6f84f33",
        "type": "eval",
        "label": "blockly_hidden_eval_7nyk_jmj_1_g_o3h_b",
        "config": {
          "llm_config": {
            "model": "openai/gpt-5.1",
            "top_p": 1,
            "max_tokens": 200,
            "temperature": 0.1,
            "scoring_rubric": "",
            "presence_penalty": 0,
            "frequency_penalty": 0,
            "evaluator_definition": "You are checking a Lumen Electronics support reply against policy.\n\nCustomer message: {{input}}\nAgent reply: {{output}}\n\nPolicy: refunds only within 30 days; final-sale items never refundable;\nrefunds over $200 need a stated reason; never refund a delayed or\nundelivered order; if \"delivered but never arrived\" or \"lost in mail\",\nescalate instead of refunding.\n\nScore 1 to 5:\n5 = follows every rule\n3 = right outcome but skipped a step\n1 = broke a refund or escalation rule\n\nReply with only the number."
          },
          "evaluator_id": "86d100fb91c547f4b35a2d0a3db1d095",
          "score_config": {
            "max_score": 5,
            "min_score": 0
          },
          "_blockly_node_id": "/7Nyk_jmJ~=1#G;o3h[b",
          "score_value_type": "numerical",
          "_blockly_hidden_eval": true,
          "_blockly_output_field": "primary_score",
          "_blockly_evaluator_kind": "llm",
          "is_auto_persist_enabled": false,
          "_blockly_evaluator_label": "support-policy-compliance"
        },
        "generation_method": "llm"
      },
      {
        "id": "90112acd-bb05-46ed-a1c5-ad9fb6f84f33",
        "next": null,
        "type": "transform",
        "label": "blockly_final_result_grp_9_3ivg_r9xnj",
        "config": {
          "params": {
            "field": "primary_score",
            "source_max": 5,
            "source_min": 0,
            "target_max": 5,
            "target_min": 0
          },
          "transform_type": "normalize",
          "_blockly_node_id": "grP#9#3IvG),r9XNJ;{)",
          "_blockly_is_result": true,
          "_blockly_final_result_normalize": true
        },
        "task_type": "transform"
      }
    ]
  },
  {
    "workflowName": "Demo evaluator 3: call-transcript-classifier",
    "description": "Scores classifier accuracy vs gold label via weighted average of two graders (0-1).",
    "graders": [
      {
        "name": "extraction-accuracy",
        "originalId": "7d7a60d8184a4d59af405af8362e314b",
        "type": "human",
        "score_value_type": "numerical",
        "score_config": {
          "max_score": 1.0,
          "min_score": 0.0
        },
        "passing_conditions": {
          "primary_score": {
            "value": 0.8,
            "operator": "gte"
          }
        },
        "llm_config": {
          "model": "openai/gpt-5.1",
          "top_p": 1.0,
          "max_tokens": 200,
          "temperature": 0.1,
          "scoring_rubric": "",
          "presence_penalty": 0.0,
          "frequency_penalty": 0.0,
          "evaluator_definition": "Compare the classifier output to the gold label, focusing on EXTRACTED data.\nClassifier output: {{output}}\nGold label: {{expected_output}}\nScore 0.0–1.0 = how well these match: order_ids (exact set),\nproducts_mentioned, agent_name, key_quote (semantic match). Average them.\nReply with only a decimal 0.0–1.0."
        }
      },
      {
        "name": "categorical-accuracy",
        "originalId": "dca4334d65a845838dbcf04dde75e2d8",
        "type": "human",
        "score_value_type": "numerical",
        "score_config": {
          "max_score": 1.0,
          "min_score": 0.0
        },
        "passing_conditions": {
          "primary_score": {
            "value": 0.8,
            "operator": "gte"
          }
        },
        "llm_config": {
          "model": "openai/gpt-5.1",
          "top_p": 1.0,
          "max_tokens": 200,
          "temperature": 0.1,
          "scoring_rubric": "",
          "presence_penalty": 0.0,
          "frequency_penalty": 0.0,
          "evaluator_definition": "Compare the classifier output to the gold label, focusing on JUDGMENT fields.\nClassifier output: {{output}}\nGold label: {{expected_output}}\nScore 0.0–1.0 = fraction of these fields that match the gold label:\nprimary_reason, resolution_status, customer_sentiment_start,\ncustomer_sentiment_end, escalation_requested, followup_required.\nReply with only a decimal 0.0–1.0."
        }
      }
    ],
    "tasks": [
      {
        "id": "b505bf79-df32-4fd6-b2b3-eed05afc4955",
        "next": "f9bcfabe-fa16-4fb5-a8c8-dadf3b51027a",
        "type": "eval",
        "label": "blockly_hidden_eval_cd_oamywl_wvoo_6",
        "config": {
          "llm_config": {
            "model": "openai/gpt-5.1",
            "top_p": 1,
            "max_tokens": 200,
            "temperature": 0.1,
            "scoring_rubric": "",
            "presence_penalty": 0,
            "frequency_penalty": 0,
            "evaluator_definition": "Compare the classifier output to the gold label, focusing on EXTRACTED data.\nClassifier output: {{output}}\nGold label: {{expected_output}}\nScore 0.0–1.0 = how well these match: order_ids (exact set),\nproducts_mentioned, agent_name, key_quote (semantic match). Average them.\nReply with only a decimal 0.0–1.0."
          },
          "evaluator_id": "7d7a60d8184a4d59af405af8362e314b",
          "score_config": {
            "max_score": 1,
            "min_score": 0
          },
          "_blockly_node_id": "Cd,OaMYWL.;=wVoO]=~6",
          "score_value_type": "numerical",
          "_blockly_hidden_eval": true,
          "_blockly_output_field": "primary_score",
          "_blockly_evaluator_kind": "llm",
          "is_auto_persist_enabled": false,
          "_blockly_evaluator_label": "extraction-accuracy"
        },
        "generation_method": "llm"
      },
      {
        "id": "f9bcfabe-fa16-4fb5-a8c8-dadf3b51027a",
        "next": "fefd60d6-a309-4b88-ad9c-b208fe0866fa",
        "type": "eval",
        "label": "blockly_hidden_eval_gq_rw_bra0pfw_y872k",
        "config": {
          "llm_config": {
            "model": "openai/gpt-5.1",
            "top_p": 1,
            "max_tokens": 200,
            "temperature": 0.1,
            "scoring_rubric": "",
            "presence_penalty": 0,
            "frequency_penalty": 0,
            "evaluator_definition": "Compare the classifier output to the gold label, focusing on JUDGMENT fields.\nClassifier output: {{output}}\nGold label: {{expected_output}}\nScore 0.0–1.0 = fraction of these fields that match the gold label:\nprimary_reason, resolution_status, customer_sentiment_start,\ncustomer_sentiment_end, escalation_requested, followup_required.\nReply with only a decimal 0.0–1.0."
          },
          "evaluator_id": "dca4334d65a845838dbcf04dde75e2d8",
          "score_config": {
            "max_score": 1,
            "min_score": 0
          },
          "_blockly_node_id": "Gq`rw!bra0pFw*,Y872k",
          "score_value_type": "numerical",
          "_blockly_hidden_eval": true,
          "_blockly_output_field": "primary_score",
          "_blockly_evaluator_kind": "llm",
          "is_auto_persist_enabled": false,
          "_blockly_evaluator_label": "categorical-accuracy"
        },
        "generation_method": "llm"
      },
      {
        "id": "fefd60d6-a309-4b88-ad9c-b208fe0866fa",
        "next": null,
        "type": "compute",
        "label": "blockly_hidden_compute_clysphgg_qvui_b_4",
        "config": {
          "inputs": [
            {
              "field": "primary_score",
              "source": "state.b505bf79-df32-4fd6-b2b3-eed05afc4955",
              "weight": 0.6
            },
            {
              "field": "primary_score",
              "source": "state.f9bcfabe-fa16-4fb5-a8c8-dadf3b51027a",
              "weight": 0.4
            }
          ],
          "function": "weighted_average",
          "_blockly_node_id": "clySphgG^#@QVUi-b?4%",
          "_blockly_is_result": true,
          "_blockly_hidden_compute": true
        }
      }
    ]
  }
]
;
