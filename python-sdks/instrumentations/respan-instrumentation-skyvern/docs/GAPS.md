# Skyvern observability: coverage, parity with Laminar, open gaps

Goal: Respan traces for Skyvern that match what
[Laminar's Skyvern integration](https://laminar.sh/docs/integrations/skyvern) offers:
the task / workflow steps, every LLM call with its screenshots, tokens and cost,
exceptions, and a browser recording synced to the steps.

## How Laminar does it

- `Laminar.initialize()` patches Skyvern from outside: `Skyvern.run_task`,
  `ForgeAgent.execute_step`, the task-v2 service, a workflow-block helper, and it
  replaces the primary `LLM_API_HANDLER` (one span per prompt name). LLM content
  comes from its litellm instrumentation; the recording comes from an rrweb recorder
  injected into the Playwright page.
- Skyvern also ships a Laminar hook (`skyvern/forge/sdk/trace/lmnr.py`) and its own
  OpenTelemetry spans (tracer `skyvern`). Neither carries Respan's span kind, so
  Respan would drop them; this package wraps Skyvern's seams directly, the same way
  the browser-use instrumentation does.

## What this package covers

| | Laminar | Respan |
|---|---|---|
| Run span (task / workflow / login / download) | yes | yes, local and cloud, with status, output, failure reason, recording link |
| `page.act / extract / validate / ai_*` | no | yes, one span per AI action |
| Step spans | yes (`execute_step`) | yes (`agent_step`), with page URL, actions + results and the step screenshot |
| Browser actions | no (inside Skyvern's own spans) | yes, one tool span per action; failed actions are error spans |
| Workflow blocks | block description helper only | yes, one span per block with status and output |
| LLM calls | primary handler only | **every** Skyvern model call (all handlers end in litellm), named by Skyvern's prompt name, with messages, screenshots, completion and tokens |
| Grouping | trace only | thread / session / trace group = Skyvern run id; customer values win |
| Secrets | - | values Skyvern resolves as secrets are masked in every span |
| Session recording | rrweb replay synced to steps | **not yet**: step screenshots plus Skyvern's own `recording_url` on the run span (see below) |

## Known limitations

| # | Limitation | Impact |
|---|---|---|
| L1 | **Session replay.** Respan has no rrweb-style recording player yet; this is a known gap shared by all browser-agent integrations. Interim: a JPEG screenshot on every step span (the image the model saw), and Skyvern's own `recording_url` on the run span. | no sub-step replay |
| L2 | **Cloud runs are one span.** Against Skyvern Cloud the engine runs remotely, so only the run span (status, output, failure reason, recording link) is captured. Steps and actions of a cloud run could be fetched afterwards from Skyvern's runs / artifacts API; not done yet. | no step tree for cloud runs |
| L3 | **Work after the last step sits under the run span.** Skyvern checks the goal and pre-plans the next step concurrently right after a step, and runs the final `complete` / `terminate` action outside the step; those spans hang off the run span, not a step. This is Skyvern's own structure (its native spans show the same). | the step count in Respan can be lower than Skyvern's `step_count` |
| L4 | **Workflows, `login`, `download_files`, the `skyvern_v2` engine and `page.act / extract / validate / ai_*` in local mode** need Skyvern's persistent database: in Skyvern's in-memory mode (`use_in_memory_db=True`) they fail inside Skyvern itself, with or without tracing. Their run, page-action and block spans are covered by tests against a fake Skyvern server and the block wrapper, not by a live local run. | verified on the client and block level only |
| L5 | **Concurrent local runs.** Skyvern's in-memory mode cannot run tasks concurrently (shared SQLite session errors, independent of tracing). Parallel-run isolation is verified against a fake Skyvern server. | - |
| L6 | **CUA engines** (`openai_cua`, `anthropic_cua`, `ui_tars`) call their provider SDKs directly, not litellm, so their model calls are not chat spans here (the provider's own Respan instrumentation can log them). Steps and actions are still traced. Not verified live (needs provider keys). | no prompt-named LLM spans for CUA engines |
| L10 | **`run_task(wait_for_completion=False)`** returns while the run is `created`; the run span records that status. Locally Skyvern has in fact finished by then. Use `wait_for_completion=True` (or `page.agent.run_task`, which waits) to record the final status and output. | run span shows `created` |
| L11 | The `skyvern_v3` engine runs one tool-calling loop, not agent steps: its trace is the run span plus one LLM span per loop turn (prompt `taskv3-agent-loop`, tool calls in the completion), without step or action spans. | flatter tree for v3 |
| L7 | Respan names task spans by kind only, so steps and blocks show as `task` in the span list; the step number is in metadata (`skyvern.step.number`), the block label in `skyvern.block.label`. | steps are told apart by metadata |
| L8 | Token counts come from litellm's `usage`. Anthropic models through litellm report cache writes inside `prompt_tokens`; cache reads are reported separately as cached tokens. | none expected; listed for audits |
| L9 | Version pin `skyvern >=1.0.56,<2`. The wrapped seams (`ForgeAgent.agent_step`, `ActionHandler.handle_action`, `Block.execute_safe`, `run_sdk_action`) are internal to Skyvern and can move; missing seams are skipped, never fatal. | partial trees on a future Skyvern that moves a seam |

## Verification

- **Baseline** (`tests/test_baseline.py`, runs in CI): Skyvern's real in-process
  engine and headless Chromium on a local page, a scripted model behind
  `litellm.acompletion`, spans captured through the real Respan pipeline. Asserts one
  root per run and that every exported span's parent is an exported span (Skyvern's
  background task and its own spans in between), customer metadata flat on every
  span, customer id, grouping defaults and overrides, parallel-run isolation, content
  off, failed actions as error spans, and secret masking. Each assertion was checked
  to fail with the corresponding code removed.
- **Live scenarios** (`python-sdks/examples/skyvern`): verified live against a
  Respan environment, with every trace read back through the Respan logs API.
