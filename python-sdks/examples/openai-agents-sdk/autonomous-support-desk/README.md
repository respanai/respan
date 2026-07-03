# Autonomous Support Desk - OpenAI Agents SDK × Respan

A standalone demo: an **autonomous AI support agent** for a multi-tenant help desk,
built on the **OpenAI Agents SDK** and instrumented with **Respan**. A customer
ticket arrives and a single self-orchestrating agent decides - on its own - what to
do: search the knowledge base, look up an order/account, issue a refund, resolve
directly, or hand off to a billing specialist who escalates. Every decision is a
tool call, every call routes through the Respan gateway, and each ticket becomes
**one nested trace**.

In one run it exercises **five Respan services together** - Tracing, Gateway,
Embeddings, Prompt management, and Datasets - plus a catalog of saved **Views** over
a genuinely diverse set of traces (different paths, costs, and outcomes per ticket).

## Prerequisites

- **Python 3.11-3.13** (the tracing SDK requires `>=3.11,<3.14`).
- A **Respan account + API key** - https://platform.respan.ai/settings/api-keys
- The gateway account needs **credits / a provider key**, and these models enabled:
  `gpt-4.1-nano`, `gpt-4.1-mini`, `text-embedding-3-small`.

## Quickstart

```bash
git clone <this-repo> && cd openai-agents-demo

python3.13 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env            # then paste your RESPAN_API_KEY into .env

python seed_prompts.py          # run once: creates the managed prompts in your
                                # account and writes prompt_ids.json

python web/app.py               # open http://127.0.0.1:8000 → "Run tickets"
```

`seed_prompts.py` is idempotent (it dedupes by prompt name) and writes a
git-ignored `prompt_ids.json` so the demo uses your account's prompt IDs - no
hand-editing. If you skip it, the demo still runs: it resolves the prompts by name,
and falls back to built-in inline prompts if it can't find them.

## Ways to run it

```bash
python web/app.py        # visual demo: animated cards, one per ticket  (recommended)
python run_demo.py       # batch: run 4 tickets across 2 tenants, print outcomes
python run_stream.py     # terminal animation of the per-step events (+ events.ndjson)
python export_dataset.py # run, then export the handled tickets into a Respan dataset
```

Then open **https://platform.respan.ai** → Traces (one nested trace per ticket).

### Visual demo (`web/app.py`)

A single page that animates each ticket through its steps (agent → tools → handoff →
resolution), styled like the Respan example cards. **No frontend build / no npm** -
it streams events over Server-Sent Events from `support_desk.stream`. Each step is
tagged with the Respan service it uses; the top links jump to that service's platform
page; and each card has a **copy `trace_id`** button so you can paste it into Traces
and land on that exact ticket.

## The five services

| Service | Where it shows up |
|---|---|
| **Tracing** | Agents SDK + instrumentation → one nested trace per ticket (cost + tokens) |
| **Gateway** | `AsyncOpenAI(base_url=…/api)` - all model + embedding traffic routed + traced |
| **Embeddings** | `search_kb` makes a direct `embeddings.create` call (`log_type=embedding`) |
| **Prompt mgmt** | agent instructions come from a deployed, versioned managed prompt |
| **Datasets** | `export_dataset.py` bulk-exports handled tickets as dataset rows |

## Architecture

```
   Scenarios: 2 tenants, varied tickets ──asyncio.gather──► run_ticket (one per ticket)
                                                              │  propagate_attributes + trace()
                                                              ▼
        ┌──────────  Autonomous Support Agent  (OpenAI Agents SDK)  ──────────┐
        │  instructions ← managed prompt (rendered per tenant)                │
        │   reason ─► tool ─┬─ search_kb ─► embeddings.create   (③ Embeddings) │
        │     ▲             ├─ lookup_order / lookup_account                   │
        │     │ (loop)      ├─ process_refund     → refunded                   │
        │     └─────────────┼─ close_resolved     → self_resolved             │
        │                   └─ handoff ─► Billing Specialist ─► escalate_to_human
        └───────────────────────────────┬─────────────────────────────────────┘
                                        │ every llm / tool / embedding call
                                        ▼
                              Respan Gateway  ──►  Respan platform
                                                   (Tracing · Prompt mgmt · Datasets · Views)
```

One ticket's trace nests as:
`support-session.workflow → agent → turn → openai.chat / tool` (with `handoff.task →
billing_specialist.agent` when it escalates). The emergent outcome is stamped onto the
trace, so it's filterable by `metadata__resolution`.

## Views catalog

See **[docs/VIEWS.md](docs/VIEWS.md)** - saved Views with exact, verified filters (per
tenant, cost, outcome, refund actions, KB searches, embeddings, prompt traffic, handoffs).
There's no Views API, so each is saved by hand in the UI (Traces/Logs → Filter → Save as
view).

## How it works

- **Autonomous, not a fixed pipeline.** One agent decides per ticket; because each
  decision is a tool call, the autonomy leaves filterable breadcrumbs - and the traces
  genuinely differ (different tools, cost, and outcomes), which is what makes the Views
  worthwhile.
- **One trace per ticket.** `run_ticket` opens `propagate_attributes(...)` (customer,
  tenant, prompt) + an Agents `trace()`, runs the agent, and stamps the resolution at
  `trace.finish()`.
- **Mocked backends, real AI.** The agent and its LLM/embedding calls are real (through
  the gateway); the tools return canned data so the demo runs deterministically without
  wiring real backends.

## Notes

- Depends on the OpenAI-Agents instrumentation **nested-trace / cost fix**
  (`respan-instrumentation-openai-agents>=1.1.1`) - without it, Agents-SDK traces render
  flat and cost reads `$0`.
- The dataset is created but not consumed (evals/experiments are out of scope) - it's the
  seed you'd run evals against next.
