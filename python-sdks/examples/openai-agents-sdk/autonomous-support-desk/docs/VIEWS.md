# Views catalog - the filtering deliverable (plan §7, build step 10)

Saved **Views** are how the demo turns a pile of diverse traces into a navigable
story. There is **no API to create Views** in the current SDK (`/api/views/`,
`/api/saved-views/` both 404), so each View is saved by hand in the platform UI:

> **Traces** or **Logs** page → **Filter** → enter the filter below → **Save as view** → name it.

Every filter here is **verified against live demo data** - the ✓ count is what it
returned at the time of writing (counts scale with how many times you run
`run_demo.py`; each run = 4 tickets across 2 tenants).

**Filter syntax:** `field:operator:value` (empty operator = exact match). Custom
dimensions use `metadata__<key>` (double underscore). Operators: *(exact)*, `not`,
`gt`/`gte`/`lt`/`lte`, `contains`/`icontains`, `startswith`/`endswith`, `in`,
`isnull`, `iexact`.

---

## Trace-level Views (Traces page)

| # | View | Filter | ✓ | Shows |
|---|------|--------|---|-------|
| 1 | **One tenant - Acme** | `customer_identifier::acme-retail` | 16 | per-tenant isolation |
| 2 | **One tenant - Globex** | `customer_identifier::globex-saas` | ✓ | per-tenant isolation |
| 3 | **Expensive tickets** | `total_cost:gt:0.0003` | 4 | cost-per-ticket actually varies |
| 4 | **Cost ranking** | *(no filter)* sort by `-total_cost` | ✓ | most expensive tickets first |
| 5 | **Refunded tickets** | `metadata__resolution::refunded` | 4 | outcome filtering |
| 6 | **Escalated tickets** | `metadata__resolution::escalated` | 2 | outcome filtering |
| 7 | **Self-resolved tickets** | `metadata__resolution::self_resolved` | 2 | outcome filtering |
| 8 | **Traffic on a prompt** | `metadata__prompt_id::08edfb89740e47ffa69c025d59a6d29f` | 17 | prompt-management traffic |
| 9 | **Errored tickets** | `error_count:gt:0` | 0* | error triage (empty until an error occurs) |

\* No errors in the happy-path demo - the View is correct, it just has nothing to
show yet. Same for a **Slow tickets** view (`duration:gt:<seconds>`): set the
threshold from the cost-ranking view.

## Span-level Views (Logs page)

| # | View | Filter | ✓ | Shows |
|---|------|--------|---|-------|
| 10 | **Refund actions** | `span_name::process_refund.tool` | 4 | every refund the agents issued |
| 11 | **KB-search calls** | `span_name::search_kb.tool` | 2 | knowledge-base lookups |
| 12 | **All tool calls** | `log_type::tool` | 17 | every decision the agents made |
| 13 | **Embeddings only** | `log_type::embedding` | 2 | the embedding service in action |
| 14 | **Per-agent: specialist** | `metadata__agent_name::billing_specialist` | 14 | the specialist's agent + turn spans |
| 15 | **Handoffs** | `span_name::handoff.task` | 6 | tickets routed support → specialist |

View 14 scopes to the specialist's **agent and turn** spans (they carry
`agent_name`); the specialist's leaf LLM/tool spans don't, so a full flat-subtree
filter would still need the optional span processor (plan §4.5). Good enough to
isolate "what the specialist handled."

---

## Notes & gotchas (learned while verifying)

- **Tool span names carry a `.tool` suffix.** The emitter names function-tool spans
  `<name>.tool`, so the exact filter is `span_name::process_refund.tool`, *not*
  `process_refund`. `span_name:startswith:process_refund` also works.
- **Embedding spans have an empty `span_name`** - filter them by `log_type::embedding`,
  not by name.
- **`metadata__prompt_id` filters; `metadata__prompt_version_number` does not.** The
  numeric-string version value (`"1"`) doesn't match a trace-level `::1` filter. Use
  `prompt_id` for the "traffic on a prompt" view. (Residual to investigate.)
- **No boolean metadata values** - they cause a backend error. `resolution` is a
  string enum (`refunded` / `escalated` / `self_resolved`) by design.
- **`logs list --json` nests results under `response.results`** (traces list is flat
  `results`) - relevant if you script around these filters.
- **Resolution is a per-trace stamp** (set at `trace.finish()`); there is no native
  "trace contains span X" filter, which is why outcomes are filtered at trace level
  and actions (refund/KB) at span level.
