/**
 * Traces seeder for `respan demo`.
 *
 * Ingests the supplied multi-agent service-desk traces (see trace-fixtures.ts)
 * into the authenticated account. All trace-specific logic lives here; the
 * orchestrator stays service-agnostic.
 *
 * Traces are treated as append-only run history: every `respan demo` run ingests a
 * fresh batch stamped at the current time, with freshly-generated trace/span ids.
 * Trace id + timestamps differ on every run, so runs never collide — there is no
 * dedup-skip and no teardown (`respan demo clear` leaves traces in place).
 *
 * Two workflow identifiers, on purpose:
 *   - `traceloop.workflow.name` is set to the shared constant `demo`
 *     (`span_workflow_name`, a filterable field) on every span, so one marker
 *     drives the dashboard filter and link across all demo traces.
 *   - the root span's NAME is set to the fixture's unique `workflow`
 *     (e.g. `demo.meridian-portal_outage-12`) — shown in the dashboard Workflow
 *     column, and unique per scenario. The platform dedups identical span CONTENT
 *     within a run (independent of trace id / timestamp), and the dataset repeats
 *     some scenarios with a byte-identical root span; a unique root name keeps each
 *     one distinct so all 12 ingest without a rejected span.
 *
 * Ingestion is a raw OTLP POST to `/v2/traces` (the SDK does not expose it).
 */

import { DEMO_TRACES } from './trace-fixtures.js';
import {
  Seeder,
  SeederContext,
  SeederResult,
  SeedItemResult,
  ExistingResource,
} from './types.js';
import { randomBytes } from 'node:crypto';

/** Shared, filterable workflow marker stamped on every demo span. */
const DEMO_WORKFLOW = 'demo';
/** OTLP attribute that becomes the trace's `span_workflow_name`. */
const WORKFLOW_ATTR = 'traceloop.workflow.name';

/** Spread the whole batch across the last few minutes so the traces read as fresh,
 * distinct activity and ALL land inside a "last 5 minutes" view. We target a
 * 4-minute window (under 5, with margin) and divide it evenly across the traces,
 * so the newest ends ~now and the oldest ends ~4 minutes ago. */
const SEED_WINDOW_MS = 4 * 60 * 1000;
const STAGGER_MS =
  DEMO_TRACES.length > 1 ? Math.floor(SEED_WINDOW_MS / (DEMO_TRACES.length - 1)) : 0;

/** Minimal OTLP shape we mutate during re-stamping. */
interface OtlpAttr {
  key: string;
  value: Record<string, unknown>;
}
interface RawSpan {
  traceId: string;
  spanId: string;
  parentSpanId?: string;
  name?: string;
  startTimeUnixNano: string;
  endTimeUnixNano: string;
  attributes?: OtlpAttr[];
}
interface RawBody {
  resourceSpans: { scopeSpans: { spans: RawSpan[] }[] }[];
}

/** Set (or replace) a string OTLP attribute on an attributes array. */
function setStringAttr(attrs: OtlpAttr[], key: string, value: string): void {
  const existing = attrs.find((a) => a.key === key);
  if (existing) existing.value = { stringValue: value };
  else attrs.push({ key, value: { stringValue: value } });
}

/** Resolve the OTLP ingest endpoint from the API base URL. Mirrors the hooks'
 * resolver: append `/v2/traces`, inserting `/api` when the base lacks it. */
function resolveTracesEndpoint(baseUrl: string): string {
  const normalized = (baseUrl || 'https://api.respan.ai').replace(/\/+$/, '');
  return normalized.endsWith('/api')
    ? `${normalized}/v2/traces`
    : `${normalized}/api/v2/traces`;
}

/** Deep-clone a fixture body, stamp the shared `demo` workflow attribute on every
 * span, set the root span's name to `rootName` (unique per trace — shown in the
 * Workflow column and keeps the root span from colliding with a repeated scenario),
 * assign a fresh trace id, remap every span id, and shift all timestamps so the
 * trace ends at `targetEndMs`. Bodies are pure JSON, so JSON round-trip is a safe,
 * fully-typed clone. */
function restamp(
  body: unknown,
  targetEndMs: number,
  rootName: string,
): { body: unknown; traceId: string } {
  const clone = JSON.parse(JSON.stringify(body)) as RawBody;
  const traceId = randomBytes(16).toString('hex'); // 32 hex chars
  const spans: RawSpan[] = [];
  for (const rs of clone.resourceSpans ?? []) {
    for (const ss of rs.scopeSpans ?? []) {
      for (const s of ss.spans ?? []) {
        // Shared filterable marker on every span...
        s.attributes = s.attributes ?? [];
        setStringAttr(s.attributes, WORKFLOW_ATTR, DEMO_WORKFLOW);
        // ...unique display name on the root span.
        if (!s.parentSpanId) s.name = rootName;
        spans.push(s);
      }
    }
  }

  const idMap = new Map<string, string>();
  for (const s of spans) {
    if (!idMap.has(s.spanId)) idMap.set(s.spanId, randomBytes(8).toString('hex')); // 16 hex chars
  }

  let maxEnd = 0n;
  for (const s of spans) {
    const e = BigInt(s.endTimeUnixNano);
    if (e > maxEnd) maxEnd = e;
  }
  const targetEndNanos = BigInt(Math.round(targetEndMs)) * 1_000_000n;
  const delta = targetEndNanos - maxEnd;

  for (const s of spans) {
    s.traceId = traceId;
    s.spanId = idMap.get(s.spanId) as string;
    if (s.parentSpanId) s.parentSpanId = idMap.get(s.parentSpanId) ?? s.parentSpanId;
    s.startTimeUnixNano = String(BigInt(s.startTimeUnixNano) + delta);
    s.endTimeUnixNano = String(BigInt(s.endTimeUnixNano) + delta);
  }

  return { body: clone, traceId };
}

/** POST one OTLP body to the ingest endpoint. Throws on a 4xx/5xx response, and
 * also on an OTLP `partialSuccess` that rejected spans — so a dropped span surfaces
 * as a loud error instead of a silently span-short trace. */
async function ingest(endpoint: string, authHeader: string, body: unknown): Promise<void> {
  const res = await fetch(endpoint, {
    method: 'POST',
    headers: {
      Authorization: authHeader,
      'Content-Type': 'application/json',
      'X-Respan-Dogfood': '1', // anti-recursion: don't trace the ingest itself
    },
    body: JSON.stringify(body),
  });
  const text = await res.text().catch(() => '');
  if (res.status >= 400) {
    throw new Error(`Trace ingest failed (HTTP ${res.status}): ${text.slice(0, 200)}`);
  }
  // OTLP returns 200 with a partialSuccess block when some spans are rejected.
  let rejected = 0;
  let message = '';
  try {
    const parsed = JSON.parse(text) as {
      partialSuccess?: { rejectedSpans?: number | string; errorMessage?: string };
    };
    rejected = Number(parsed.partialSuccess?.rejectedSpans ?? 0);
    message = parsed.partialSuccess?.errorMessage ?? '';
  } catch {
    // Non-JSON success body — nothing to inspect.
  }
  if (rejected > 0) {
    throw new Error(`Trace ingest rejected ${rejected} span(s)${message ? `: ${message}` : ''}.`);
  }
}

export const tracesSeeder: Seeder = {
  service: 'traces',
  label: 'Traces',

  // Traces are append-only run history — nothing to find for dedup or teardown.
  findExisting(): Promise<ExistingResource[]> {
    return Promise.resolve([]);
  },

  async run(ctx: SeederContext): Promise<SeederResult> {
    const endpoint = resolveTracesEndpoint(ctx.baseUrl);
    const now = Date.now();
    const items: SeedItemResult[] = [];
    for (let i = 0; i < DEMO_TRACES.length; i++) {
      const fixture = DEMO_TRACES[i];
      // Fresh trace id + timestamps every run (so runs never collide), staggered
      // backwards from "now", each with a unique root-span name.
      const { body, traceId } = restamp(fixture.body, now - i * STAGGER_MS, fixture.workflow);
      await ingest(endpoint, ctx.authHeader, body);
      items.push({ name: fixture.name, status: 'created', id: traceId });
    }
    return { service: this.service, label: this.label, items };
  },

  // Traces are intentionally left in place — each run is timestamped history, and
  // `respan demo clear` only removes deletable resources (prompts).
  clear(): Promise<SeederResult> {
    return Promise.resolve({ service: this.service, label: this.label, items: [] });
  },
};
