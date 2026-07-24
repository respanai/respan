export interface TraceSpanRecord {
  span_name?: string;
  log_type?: string;
  model?: string;
  status?: string;
  status_code?: number;
  children?: unknown;
  [key: string]: unknown;
}

export interface TraceRecord {
  trace_unique_id?: string;
  name?: string;
  model?: string;
  span_count?: number;
  error_count?: number;
  span_tree?: unknown;
  [key: string]: unknown;
}

export interface TraceExpectations {
  minSpans?: number;
  maxErrors?: number;
  types?: Record<string, number>;
  names?: Record<string, number>;
  models?: string[];
}

export interface TraceInspection {
  traceId?: string;
  name?: string;
  spanCount: number;
  errorCount: number;
  typeCounts: Record<string, number>;
  nameCounts: Record<string, number>;
  models: string[];
}

export interface TraceAssertionIssue {
  field: string;
  expected: string;
  actual: string;
}

export interface TraceAssertionResult {
  passed: boolean;
  inspection: TraceInspection;
  issues: TraceAssertionIssue[];
}

export interface WaitForTraceOptions {
  timeoutMs: number;
  intervalMs: number;
  ready?: (trace: TraceRecord) => boolean;
}

export interface TraceFetchContext {
  abortSignal: AbortSignal;
  timeoutMs: number;
}

export class TraceWaitTimeoutError extends Error {
  readonly lastTrace?: TraceRecord;
  readonly lastError?: unknown;

  constructor(timeoutMs: number, lastTrace?: TraceRecord, lastError?: unknown) {
    super(`Trace was not ready within ${Math.ceil(timeoutMs / 1000)} seconds.`);
    this.name = 'TraceWaitTimeoutError';
    this.lastTrace = lastTrace;
    this.lastError = lastError;
  }
}

function asSpanArray(value: unknown): TraceSpanRecord[] {
  if (!Array.isArray(value)) return [];
  return value.filter(
    (item): item is TraceSpanRecord => Boolean(item) && typeof item === 'object' && !Array.isArray(item),
  );
}

export function flattenTraceSpans(trace: TraceRecord): TraceSpanRecord[] {
  const flattened: TraceSpanRecord[] = [];
  const queue = [...asSpanArray(trace.span_tree)];

  while (queue.length > 0) {
    const span = queue.shift();
    if (!span) continue;
    flattened.push(span);
    queue.push(...asSpanArray(span.children));
  }

  return flattened;
}

function increment(counts: Record<string, number>, key: string): void {
  counts[key] = (counts[key] || 0) + 1;
}

function isErrorSpan(span: TraceSpanRecord): boolean {
  const status = span.status?.toLowerCase();
  return status === 'error' || status === 'failed' || (span.status_code ?? 0) >= 400;
}

export function inspectTrace(trace: TraceRecord): TraceInspection {
  const spans = flattenTraceSpans(trace);
  const typeCounts: Record<string, number> = {};
  const nameCounts: Record<string, number> = {};
  const models = new Set<string>();

  if (trace.model) models.add(trace.model);

  for (const span of spans) {
    if (span.log_type) increment(typeCounts, span.log_type.toLowerCase());
    if (span.span_name) increment(nameCounts, span.span_name);
    if (span.model) models.add(span.model);
  }

  return {
    traceId: trace.trace_unique_id,
    name: trace.name,
    spanCount: trace.span_count ?? spans.length,
    errorCount: trace.error_count ?? spans.filter(isErrorSpan).length,
    typeCounts,
    nameCounts,
    models: [...models].sort(),
  };
}

export function evaluateTrace(
  trace: TraceRecord,
  expectations: TraceExpectations,
): TraceAssertionResult {
  const inspection = inspectTrace(trace);
  const issues: TraceAssertionIssue[] = [];

  if (expectations.minSpans !== undefined && inspection.spanCount < expectations.minSpans) {
    issues.push({
      field: 'span_count',
      expected: `>= ${expectations.minSpans}`,
      actual: String(inspection.spanCount),
    });
  }

  if (expectations.maxErrors !== undefined && inspection.errorCount > expectations.maxErrors) {
    issues.push({
      field: 'error_count',
      expected: `<= ${expectations.maxErrors}`,
      actual: String(inspection.errorCount),
    });
  }

  for (const [type, minimum] of Object.entries(expectations.types || {})) {
    const actual = inspection.typeCounts[type.toLowerCase()] || 0;
    if (actual < minimum) {
      issues.push({ field: `type:${type}`, expected: `>= ${minimum}`, actual: String(actual) });
    }
  }

  for (const [name, minimum] of Object.entries(expectations.names || {})) {
    const actual = inspection.nameCounts[name] || 0;
    if (actual < minimum) {
      issues.push({ field: `name:${name}`, expected: `>= ${minimum}`, actual: String(actual) });
    }
  }

  for (const model of expectations.models || []) {
    if (!inspection.models.includes(model)) {
      issues.push({
        field: `model:${model}`,
        expected: 'present',
        actual: inspection.models.length > 0 ? inspection.models.join(', ') : 'none',
      });
    }
  }

  return { passed: issues.length === 0, inspection, issues };
}

export function parseCountExpectations(values: string[] | undefined): Record<string, number> {
  const parsed: Record<string, number> = {};

  for (const rawValue of values || []) {
    const value = rawValue.trim();
    if (!value) throw new Error('Expectation values cannot be empty.');

    const match = value.match(/^(.*):([0-9]+)$/);
    const key = (match?.[1] || value).trim();
    const count = match ? Number(match[2]) : 1;
    if (!key || count < 1) {
      throw new Error(`Invalid expectation "${rawValue}". Use NAME or NAME:COUNT.`);
    }
    parsed[key] = Math.max(parsed[key] || 0, count);
  }

  return parsed;
}

function getStatusCode(error: unknown): number | undefined {
  if (!error || typeof error !== 'object') return undefined;
  const candidate = error as Record<string, unknown>;
  for (const key of ['statusCode', 'status', 'status_code']) {
    if (typeof candidate[key] === 'number') return candidate[key];
  }
  return undefined;
}

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

class TraceFetchDeadlineError extends Error {}

export async function waitForTrace(
  fetchTrace: (context: TraceFetchContext) => Promise<TraceRecord>,
  options: WaitForTraceOptions,
): Promise<TraceRecord> {
  const startedAt = Date.now();
  let lastTrace: TraceRecord | undefined;
  let lastError: unknown;

  while (true) {
    const elapsedBeforeFetch = Date.now() - startedAt;
    const remainingMs = options.timeoutMs - elapsedBeforeFetch;
    if (remainingMs <= 0) {
      throw new TraceWaitTimeoutError(options.timeoutMs, lastTrace, lastError);
    }

    const abortController = new AbortController();
    let deadlineTimer: ReturnType<typeof setTimeout> | undefined;
    try {
      const deadline = new Promise<never>((_resolve, reject) => {
        deadlineTimer = setTimeout(() => {
          abortController.abort();
          reject(new TraceFetchDeadlineError());
        }, remainingMs);
      });
      lastTrace = await Promise.race([
        fetchTrace({ abortSignal: abortController.signal, timeoutMs: remainingMs }),
        deadline,
      ]);
      lastError = undefined;
      if (!options.ready || options.ready(lastTrace)) return lastTrace;
    } catch (error) {
      if (error instanceof TraceFetchDeadlineError) {
        throw new TraceWaitTimeoutError(options.timeoutMs, lastTrace, lastError);
      }
      const statusCode = getStatusCode(error);
      const retryableClientStatuses = new Set([404, 408, 409, 425, 429]);
      if (
        statusCode !== undefined
        && statusCode >= 400
        && statusCode < 500
        && !retryableClientStatuses.has(statusCode)
      ) {
        throw error;
      }
      lastError = error;
    } finally {
      if (deadlineTimer) clearTimeout(deadlineTimer);
    }

    const elapsed = Date.now() - startedAt;
    if (elapsed >= options.timeoutMs) {
      throw new TraceWaitTimeoutError(options.timeoutMs, lastTrace, lastError);
    }

    await delay(Math.min(options.intervalMs, options.timeoutMs - elapsed));
  }
}
