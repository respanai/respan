/**
 * Shared utilities for Respan CLI hooks.
 *
 * Provides auth loading, config loading, state management, span construction,
 * and API submission. Used by claude-code, codex-cli, and gemini-cli hooks.
 */
import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';

// Resolve the v2/traces endpoint (OTLP JSON format)
export function resolveTracesEndpoint(baseUrl?: string): string {
  const DEFAULT = 'https://api.respan.ai/api/v2/traces';
  if (!baseUrl) return DEFAULT;
  const normalized = baseUrl.replace(/\/+$/, '');
  if (normalized.endsWith('/api')) return `${normalized}/v2/traces`;
  return `${normalized}/api/v2/traces`;
}

// Keep old name for backward compat in gemini-cli detached senders
export const resolveTracingIngestEndpoint = resolveTracesEndpoint;

// ── Types ─────────────────────────────────────────────────────────

export interface RespanConfig {
  fields: Record<string, string>;
  properties: Record<string, string>;
}

export interface Credentials {
  apiKey: string;
  baseUrl: string;
}

export interface SpanData {
  trace_unique_id: string;
  span_unique_id: string;
  span_parent_id?: string;
  span_name: string;
  span_workflow_name: string;
  span_path?: string;
  session_identifier?: string;
  thread_identifier?: string;
  customer_identifier?: string;
  model?: string;
  provider_id?: string;
  input?: string;
  output?: string;
  timestamp: string;
  start_time: string;
  latency?: number;
  metadata?: Record<string, unknown>;
  prompt_tokens?: number;
  completion_tokens?: number;
  total_tokens?: number;
  prompt_tokens_details?: Record<string, number>;
  prompt_messages?: Array<Record<string, unknown>>;
  completion_message?: Record<string, unknown> | null;
  // Required Respan platform fields
  warnings?: string;
  encoding_format?: string;
  disable_fallback?: boolean;
  respan_params?: Record<string, unknown>;
  field_name?: string;
  delimiter?: string;
  disable_log?: boolean;
  request_breakdown?: boolean;
}

// ── Logging ───────────────────────────────────────────────────────

let _logFile: string | null = null;
let _debug = false;

export function initLogging(logFile: string, debug: boolean): void {
  _logFile = logFile;
  _debug = debug;
  const dir = path.dirname(logFile);
  fs.mkdirSync(dir, { recursive: true });
}

export function log(level: string, message: string): void {
  if (!_logFile) return;
  const ts = new Date().toISOString().replace('T', ' ').slice(0, 19);
  fs.appendFileSync(_logFile, `${ts} [${level}] ${message}\n`);
}

export function debug(message: string): void {
  if (_debug) log('DEBUG', message);
}

// ── Credentials ───────────────────────────────────────────────────

const DEFAULT_BASE_URL = 'https://api.respan.ai/api';

export function resolveCredentials(): Credentials | null {
  let apiKey = process.env.RESPAN_API_KEY ?? '';
  let baseUrl = process.env.RESPAN_BASE_URL ?? DEFAULT_BASE_URL;

  if (!apiKey) {
    const credsFile = path.join(os.homedir(), '.respan', 'credentials.json');
    if (fs.existsSync(credsFile)) {
      try {
        const creds = JSON.parse(fs.readFileSync(credsFile, 'utf-8'));
        const configFile = path.join(os.homedir(), '.respan', 'config.json');
        let profile = 'default';
        if (fs.existsSync(configFile)) {
          const cfg = JSON.parse(fs.readFileSync(configFile, 'utf-8'));
          profile = cfg.activeProfile ?? 'default';
        }
        const cred = creds[profile] ?? {};
        apiKey = cred.apiKey ?? cred.accessToken ?? '';
        if (!baseUrl || baseUrl === DEFAULT_BASE_URL) {
          baseUrl = cred.baseUrl ?? baseUrl;
        }
        // Ensure base_url ends with /api
        if (baseUrl && !baseUrl.replace(/\/+$/, '').endsWith('/api')) {
          baseUrl = baseUrl.replace(/\/+$/, '') + '/api';
        }
        if (apiKey) {
          debug(`Using API key from credentials.json (profile: ${profile})`);
        }
      } catch (e) {
        debug(`Failed to read credentials.json: ${e}`);
      }
    }
  }

  if (!apiKey) return null;
  return { apiKey, baseUrl };
}

// ── Config loading ────────────────────────────────────────────────

const KNOWN_CONFIG_KEYS = new Set([
  'customer_id', 'span_name', 'workflow_name', 'base_url', 'project_id',
]);

export function loadRespanConfig(configPath: string): RespanConfig {
  if (!fs.existsSync(configPath)) {
    return { fields: {}, properties: {} };
  }
  try {
    const raw = JSON.parse(fs.readFileSync(configPath, 'utf-8'));
    if (typeof raw !== 'object' || raw === null || Array.isArray(raw)) {
      return { fields: {}, properties: {} };
    }
    const fields: Record<string, string> = {};
    const properties: Record<string, string> = {};
    for (const [k, v] of Object.entries(raw)) {
      if (KNOWN_CONFIG_KEYS.has(k)) {
        fields[k] = String(v);
      } else {
        properties[k] = String(v);
      }
    }
    return { fields, properties };
  } catch (e) {
    debug(`Failed to load config from ${configPath}: ${e}`);
    return { fields: {}, properties: {} };
  }
}

// ── State management ──────────────────────────────────────────────

export function loadState(statePath: string): Record<string, unknown> {
  if (!fs.existsSync(statePath)) return {};
  try {
    return JSON.parse(fs.readFileSync(statePath, 'utf-8'));
  } catch {
    return {};
  }
}

export function saveState(statePath: string, state: Record<string, unknown>): void {
  const dir = path.dirname(statePath);
  fs.mkdirSync(dir, { recursive: true });
  const tmpPath = statePath + '.tmp.' + process.pid;
  try {
    fs.writeFileSync(tmpPath, JSON.stringify(state, null, 2));
    fs.renameSync(tmpPath, statePath);
  } catch (e) {
    try { fs.unlinkSync(tmpPath); } catch {}
    // Fallback to direct write
    fs.writeFileSync(statePath, JSON.stringify(state, null, 2));
  }
}

// ── File locking ──────────────────────────────────────────────────

/**
 * Simple advisory file lock using mkdir (atomic on all platforms).
 * Returns an unlock function, or null if lock couldn't be acquired.
 */
export function acquireLock(lockPath: string, timeoutMs = 5000): (() => void) | null {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    try {
      fs.mkdirSync(lockPath);
      return () => {
        try { fs.rmdirSync(lockPath); } catch {}
      };
    } catch {
      // Lock held by another process, wait
      const waitMs = Math.min(100, deadline - Date.now());
      if (waitMs > 0) {
        Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, waitMs);
      }
    }
  }
  debug('Could not acquire lock within timeout, proceeding without lock');
  return () => {}; // No-op unlock
}

// ── Timestamp helpers ─────────────────────────────────────────────

export function nowISO(): string {
  return new Date().toISOString();
}

export function parseTimestamp(ts: string): Date | null {
  try {
    const d = new Date(ts);
    return isNaN(d.getTime()) ? null : d;
  } catch {
    return null;
  }
}

export function latencySeconds(start: string, end: string): number | undefined {
  const s = parseTimestamp(start);
  const e = parseTimestamp(end);
  if (s && e) return (e.getTime() - s.getTime()) / 1000;
  return undefined;
}

// ── Truncation ────────────────────────────────────────────────────

export function truncate(text: string, maxChars = 4000): string {
  if (text.length <= maxChars) return text;
  return text.slice(0, maxChars) + '\n... (truncated)';
}

// ── Span construction helpers ─────────────────────────────────────

/**
 * No-op — v1 platform defaults are no longer needed with OTLP v2 format.
 * Kept for API compatibility with hook files.
 */
export function addDefaults(span: SpanData): SpanData {
  return span;
}

export function addDefaultsToAll(spans: SpanData[]): SpanData[] {
  return spans;
}

/**
 * Resolve config overrides for span fields. Env vars take precedence over config file.
 */
export function resolveSpanFields(config: RespanConfig | null, defaults: {
  workflowName: string;
  spanName: string;
}): { workflowName: string; spanName: string; customerId: string } {
  const fields = config?.fields ?? {};
  return {
    workflowName: process.env.RESPAN_WORKFLOW_NAME ?? fields.workflow_name ?? defaults.workflowName,
    spanName: process.env.RESPAN_SPAN_NAME ?? fields.span_name ?? defaults.spanName,
    customerId: process.env.RESPAN_CUSTOMER_ID ?? fields.customer_id ?? '',
  };
}

/**
 * Build metadata from config properties + env overrides.
 */
export function buildMetadata(
  config: RespanConfig | null,
  base: Record<string, unknown> = {},
): Record<string, unknown> {
  const metadata: Record<string, unknown> = { ...base };
  if (config?.properties) {
    Object.assign(metadata, config.properties);
  }
  const envMetadata = process.env.RESPAN_METADATA;
  if (envMetadata) {
    try {
      const extra = JSON.parse(envMetadata);
      if (typeof extra === 'object' && extra !== null) {
        Object.assign(metadata, extra);
      }
    } catch {}
  }
  return metadata;
}

// ── OTLP JSON conversion ──────────────────────────────────────────

/** Convert a JS value to an OTLP attribute value object. */
function toOtlpValue(value: unknown): Record<string, unknown> | null {
  if (value === null || value === undefined) return null;
  if (typeof value === 'string') return { stringValue: value };
  if (typeof value === 'number') {
    return Number.isInteger(value) ? { intValue: String(value) } : { doubleValue: value };
  }
  if (typeof value === 'boolean') return { boolValue: value };
  if (Array.isArray(value)) {
    const values = value.map(toOtlpValue).filter(Boolean);
    return { arrayValue: { values } };
  }
  if (typeof value === 'object') {
    // Convert object to kvlist
    const values = Object.entries(value as Record<string, unknown>)
      .map(([k, v]) => {
        const converted = toOtlpValue(v);
        return converted ? { key: k, value: converted } : null;
      })
      .filter(Boolean);
    return { kvlistValue: { values } };
  }
  return { stringValue: String(value) };
}

/** Convert a flat key-value map to OTLP attribute list. */
function toOtlpAttributes(attrs: Record<string, unknown>): Array<{ key: string; value: unknown }> {
  const result: Array<{ key: string; value: unknown }> = [];
  for (const [key, value] of Object.entries(attrs)) {
    if (value === null || value === undefined) continue;
    const converted = toOtlpValue(value);
    if (converted) result.push({ key, value: converted });
  }
  return result;
}

/** Convert ISO timestamp to nanosecond string. */
function isoToNanos(iso: string): string {
  const d = new Date(iso);
  if (isNaN(d.getTime())) return '0';
  return String(BigInt(d.getTime()) * 1_000_000n);
}

/** Generate a 32-char hex trace ID from a string (MD5-like hash). */
function stringToTraceId(s: string): string {
  let hash = 0;
  for (let i = 0; i < s.length; i++) {
    hash = ((hash << 5) - hash + s.charCodeAt(i)) | 0;
  }
  // Produce 32 hex chars from two different hash seeds
  let hash2 = 0;
  for (let i = s.length - 1; i >= 0; i--) {
    hash2 = ((hash2 << 7) - hash2 + s.charCodeAt(i)) | 0;
  }
  const hex1 = (hash >>> 0).toString(16).padStart(8, '0');
  const hex2 = (hash2 >>> 0).toString(16).padStart(8, '0');
  // Use both hashes + length-based component for 32 chars
  const hex3 = (s.length * 2654435761 >>> 0).toString(16).padStart(8, '0');
  const hex4 = ((hash ^ hash2) >>> 0).toString(16).padStart(8, '0');
  return (hex1 + hex2 + hex3 + hex4).slice(0, 32);
}

/** Generate a 16-char hex span ID from a string. */
function stringToSpanId(s: string): string {
  let hash = 0;
  for (let i = 0; i < s.length; i++) {
    hash = ((hash << 5) - hash + s.charCodeAt(i)) | 0;
  }
  const hex1 = (hash >>> 0).toString(16).padStart(8, '0');
  let hash2 = 0;
  for (let i = s.length - 1; i >= 0; i--) {
    hash2 = ((hash2 << 3) - hash2 + s.charCodeAt(i)) | 0;
  }
  const hex2 = (hash2 >>> 0).toString(16).padStart(8, '0');
  return (hex1 + hex2).slice(0, 16);
}

/** Convert SpanData[] to OTLP JSON payload for /v2/traces. */
export function toOtlpPayload(spans: SpanData[]): Record<string, unknown> {
  const otlpSpans = spans.map((span) => {
    // Build OTEL-compatible attributes from SpanData fields
    const attrs: Record<string, unknown> = {};

    // Respan-specific attributes
    if (span.session_identifier) attrs['respan.sessions.session_identifier'] = span.session_identifier;
    if (span.thread_identifier) attrs['respan.threads.thread_identifier'] = span.thread_identifier;
    if (span.customer_identifier) attrs['respan.customer_params.customer_identifier'] = span.customer_identifier;
    if (span.span_workflow_name) attrs['traceloop.workflow.name'] = span.span_workflow_name;
    if (span.span_path) attrs['traceloop.entity.path'] = span.span_path;

    // Span kind mapping
    const isRoot = !span.span_parent_id;
    const isLlm = span.span_name.includes('.chat');
    const isTool = span.span_name.startsWith('Tool:');
    const isThinking = span.span_name.startsWith('Thinking') || span.span_name === 'Reasoning';

    if (isLlm) {
      attrs['traceloop.span.kind'] = 'task';
      attrs['llm.request.type'] = 'chat';
    } else if (isTool) {
      attrs['traceloop.span.kind'] = 'tool';
    } else if (isRoot) {
      attrs['traceloop.span.kind'] = 'workflow';
    } else if (isThinking) {
      attrs['traceloop.span.kind'] = 'task';
    }

    // Model and provider
    if (span.model) attrs['gen_ai.request.model'] = span.model;
    if (span.provider_id) attrs['gen_ai.system'] = span.provider_id;

    // Token usage
    if (span.prompt_tokens !== undefined) attrs['gen_ai.usage.prompt_tokens'] = span.prompt_tokens;
    if (span.completion_tokens !== undefined) attrs['gen_ai.usage.completion_tokens'] = span.completion_tokens;
    if (span.total_tokens !== undefined) attrs['llm.usage.total_tokens'] = span.total_tokens;

    // Input/output as traceloop entity fields
    if (span.input) attrs['traceloop.entity.input'] = span.input;
    if (span.output) attrs['traceloop.entity.output'] = span.output;

    // Metadata as respan.metadata JSON
    if (span.metadata && Object.keys(span.metadata).length > 0) {
      attrs['respan.metadata'] = JSON.stringify(span.metadata);
    }

    // Environment
    attrs['respan.entity.log_method'] = 'ts_tracing';

    // Compute start/end nanos
    const startNanos = isoToNanos(span.start_time);
    let endNanos = isoToNanos(span.timestamp);
    // If start == end and we have latency, compute end
    if (startNanos === endNanos && span.latency && span.latency > 0) {
      const startMs = new Date(span.start_time).getTime();
      endNanos = String(BigInt(Math.round(startMs + span.latency * 1000)) * 1_000_000n);
    }

    const otlpSpan: Record<string, unknown> = {
      traceId: stringToTraceId(span.trace_unique_id),
      spanId: stringToSpanId(span.span_unique_id),
      name: span.span_name,
      kind: isLlm ? 3 : 1, // 3=CLIENT (LLM calls), 1=INTERNAL
      startTimeUnixNano: startNanos,
      endTimeUnixNano: endNanos,
      attributes: toOtlpAttributes(attrs),
      status: { code: 1 }, // STATUS_CODE_OK
    };

    if (span.span_parent_id) {
      otlpSpan.parentSpanId = stringToSpanId(span.span_parent_id);
    }

    return otlpSpan;
  });

  return {
    resourceSpans: [{
      resource: {
        attributes: toOtlpAttributes({
          'service.name': 'respan-cli-hooks',
        }),
      },
      scopeSpans: [{
        scope: { name: 'respan-cli-hooks', version: '0.7.1' },
        spans: otlpSpans,
      }],
    }],
  };
}

// ── API submission ────────────────────────────────────────────────

export async function sendSpans(
  spans: SpanData[],
  apiKey: string,
  baseUrl: string,
  context: string,
): Promise<void> {
  const url = resolveTracesEndpoint(baseUrl);
  const headers: Record<string, string> = {
    'Authorization': `Bearer ${apiKey}`,
    'Content-Type': 'application/json',
    'X-Respan-Dogfood': '1', // Anti-recursion: prevent trace loops on ingest
  };

  // Convert to OTLP JSON format for v2/traces
  const payload = toOtlpPayload(spans);
  const body = JSON.stringify(payload);
  const spanNames = spans.map(s => s.span_name);
  debug(`Sending ${spans.length} spans (${body.length} bytes) to ${url} for ${context}: ${spanNames.join(', ')}`);

  if (_debug) {
    const debugDir = _logFile ? path.dirname(_logFile) : os.tmpdir();
    const debugFile = path.join(debugDir, `respan_spans_${context.replace(/\s+/g, '_')}.json`);
    fs.writeFileSync(debugFile, body);
    debug(`Dumped OTLP payload to ${debugFile}`);
  }

  for (let attempt = 0; attempt < 3; attempt++) {
    try {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 30_000);
      const response = await fetch(url, {
        method: 'POST',
        headers,
        body,
        signal: controller.signal,
      });
      clearTimeout(timeout);

      if (response.status < 400) {
        const text = await response.text();
        debug(`Sent ${spans.length} spans for ${context} (attempt ${attempt + 1}): ${text.slice(0, 300)}`);
        return;
      }
      if (response.status < 500) {
        const text = await response.text();
        log('ERROR', `Spans rejected for ${context}: HTTP ${response.status} - ${text.slice(0, 200)}`);
        return;
      }
      // 5xx — retry
      debug(`Server error for ${context} (attempt ${attempt + 1}), retrying...`);
      await sleep(1000);
    } catch (e) {
      if (attempt < 2) {
        await sleep(1000);
      } else {
        log('ERROR', `Failed to send spans for ${context}: ${e}`);
      }
    }
  }
  log('ERROR', `Failed to send ${spans.length} spans for ${context} after 3 attempts`);
}

function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}
