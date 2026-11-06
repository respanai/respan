// Sends this session's prompts, responses, tool calls, and token usage to Respan as traces.
// This plugin holds nothing else, so installing it is the user's opt-in.
//
// Tracing must never get in the way of the session, so hook bodies only record and exports
// are not awaited. Only session.end waits, briefly, so a `claude -p` run can finish sending.
// $.http.fetch cannot be cancelled: if Respan hangs, a `claude -p` process stays alive until
// that call's 30-second timeout. Interactive exit does not wait for it.

import { buildBodies } from './lib/otlp.mjs';
import { SessionTracer } from './lib/tracer.mjs';

// The same two deployments `respan auth login` offers.
const INGEST_URLS = {
  platform: 'https://api.respan.ai/api/v2/traces',
  enterprise: 'https://endpoint.respan.ai/api/v2/traces',
};
const FLUSH_INTERVAL_MS = 10_000;
const SHUTDOWN_WAIT_MS = 1_000;
// $.http.fetch refuses a request body over 4 MiB.
const MAX_BODY_CHARS = 3 * 1024 * 1024;
// What Respan's absence can cost in memory: the oldest unsent bodies go first.
const MAX_QUEUED_CHARS = 8 * 1024 * 1024;

function createState(options) {
  return {
    options: options ?? {},
    enabled: undefined, // undefined until resolved
    resolving: null,
    tracer: null,
    apiKey: '',
    url: '',
    version: '',
    queue: [],
    sending: null,
    authFailed: false,
  };
}

async function readJsonFile($, path) {
  try {
    return JSON.parse(await $.fs.read(path));
  } catch {
    return null;
  }
}

/** The non-secret per-project settings `.claude/respan.json` holds. Unknown keys become metadata. */
async function readProjectConfig($) {
  const raw = await readJsonFile($, '.claude/respan.json');
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) return {};
  // base_url is dropped on purpose: a repository must not choose where traces and the key go.
  const { customer_id: customerId, span_name: spanName, workflow_name: workflowName, base_url: ignored, ...metadata } = raw;
  return { customerId, spanName, workflowName, metadata };
}

async function resolveEnabled($, state) {
  const { options } = state;
  // A repository can set environment variables through its settings, so the environment can
  // turn tracing off, and never names a key or a destination.
  if ((await $.env.get('RESPAN_TRACING')) === 'false') return false;

  state.apiKey = options.api_key || '';
  if (!state.apiKey) {
    $.ui.log('No Respan API key is set, so sessions are not being traced. Add one with /plugin configure.');
    return false;
  }

  const project = await readProjectConfig($);
  state.tracer = new SessionTracer(project);
  state.url = INGEST_URLS[options.deployment] ?? INGEST_URLS.platform;
  state.version = (await $.session.version())?.version ?? '';
  // The hook that `respan integrate claude-code` installed sends only when this is true.
  // Without this, a machine that still has that hook would report every turn twice.
  await $.env.set('TRACE_TO_RESPAN', 'false');
  return true;
}

/** Concurrent hooks share one resolution, so no event is recorded by a second tracer. */
async function isEnabled($, state) {
  if (state.enabled !== undefined) return state.enabled;
  state.resolving ??= resolveEnabled($, state).catch(() => false);
  state.enabled = await state.resolving;
  return state.enabled;
}

/** Sends one body. Resolves to whether the queue can move on from it. */
async function postBody($, state, body) {
  try {
    const response = await $.http.fetch(state.url, {
      method: 'POST',
      headers: { Authorization: `Bearer ${state.apiKey}`, 'Content-Type': 'application/json' },
      body,
    });
    if (response.ok) return true;
    if (response.status === 401 || response.status === 403) {
      $.ui.log(`Respan rejected the API key (HTTP ${response.status}). Traces are not being sent.`);
      state.authFailed = true;
      state.queue = [];
      return true;
    }
    if (response.status < 500 && response.status !== 429) {
      $.ui.log(`Respan rejected a trace batch (HTTP ${response.status})`, { to: 'debug' });
      return true;
    }
  } catch (error) {
    $.ui.log(`Could not reach Respan: ${error?.message ?? error}`, { to: 'debug' });
  }
  return false;
}

/** One request at a time, so an outage costs one request per flush and not the whole backlog. */
async function drainQueue($, state) {
  while (state.queue.length > 0) {
    // Out of the queue while in flight, so trimming the queue can't drop or repeat it.
    const body = state.queue.shift();
    if (!(await postBody($, state, body))) {
      if (!state.authFailed) state.queue.unshift(body);
      return;
    }
  }
}

function trimQueue(state) {
  let chars = state.queue.reduce((total, body) => total + body.length, 0);
  while (chars > MAX_QUEUED_CHARS && state.queue.length > 1) chars -= state.queue.shift().length;
}

/** Start sending finished spans and earlier failures. Never waits for the network. */
function flush($, state) {
  if (!state.enabled || state.authFailed) return;
  state.queue.push(...buildBodies(state.tracer.drain(), { maxBodyChars: MAX_BODY_CHARS, serviceVersion: state.version }));
  trimQueue(state);
  if (state.sending || state.queue.length === 0) return;
  state.sending = drainQueue($, state).finally(() => {
    state.sending = null;
  });
}

function record($, action) {
  try {
    action();
  } catch (error) {
    $.ui.log(`Tracing error: ${error?.message ?? error}`, { to: 'debug' });
  }
}

export function register(on, options) {
  const state = createState(options);

  on('session.start', async ($, e, next) => {
    if (await isEnabled($, state)) {
      $.clock.every(FLUSH_INTERVAL_MS, () => flush($, state));
    }
    return next(e);
  });

  on('prompt.compose', async ($, e, next) => {
    const result = await next(e);
    if (await isEnabled($, state)) record($, () => state.tracer.onPromptComposed(result?.sections));
    return result;
  });

  on('turn.start', async ($, e, next) => {
    if (await isEnabled($, state)) {
      const sessionId = await $.session.id();
      record($, () => {
        state.tracer.setSessionId(sessionId);
        state.tracer.onTurnStart(Date.now(), e);
      });
    }
    return next(e);
  });

  on('session.append', async ($, e, next) => {
    if (await isEnabled($, state)) record($, () => state.tracer.onAppend(e));
    return next(e);
  });

  on('agent.spawn', async ($, e, next) => {
    if (!(await isEnabled($, state))) return next(e);
    let pending = null;
    record($, () => {
      pending = state.tracer.onAgentSpawning(e);
    });
    const result = await next(e);
    record($, () => state.tracer.onAgentSpawned(pending, result));
    return result;
  });

  on('turn.step', async function* ($, e, next) {
    if (!(await isEnabled($, state))) return yield* next(e);
    let step = null;
    record($, () => {
      step = state.tracer.onStepStart(Date.now(), e);
    });
    const stream = next(e);
    let item;
    try {
      item = await stream.next();
      while (!item.done) {
        const chunk = item.value;
        record($, () => state.tracer.onStepChunk(step, Date.now(), chunk));
        yield chunk;
        item = await stream.next();
      }
    } finally {
      if (!item?.done) await stream.return?.();
    }
    const result = item.value;
    record($, () => state.tracer.onStepEnd(step, Date.now(), result));
    return result;
  });

  on('tool.call', async ($, e, next) => {
    if (!(await isEnabled($, state))) return next(e);
    let call = null;
    record($, () => {
      call = state.tracer.onToolStart(Date.now(), e);
    });
    const result = await next(e);
    record($, () => state.tracer.onToolEnd(call, Date.now(), result));
    return result;
  });

  on('turn.complete', async ($, e, next) => {
    const result = await next(e);
    if (await isEnabled($, state)) {
      record($, () => state.tracer.onTurnComplete(Date.now(), e));
      // Serializing a long turn's spans takes a few milliseconds; do it after the event is done.
      $.clock.after(0, () => flush($, state));
    }
    return result;
  });

  on('session.end', async ($, e, next) => {
    if (await isEnabled($, state)) {
      flush($, state);
      if (state.sending) await Promise.race([state.sending, $.clock.sleep(SHUTDOWN_WAIT_MS)]);
    }
    return next(e);
  });
}
