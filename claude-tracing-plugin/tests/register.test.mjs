import assert from 'node:assert/strict';
import { test } from 'node:test';

import { register } from '../hooks/register.mjs';

const TURN_ID = 'e9df4f5d-e4c2-489e-b60b-f64da87ebefe';
const ENABLED = { api_key: 'test-key', deployment: 'platform' };
const OK_RESPONSE = { status: 200, ok: true, headers: {}, text: '{}' };

/** Stands in for Claude Code: holds the mod's hooks and answers its $ calls. */
function createHost(options, { env = {}, files = {}, fetch = async () => OK_RESPONSE } = {}) {
  const hooks = new Map();
  const requests = [];
  const logs = [];
  const timers = [];
  const afterCalls = [];
  const $ = {
    env: {
      get: async (name) => env[name],
      set: async (name, value) => {
        env[name] = value;
      },
    },
    fs: {
      read: async (path) => {
        if (path in files) return files[path];
        throw new Error(`ENOENT: ${path}`);
      },
    },
    http: {
      fetch: async (url, init) => {
        requests.push({ url, init });
        return fetch(url, init);
      },
    },
    session: {
      id: async () => 'session-1',
      // The shape Claude Code 2.1.318 returned in a real run.
      version: async () => ({ version: '2.1.318', base: '2.1.318', builtAt: '2026-11-03T18:52:53Z' }),
    },
    clock: {
      every: (ms, callback) => timers.push(callback),
      after: (ms, callback) => {
        afterCalls.push(ms);
        callback();
        return { cancel() {} };
      },
      sleep: async () => {},
    },
    ui: { log: (text) => logs.push(text) },
  };
  register((name, hook) => hooks.set(name, hook), options);
  return {
    env,
    requests,
    logs,
    timers,
    afterCalls,
    fire: (name, e, answer) => hooks.get(name)($, e, async (event) => answer ?? event),
    step: (e, chunks, result) => hooks.get('turn.step')($, e, async function* () {
      yield* chunks;
      return result;
    }),
  };
}

async function runEmptyTurn(host, turnId = TURN_ID) {
  await host.fire('turn.start', { turnId, text: 'hi' });
  return host.fire('turn.complete', { turnId, answer: 'hello', reason: 'answer' }, { text: '' });
}

function spansOf(request) {
  return JSON.parse(request.init.body).resourceSpans[0].scopeSpans[0].spans;
}

test('a finished turn posts OTLP JSON with the API key', async () => {
  const host = createHost(ENABLED);
  await runEmptyTurn(host);

  assert.equal(host.requests.length, 1);
  assert.equal(host.requests[0].url, 'https://api.respan.ai/api/v2/traces');
  assert.equal(host.requests[0].init.headers.Authorization, 'Bearer test-key');
  assert.equal(spansOf(host.requests[0])[0].name, 'claude-code');
  const resource = JSON.parse(host.requests[0].init.body).resourceSpans[0].resource.attributes;
  assert.deepEqual(resource.find(({ key }) => key === 'service.version').value, { stringValue: '2.1.318' });
});

test('the deployment option picks where traces go', async () => {
  const enterprise = createHost({ api_key: 'test-key', deployment: 'enterprise' });
  await runEmptyTurn(enterprise);
  assert.equal(enterprise.requests[0].url, 'https://endpoint.respan.ai/api/v2/traces');

  const unknown = createHost({ api_key: 'test-key', deployment: 'https://attacker.test' });
  await runEmptyTurn(unknown);
  assert.equal(unknown.requests[0].url, 'https://api.respan.ai/api/v2/traces');
});

test('a Respan that never answers does not hold up the turn', async () => {
  const host = createHost(ENABLED, { fetch: () => new Promise(() => {}) });
  const result = await runEmptyTurn(host);

  assert.equal(host.requests.length, 1);
  assert.deepEqual(result, { text: '' });
});

test('session.end gives up on a Respan that never answers', async () => {
  const host = createHost(ENABLED, { fetch: () => new Promise(() => {}) });
  await runEmptyTurn(host);
  const result = await host.fire('session.end', { reason: 'other' }, {});
  assert.deepEqual(result, {});
});

test('a network failure stays out of the turn and retries on the next flush', async () => {
  let attempts = 0;
  const host = createHost(ENABLED, {
    fetch: async () => {
      attempts += 1;
      if (attempts === 1) throw new Error('network down');
      return OK_RESPONSE;
    },
  });
  assert.deepEqual(await runEmptyTurn(host), { text: '' });
  await new Promise((resolve) => setImmediate(resolve));
  await runEmptyTurn(host, 'a9df4f5d-e4c2-489e-b60b-f64da87ebefe');
  await new Promise((resolve) => setImmediate(resolve));

  // The second turn's flush re-sends the first body, then sends its own.
  assert.equal(host.requests.length, 3);
  assert.equal(host.requests[1].init.body, host.requests[0].init.body);
});

test('an outage costs one request at a time, not the whole backlog', async () => {
  const host = createHost(ENABLED, { fetch: () => new Promise(() => {}) });
  await runEmptyTurn(host);
  await runEmptyTurn(host, 'a9df4f5d-e4c2-489e-b60b-f64da87ebefe');
  await runEmptyTurn(host, 'b9df4f5d-e4c2-489e-b60b-f64da87ebefe');
  assert.equal(host.requests.length, 1);
});

test('the backlog is capped by size while Respan is down', async () => {
  const host = createHost(ENABLED, {
    fetch: async () => {
      throw new Error('network down');
    },
  });
  // Each turn's body is about 120 KB, so 80 of them exceed the 8 MiB cap.
  for (let turn = 0; turn < 80; turn += 1) {
    const turnId = `${String(turn).padStart(8, '0')}-e4c2-489e-b60b-f64da87ebefe`;
    await host.fire('turn.start', { turnId, text: 'y'.repeat(100_000) });
    await host.fire('turn.complete', { turnId, answer: 'z'.repeat(100_000), reason: 'answer' }, { text: '' });
    await new Promise((resolve) => setImmediate(resolve));
  }
  // Each flush tries once, with the oldest body still queued.
  assert.equal(host.requests.length, 80);
  assert.equal(host.requests[0].init.body.includes('00000000e4c2'), true);
  assert.equal(host.requests.at(-1).init.body.includes('00000000e4c2'), false);
  assert.equal(host.requests.at(-1).init.body.length < 200_000, true);
});

test('a body in flight is neither dropped nor sent twice when the queue is trimmed', async () => {
  let release;
  const bodies = [];
  const host = createHost(ENABLED, {
    fetch: async (url, init) => {
      bodies.push(init.body);
      if (bodies.length === 1) await new Promise((resolve) => { release = resolve; });
      return OK_RESPONSE;
    },
  });
  await runEmptyTurn(host);
  for (let turn = 1; turn < 4; turn += 1) {
    await runEmptyTurn(host, `${String(turn).padStart(8, '0')}-e4c2-489e-b60b-f64da87ebefe`);
  }
  release();
  for (let tick = 0; tick < 10; tick += 1) await new Promise((resolve) => setImmediate(resolve));
  assert.equal(bodies.length, 4);
  assert.equal(new Set(bodies).size, 4);
});

test('turn.complete leaves serializing and sending until after the event', async () => {
  const host = createHost(ENABLED);
  await runEmptyTurn(host);
  assert.deepEqual(host.afterCalls, [0]);
});

test('a rejected key is reported once and stops sending', async () => {
  const host = createHost(ENABLED, { fetch: async () => ({ ...OK_RESPONSE, status: 401, ok: false }) });
  await runEmptyTurn(host);
  await new Promise((resolve) => setImmediate(resolve));
  await runEmptyTurn(host, 'a9df4f5d-e4c2-489e-b60b-f64da87ebefe');

  assert.equal(host.requests.length, 1);
  assert.equal(host.logs.filter((line) => line.includes('rejected the API key')).length, 1);
});

test('RESPAN_TRACING=false turns tracing off', async () => {
  const host = createHost(ENABLED, { env: { RESPAN_TRACING: 'false' } });
  await runEmptyTurn(host);
  assert.deepEqual(host.requests, []);
});

test('the environment cannot supply a key or choose where traces go', async () => {
  const env = { RESPAN_TRACING: 'true', RESPAN_API_KEY: 'attacker', RESPAN_BASE_URL: 'https://attacker.test' };
  const withoutKey = createHost({}, { env: { ...env } });
  await runEmptyTurn(withoutKey);
  assert.deepEqual(withoutKey.requests, []);

  const withKey = createHost({ api_key: 'test-key' }, { env: { ...env } });
  await runEmptyTurn(withKey);
  assert.equal(withKey.requests[0].url, 'https://api.respan.ai/api/v2/traces');
  assert.equal(withKey.requests[0].init.headers.Authorization, 'Bearer test-key');
});

test('a project file adds customer and metadata but cannot redirect traces', async () => {
  const host = createHost({ api_key: 'test-key' }, {
    files: {
      '.claude/respan.json': JSON.stringify({ base_url: 'https://attacker.test', customer_id: 'frank', team: 'sdk' }),
    },
  });
  await runEmptyTurn(host);

  const { body } = host.requests[0].init;
  assert.equal(host.requests[0].url, 'https://api.respan.ai/api/v2/traces');
  assert.equal(body.includes('respan.customer_params.customer_identifier'), true);
  assert.equal(body.includes('respan.metadata.team'), true);
  assert.equal(body.includes('attacker'), false);
});

test('the Respan CLI login plays no part in where traces go', async () => {
  const files = {
    '/home/test/.respan/credentials.json': JSON.stringify({ default: { apiKey: 'cli-key', baseUrl: 'https://respan.internal' } }),
  };
  const withKey = createHost({ api_key: 'plugin-key' }, { env: { HOME: '/home/test' }, files });
  await runEmptyTurn(withKey);
  assert.equal(withKey.requests[0].url, 'https://api.respan.ai/api/v2/traces');
  assert.equal(withKey.requests[0].init.headers.Authorization, 'Bearer plugin-key');

  const withoutKey = createHost({}, { env: { HOME: '/home/test' }, files });
  await runEmptyTurn(withoutKey);
  assert.deepEqual(withoutKey.requests, []);
});

test('no key says so once and sends nothing', async () => {
  const host = createHost({});
  await runEmptyTurn(host);
  await runEmptyTurn(host, 'a9df4f5d-e4c2-489e-b60b-f64da87ebefe');

  assert.deepEqual(host.requests, []);
  assert.equal(host.logs.filter((line) => line.includes('No Respan API key')).length, 1);
});

test('the legacy hook is silenced only while tracing runs', async () => {
  const host = createHost(ENABLED, { env: { TRACE_TO_RESPAN: 'true' } });
  await runEmptyTurn(host);
  assert.equal(host.env.TRACE_TO_RESPAN, 'false');

  const off = createHost(ENABLED, { env: { TRACE_TO_RESPAN: 'true', RESPAN_TRACING: 'false' } });
  await runEmptyTurn(off);
  assert.equal(off.env.TRACE_TO_RESPAN, 'true');
});

test('turn.step passes every chunk and the result through unchanged', async () => {
  const host = createHost(ENABLED);
  await host.fire('turn.start', { turnId: TURN_ID, text: 'hi' });
  const chunks = [{ kind: 'engine' }, { kind: 'text', index: 0, text: 'he' }, { kind: 'text', index: 0, text: 'llo' }];
  const expected = { turnId: TURN_ID, index: 0, answer: 'hello', toolUses: [], stopReason: 'end_turn', usage: null };
  const stream = host.step({ turnId: TURN_ID, index: 0, model: 'claude-haiku-5-5', messageCount: 1 }, chunks, expected);

  const seen = [];
  let item = await stream.next();
  while (!item.done) {
    seen.push(item.value);
    item = await stream.next();
  }

  assert.deepEqual(seen, chunks);
  assert.equal(item.value, expected);
});

test('a broken tracer never reaches the tool call or the turn', async () => {
  const host = createHost(ENABLED);
  await host.fire('turn.start', { turnId: TURN_ID, text: 'hi' });
  // A row whose content is not iterable, and a result that is not an object.
  await host.fire('session.append', { door: 'response', message: { content: 42 } });
  const answer = await host.fire('tool.call', { tool: 'Read', tool_use_id: 't1', get file_path() { throw new Error('boom'); } }, { text: 'ok' });
  assert.deepEqual(answer, { text: 'ok' });
});

test('the timer flushes finished spans during a long turn', async () => {
  const host = createHost(ENABLED);
  await host.fire('session.start', { cwd: '/work' }, { cwd: '/work' });
  await host.fire('turn.start', { turnId: TURN_ID, text: 'hi' });
  await host.fire('tool.call', { tool: 'Read', tool_use_id: 't1', file_path: 'a' }, { text: 'ok' });
  assert.equal(host.requests.length, 0);

  host.timers[0]();
  assert.equal(host.requests.length, 1);
  assert.equal(spansOf(host.requests[0])[0].name, 'Tool: Read');
});
