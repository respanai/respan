import assert from 'node:assert/strict';
import test from 'node:test';

import {
  evaluateTrace,
  flattenTraceSpans,
  inspectTrace,
  parseCountExpectations,
  waitForTrace,
} from '../dist/lib/trace-validation.js';

const trace = {
  trace_unique_id: '0123456789abcdef0123456789abcdef',
  name: 'smoke',
  span_count: 7,
  error_count: 0,
  span_tree: [
    {
      span_name: 'smoke.workflow',
      log_type: 'workflow',
      children: [
        {
          span_name: 'Triage Agent.agent',
          log_type: 'agent',
          children: [
            { span_name: 'openai.chat', log_type: 'chat', model: 'gpt-test' },
            { span_name: 'handoff.task', log_type: 'task' },
          ],
        },
        { span_name: 'Arithmetic Specialist.agent', log_type: 'agent' },
        { span_name: 'add_numbers.tool', log_type: 'tool' },
        { span_name: 'openai.chat', log_type: 'chat', model: 'gpt-test' },
      ],
    },
  ],
};

test('trace inspection flattens nested span trees and counts contract fields', () => {
  assert.equal(flattenTraceSpans(trace).length, 7);
  assert.deepEqual(inspectTrace(trace).typeCounts, {
    workflow: 1,
    agent: 2,
    chat: 2,
    task: 1,
    tool: 1,
  });
});

test('trace assertions report missing types and pass complete traces', () => {
  const expectations = {
    minSpans: 7,
    maxErrors: 0,
    types: { workflow: 1, agent: 2, chat: 2, tool: 1, task: 1 },
    names: { 'handoff.task': 1 },
    models: ['gpt-test'],
  };
  assert.equal(evaluateTrace(trace, expectations).passed, true);

  const failed = evaluateTrace(trace, { types: { guardrail: 1 } });
  assert.equal(failed.passed, false);
  assert.equal(failed.issues[0].field, 'type:guardrail');
});

test('count expectations support names and minimum counts', () => {
  assert.deepEqual(parseCountExpectations(['agent:2', 'handoff.task', 'guardrail:input']), {
    agent: 2,
    'handoff.task': 1,
    'guardrail:input': 1,
  });
});

test('trace polling waits for the complete exported trace', async () => {
  let calls = 0;
  const result = await waitForTrace(
    async () => ({ ...trace, span_count: ++calls }),
    {
      timeoutMs: 100,
      intervalMs: 1,
      ready: (candidate) => candidate.span_count >= 3,
    },
  );

  assert.equal(result.span_count, 3);
  assert.equal(calls, 3);
});

test('trace polling fails fast for non-retryable client errors', async () => {
  let calls = 0;
  const requestError = Object.assign(new Error('bad request'), { statusCode: 400 });
  await assert.rejects(
    waitForTrace(
      async () => {
        calls += 1;
        throw requestError;
      },
      { timeoutMs: 100, intervalMs: 1 },
    ),
    /bad request/,
  );
  assert.equal(calls, 1);
});

test('trace polling aborts a hung fetch at the overall deadline', async () => {
  const startedAt = Date.now();
  await assert.rejects(
    waitForTrace(
      async () => new Promise(() => {}),
      { timeoutMs: 20, intervalMs: 1 },
    ),
    /not ready within 1 seconds/,
  );
  assert.ok(Date.now() - startedAt < 200);
});
