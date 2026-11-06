import assert from 'node:assert/strict';
import { test } from 'node:test';

import { buildBodies, spanIdFrom } from '../hooks/lib/otlp.mjs';
import { SessionTracer } from '../hooks/lib/tracer.mjs';

const TURN_ID = 'e9df4f5d-e4c2-489e-b60b-f64da87ebefe';
const TRACE_ID = 'e9df4f5de4c2489eb60bf64da87ebefe';

function attributesOf(span) {
  return Object.fromEntries(span.attributes.map(({ key, value }) => [key, Object.values(value)[0]]));
}

function spanNamed(spans, name) {
  return spans.find((span) => span.name === name);
}

function runTurn(tracer) {
  tracer.setSessionId('session-1');
  tracer.onPromptComposed([{ id: 'a', text: 'You are Claude Code.' }]);
  tracer.onAppend({ door: 'queue', message: { role: 'user', content: [{ type: 'text', text: 'read it' }] } });
  tracer.onAppend({ door: 'prompt', message: { role: 'user', content: [{ type: 'text', text: 'read it' }] } });
  tracer.onTurnStart(1_000, { turnId: TURN_ID, text: 'read it' });

  const first = tracer.onStepStart(1_100, { turnId: TURN_ID, index: 0, model: 'claude-haiku-5-5', messageCount: 2 });
  tracer.onStepChunk(first, 1_150, { kind: 'engine' });
  tracer.onStepChunk(first, 1_400, { kind: 'tool', index: 0 });
  tracer.onAppend({
    door: 'response',
    message: { role: 'assistant', content: [{ type: 'tool_use', id: 'toolu_1', name: 'Read', input: { file_path: 'a.txt' } }] },
  });
  const call = tracer.onToolStart(1_450, { tool: 'Read', tool_use_id: 'toolu_1', file_path: 'a.txt' });
  tracer.onStepEnd(first, 1_500, {
    stopReason: 'tool_use',
    answer: '',
    toolUses: [],
    usage: { input_tokens: 2, output_tokens: 10, cache_read_input_tokens: 100, cache_creation_input_tokens: 50, model: 'claude-haiku-5-5' },
  });
  tracer.onToolEnd(call, 1_600, { text: 'hello' });
  tracer.onAppend({
    door: 'tool-result',
    message: { role: 'user', content: [{ type: 'tool_result', tool_use_id: 'toolu_1', content: 'hello' }] },
  });

  const second = tracer.onStepStart(1_700, { turnId: TURN_ID, index: 1, model: 'claude-haiku-5-5', messageCount: 4 });
  tracer.onAppend({ door: 'response', message: { role: 'assistant', content: [{ type: 'text', text: 'done' }] } });
  tracer.onStepEnd(second, 1_900, { stopReason: 'end_turn', answer: 'done', toolUses: [], usage: null });
  tracer.onTurnComplete(2_000, { turnId: TURN_ID, answer: 'done', reason: 'answer' });
  return tracer.drain();
}

test('a turn becomes a workflow span with chat and tool spans under it', () => {
  const spans = runTurn(new SessionTracer());
  const root = spanNamed(spans, 'claude-code');
  const attributes = attributesOf(root);

  assert.equal(root.traceId, TRACE_ID);
  assert.equal(root.parentSpanId, undefined);
  assert.equal(attributes['respan.entity.log_type'], 'workflow');
  assert.equal(attributes['traceloop.entity.path'], '');
  assert.equal(attributes['traceloop.entity.input'], 'read it');
  assert.equal(attributes['traceloop.entity.output'], 'done');
  assert.equal(attributes['respan.sessions.session_identifier'], 'claudecode_session-1');
  assert.equal(spans.filter((span) => span.parentSpanId === root.spanId).length, 3);
});

test('a chat span follows the contract and does not set traceloop.span.kind', () => {
  const attributes = attributesOf(spanNamed(runTurn(new SessionTracer()), 'claude.chat'));

  assert.equal(attributes['traceloop.span.kind'], undefined);
  assert.equal(attributes['llm.request.type'], 'chat');
  assert.equal(attributes['gen_ai.system'], 'anthropic');
  assert.equal(attributes['gen_ai.request.model'], 'claude-haiku-5-5');
  assert.equal(attributes['gen_ai.response.time_to_first_chunk'], 0.3);
  assert.equal(attributes['respan.metadata.time_to_first_token_ms'], undefined);
});

test('input tokens include the cache so the backend does not have to guess', () => {
  const attributes = attributesOf(spanNamed(runTurn(new SessionTracer()), 'claude.chat'));

  assert.equal(attributes['gen_ai.usage.input_tokens'], '152');
  assert.equal(attributes['gen_ai.usage.prompt_tokens'], '152');
  assert.equal(attributes['gen_ai.usage.output_tokens'], '10');
  assert.equal(attributes['llm.usage.total_tokens'], '162');
  assert.equal(attributes['gen_ai.usage.cache_read_input_tokens'], '100');
  assert.equal(attributes['gen_ai.usage.cache_creation_input_tokens'], '50');
});

test('each chat span carries only what is new since the last request', () => {
  const chats = runTurn(new SessionTracer()).filter((span) => span.name === 'claude.chat');
  const first = attributesOf(chats[0]);
  const second = attributesOf(chats[1]);

  assert.equal(first['gen_ai.prompt.0.role'], 'system');
  assert.equal(first['gen_ai.prompt.1.content'], 'read it');
  assert.equal(first['gen_ai.prompt.2.role'], undefined);
  assert.equal(JSON.parse(first['gen_ai.completion.0.tool_calls'])[0].function.name, 'Read');

  assert.equal(second['gen_ai.prompt.0.role'], 'tool');
  assert.equal(second['gen_ai.prompt.0.content'], 'hello');
  assert.equal(second['gen_ai.prompt.0.tool_call_id'], 'toolu_1');
  assert.equal(second['gen_ai.prompt.1.role'], undefined);
  assert.equal(second['gen_ai.completion.0.content'], 'done');
});

test('a tool span holds its arguments and result', () => {
  const span = spanNamed(runTurn(new SessionTracer()), 'Tool: Read');
  const attributes = attributesOf(span);

  assert.equal(attributes['respan.entity.log_type'], 'tool');
  assert.equal(attributes['gen_ai.tool.call.id'], 'toolu_1');
  assert.deepEqual(JSON.parse(attributes['traceloop.entity.input']), { name: 'Read', arguments: { file_path: 'a.txt' } });
  assert.equal(attributes['traceloop.entity.output'], 'hello');
  assert.equal(span.endTimeUnixNano, '1600000000');
});

test('time to first chunk stays a double when it is a whole number of seconds', () => {
  const tracer = new SessionTracer();
  tracer.onTurnStart(1_000, { turnId: TURN_ID, text: 'hi' });
  const step = tracer.onStepStart(1_000, { turnId: TURN_ID, index: 0, model: 'claude-haiku-5-5', messageCount: 1 });
  tracer.onStepChunk(step, 3_000, { kind: 'text', index: 0, text: 'ok' });
  tracer.onStepEnd(step, 3_100, { stopReason: 'end_turn', answer: 'ok', toolUses: [], usage: null });

  const span = spanNamed(tracer.drain(), 'claude.chat');
  const value = span.attributes.find(({ key }) => key === 'gen_ai.response.time_to_first_chunk').value;
  assert.deepEqual(value, { doubleValue: 2 });
});

test('thinking gets its own span', () => {
  const tracer = new SessionTracer();
  tracer.onTurnStart(1_000, { turnId: TURN_ID, text: 'hard question' });
  const step = tracer.onStepStart(1_100, { turnId: TURN_ID, index: 0, model: 'claude-sonnet-5-5', messageCount: 1 });
  tracer.onStepChunk(step, 1_200, { kind: 'thinking', index: 0, text: 'let me ' });
  tracer.onStepChunk(step, 1_300, { kind: 'thinking', index: 0, text: 'think' });
  tracer.onStepEnd(step, 1_400, { stopReason: 'end_turn', answer: 'ok', toolUses: [], usage: null });

  assert.equal(attributesOf(spanNamed(tracer.drain(), 'Thinking'))['traceloop.entity.output'], 'let me think');
});

test('a subagent turn joins its parent trace under the Agent tool span', () => {
  const tracer = new SessionTracer();
  tracer.onTurnStart(1_000, { turnId: TURN_ID, text: 'delegate' });
  const pending = tracer.onAgentSpawning({ tool_use_id: 'toolu_agent', prompt: 'read a.txt', subagentType: 'Explore' });
  // The subagent's turn starts before its agent.spawn call resolves.
  tracer.onTurnStart(1_100, { turnId: 'sub-turn', text: 'read a.txt', agentId: 'agent-1' });
  tracer.onAgentSpawned(pending, { agentId: 'agent-1', model: 'claude-haiku-5-5' });
  const step = tracer.onStepStart(1_200, { turnId: 'sub-turn', index: 0, model: 'claude-haiku-5-5', messageCount: 1, agentId: 'agent-1' });
  tracer.onStepEnd(step, 1_300, { stopReason: 'end_turn', answer: 'ok', toolUses: [], usage: null });
  tracer.onTurnComplete(1_400, { turnId: 'sub-turn', agentId: 'agent-1', answer: 'ok', reason: 'answer' });

  const spans = tracer.drain();
  const agent = spanNamed(spans, 'Agent: Explore');
  assert.equal(agent.traceId, TRACE_ID);
  assert.equal(agent.parentSpanId, spanIdFrom('tool:toolu_agent'));
  assert.equal(attributesOf(agent)['respan.entity.log_type'], 'agent');
  assert.equal(spanNamed(spans, 'claude.chat').parentSpanId, agent.spanId);
  assert.equal(spanNamed(spans, 'claude.chat').traceId, TRACE_ID);
});

test('a resumed subagent stays under its original Agent tool span', () => {
  const tracer = new SessionTracer();
  tracer.onTurnStart(1_000, { turnId: TURN_ID, text: 'delegate' });
  const pending = tracer.onAgentSpawning({ tool_use_id: 'toolu_agent', prompt: 'first', subagentType: 'Explore' });
  tracer.onAgentSpawned(pending, { agentId: 'agent-1' });
  tracer.onTurnStart(1_100, { turnId: 'sub-1', text: 'first', agentId: 'agent-1' });
  tracer.onTurnComplete(1_200, { turnId: 'sub-1', agentId: 'agent-1', answer: 'ok', reason: 'answer' });
  assert.equal(tracer.onAgentSpawning({ tool_use_id: 'x', prompt: 'more', resume: { agentId: 'agent-1' } }), null);
  tracer.onTurnStart(1_300, { turnId: 'sub-2', text: 'more', agentId: 'agent-1' });
  tracer.onTurnComplete(1_400, { turnId: 'sub-2', agentId: 'agent-1', answer: 'ok', reason: 'answer' });

  const agents = tracer.drain().filter((span) => span.name === 'Agent: Explore');
  assert.equal(agents.length, 2);
  assert.equal(agents[1].parentSpanId, spanIdFrom('tool:toolu_agent'));
});

test('a turn interrupted mid-request reports the open request and is not an error', () => {
  const tracer = new SessionTracer();
  tracer.onTurnStart(1_000, { turnId: TURN_ID, text: 'long task' });
  tracer.onStepStart(1_100, { turnId: TURN_ID, index: 0, model: 'claude-haiku-5-5', messageCount: 1 });
  tracer.onTurnComplete(1_500, { turnId: TURN_ID, answer: '', reason: 'aborted', isAborted: true });

  const spans = tracer.drain();
  assert.equal(spanNamed(spans, 'claude.chat').status.code, 2);
  assert.equal(spanNamed(spans, 'claude-code').status.code, 1);
  assert.equal(attributesOf(spanNamed(spans, 'claude-code'))['respan.metadata.turn_reason'], 'aborted');
});

test('a failed tool call is an error span', () => {
  const tracer = new SessionTracer();
  tracer.onTurnStart(1_000, { turnId: TURN_ID, text: 'run it' });
  tracer.onToolEnd(tracer.onToolStart(1_100, { tool: 'Bash', tool_use_id: 'toolu_2', command: 'false' }), 1_200, { isError: true, text: 'exit 1' });

  const span = spanNamed(tracer.drain(), 'Tool: Bash');
  assert.equal(span.status.code, 2);
  assert.equal(attributesOf(span)['traceloop.entity.output'], 'exit 1');
});

test('long content is clipped with a marker', () => {
  const tracer = new SessionTracer({ maxContentChars: 10 });
  tracer.onTurnStart(1_000, { turnId: TURN_ID, text: 'x'.repeat(25) });
  tracer.onTurnComplete(1_500, { turnId: TURN_ID, answer: '', reason: 'answer' });

  const input = attributesOf(tracer.drain()[0])['traceloop.entity.input'];
  assert.equal(input, `${'x'.repeat(10)} ...[truncated 15 chars]`);
});

test('bodies stay under the limit and an oversized span loses only its content', () => {
  const tracer = new SessionTracer();
  tracer.onTurnStart(1_000, { turnId: TURN_ID, text: 'y'.repeat(5_000) });
  tracer.onTurnComplete(1_500, { turnId: TURN_ID, answer: '', reason: 'answer' });

  const [body] = buildBodies(tracer.drain(), { maxBodyChars: 1_000, serviceVersion: '1.0.0' });
  assert.equal(body.length < 2_000, true);
  assert.equal(body.includes('yyyy'), false);
  assert.equal(body.includes('respan.metadata.content_dropped'), true);

  const spans = runTurn(new SessionTracer());
  const single = JSON.stringify(spans[0]).length;
  assert.equal(buildBodies(spans, { maxBodyChars: single + 1, serviceVersion: '1.0.0' }).length > 1, true);
});
