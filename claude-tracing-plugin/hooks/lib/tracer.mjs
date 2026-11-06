// Turns Claude Code's mod events into Respan spans. It does no I/O and takes every
// timestamp as an argument, so register.mjs owns the clock and the network.
//
// One trace per main-conversation turn:
//   claude-code                workflow, from turn.start to turn.complete
//     claude.chat              chat, one per request to the model
//     Tool: Read               tool
//     Tool: Agent              tool
//       Agent: general-purpose agent, a subagent's turn
//         claude.chat
//         Tool: Read

import { ATTR, Double, LOG_METHOD_TS_TRACING, buildSpan, spanIdFrom, traceIdFrom } from './otlp.mjs';

// Matches the content limit of Claude Code's own telemetry.
const DEFAULT_MAX_CONTENT_CHARS = 61_440;
const DEFAULT_NAME = 'claude-code';

// `session.append` rows the model reads as input. `queue` repeats `prompt`, and `note` and
// `notice` are for the person at the screen.
const INPUT_DOORS = new Set([
  'prompt',
  'command',
  'attachment',
  'tool-result',
  'tool-message',
  'delivery',
  'hook-context',
  'compaction',
]);
const STREAMED_CONTENT_KINDS = new Set(['text', 'tool', 'input', 'thinking']);
const MAIN_LOOP = 'main';

function metadataAttributes(metadata) {
  const attributes = {};
  for (const [key, value] of Object.entries(metadata)) {
    attributes[`${ATTR.METADATA_PREFIX}${key}`] = value;
  }
  return attributes;
}

function stringify(value) {
  return typeof value === 'string' ? value : JSON.stringify(value);
}

function blockText(block) {
  if (block?.type === 'text') return block.text ?? '';
  if (typeof block === 'string') return block;
  return `[${block?.type ?? 'unknown'}]`;
}

function toolResultText(content) {
  if (Array.isArray(content)) return content.map(blockText).join('\n');
  return stringify(content ?? '');
}

/** A row's blocks as the messages a request carries. */
function rowToMessages(message) {
  const content = message?.content;
  const role = message?.role ?? 'user';
  if (typeof content === 'string') return content ? [{ role, content }] : [];
  if (!Array.isArray(content)) return [];

  const messages = [];
  const text = [];
  for (const block of content) {
    if (block?.type === 'tool_result') {
      messages.push({ role: 'tool', tool_call_id: block.tool_use_id, content: toolResultText(block.content) });
    } else {
      text.push(blockText(block));
    }
  }
  if (text.length > 0) messages.unshift({ role, content: text.join('\n') });
  return messages;
}

function toToolCall(block) {
  return {
    id: block.id,
    type: 'function',
    function: { name: block.name, arguments: stringify(block.input ?? {}) },
  };
}

function usageAttributes(usage) {
  if (!usage) return {};
  const cacheRead = usage.cache_read_input_tokens ?? 0;
  const cacheCreation = usage.cache_creation_input_tokens ?? 0;
  // Anthropic reports input_tokens without the cache. The contract's input count includes it,
  // which also keeps the backend from guessing when the cached part is the smaller one.
  const input = (usage.input_tokens ?? 0) + cacheRead + cacheCreation;
  const output = usage.output_tokens ?? 0;
  return {
    [ATTR.INPUT_TOKENS]: input,
    [ATTR.PROMPT_TOKENS]: input,
    [ATTR.OUTPUT_TOKENS]: output,
    [ATTR.COMPLETION_TOKENS]: output,
    [ATTR.LLM_TOTAL_TOKENS]: input + output,
    [ATTR.CACHE_READ_TOKENS]: cacheRead,
    [ATTR.CACHE_CREATION_TOKENS]: cacheCreation,
    [ATTR.RESPONSE_MODEL]: usage.model,
  };
}

export class SessionTracer {
  constructor(options = {}) {
    this.maxContentChars = options.maxContentChars ?? DEFAULT_MAX_CONTENT_CHARS;
    this.customerId = options.customerId || '';
    this.workflowName = options.workflowName || DEFAULT_NAME;
    this.spanName = options.spanName || DEFAULT_NAME;
    this.metadata = options.metadata ?? {};
    this.sessionId = '';
    this.mainTurnCount = 0;
    this.pendingSystem = null;
    this.loops = new Map();
    this.pendingSpawns = [];
    this.agentLinks = new Map();
    this.finished = [];
  }

  setSessionId(sessionId) {
    this.sessionId = sessionId || '';
  }

  /** Finished spans since the last call. */
  drain() {
    const spans = this.finished;
    this.finished = [];
    return spans;
  }

  onPromptComposed(sections) {
    this.pendingSystem = (sections ?? []).map((section) => section.text ?? '').filter(Boolean).join('\n\n') || null;
  }

  onTurnStart(at, e) {
    const loop = this.loopOf(e.agentId);
    const link = e.agentId ? this.linkFor(e.agentId, e.text) : null;
    if (!e.agentId) this.mainTurnCount += 1;
    loop.turn = {
      id: e.turnId,
      agentId: e.agentId,
      number: this.mainTurnCount,
      traceId: link?.traceId ?? traceIdFrom(e.turnId),
      spanId: spanIdFrom(`${e.turnId}:turn`),
      parentSpanId: link?.spanId,
      subagentType: link?.subagentType,
      startedAt: at,
      input: e.text ?? '',
    };
  }

  onAgentSpawning(e) {
    const turn = this.loopOf(e.parentAgentId).turn;
    if (!turn || e.resume) return null;
    const pending = {
      prompt: e.prompt,
      link: {
        traceId: turn.traceId,
        spanId: this.toolSpanId(e.tool_use_id),
        subagentType: e.subagentType,
      },
    };
    this.pendingSpawns.push(pending);
    return pending;
  }

  onAgentSpawned(pending, result) {
    if (!pending) return;
    this.pendingSpawns = this.pendingSpawns.filter((candidate) => candidate !== pending);
    if (result?.agentId) this.agentLinks.set(result.agentId, pending.link);
  }

  onStepStart(at, e) {
    const loop = this.loopOf(e.agentId);
    const step = {
      turn: loop.turn,
      index: e.index,
      model: e.model,
      effort: e.effort,
      messageCount: e.messageCount,
      startedAt: at,
      firstContentAt: null,
      input: loop.input,
      response: [],
      thinking: '',
      system: e.agentId ? null : this.pendingSystem,
    };
    if (!e.agentId) this.pendingSystem = null;
    loop.input = [];
    loop.step = step;
    return step;
  }

  onStepChunk(step, at, chunk) {
    if (step.firstContentAt === null && STREAMED_CONTENT_KINDS.has(chunk?.kind)) step.firstContentAt = at;
    if (chunk?.kind === 'thinking') step.thinking += chunk.text ?? '';
  }

  onStepEnd(step, at, result) {
    const loop = this.loops.get(step.turn?.agentId ?? MAIN_LOOP);
    if (loop?.step === step) loop.step = null;
    if (!step.turn) return;

    const error = result?.stopReason === null ? 'No response from the model' : undefined;
    const attributes = {
      ...this.commonAttributes(),
      [ATTR.LOG_TYPE]: 'chat',
      [ATTR.LLM_REQUEST_TYPE]: 'chat',
      [ATTR.ENTITY_NAME]: 'claude.chat',
      [ATTR.ENTITY_PATH]: 'claude.chat',
      [ATTR.SYSTEM]: 'anthropic',
      [ATTR.REQUEST_MODEL]: step.model,
      // OTel GenAI's time to first chunk: seconds until the first chunk carrying new output.
      [ATTR.TIME_TO_FIRST_CHUNK]:
        step.firstContentAt === null ? undefined : new Double((step.firstContentAt - step.startedAt) / 1000),
      ...usageAttributes(result?.usage),
      ...metadataAttributes({
        step_index: step.index,
        message_count: step.messageCount,
        effort: step.effort,
        stop_reason: result?.stopReason ?? undefined,
      }),
      ...this.stepContentAttributes(step, result),
    };
    const spanId = spanIdFrom(`${step.turn.id}:step:${step.index}`);
    this.finished.push(buildSpan({
      traceId: step.turn.traceId,
      spanId,
      parentSpanId: step.turn.spanId,
      name: 'claude.chat',
      isLlm: true,
      startMs: step.startedAt,
      endMs: at,
      attributes,
      error,
    }));

    if (step.thinking) {
      this.finished.push(buildSpan({
        traceId: step.turn.traceId,
        spanId: spanIdFrom(`${step.turn.id}:thinking:${step.index}`),
        parentSpanId: step.turn.spanId,
        name: 'Thinking',
        startMs: step.startedAt,
        endMs: step.firstContentAt ?? at,
        attributes: {
          ...this.commonAttributes(),
          [ATTR.LOG_TYPE]: 'task',
          [ATTR.ENTITY_NAME]: 'Thinking',
          [ATTR.ENTITY_PATH]: 'thinking',
          [ATTR.ENTITY_OUTPUT]: this.clip(step.thinking),
        },
      }));
    }
  }

  onAppend(e) {
    const loop = this.loopOf(e.agentId);
    if (e.door === 'response') {
      const content = e.message?.content;
      if (loop.step && Array.isArray(content)) loop.step.response.push(...content);
    } else if (INPUT_DOORS.has(e.door)) {
      loop.input.push(...rowToMessages(e.message));
    }
  }

  onToolStart(at, e) {
    const turn = this.loopOf(e.agentId).turn;
    if (!turn) return null;
    return { turn, tool: e.tool, toolUseId: e.tool_use_id, input: e, startedAt: at };
  }

  onToolEnd(call, at, result) {
    if (!call) return;
    const failure = result?.deny ?? (result?.isError ? 'Tool call failed' : undefined);
    const { tool, tool_use_id: toolUseId, agentId, ...args } = call.input;
    const name = `Tool: ${call.tool}`;
    const attributes = {
      ...this.commonAttributes(),
      [ATTR.LOG_TYPE]: 'tool',
      [ATTR.ENTITY_NAME]: call.tool,
      [ATTR.ENTITY_PATH]: `tool_${String(call.tool).toLowerCase()}`,
      [ATTR.TOOL_CALL_ID]: call.toolUseId,
      [ATTR.ENTITY_INPUT]: this.clip(JSON.stringify({ name: tool, arguments: args })),
      [ATTR.ENTITY_OUTPUT]: this.clip(stringify(result?.text ?? result?.deny ?? result?.result ?? '')),
    };
    this.finished.push(buildSpan({
      traceId: call.turn.traceId,
      spanId: this.toolSpanId(call.toolUseId),
      parentSpanId: call.turn.spanId,
      name,
      startMs: call.startedAt,
      endMs: at,
      attributes,
      error: failure,
    }));
  }

  onTurnComplete(at, e) {
    const loop = this.loopOf(e.agentId);
    const turn = loop.turn;
    if (!turn || turn.id !== e.turnId) return;
    // A turn interrupted mid-request can end with its step still open.
    if (loop.step) this.onStepEnd(loop.step, at, { stopReason: null, usage: null });
    loop.turn = null;

    const name = turn.agentId ? `Agent: ${turn.subagentType ?? 'subagent'}` : this.spanName;
    const failed = e.reason === 'error' || e.reason === 'refusal';
    const attributes = {
      ...this.commonAttributes(),
      [ATTR.LOG_TYPE]: turn.agentId ? 'agent' : 'workflow',
      [ATTR.ENTITY_NAME]: name,
      [ATTR.ENTITY_PATH]: turn.parentSpanId ? 'agent' : '',
      ...metadataAttributes({
        ...(turn.agentId ? { agent_id: turn.agentId } : { ...this.metadata, claude_code_turn: turn.number }),
        turn_reason: e.reason,
      }),
      [ATTR.ENTITY_INPUT]: this.clip(turn.input),
      [ATTR.ENTITY_OUTPUT]: this.clip(e.answer ?? ''),
    };
    this.finished.push(buildSpan({
      traceId: turn.traceId,
      spanId: turn.spanId,
      parentSpanId: turn.parentSpanId,
      name,
      startMs: turn.startedAt,
      endMs: at,
      attributes,
      error: failed ? `Turn ended: ${e.reason}` : undefined,
    }));
  }

  loopOf(agentId) {
    const key = agentId ?? MAIN_LOOP;
    let loop = this.loops.get(key);
    if (!loop) {
      loop = { turn: null, step: null, input: [] };
      this.loops.set(key, loop);
    }
    return loop;
  }

  /** The parent of a subagent's turn. A turn can start before its agent.spawn call resolves. */
  linkFor(agentId, text) {
    const known = this.agentLinks.get(agentId);
    if (known) return known;
    const pending = this.pendingSpawns.find((candidate) => candidate.prompt === text) ?? this.pendingSpawns[0];
    if (!pending) return null;
    this.pendingSpawns = this.pendingSpawns.filter((candidate) => candidate !== pending);
    this.agentLinks.set(agentId, pending.link);
    return pending.link;
  }

  toolSpanId(toolUseId) {
    return spanIdFrom(`tool:${toolUseId}`);
  }

  commonAttributes() {
    const attributes = {
      [ATTR.WORKFLOW]: this.workflowName,
      [ATTR.LOG_METHOD]: LOG_METHOD_TS_TRACING,
    };
    if (this.sessionId) {
      attributes[ATTR.SESSION] = `claudecode_${this.sessionId}`;
      attributes[ATTR.THREAD] = `claudecode_${this.sessionId}`;
    }
    if (this.customerId) attributes[ATTR.CUSTOMER] = this.customerId;
    return attributes;
  }

  stepContentAttributes(step, result) {
    const attributes = {};
    const messages = step.system ? [{ role: 'system', content: step.system }, ...step.input] : step.input;
    messages.forEach((message, index) => {
      const prefix = `${ATTR.PROMPT_PREFIX}${index}`;
      attributes[`${prefix}.role`] = message.role;
      attributes[`${prefix}.content`] = this.clip(message.content);
      if (message.tool_call_id) attributes[`${prefix}.tool_call_id`] = message.tool_call_id;
    });

    const text = step.response.filter((block) => block?.type === 'text').map((block) => block.text ?? '').join('\n');
    const toolCalls = step.response.filter((block) => block?.type === 'tool_use').map(toToolCall);
    attributes[`${ATTR.COMPLETION_PREFIX}0.role`] = 'assistant';
    attributes[`${ATTR.COMPLETION_PREFIX}0.content`] = this.clip(text || result?.answer || '');
    if (toolCalls.length > 0) attributes[`${ATTR.COMPLETION_PREFIX}0.tool_calls`] = this.clip(JSON.stringify(toolCalls));
    return attributes;
  }

  clip(value) {
    const text = stringify(value);
    if (text.length <= this.maxContentChars) return text;
    return `${text.slice(0, this.maxContentChars)} ...[truncated ${text.length - this.maxContentChars} chars]`;
  }
}
