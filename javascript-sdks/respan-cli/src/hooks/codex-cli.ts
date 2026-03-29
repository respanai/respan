/**
 * Respan Hook for Codex CLI
 *
 * Sends Codex CLI conversation traces to Respan after each agent turn.
 * Uses Codex CLI's notify hook to capture session JSONL files and convert
 * them to Respan spans.
 *
 * Span tree per turn:
 *   Root: codex-cli
 *     ├── openai.chat   (generation — model, tokens, messages)
 *     ├── Reasoning      (if reasoning_output_tokens > 0)
 *     ├── Tool: Shell    (if exec_command)
 *     ├── Tool: File Edit (if apply_patch)
 *     └── Tool: Web Search (if web_search_call)
 */
import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';
import { execFile } from 'node:child_process';

import {
  initLogging,
  log,
  debug,
  resolveCredentials,
  loadRespanConfig,
  loadState,
  saveState,
  acquireLock,
  sendSpans,
  addDefaultsToAll,
  resolveSpanFields,
  buildMetadata,
  nowISO,
  parseTimestamp,
  latencySeconds,
  truncate,
  type SpanData,
  type RespanConfig,
} from './shared.js';

// ── Config ────────────────────────────────────────────────────────

const STATE_DIR = path.join(os.homedir(), '.codex', 'state');
const LOG_FILE = path.join(STATE_DIR, 'respan_hook.log');
const STATE_FILE = path.join(STATE_DIR, 'respan_state.json');
const LOCK_PATH = path.join(STATE_DIR, 'respan_hook.lock');
const DEBUG_MODE = (process.env.CODEX_RESPAN_DEBUG ?? '').toLowerCase() === 'true';
const MAX_CHARS = parseInt(process.env.CODEX_RESPAN_MAX_CHARS ?? '4000', 10) || 4000;

initLogging(LOG_FILE, DEBUG_MODE);

// ── Types ─────────────────────────────────────────────────────────

type Msg = Record<string, unknown>;

interface Turn {
  turn_id: string;
  start_time: string;
  end_time: string;
  model: string;
  cwd: string;
  user_message: string;
  assistant_message: string;
  commentary: string[];
  tool_calls: Array<{ name: string; arguments: string; call_id: string; timestamp: string }>;
  tool_outputs: Array<{ call_id: string; output: string; timestamp: string }>;
  reasoning: boolean;
  reasoning_text: string;
  token_usage: Msg;
}

// ── Tool display names ────────────────────────────────────────────

const TOOL_DISPLAY_NAMES: Record<string, string> = {
  exec_command: 'Shell',
  apply_patch: 'File Edit',
  web_search: 'Web Search',
};

function toolDisplayName(name: string): string {
  return TOOL_DISPLAY_NAMES[name] ?? name;
}

function formatToolInput(toolName: string, args: string): string {
  if (!args) return '';
  try {
    const parsed = typeof args === 'string' ? JSON.parse(args) : args;
    if (toolName === 'exec_command' && typeof parsed === 'object' && parsed !== null) {
      const cmd = parsed.cmd ?? '';
      const workdir = parsed.workdir ?? '';
      return truncate(workdir ? `[${workdir}] Command: ${cmd}` : `Command: ${cmd}`, MAX_CHARS);
    }
    if (toolName === 'apply_patch') return truncate(args, MAX_CHARS);
    if (typeof parsed === 'object') return truncate(JSON.stringify(parsed, null, 2), MAX_CHARS);
  } catch {}
  return truncate(String(args), MAX_CHARS);
}

function formatToolOutput(output: string): string {
  if (!output) return '';
  try {
    const parsed = JSON.parse(output);
    if (typeof parsed === 'object' && parsed !== null && 'output' in parsed) {
      return truncate(String(parsed.output), MAX_CHARS);
    }
  } catch {}
  return truncate(output, MAX_CHARS);
}

// ── Session parsing ───────────────────────────────────────────────

function parseSession(lines: string[]): Msg[] {
  const events: Msg[] = [];
  for (const line of lines) {
    if (!line.trim()) continue;
    try { events.push(JSON.parse(line)); } catch {}
  }
  return events;
}

function extractTurns(events: Msg[]): Turn[] {
  const turns: Turn[] = [];
  let current: Turn | null = null;

  for (const event of events) {
    const evtType = String(event.type ?? '');
    const payload = (event.payload ?? {}) as Msg;
    const timestamp = String(event.timestamp ?? '');

    if (evtType === 'event_msg') {
      const msgType = String(payload.type ?? '');

      if (msgType === 'task_started') {
        current = {
          turn_id: String(payload.turn_id ?? ''),
          start_time: timestamp,
          end_time: '',
          model: '',
          cwd: '',
          user_message: '',
          assistant_message: '',
          commentary: [],
          tool_calls: [],
          tool_outputs: [],
          reasoning: false,
          reasoning_text: '',
          token_usage: {},
        };
      } else if (msgType === 'task_complete' && current) {
        current.end_time = timestamp;
        turns.push(current);
        current = null;
      } else if (msgType === 'user_message' && current) {
        current.user_message = String(payload.message ?? '');
      } else if (msgType === 'agent_message' && current) {
        const phase = String(payload.phase ?? '');
        const message = String(payload.message ?? '');
        if (phase === 'final_answer') current.assistant_message = message;
        else if (phase === 'commentary') current.commentary.push(message);
      } else if (msgType === 'token_count' && current) {
        const info = (payload.info ?? {}) as Msg;
        const lastUsage = (info.last_token_usage ?? {}) as Msg;
        if (Object.keys(lastUsage).length) current.token_usage = lastUsage;
      }
    } else if (evtType === 'turn_context' && current) {
      current.model = String(payload.model ?? '');
      current.cwd = String(payload.cwd ?? '');
    } else if (evtType === 'response_item' && current) {
      const itemType = String(payload.type ?? '');
      if (itemType === 'function_call' || itemType === 'custom_tool_call') {
        current.tool_calls.push({
          name: String(payload.name ?? 'unknown'),
          arguments: String(itemType === 'custom_tool_call' ? (payload.input ?? '') : (payload.arguments ?? '')),
          call_id: String(payload.call_id ?? ''),
          timestamp,
        });
      } else if (itemType === 'function_call_output' || itemType === 'custom_tool_call_output') {
        current.tool_outputs.push({
          call_id: String(payload.call_id ?? ''),
          output: String(payload.output ?? ''),
          timestamp,
        });
      } else if (itemType === 'reasoning') {
        if (current) {
          current.reasoning = true;
          const summary = String(payload.summary ?? payload.text ?? '');
          if (summary) current.reasoning_text += (current.reasoning_text ? '\n' : '') + summary;
        }
      } else if (itemType === 'web_search_call') {
        const action = (payload.action ?? {}) as Msg;
        const syntheticId = `web_search_${timestamp}`;
        current.tool_calls.push({
          name: 'web_search',
          arguments: JSON.stringify({ query: action.query ?? '' }),
          call_id: syntheticId,
          timestamp,
        });
        // Web search has no separate output event; record query as output
        current.tool_outputs.push({
          call_id: syntheticId,
          output: `Search: ${action.query ?? ''}`,
          timestamp,
        });
      }
    }
  }
  return turns;
}

// ── Span creation ─────────────────────────────────────────────────

function createSpans(
  sessionId: string,
  turnNum: number,
  turn: Turn,
  config: RespanConfig | null,
): SpanData[] {
  const spans: SpanData[] = [];

  const now = nowISO();
  const startTimeStr = turn.start_time || now;
  const endTimeStr = turn.end_time || now;
  const lat = latencySeconds(startTimeStr, endTimeStr);

  const promptMessages: Msg[] = [];
  if (turn.user_message) promptMessages.push({ role: 'user', content: turn.user_message });
  const completionMessage = turn.assistant_message
    ? { role: 'assistant', content: turn.assistant_message }
    : null;

  const { workflowName, spanName, customerId } = resolveSpanFields(config, {
    workflowName: 'codex-cli',
    spanName: 'codex-cli',
  });
  const traceUniqueId = `${sessionId}_turn_${turnNum}`;
  const threadId = `codexcli_${sessionId}`;

  // Metadata
  const baseMeta: Record<string, unknown> = { codex_cli_turn: turnNum };
  if (turn.cwd) baseMeta.cwd = turn.cwd;
  if (turn.commentary.length) baseMeta.commentary = turn.commentary.join('\n');
  const metadata = buildMetadata(config, baseMeta);

  // Token usage
  const usageFields: Partial<SpanData> = {};
  const tu = turn.token_usage;
  if (Object.keys(tu).length) {
    const pt = Number(tu.input_tokens ?? 0);
    const ct = Number(tu.output_tokens ?? 0);
    usageFields.prompt_tokens = pt;
    usageFields.completion_tokens = ct;
    usageFields.total_tokens = Number(tu.total_tokens ?? pt + ct) || pt + ct;
    const cached = Number(tu.cached_input_tokens ?? 0);
    if (cached > 0) usageFields.prompt_tokens_details = { cached_tokens: cached };
    const reasoning = Number(tu.reasoning_output_tokens ?? 0);
    if (reasoning > 0) metadata.reasoning_tokens = reasoning;
  }

  // Root span
  const rootSpanId = `codexcli_${traceUniqueId}_root`;
  spans.push({
    trace_unique_id: traceUniqueId,
    session_identifier: threadId,
    thread_identifier: threadId,
    customer_identifier: customerId,
    span_unique_id: rootSpanId,
    span_name: spanName,
    span_workflow_name: workflowName,
    model: turn.model || 'gpt-5.4',
    provider_id: '',
    span_path: '',
    input: promptMessages.length ? JSON.stringify(promptMessages) : '',
    output: turn.assistant_message,
    timestamp: endTimeStr,
    start_time: startTimeStr,
    metadata,
    ...(lat !== undefined ? { latency: lat } : {}),
  });

  // LLM generation child span
  spans.push({
    trace_unique_id: traceUniqueId,
    span_unique_id: `codexcli_${traceUniqueId}_gen`,
    span_parent_id: rootSpanId,
    span_name: 'openai.chat',
    span_workflow_name: workflowName,
    span_path: 'openai_chat',
    model: turn.model || 'gpt-5.4',
    provider_id: 'openai',
    metadata: {},
    input: promptMessages.length ? JSON.stringify(promptMessages) : '',
    output: turn.assistant_message,
    prompt_messages: promptMessages,
    completion_message: completionMessage,
    timestamp: endTimeStr,
    start_time: startTimeStr,
    ...(lat !== undefined ? { latency: lat } : {}),
    ...usageFields,
  });

  // Reasoning child span
  const reasoningTokens = Number(tu.reasoning_output_tokens ?? 0);
  if (turn.reasoning || reasoningTokens > 0) {
    spans.push({
      trace_unique_id: traceUniqueId,
      span_unique_id: `codexcli_${traceUniqueId}_reasoning`,
      span_parent_id: rootSpanId,
      span_name: 'Reasoning',
      span_workflow_name: workflowName,
      span_path: 'reasoning',
      provider_id: '',
      metadata: reasoningTokens > 0 ? { reasoning_tokens: reasoningTokens } : {},
      input: '',
      output: turn.reasoning_text || (reasoningTokens > 0 ? `[Reasoning: ${reasoningTokens} tokens]` : '[Reasoning]'),
      timestamp: endTimeStr,
      start_time: startTimeStr,
    });
  }

  // Tool child spans
  const outputMap = new Map<string, { output: string; timestamp: string }>();
  for (const to of turn.tool_outputs) {
    if (to.call_id) outputMap.set(to.call_id, to);
  }

  let toolNum = 0;
  for (const tc of turn.tool_calls) {
    toolNum++;
    const display = toolDisplayName(tc.name);
    const outputData = outputMap.get(tc.call_id);
    const toolEnd = outputData?.timestamp ?? endTimeStr;
    const toolLat = latencySeconds(tc.timestamp, toolEnd);

    spans.push({
      trace_unique_id: traceUniqueId,
      span_unique_id: `codexcli_${traceUniqueId}_tool_${toolNum}`,
      span_parent_id: rootSpanId,
      span_name: `Tool: ${display}`,
      span_workflow_name: workflowName,
      span_path: `tool_${tc.name.toLowerCase()}`,
      provider_id: '',
      metadata: {},
      input: formatToolInput(tc.name, tc.arguments),
      output: formatToolOutput(outputData?.output ?? ''),
      timestamp: toolEnd,
      start_time: tc.timestamp || startTimeStr,
      ...(toolLat !== undefined ? { latency: toolLat } : {}),
    });
  }

  return addDefaultsToAll(spans);
}

// ── Session file finding ──────────────────────────────────────────

function findSessionFile(sessionId: string): string | null {
  const sessionsDir = path.join(os.homedir(), '.codex', 'sessions');
  if (!fs.existsSync(sessionsDir)) return null;

  // Search date dirs in reverse order (newest first)
  const walk = (dir: string): string | null => {
    const entries = fs.readdirSync(dir).sort().reverse();
    for (const entry of entries) {
      const full = path.join(dir, entry);
      if (fs.statSync(full).isDirectory()) {
        const result = walk(full);
        if (result) return result;
      } else if (entry.endsWith('.jsonl') && entry.includes(sessionId)) {
        return full;
      }
    }
    return null;
  };
  return walk(sessionsDir);
}

function findLatestSessionFile(): { sessionId: string; sessionFile: string } | null {
  const sessionsDir = path.join(os.homedir(), '.codex', 'sessions');
  if (!fs.existsSync(sessionsDir)) return null;

  let latestFile: string | null = null;
  let latestMtime = 0;

  const walk = (dir: string) => {
    for (const entry of fs.readdirSync(dir)) {
      const full = path.join(dir, entry);
      const stat = fs.statSync(full);
      if (stat.isDirectory()) walk(full);
      else if (entry.endsWith('.jsonl') && stat.mtimeMs > latestMtime) {
        latestMtime = stat.mtimeMs;
        latestFile = full;
      }
    }
  };
  walk(sessionsDir);

  if (!latestFile) return null;
  try {
    const firstLine = fs.readFileSync(latestFile, 'utf-8').split('\n')[0];
    if (!firstLine) return null;
    const firstMsg = JSON.parse(firstLine) as Msg;
    const payload = (firstMsg.payload ?? {}) as Msg;
    const sessionId = String(payload.id ?? path.basename(latestFile, '.jsonl'));
    return { sessionId, sessionFile: latestFile };
  } catch {
    return null;
  }
}

// ── Main ──────────────────────────────────────────────────────────

async function mainWorker(): Promise<void> {
  const scriptStart = Date.now();
  debug('Worker started');

  const sessionId = process.env._RESPAN_CODEX_SESSION!;
  const sessionFile = process.env._RESPAN_CODEX_FILE!;
  const cwd = process.env._RESPAN_CODEX_CWD ?? '';

  const creds = resolveCredentials();
  if (!creds) { log('ERROR', 'No API key'); return; }

  const config = cwd ? loadRespanConfig(path.join(cwd, '.codex', 'respan.json')) : null;

  const maxAttempts = 3;
  let turns = 0;
  try {
    for (let attempt = 0; attempt < maxAttempts; attempt++) {
      const unlock = acquireLock(LOCK_PATH);
      try {
        const state = loadState(STATE_FILE);
        const sessionState = (state[sessionId] ?? {}) as Msg;
        const lastTurnCount = Number(sessionState.turn_count ?? 0);

        const lines = fs.readFileSync(sessionFile, 'utf-8').trim().split('\n');
        const events = parseSession(lines);
        const allTurns = extractTurns(events);

        if (allTurns.length > lastTurnCount) {
          const newTurns = allTurns.slice(lastTurnCount);
          for (const turn of newTurns) {
            turns++;
            const turnNum = lastTurnCount + turns;
            const spans = createSpans(sessionId, turnNum, turn, config);
            await sendSpans(spans, creds.apiKey, creds.baseUrl, `turn_${turnNum}`);
          }
          state[sessionId] = {
            turn_count: lastTurnCount + turns,
            updated: nowISO(),
          };
          saveState(STATE_FILE, state);
        }
      } finally {
        unlock?.();
      }
      if (turns > 0) break;
      if (attempt < maxAttempts - 1) {
        await new Promise((r) => setTimeout(r, 500 * (attempt + 1)));
      }
    }
    const duration = (Date.now() - scriptStart) / 1000;
    log('INFO', `Processed ${turns} turns in ${duration.toFixed(1)}s`);
  } catch (e) {
    log('ERROR', `Failed to process session: ${e}`);
  }
}

function main(): void {
  // Worker mode: re-invoked as detached subprocess
  if (process.env._RESPAN_CODEX_WORKER === '1') {
    mainWorker().catch((e) => log('ERROR', `Worker crashed: ${e}`));
    return;
  }

  debug('Codex hook started');

  if (process.argv.length < 3) {
    debug('No argument provided');
    process.exit(0);
  }

  let payload: Msg;
  try {
    payload = JSON.parse(process.argv[2]);
  } catch (e) {
    debug(`Invalid JSON in argv[2]: ${e}`);
    process.exit(0);
  }

  const eventType = String(payload.type ?? '');
  if (eventType !== 'agent-turn-complete') {
    debug(`Ignoring event type: ${eventType}`);
    process.exit(0);
  }

  let sessionId = String(payload['thread-id'] ?? '');
  if (!sessionId) {
    debug('No thread-id in notify payload');
    process.exit(0);
  }

  // Find session file
  let sessionFile = findSessionFile(sessionId);
  if (!sessionFile) {
    const latest = findLatestSessionFile();
    if (latest) {
      sessionId = latest.sessionId;
      sessionFile = latest.sessionFile;
    } else {
      debug('No session file found');
      process.exit(0);
    }
  }

  // Fork self as detached worker so Codex CLI doesn't block
  const cwd = String(payload.cwd ?? '');
  debug(`Forking worker for session: ${sessionId}`);
  try {
    const scriptPath = __filename || process.argv[1];
    const child = execFile('node', [scriptPath], {
      env: {
        ...process.env,
        _RESPAN_CODEX_WORKER: '1',
        _RESPAN_CODEX_SESSION: sessionId,
        _RESPAN_CODEX_FILE: sessionFile!,
        _RESPAN_CODEX_CWD: cwd,
      },
      stdio: 'ignore' as any,
      detached: true,
    } as any);
    child.unref();
    debug('Worker launched');
  } catch (e) {
    log('ERROR', `Failed to fork worker: ${e}`);
  }
  process.exit(0);
}

main();
