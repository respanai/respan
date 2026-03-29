/**
 * Respan Hook for Claude Code
 *
 * Sends Claude Code conversation traces to Respan after each response.
 * Uses Claude Code's Stop hook to capture transcripts and convert them to spans.
 *
 * Span tree per turn:
 *   Root (claude-code)
 *     ├── claude.chat  (generation — model, tokens, messages)
 *     ├── Thinking 1   (if extended thinking is present)
 *     ├── Tool: Read   (if tool use occurred)
 *     └── Tool: Write  (if tool use occurred)
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

const STATE_DIR = path.join(os.homedir(), '.claude', 'state');
const LOG_FILE = path.join(STATE_DIR, 'respan_hook.log');
const STATE_FILE = path.join(STATE_DIR, 'respan_state.json');
const LOCK_PATH = path.join(STATE_DIR, 'respan_hook.lock');
const DEBUG_MODE = (process.env.CC_RESPAN_DEBUG ?? '').toLowerCase() === 'true';
const MAX_CHARS = parseInt(process.env.CC_RESPAN_MAX_CHARS ?? '4000', 10) || 4000;

initLogging(LOG_FILE, DEBUG_MODE);

// ── Message helpers ───────────────────────────────────────────────

type Msg = Record<string, unknown>;

function getContent(msg: Msg): unknown {
  if (msg.message && typeof msg.message === 'object') {
    return (msg.message as Msg).content;
  }
  return msg.content;
}

function isToolResult(msg: Msg): boolean {
  const content = getContent(msg);
  if (Array.isArray(content)) {
    return content.some(
      (item) => typeof item === 'object' && item !== null && (item as Msg).type === 'tool_result',
    );
  }
  return false;
}

function getToolCalls(msg: Msg): Msg[] {
  const content = getContent(msg);
  if (Array.isArray(content)) {
    return content.filter(
      (item) => typeof item === 'object' && item !== null && (item as Msg).type === 'tool_use',
    ) as Msg[];
  }
  return [];
}

function getTextContent(msg: Msg): string {
  const content = getContent(msg);
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    return content
      .map((item) => {
        if (typeof item === 'string') return item;
        if (typeof item === 'object' && item !== null) {
          if ((item as Msg).type === 'text') return String((item as Msg).text ?? '');
        }
        return '';
      })
      .filter(Boolean)
      .join('\n');
  }
  return '';
}

function mergeAssistantParts(parts: Msg[]): Msg {
  if (parts.length === 0) return {};
  const merged: unknown[] = [];
  for (const part of parts) {
    const content = getContent(part);
    if (Array.isArray(content)) merged.push(...content);
    else if (content) merged.push({ type: 'text', text: String(content) });
  }
  const result = { ...parts[0] };
  if (result.message && typeof result.message === 'object') {
    result.message = { ...(result.message as Msg), content: merged };
  } else {
    result.content = merged;
  }
  return result;
}

// ── Tool formatting ───────────────────────────────────────────────

function formatToolInput(toolName: string, toolInput: unknown): string {
  if (!toolInput) return '';
  const input = toolInput as Msg;

  if (['Write', 'Edit', 'MultiEdit'].includes(toolName) && typeof input === 'object') {
    const filePath = input.file_path ?? input.path ?? '';
    const content = String(input.content ?? '');
    let result = `File: ${filePath}\n`;
    if (content) {
      const preview = content.length > 2000 ? content.slice(0, 2000) + '...' : content;
      result += `Content:\n${preview}`;
    }
    return truncate(result, MAX_CHARS);
  }
  if (toolName === 'Read' && typeof input === 'object') {
    return `File: ${input.file_path ?? input.path ?? ''}`;
  }
  if (['Bash', 'Shell'].includes(toolName) && typeof input === 'object') {
    return `Command: ${input.command ?? ''}`;
  }
  try {
    return truncate(JSON.stringify(toolInput, null, 2), MAX_CHARS);
  } catch {
    return truncate(String(toolInput), MAX_CHARS);
  }
}

function formatToolOutput(toolName: string, toolOutput: unknown): string {
  if (!toolOutput) return '';
  if (typeof toolOutput === 'string') return truncate(toolOutput, MAX_CHARS);
  if (Array.isArray(toolOutput)) {
    const parts: string[] = [];
    let total = 0;
    for (const item of toolOutput) {
      if (typeof item === 'object' && item !== null) {
        const obj = item as Msg;
        if (obj.type === 'text') {
          const text = String(obj.text ?? '');
          if (total + text.length > MAX_CHARS) {
            const remaining = MAX_CHARS - total;
            if (remaining > 100) parts.push(text.slice(0, remaining) + '... (truncated)');
            break;
          }
          parts.push(text);
          total += text.length;
        } else if (obj.type === 'image') {
          parts.push('[Image output]');
        }
      } else if (typeof item === 'string') {
        if (total + item.length > MAX_CHARS) break;
        parts.push(item);
        total += item.length;
      }
    }
    return parts.join('\n');
  }
  try {
    return truncate(JSON.stringify(toolOutput, null, 2), MAX_CHARS);
  } catch {
    return truncate(String(toolOutput), MAX_CHARS);
  }
}

// ── Span creation ─────────────────────────────────────────────────

function createSpans(
  sessionId: string,
  turnNum: number,
  userMsg: Msg,
  assistantMsgs: Msg[],
  toolResults: Msg[],
  config: RespanConfig | null,
): SpanData[] {
  const spans: SpanData[] = [];

  // Extract user data
  const userText = getTextContent(userMsg);
  const userTimestamp = String(userMsg.timestamp ?? '');

  // Collect assistant text
  const textParts = assistantMsgs.map(getTextContent).filter(Boolean);
  const finalOutput = textParts.join('\n');

  // Aggregate model, usage, timing
  let model = 'claude';
  let usage: Msg | null = null;
  let requestId: string | undefined;
  let stopReason: string | undefined;
  let firstAssistantTs: string | undefined;
  let lastAssistantTs: string | undefined;

  for (const aMsg of assistantMsgs) {
    if (typeof aMsg !== 'object' || !aMsg.message) continue;
    const msgObj = aMsg.message as Msg;
    model = String(msgObj.model ?? model);
    requestId = String(aMsg.requestId ?? requestId ?? '');
    stopReason = String(msgObj.stop_reason ?? stopReason ?? '');
    const ts = String(aMsg.timestamp ?? '');
    if (ts) {
      if (!firstAssistantTs) firstAssistantTs = ts;
      lastAssistantTs = ts;
    }
    const msgUsage = msgObj.usage as Msg | undefined;
    if (msgUsage) {
      if (!usage) {
        usage = { ...msgUsage };
      } else {
        for (const key of ['input_tokens', 'output_tokens', 'cache_creation_input_tokens', 'cache_read_input_tokens']) {
          if (key in msgUsage) {
            (usage as any)[key] = ((usage as any)[key] ?? 0) + Number(msgUsage[key]);
          }
        }
        if (msgUsage.service_tier) usage.service_tier = msgUsage.service_tier;
      }
    }
  }

  // Timing
  const now = nowISO();
  const startTimeStr = userTimestamp || firstAssistantTs || now;
  const timestampStr = lastAssistantTs || firstAssistantTs || now;
  const lat = latencySeconds(startTimeStr, timestampStr);

  // Messages
  const promptMessages: Msg[] = [];
  if (userText) promptMessages.push({ role: 'user', content: userText });
  const completionMessage = finalOutput ? { role: 'assistant', content: finalOutput } : null;

  // IDs & fields
  const { workflowName, spanName, customerId } = resolveSpanFields(config, {
    workflowName: 'claude-code',
    spanName: 'claude-code',
  });
  const traceUniqueId = `${sessionId}_turn_${turnNum}`;
  const threadId = `claudecode_${sessionId}`;

  // Metadata
  const metadata = buildMetadata(config, { claude_code_turn: turnNum });
  if (requestId) metadata.request_id = requestId;
  if (stopReason) metadata.stop_reason = stopReason;

  // Usage
  const usageFields: Partial<SpanData> = {};
  if (usage) {
    const pt = Number(usage.input_tokens ?? 0);
    const ct = Number(usage.output_tokens ?? 0);
    usageFields.prompt_tokens = pt;
    usageFields.completion_tokens = ct;
    if (pt + ct > 0) usageFields.total_tokens = pt + ct;
    const cacheCreation = Number(usage.cache_creation_input_tokens ?? 0);
    const cacheRead = Number(usage.cache_read_input_tokens ?? 0);
    if (cacheCreation > 0) usageFields.prompt_tokens_details = { cache_creation_tokens: cacheCreation };
    if (cacheRead > 0) {
      usageFields.prompt_tokens_details = {
        ...usageFields.prompt_tokens_details,
        cached_tokens: cacheRead,
      };
    }
    if (usage.service_tier) metadata.service_tier = String(usage.service_tier);
  }

  // Root span
  const rootSpanId = `claudecode_${traceUniqueId}_root`;
  spans.push({
    trace_unique_id: traceUniqueId,
    session_identifier: threadId,
    thread_identifier: threadId,
    customer_identifier: customerId,
    span_unique_id: rootSpanId,
    span_name: spanName,
    span_workflow_name: workflowName,
    model,
    provider_id: '',
    span_path: '',
    input: promptMessages.length ? JSON.stringify(promptMessages) : '',
    output: finalOutput,
    timestamp: timestampStr,
    start_time: startTimeStr,
    metadata,
    ...(lat !== undefined ? { latency: lat } : {}),
  });

  // LLM generation child span
  const genStart = firstAssistantTs || startTimeStr;
  const genEnd = lastAssistantTs || timestampStr;
  const genLat = latencySeconds(genStart, genEnd);
  spans.push({
    trace_unique_id: traceUniqueId,
    span_unique_id: `claudecode_${traceUniqueId}_gen`,
    span_parent_id: rootSpanId,
    span_name: 'claude.chat',
    span_workflow_name: workflowName,
    span_path: 'claude_chat',
    model,
    provider_id: 'anthropic',
    metadata: {},
    input: promptMessages.length ? JSON.stringify(promptMessages) : '',
    output: finalOutput,
    prompt_messages: promptMessages,
    completion_message: completionMessage,
    timestamp: genEnd,
    start_time: genStart,
    ...(genLat !== undefined ? { latency: genLat } : {}),
    ...usageFields,
  });

  // Thinking child spans
  let thinkingNum = 0;
  for (const aMsg of assistantMsgs) {
    if (typeof aMsg !== 'object' || !aMsg.message) continue;
    const content = (aMsg.message as Msg).content;
    if (!Array.isArray(content)) continue;
    for (const item of content) {
      if (typeof item === 'object' && item !== null && (item as Msg).type === 'thinking') {
        const thinkingText = String((item as Msg).thinking ?? '');
        if (!thinkingText) continue;
        thinkingNum++;
        const thinkingTs = String(aMsg.timestamp ?? timestampStr);
        spans.push({
          trace_unique_id: traceUniqueId,
          span_unique_id: `claudecode_${traceUniqueId}_thinking_${thinkingNum}`,
          span_parent_id: rootSpanId,
          span_name: `Thinking ${thinkingNum}`,
          span_workflow_name: workflowName,
          span_path: 'thinking',
          provider_id: '',
          metadata: {},
          input: '',
          output: thinkingText,
          timestamp: thinkingTs,
          start_time: thinkingTs,
        });
      }
    }
  }

  // Tool child spans
  const toolCallMap = new Map<string, Msg>();
  for (const aMsg of assistantMsgs) {
    for (const tc of getToolCalls(aMsg)) {
      const id = String(tc.id ?? '');
      toolCallMap.set(id, {
        name: tc.name ?? 'unknown',
        input: tc.input,
        id,
        timestamp: aMsg.timestamp,
      });
    }
  }

  for (const tr of toolResults) {
    const trContent = getContent(tr);
    const trMeta: Msg = {};
    if (typeof tr === 'object' && tr.toolUseResult && typeof tr.toolUseResult === 'object') {
      const tur = tr.toolUseResult as Msg;
      for (const [src, dst] of [['durationMs', 'duration_ms'], ['numFiles', 'num_files'], ['filenames', 'filenames'], ['truncated', 'truncated']]) {
        if (src in tur) trMeta[dst] = tur[src];
      }
    }
    if (Array.isArray(trContent)) {
      for (const item of trContent) {
        if (typeof item === 'object' && item !== null && (item as Msg).type === 'tool_result') {
          const toolUseId = String((item as Msg).tool_use_id ?? '');
          const existing = toolCallMap.get(toolUseId);
          if (existing) {
            existing.output = (item as Msg).content;
            existing.result_metadata = trMeta;
            existing.result_timestamp = tr.timestamp;
          }
        }
      }
    }
  }

  let toolNum = 0;
  for (const [, td] of toolCallMap) {
    toolNum++;
    const toolTs = String(td.result_timestamp ?? td.timestamp ?? timestampStr);
    const toolStart = String(td.timestamp ?? startTimeStr);
    const toolLat = latencySeconds(toolStart, toolTs);
    const durationMs = (td.result_metadata as Msg | undefined)?.duration_ms;
    spans.push({
      trace_unique_id: traceUniqueId,
      span_unique_id: `claudecode_${traceUniqueId}_tool_${toolNum}`,
      span_parent_id: rootSpanId,
      span_name: `Tool: ${td.name}`,
      span_workflow_name: workflowName,
      span_path: `tool_${String(td.name).toLowerCase()}`,
      provider_id: '',
      metadata: (td.result_metadata as Record<string, unknown>) ?? {},
      input: formatToolInput(String(td.name), td.input),
      output: formatToolOutput(String(td.name), td.output),
      timestamp: toolTs,
      start_time: toolStart,
      ...(durationMs ? { latency: Number(durationMs) / 1000 } : toolLat !== undefined ? { latency: toolLat } : {}),
    });
  }

  return addDefaultsToAll(spans);
}

// ── Transcript processing ─────────────────────────────────────────

function processTranscript(
  sessionId: string,
  transcriptFile: string,
  state: Record<string, unknown>,
  apiKey: string,
  baseUrl: string,
  config: RespanConfig | null,
): { turnsProcessed: number; lastCommittedLine: number } {
  const sessionState = (state[sessionId] ?? {}) as Msg;
  const lastLine = Number(sessionState.last_line ?? 0);
  const turnCount = Number(sessionState.turn_count ?? 0);

  const content = fs.readFileSync(transcriptFile, 'utf-8');
  const lines = content.trim().split('\n');
  const totalLines = lines.length;

  if (lastLine >= totalLines) {
    debug(`No new lines to process (last: ${lastLine}, total: ${totalLines})`);
    return { turnsProcessed: 0, lastCommittedLine: lastLine };
  }

  const newMessages: (Msg & { _lineIdx: number })[] = [];
  for (let i = lastLine; i < totalLines; i++) {
    try {
      if (lines[i].trim()) {
        const msg = JSON.parse(lines[i]) as Msg;
        newMessages.push({ ...msg, _lineIdx: i });
      }
    } catch {}
  }

  if (newMessages.length === 0) return { turnsProcessed: 0, lastCommittedLine: lastLine };
  debug(`Processing ${newMessages.length} new messages`);

  // Group into turns
  let turnsProcessed = 0;
  let lastCommittedLine = lastLine;
  let currentUser: Msg | null = null;
  let currentUserLine = lastLine;
  let currentAssistants: Msg[] = [];
  let currentAssistantParts: Msg[] = [];
  let currentMsgId: string | null = null;
  let currentToolResults: Msg[] = [];

  const commitTurn = () => {
    turnsProcessed++;
    const turnNum = turnCount + turnsProcessed;
    const spans = createSpans(sessionId, turnNum, currentUser!, currentAssistants, currentToolResults, config);
    sendSpans(spans, apiKey, baseUrl, `turn_${turnNum}`);
    lastCommittedLine = totalLines;
  };

  for (const msg of newMessages) {
    const lineIdx = msg._lineIdx;
    delete (msg as any)._lineIdx;
    const role = String(msg.type ?? (msg.message as Msg | undefined)?.role ?? '');

    if (role === 'user') {
      if (isToolResult(msg)) {
        currentToolResults.push(msg);
        continue;
      }
      // New user message — finalize previous turn
      if (currentMsgId && currentAssistantParts.length) {
        currentAssistants.push(mergeAssistantParts(currentAssistantParts));
        currentAssistantParts = [];
        currentMsgId = null;
      }
      if (currentUser && currentAssistants.length) {
        commitTurn();
        lastCommittedLine = lineIdx;
      }
      currentUser = msg;
      currentUserLine = lineIdx;
      currentAssistants = [];
      currentAssistantParts = [];
      currentMsgId = null;
      currentToolResults = [];
    } else if (role === 'assistant') {
      let msgId: string | null = null;
      if (typeof msg === 'object' && msg.message) {
        msgId = String((msg.message as Msg).id ?? '') || null;
      }
      if (!msgId) {
        currentAssistantParts.push(msg);
      } else if (msgId === currentMsgId) {
        currentAssistantParts.push(msg);
      } else {
        if (currentMsgId && currentAssistantParts.length) {
          currentAssistants.push(mergeAssistantParts(currentAssistantParts));
        }
        currentMsgId = msgId;
        currentAssistantParts = [msg];
      }
    }
  }

  // Process final turn
  if (currentMsgId && currentAssistantParts.length) {
    currentAssistants.push(mergeAssistantParts(currentAssistantParts));
  }
  if (currentUser && currentAssistants.length) {
    const hasText = currentAssistants.some((m) => getTextContent(m));
    if (hasText) {
      commitTurn();
      lastCommittedLine = totalLines;
    } else {
      lastCommittedLine = currentUserLine;
      debug('Turn has assistant msgs but no text output yet, will retry');
    }
  } else {
    if (currentUser) {
      lastCommittedLine = currentUserLine;
      debug(`Incomplete turn at line ${currentUserLine}, will retry next run`);
    } else if (lastCommittedLine === lastLine) {
      lastCommittedLine = totalLines;
    }
  }

  return { turnsProcessed, lastCommittedLine };
}

// ── Stdin payload ─────────────────────────────────────────────────

function readStdinPayload(): { sessionId: string; transcriptPath: string } | null {
  if (process.stdin.isTTY) return null;
  try {
    const raw = fs.readFileSync(0, 'utf-8');
    if (!raw.trim()) return null;
    const payload = JSON.parse(raw) as Msg;
    const sessionId = String(payload.session_id ?? '');
    const transcriptPath = String(payload.transcript_path ?? '');
    if (!sessionId || !transcriptPath) return null;
    if (!fs.existsSync(transcriptPath)) return null;
    debug(`Got transcript from stdin: session=${sessionId}, path=${transcriptPath}`);
    return { sessionId, transcriptPath };
  } catch {
    return null;
  }
}

function findLatestTranscript(): { sessionId: string; transcriptPath: string } | null {
  const projectsDir = path.join(os.homedir(), '.claude', 'projects');
  if (!fs.existsSync(projectsDir)) return null;

  let latestFile: string | null = null;
  let latestMtime = 0;

  for (const projEntry of fs.readdirSync(projectsDir)) {
    const projDir = path.join(projectsDir, projEntry);
    if (!fs.statSync(projDir).isDirectory()) continue;
    for (const file of fs.readdirSync(projDir)) {
      if (!file.endsWith('.jsonl')) continue;
      const full = path.join(projDir, file);
      const mtime = fs.statSync(full).mtimeMs;
      if (mtime > latestMtime) {
        latestMtime = mtime;
        latestFile = full;
      }
    }
  }

  if (!latestFile) return null;
  try {
    const firstLine = fs.readFileSync(latestFile, 'utf-8').split('\n')[0];
    if (!firstLine) return null;
    const firstMsg = JSON.parse(firstLine) as Msg;
    const sessionId = String(firstMsg.sessionId ?? path.basename(latestFile, '.jsonl'));
    return { sessionId, transcriptPath: latestFile };
  } catch {
    return null;
  }
}

// ── Main ──────────────────────────────────────────────────────────

async function mainWorker(): Promise<void> {
  const scriptStart = Date.now();
  debug('Worker started');

  const creds = resolveCredentials();
  if (!creds) {
    log('ERROR', 'No API key found. Run: respan auth login');
    return;
  }

  const sessionId = process.env._RESPAN_SESSION_ID!;
  const transcriptPath = process.env._RESPAN_TRANSCRIPT_PATH!;
  debug(`Processing session: ${sessionId}`);

  let config: RespanConfig | null = null;
  try {
    const content = fs.readFileSync(transcriptPath, 'utf-8');
    const lines = content.split('\n');
    let cwd = '';
    for (const line of lines.slice(0, 5)) {
      if (!line.trim()) continue;
      try {
        const msg = JSON.parse(line) as Msg;
        if (msg.cwd) { cwd = String(msg.cwd); break; }
      } catch {}
    }
    if (cwd) {
      config = loadRespanConfig(path.join(cwd, '.claude', 'respan.json'));
    }
  } catch {}

  const maxAttempts = 3;
  let turns = 0;
  try {
    for (let attempt = 0; attempt < maxAttempts; attempt++) {
      const unlock = acquireLock(LOCK_PATH);
      try {
        const state = loadState(STATE_FILE);
        const result = processTranscript(sessionId, transcriptPath, state, creds.apiKey, creds.baseUrl, config);
        turns = result.turnsProcessed;
        state[sessionId] = {
          last_line: result.lastCommittedLine,
          turn_count: (Number((state[sessionId] as Msg)?.turn_count ?? 0)) + turns,
          updated: nowISO(),
        };
        saveState(STATE_FILE, state);
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
    log('ERROR', `Failed to process transcript: ${e}`);
  }
}

function main(): void {
  // Worker mode: re-invoked as detached subprocess
  if (process.env._RESPAN_WORKER === '1') {
    mainWorker().catch((e) => log('ERROR', `Worker crashed: ${e}`));
    return;
  }

  debug('Hook started');

  if ((process.env.TRACE_TO_RESPAN ?? '').toLowerCase() !== 'true') {
    debug('Tracing disabled (TRACE_TO_RESPAN != true)');
    process.exit(0);
  }

  const payload = readStdinPayload() ?? findLatestTranscript();
  if (!payload) {
    debug('No transcript file found');
    process.exit(0);
  }

  // Fork self as detached worker so Claude Code doesn't block
  const { sessionId, transcriptPath } = payload;
  debug(`Forking worker for session: ${sessionId}`);
  try {
    const scriptPath = __filename || process.argv[1];
    const child = execFile('node', [scriptPath], {
      env: {
        ...process.env,
        _RESPAN_WORKER: '1',
        _RESPAN_SESSION_ID: sessionId,
        _RESPAN_TRANSCRIPT_PATH: transcriptPath,
      },
      stdio: 'ignore' as any,
      detached: true,
    } as any);
    child.unref();
    debug('Worker launched');
  } catch (e) {
    log('ERROR', `Failed to fork worker: ${e}`);
  }
  // Exit immediately so Claude Code doesn't block
  process.exit(0);
}

main();
