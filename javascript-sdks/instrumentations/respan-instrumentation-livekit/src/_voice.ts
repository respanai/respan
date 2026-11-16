import { createHash } from "node:crypto";
import { RespanSpanAttributes } from "@respan/respan-sdk";
import {
  LIVEKIT_SPAN_NAMES,
  LK_E2E_LATENCY,
  LK_RESPONSE_TTFB,
  LK_RESPONSE_TTFT,
  LK_SPEECH_INTERRUPTED,
  LK_TRANSCRIPTION_DELAY,
  VOICE_KEYS,
  VOICE_METADATA_SEGMENT,
} from "./_constants.js";

/**
 * A LiveKit call as a Respan session: one trace per turn, timings on its root.
 *
 * LiveKit traces a whole call as one trace (`agent_session` at the root, one
 * `agent_turn` per reply). Respan reads a session as its traces, one per turn,
 * so each `agent_turn` subtree is exported under a trace of its own (derived
 * from the call's trace id and the turn's span id, so every span of a turn
 * agrees without waiting for the others), the turn root is exported as a root,
 * and every span of the call carries the call's trace id as its session id
 * unless the user set one. The turn root gets `respan.metadata.voice.*` in ms.
 */
const VOICE_METADATA_PREFIX = `${RespanSpanAttributes.RESPAN_METADATA}.${VOICE_METADATA_SEGMENT}.`;
// LiveKit attribute (seconds) -> voice key (ms). On the agent_turn itself:
const TURN_TIMINGS: ReadonlyArray<readonly [string, string]> = [
  [LK_E2E_LATENCY, VOICE_KEYS.WAIT_MS],
  [LK_TRANSCRIPTION_DELAY, VOICE_KEYS.STT_MS],
];
// On a child of the turn; the first one seen wins (what the user waited on):
const CHILD_TIMINGS: ReadonlyArray<readonly [string, string]> = [
  [LK_RESPONSE_TTFT, VOICE_KEYS.LLM_TTFT_MS],
  [LK_RESPONSE_TTFB, VOICE_KEYS.TTS_TTFA_MS],
];
const MAX_CALLS = 4096;

type Attrs = Record<string, unknown>;
interface Identity {
  readonly key: string;
  readonly spanId: string;
  readonly traceId: string;
}

function secondsToMs(value: unknown): number | undefined {
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0) {
    return undefined;
  }
  return Math.round(value * 10000) / 10;
}

export function turnTraceId(callTraceId: string, turnSpanId: string): string {
  return createHash("sha256")
    .update(`${callTraceId}${turnSpanId}`)
    .digest("hex")
    .slice(0, 32);
}

export class VoiceTurns {
  // span key -> its turn's identity
  private readonly _turnOf = new Map<string, Identity>();
  // turn key -> timings its children reported, plus its position
  private readonly _timings = new Map<string, Record<string, unknown>>();
  // call trace id -> turns started so far (insertion order bounds it)
  private readonly _turnCounts = new Map<string, number>();

  start(
    identity: Identity | undefined,
    parentKey: string | undefined,
    spanName: string,
  ): void {
    if (!identity) return;
    if (spanName === LIVEKIT_SPAN_NAMES.AGENT_TURN) {
      this._turnOf.set(identity.key, identity);
      const index = this._turnCounts.get(identity.traceId) ?? 0;
      this._turnCounts.delete(identity.traceId);
      this._turnCounts.set(identity.traceId, index + 1);
      while (this._turnCounts.size > MAX_CALLS) {
        const oldest = this._turnCounts.keys().next().value;
        if (oldest === undefined) break;
        this._turnCounts.delete(oldest);
      }
      this._timings.set(identity.key, { [VOICE_KEYS.TURN_INDEX]: index });
      return;
    }
    const turn = parentKey ? this._turnOf.get(parentKey) : undefined;
    if (turn) this._turnOf.set(identity.key, turn);
  }

  /**
   * Group an ended span: `raw` is the span's own attributes (LiveKit keys
   * intact), `exported` the attributes it will be exported with.
   */
  end(identity: Identity | undefined, raw: Attrs, exported: Attrs): void {
    if (!identity) return;
    const turn = this._turnOf.get(identity.key);
    this._turnOf.delete(identity.key);
    if (!exported[RespanSpanAttributes.RESPAN_SESSION_ID]) {
      exported[RespanSpanAttributes.RESPAN_SESSION_ID] = identity.traceId;
    }
    if (!turn) return;
    exported[RespanSpanAttributes.RESPAN_INTERNAL_EXPORT_TRACE_ID] = turnTraceId(
      identity.traceId,
      turn.spanId,
    );
    if (turn.key !== identity.key) {
      const timings = this._timings.get(turn.key);
      if (!timings) return;
      for (const [lkKey, voiceKey] of CHILD_TIMINGS) {
        const reading = secondsToMs(raw[lkKey]);
        if (reading !== undefined && !(voiceKey in timings)) {
          timings[voiceKey] = reading;
        }
      }
      return;
    }
    // The turn's root: exported as a root, with its timings.
    exported[RespanSpanAttributes.RESPAN_INTERNAL_EXPORT_PARENT] = "";
    const timings = this._timings.get(turn.key) ?? {};
    this._timings.delete(turn.key);
    for (const [lkKey, voiceKey] of TURN_TIMINGS) {
      const reading = secondsToMs(raw[lkKey]);
      if (reading !== undefined) timings[voiceKey] = reading;
    }
    if (typeof raw[LK_SPEECH_INTERRUPTED] === "boolean") {
      timings[VOICE_KEYS.INTERRUPTED] = raw[LK_SPEECH_INTERRUPTED];
    }
    for (const [voiceKey, value] of Object.entries(timings)) {
      exported[`${VOICE_METADATA_PREFIX}${voiceKey}`] = value;
    }
  }

  dispose(): void {
    this._turnOf.clear();
    this._timings.clear();
    this._turnCounts.clear();
  }
}
