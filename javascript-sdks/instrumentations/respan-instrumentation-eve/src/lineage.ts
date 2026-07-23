import { trace, type Attributes } from "@opentelemetry/api";
import type { InstrumentationDefinition } from "eve/instrumentation";
import {
  EVE_RESPAN_BRIDGE_RUNTIME_CONTEXT_KEY,
  EVE_RESPAN_LINEAGE_PARENT_CALL_ID_ATTRIBUTE,
  EVE_RESPAN_LINEAGE_PARENT_SESSION_ID_ATTRIBUTE,
  EVE_RESPAN_LINEAGE_PARENT_TURN_ID_ATTRIBUTE,
  EVE_RESPAN_LINEAGE_PARENT_TURN_SEQUENCE_ATTRIBUTE,
  EVE_RESPAN_LINEAGE_ROOT_SESSION_ID_ATTRIBUTE,
} from "./constants/lineage.js";

type StepStartedHandler = (...args: never[]) => unknown;

interface EveInstrumentationDefinitionLike {
  readonly events?: {
    readonly "step.started"?: StepStartedHandler;
  };
}

interface EveStepStartedInputLike {
  readonly session: {
    readonly id: string;
    readonly parent?: {
      readonly callId: string;
      readonly rootSessionId: string;
      readonly sessionId: string;
      readonly turn: {
        readonly id: string;
        readonly sequence: number;
      };
    };
  };
}

interface EveLineage {
  readonly callId?: string;
  readonly rootSessionId: string;
  readonly sessionId?: string;
  readonly turn?: {
    readonly id: string;
    readonly sequence: number;
  };
}

/**
 * Adds Respan delegation lineage to an Eve instrumentation definition.
 *
 * Wrap the result of Eve's `defineInstrumentation(...)`. Any authored
 * `events["step.started"]` callback is invoked once and its runtime context is
 * preserved. The private `__respan_eve` key is helper-owned and wins over an
 * authored value with the same name.
 */
export function withEveLineage<T extends InstrumentationDefinition>(
  definition: T,
): T {
  const typedDefinition = definition as T & EveInstrumentationDefinitionLike;
  const authoredStepStarted = typedDefinition.events?.["step.started"] as
    | ((input: EveStepStartedInputLike) => unknown)
    | undefined;

  const stepStarted = (input: EveStepStartedInputLike): unknown => {
    const authoredResult = authoredStepStarted?.(input);

    // Preserve Eve's own warning-only validation for forced async or malformed
    // callback results instead of silently converting them into valid output.
    if (
      authoredResult !== undefined &&
      (!isRecord(authoredResult) || !isRecord(authoredResult.runtimeContext))
    ) {
      return authoredResult;
    }

    const lineage = buildLineage(input);
    const authoredRuntimeContext =
      authoredResult === undefined
        ? {}
        : (authoredResult.runtimeContext as Record<string, unknown>);

    stampActiveTurn(lineage);

    return {
      runtimeContext: {
        ...authoredRuntimeContext,
        [EVE_RESPAN_BRIDGE_RUNTIME_CONTEXT_KEY]: { lineage },
      },
    };
  };

  return {
    ...definition,
    events: {
      ...typedDefinition.events,
      "step.started": stepStarted,
    },
  } as T;
}

function buildLineage(input: EveStepStartedInputLike): EveLineage {
  const parent = input.session.parent;
  if (parent === undefined) {
    return { rootSessionId: input.session.id };
  }

  return {
    callId: parent.callId,
    rootSessionId: parent.rootSessionId,
    sessionId: parent.sessionId,
    turn: {
      id: parent.turn.id,
      sequence: parent.turn.sequence,
    },
  };
}

/**
 * Eve invokes the authored callback inside the active `ai.eve.turn` context on
 * the first step. Mirroring the flattened bridge attributes here lets the turn
 * root and the AI SDK children receive identical grouping without maintaining
 * cross-span state in the translator. Continuation steps expose a non-recording
 * remote parent, so this is intentionally best-effort.
 */
function stampActiveTurn(lineage: EveLineage): void {
  try {
    const activeSpan = trace.getActiveSpan();
    if (activeSpan === undefined || !activeSpan.isRecording()) {
      return;
    }

    const attributes: Attributes = {
      [EVE_RESPAN_LINEAGE_ROOT_SESSION_ID_ATTRIBUTE]:
        lineage.rootSessionId,
    };
    if (lineage.sessionId !== undefined) {
      attributes[EVE_RESPAN_LINEAGE_PARENT_SESSION_ID_ATTRIBUTE] =
        lineage.sessionId;
    }
    if (lineage.callId !== undefined) {
      attributes[EVE_RESPAN_LINEAGE_PARENT_CALL_ID_ATTRIBUTE] = lineage.callId;
    }
    if (lineage.turn !== undefined) {
      attributes[EVE_RESPAN_LINEAGE_PARENT_TURN_ID_ATTRIBUTE] = lineage.turn.id;
      attributes[EVE_RESPAN_LINEAGE_PARENT_TURN_SEQUENCE_ATTRIBUTE] =
        lineage.turn.sequence;
    }
    activeSpan.setAttributes(attributes);
  } catch {
    // Observability enrichment must never interrupt an Eve turn.
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}
