import { trace, type Attributes } from "@opentelemetry/api";
import { types } from "node:util";
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
export function withEveLineage<T extends object>(definition: T): T {
  if (types.isProxy(definition)) return definition;
  // Eve 0.62+ moved model runtime context from step.started to each OTel
  // destination. Keep this helper independent of Eve's removed legacy types
  // so the same package can serve both released API generations.
  if (Object.hasOwn(definition, "runtimeContext")) {
    const integration = definition as T & {
      runtimeContext?: (input: EveStepStartedInputLike) => unknown;
    };
    const authoredRuntimeContext = ownData(
      integration,
      "runtimeContext",
    ) as typeof integration.runtimeContext;
    return extend(definition, {
      runtimeContext(input: EveStepStartedInputLike): unknown {
        const result = authoredRuntimeContext?.(input);
        if (result !== undefined && !isRecord(result)) {
          return result;
        }
        const lineage = buildLineage(input);
        if (!lineage) return result;
        stampActiveTurn(lineage);
        return extend(result ?? {}, {
          [EVE_RESPAN_BRIDGE_RUNTIME_CONTEXT_KEY]: { lineage },
        });
      },
    });
  }
  const typedDefinition = definition as T & EveInstrumentationDefinitionLike;
  const events = ownData(typedDefinition, "events");
  const authoredStepStarted = (
    isRecord(events) ? ownData(events, "step.started") : undefined
  ) as ((input: EveStepStartedInputLike) => unknown) | undefined;

  const stepStarted = (input: EveStepStartedInputLike): unknown => {
    const authoredResult = authoredStepStarted?.(input);

    // Preserve Eve's own warning-only validation for forced async or malformed
    // callback results instead of silently converting them into valid output.
    if (
      authoredResult !== undefined &&
      (!isRecord(authoredResult) ||
        !isRecord(ownData(authoredResult, "runtimeContext")))
    ) {
      return authoredResult;
    }

    const lineage = buildLineage(input);
    if (!lineage) return authoredResult;
    const authoredRuntimeContext =
      authoredResult === undefined
        ? {}
        : (ownData(authoredResult, "runtimeContext") as Record<
            string,
            unknown
          >);

    stampActiveTurn(lineage);

    return {
      runtimeContext: extend(authoredRuntimeContext, {
        [EVE_RESPAN_BRIDGE_RUNTIME_CONTEXT_KEY]: { lineage },
      }),
    };
  };

  return extend(definition, {
    events: extend(isRecord(events) ? events : {}, {
      "step.started": stepStarted,
    }),
  });
}

function buildLineage(input: EveStepStartedInputLike): EveLineage | undefined {
  if (types.isProxy(input)) return undefined;
  const session = ownData(input, "session");
  if (!isRecord(session) || typeof ownData(session, "id") !== "string")
    return undefined;
  const parent = ownData(session, "parent");
  if (parent === undefined) {
    return { rootSessionId: ownData(session, "id") as string };
  }

  if (!isRecord(parent)) return undefined;
  const turn = ownData(parent, "turn");
  if (!isRecord(turn)) return undefined;
  const callId = ownData(parent, "callId"),
    rootSessionId = ownData(parent, "rootSessionId"),
    sessionId = ownData(parent, "sessionId");
  const turnId = ownData(turn, "id"),
    sequence = ownData(turn, "sequence");
  if (
    typeof callId !== "string" ||
    typeof rootSessionId !== "string" ||
    typeof sessionId !== "string" ||
    typeof turnId !== "string" ||
    typeof sequence !== "number"
  )
    return undefined;

  return {
    callId,
    rootSessionId,
    sessionId,
    turn: {
      id: turnId,
      sequence,
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
      [EVE_RESPAN_LINEAGE_ROOT_SESSION_ID_ATTRIBUTE]: lineage.rootSessionId,
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
  if (value === null || typeof value !== "object" || types.isProxy(value))
    return false;
  const prototype = Object.getPrototypeOf(value);
  return prototype === Object.prototype || prototype === null;
}

function ownData(value: object, key: string): unknown {
  const descriptor = Object.getOwnPropertyDescriptor(value, key);
  return descriptor && "value" in descriptor ? descriptor.value : undefined;
}

function extend<T extends object>(value: T, additions: object): T {
  const descriptors = Object.getOwnPropertyDescriptors(value);
  for (const key of Object.keys(additions)) delete descriptors[key];
  return Object.assign(
    Object.create(Object.getPrototypeOf(value), descriptors),
    additions,
  );
}
