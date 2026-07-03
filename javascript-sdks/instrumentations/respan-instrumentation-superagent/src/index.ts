/**
 * Respan instrumentation plugin for the Superagent safety-agent TypeScript SDK.
 *
 * Patches `SafetyClient` methods and emits canonical Respan spans for guardrail,
 * redaction, and repository-scan operations.
 */

import {
  SpanStatusCode,
  context,
  trace,
  type Span,
  type Tracer,
} from "@opentelemetry/api";
import { WORKFLOW_NAME_KEY } from "@respan/tracing";
import type { SafetyClient } from "safety-agent";
import type * as SafetyAgentModule from "safety-agent";
import {
  SAFETY_AGENT_MODULE_NAME,
  SUPERAGENT_INSTRUMENTATION_NAME,
  SUPPORTED_METHODS,
} from "./_constants.js";
import { buildSuperagentSpanAttributes } from "./_span_attributes.js";

type OriginalMethod = (
  this: SafetyClient,
  ...args: unknown[]
) => unknown | Promise<unknown>;

type SafetyClientPrototype = Record<string, OriginalMethod>;

export interface SuperagentInstrumentorOptions {
  methods?: string[];
  safetyAgentModule?: typeof SafetyAgentModule;
}

const ORIGINAL_METHODS = new Map<string, OriginalMethod>();

let patchedPrototype: SafetyClientPrototype | null = null;
let activeInstances = 0;

function setSpanAttributes(
  span: Span,
  attrs: Record<string, string | number | boolean | string[]>,
): void {
  for (const [key, value] of Object.entries(attrs)) {
    span.setAttribute(key, value);
  }
}

function currentWorkflowName(): string | undefined {
  const value = context.active().getValue(WORKFLOW_NAME_KEY);
  return typeof value === "string" && value.length > 0 ? value : undefined;
}

function wrapMethod(methodName: string, original: OriginalMethod): OriginalMethod {
  return async function wrappedSuperagentMethod(
    this: SafetyClient,
    ...args: unknown[]
  ): Promise<unknown> {
    const tracer: Tracer = trace.getTracer(SUPERAGENT_INSTRUMENTATION_NAME);
    const operationName = `superagent.${methodName}`;
    const workflowName = currentWorkflowName();

    return await tracer.startActiveSpan(
      operationName,
      {
        attributes: buildSuperagentSpanAttributes({
          methodName,
          args,
          workflowName,
        }),
      },
      async (span) => {
        try {
          const result = await original.apply(this, args);
          setSpanAttributes(
            span,
            buildSuperagentSpanAttributes({
              methodName,
              args,
              result,
              workflowName,
            }),
          );
          span.setStatus({ code: SpanStatusCode.OK });
          return result;
        } catch (error) {
          setSpanAttributes(
            span,
            buildSuperagentSpanAttributes({
              methodName,
              args,
              error,
              workflowName,
            }),
          );
          span.recordException(error as Error);
          span.setStatus({
            code: SpanStatusCode.ERROR,
            message: error instanceof Error ? error.message : String(error),
          });
          throw error;
        } finally {
          span.end();
        }
      },
    );
  };
}

function patchSafetyClient(
  prototype: SafetyClientPrototype,
  methodNames: string[],
): boolean {
  let patchedAny = false;

  for (const methodName of methodNames) {
    const original = prototype[methodName];
    if (typeof original !== "function") {
      continue;
    }

    if (!ORIGINAL_METHODS.has(methodName)) {
      ORIGINAL_METHODS.set(methodName, original);
      Object.defineProperty(prototype, methodName, {
        configurable: true,
        writable: true,
        value: wrapMethod(methodName, original),
      });
    }

    patchedAny = true;
  }

  if (patchedAny) {
    patchedPrototype = prototype;
  }

  return patchedAny;
}

function restoreSafetyClient(): void {
  if (!patchedPrototype) {
    ORIGINAL_METHODS.clear();
    return;
  }

  for (const [methodName, original] of ORIGINAL_METHODS.entries()) {
    Object.defineProperty(patchedPrototype, methodName, {
      configurable: true,
      writable: true,
      value: original,
    });
  }

  ORIGINAL_METHODS.clear();
  patchedPrototype = null;
}

export class SuperagentInstrumentor {
  public readonly name = SUPERAGENT_INSTRUMENTATION_NAME;

  private readonly _methods: string[];
  private readonly _safetyAgentModule?: typeof SafetyAgentModule;
  private _isInstrumented = false;

  constructor(options: SuperagentInstrumentorOptions = {}) {
    this._methods = options.methods ?? [...SUPPORTED_METHODS];
    this._safetyAgentModule = options.safetyAgentModule;
  }

  async activate(): Promise<void> {
    if (this._isInstrumented) {
      return;
    }

    let safetyAgentModule: typeof SafetyAgentModule;
    try {
      safetyAgentModule =
        this._safetyAgentModule ??
        (await import(SAFETY_AGENT_MODULE_NAME));
    } catch (error) {
      console.warn(
        `Failed to activate Superagent instrumentation; missing dependency ${SAFETY_AGENT_MODULE_NAME}:`,
        error,
      );
      return;
    }

    const prototype = safetyAgentModule.SafetyClient?.prototype as
      | unknown as SafetyClientPrototype
      | undefined;

    if (!prototype) {
      console.warn(
        "Failed to activate Superagent instrumentation; SafetyClient prototype was not found.",
      );
      return;
    }

    if (!patchSafetyClient(prototype, this._methods)) {
      console.warn(
        "Failed to activate Superagent instrumentation; no compatible SafetyClient methods found.",
      );
      return;
    }

    activeInstances += 1;
    this._isInstrumented = true;
  }

  deactivate(): void {
    if (!this._isInstrumented) {
      return;
    }

    activeInstances = Math.max(0, activeInstances - 1);
    if (activeInstances === 0) {
      restoreSafetyClient();
    }

    this._isInstrumented = false;
  }
}
