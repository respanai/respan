import { SpanAttributes } from "@traceloop/ai-semantic-conventions";
import { RespanLogType, RespanSpanAttributes } from "@respan/respan-sdk";
import {
  GUARD_METHOD,
  SUPERAGENT_INSTRUMENTATION_NAME,
  SUPERAGENT_METADATA_CLASSIFICATION,
  SUPERAGENT_METADATA_INTEGRATION,
  SUPERAGENT_METADATA_METHOD,
  SUPERAGENT_METADATA_MODEL,
  SUPERAGENT_METADATA_REDACT_FINDINGS,
} from "./_constants.js";
import {
  extractModel,
  getAttr,
  normalizeCallInput,
  safeJsonStringify,
} from "./_serialization.js";

export type SuperagentSpanAttributeValue = string | number | boolean | string[];

export type SuperagentSpanAttributes = Record<
  string,
  SuperagentSpanAttributeValue
>;

export interface BuildSuperagentSpanAttributesOptions {
  methodName: string;
  args: unknown[];
  result?: unknown;
  error?: unknown;
  workflowName?: string;
}

function operationLogType(methodName: string): RespanLogType {
  return methodName === GUARD_METHOD ? RespanLogType.GUARDRAIL : RespanLogType.TOOL;
}

function errorMessage(error: unknown): string {
  if (error instanceof Error) {
    return error.message;
  }
  return String(error);
}

function addResultMetadata(
  attrs: SuperagentSpanAttributes,
  methodName: string,
  result: unknown,
): void {
  if (methodName === GUARD_METHOD) {
    const classification = getAttr(result, "classification");
    if (typeof classification === "string" && classification.length > 0) {
      attrs[SUPERAGENT_METADATA_CLASSIFICATION] = classification;
      attrs[RespanSpanAttributes.RESPAN_METADATA_TRIGGERED] =
        classification === "block";
    }

    attrs[RespanSpanAttributes.RESPAN_METADATA_GUARDRAIL_NAME] =
      "superagent.guard";
    return;
  }

  if (methodName === "redact") {
    const findings = getAttr(result, "findings");
    if (findings !== undefined && findings !== null) {
      attrs[SUPERAGENT_METADATA_REDACT_FINDINGS] = safeJsonStringify(findings);
    }
  }
}

export function buildSuperagentSpanAttributes({
  methodName,
  args,
  result,
  error,
  workflowName,
}: BuildSuperagentSpanAttributesOptions): SuperagentSpanAttributes {
  const operationName = `superagent.${methodName}`;
  const attrs: SuperagentSpanAttributes = {
    [RespanSpanAttributes.RESPAN_LOG_TYPE]: operationLogType(methodName),
    [SpanAttributes.TRACELOOP_ENTITY_NAME]: operationName,
    [SpanAttributes.TRACELOOP_ENTITY_PATH]: operationName,
    [SpanAttributes.TRACELOOP_ENTITY_INPUT]: safeJsonStringify(
      normalizeCallInput(methodName, args),
    ),
    [SUPERAGENT_METADATA_INTEGRATION]: SUPERAGENT_INSTRUMENTATION_NAME,
    [SUPERAGENT_METADATA_METHOD]: methodName,
  };

  if (workflowName) {
    attrs[SpanAttributes.TRACELOOP_WORKFLOW_NAME] = workflowName;
  }

  const model = extractModel(args);
  if (model) {
    attrs[SUPERAGENT_METADATA_MODEL] = model;
  }

  if (error !== undefined) {
    attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = safeJsonStringify({
      error: errorMessage(error),
    });
    return attrs;
  }

  if (result !== undefined) {
    attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = safeJsonStringify(result);
    addResultMetadata(attrs, methodName, result);
  }

  return attrs;
}
