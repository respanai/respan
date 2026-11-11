import { trace, type Span } from "@opentelemetry/api";
import type { FetchFunction } from "@ai-sdk/provider-utils";

// The gateway returns the unique_id of the log row it wrote for a call in this
// header. That row already carries the call's usage, cost and content, so the
// AI SDK's model-call span carries the id instead (a link only) and Respan
// shows the call once. Mirrors RespanSpanAttributes.RESPAN_GATEWAY_LOG_ID and
// the OpenAI / pi instrumentations.
export const RESPAN_LOG_ID_RESPONSE_HEADER = "x-respan-log-id";
export const RESPAN_GATEWAY_LOG_ID = "respan.gateway.log_id";

// Attributes only the AI SDK's model-call span has: gen_ai.operation.name on
// AI SDK 7 (`chat <model>`), ai.operationId on the legacy
// `ai.<operation>.doGenerate` / `.doStream` spans.
const GEN_AI_OPERATION_NAME = "gen_ai.operation.name";
const AI_OPERATION_ID = "ai.operationId";

function isModelCallSpan(span: Span | undefined): span is Span {
  const attributes = (span as { attributes?: Record<string, unknown> } | undefined)
    ?.attributes;
  return Boolean(attributes?.[GEN_AI_OPERATION_NAME] ?? attributes?.[AI_OPERATION_ID]);
}

/**
 * Wrap fetch so each gateway response links the active AI SDK model-call span
 * (generate and stream) to the gateway's log row. Any other active span, such
 * as a user's task, is left alone, and tracing never changes the response.
 */
export function linkingGatewayLog(fetch?: FetchFunction): FetchFunction {
  const send: FetchFunction = fetch ?? ((input, init) => globalThis.fetch(input, init));
  return async (input, init) => {
    const span = trace.getActiveSpan();
    const response = await send(input, init);
    try {
      const logId = response.headers?.get(RESPAN_LOG_ID_RESPONSE_HEADER);
      if (logId && isModelCallSpan(span) && span.isRecording()) {
        span.setAttribute(RESPAN_GATEWAY_LOG_ID, logId);
      }
    } catch {
      /* tracing must not change provider results */
    }
    return response;
  };
}
