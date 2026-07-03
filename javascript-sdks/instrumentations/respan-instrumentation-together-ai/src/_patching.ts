import { hrTime } from "@opentelemetry/core";
import { emitErrorSpan, emitSuccessSpan, emitToolSpansFromMessages } from "./_span_emitter.js";
import { wrapStreamingResult } from "./_streaming.js";
import type { PatchedResourceTarget, TogetherOperationSpec } from "./_types.js";

function isStreamingRequest(spec: TogetherOperationSpec, request: any): boolean {
  return (spec.kind === "chat" || spec.kind === "text" || spec.kind === "speech") &&
    request?.stream === true;
}

export function instrumentCreateResult(
  result: any,
  spec: TogetherOperationSpec,
  request: any,
  startTime: [number, number],
): any {
  if (!result || typeof result !== "object") {
    emitSuccessSpan(spec, request, startTime, result);
    return result;
  }

  let hasHandled = false;

  const handleSuccess = (value: any) => {
    if (isStreamingRequest(spec, request)) {
      return wrapStreamingResult(value, spec, request, startTime);
    }

    if (!hasHandled) {
      hasHandled = true;
      emitSuccessSpan(spec, request, startTime, value);
    }
    return value;
  };

  const handleError = (err: unknown) => {
    if (hasHandled) return;
    hasHandled = true;
    emitErrorSpan(spec, request, startTime, err);
  };

  const originalThen = typeof result.then === "function" ? result.then.bind(result) : null;
  if (originalThen) {
    result.then = function (onfulfilled?: any, onrejected?: any) {
      return originalThen(
        (value: any) => {
          const instrumentedValue = handleSuccess(value);
          return onfulfilled ? onfulfilled(instrumentedValue) : instrumentedValue;
        },
        (reason: any) => {
          handleError(reason);
          if (onrejected) return onrejected(reason);
          throw reason;
        },
      );
    };
  }

  const originalCatch = typeof result.catch === "function" ? result.catch.bind(result) : null;
  if (originalCatch) {
    result.catch = function (onrejected?: any) {
      return originalCatch((reason: any) => {
        handleError(reason);
        if (onrejected) return onrejected(reason);
        throw reason;
      });
    };
  }

  const originalWithResponse =
    typeof result.withResponse === "function" ? result.withResponse.bind(result) : null;
  if (originalWithResponse) {
    result.withResponse = async function () {
      try {
        const response = await originalWithResponse();
        return {
          ...response,
          data: handleSuccess(response.data),
        };
      } catch (err) {
        handleError(err);
        throw err;
      }
    };
  }

  const originalAsResponse =
    typeof result.asResponse === "function" ? result.asResponse.bind(result) : null;
  if (originalAsResponse) {
    result.asResponse = async function () {
      try {
        const response = await originalAsResponse();
        handleSuccess(response);
        return response;
      } catch (err) {
        handleError(err);
        throw err;
      }
    };
  }

  if (!originalThen && !originalWithResponse && !originalAsResponse) {
    handleSuccess(result);
  }

  return result;
}

export function patchResourceMethod(
  prototype: Record<string, any>,
  spec: TogetherOperationSpec,
): PatchedResourceTarget | null {
  if (!prototype || typeof prototype[spec.method] !== "function") return null;

  const patchedTarget: PatchedResourceTarget = {
    prototype,
    method: spec.method,
    original: prototype[spec.method],
  };

  prototype[spec.method] = function (this: any, body: any, options?: any) {
    const startTime = hrTime();
    try {
      if (spec.kind === "chat") {
        emitToolSpansFromMessages(body?.messages);
      }
      const result = patchedTarget.original.call(this, body, options);
      return instrumentCreateResult(result, spec, body, startTime);
    } catch (err) {
      emitErrorSpan(spec, body, startTime, err);
      throw err;
    }
  };

  return patchedTarget;
}
