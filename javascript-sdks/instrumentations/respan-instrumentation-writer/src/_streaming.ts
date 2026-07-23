import { hrTime } from "@opentelemetry/core";
import {
  buildChatCompletionFromStreamState,
  buildCompletionFromStreamState,
  createChatStreamState,
  createTextStreamState,
  updateChatStreamState,
  updateTextStreamState,
} from "./_helpers.js";
import {
  emitOperationError,
  emitOperationSuccess,
  emitToolSpansFromMessages,
  type WriterOperationType,
} from "./_span_emitter.js";

const STREAM_INSTRUMENTED = Symbol.for("respan.writer.stream.instrumented");
const PROMISE_PROXY = Symbol.for("respan.writer.promise.proxy");

interface InstrumentationState {
  handled: boolean;
}

function emitSuccessOnce(
  state: InstrumentationState,
  type: WriterOperationType,
  body: Record<string, any>,
  startTime: [number, number],
  response: unknown,
): void {
  if (state.handled) return;
  state.handled = true;
  emitOperationSuccess({ type, body, startTime, response });
}

function emitErrorOnce(
  state: InstrumentationState,
  type: WriterOperationType,
  body: Record<string, any>,
  startTime: [number, number],
  error: unknown,
): void {
  if (state.handled) return;
  state.handled = true;
  emitOperationError({ type, body, startTime, error });
}

function instrumentAsyncIterableStream(
  streamResult: any,
  type: WriterOperationType,
  body: Record<string, any>,
  startTime: [number, number],
  state: InstrumentationState,
): any {
  if (
    !streamResult ||
    typeof streamResult !== "object" ||
    streamResult[STREAM_INSTRUMENTED]
  ) {
    return streamResult;
  }

  const originalAsyncIterator = streamResult[Symbol.asyncIterator]?.bind(streamResult);
  if (typeof originalAsyncIterator !== "function") {
    emitSuccessOnce(state, type, body, startTime, streamResult);
    return streamResult;
  }

  Object.defineProperty(streamResult, STREAM_INSTRUMENTED, {
    value: true,
    configurable: true,
    enumerable: false,
  });

  const streamState = type === "chat"
    ? createChatStreamState(body)
    : createTextStreamState(body);

  const buildResponse = () => type === "chat"
    ? buildChatCompletionFromStreamState(streamState as ReturnType<typeof createChatStreamState>, body)
    : buildCompletionFromStreamState(streamState as ReturnType<typeof createTextStreamState>, body);

  const updateState = (chunk: unknown) => {
    if (type === "chat") {
      updateChatStreamState(streamState as ReturnType<typeof createChatStreamState>, chunk);
    } else {
      updateTextStreamState(streamState as ReturnType<typeof createTextStreamState>, chunk);
    }
  };

  const emitFinalSpan = (error?: unknown) => {
    if (error) {
      emitErrorOnce(state, type, body, startTime, error);
      return;
    }
    emitSuccessOnce(state, type, body, startTime, buildResponse());
  };

  streamResult[Symbol.asyncIterator] = function () {
    const iterator = originalAsyncIterator();

    return {
      async next(...args: any[]) {
        try {
          const result = await iterator.next(...args);
          if (result.done) {
            emitFinalSpan();
          } else {
            updateState(result.value);
          }
          return result;
        } catch (err) {
          emitFinalSpan(err);
          throw err;
        }
      },

      async return(value?: any) {
        try {
          const result = typeof iterator.return === "function"
            ? await iterator.return(value)
            : { done: true, value };
          emitFinalSpan();
          return result;
        } catch (err) {
          emitFinalSpan(err);
          throw err;
        }
      },

      async throw(err?: any) {
        emitFinalSpan(err);
        if (typeof iterator.throw === "function") {
          return iterator.throw(err);
        }
        throw err;
      },

      [Symbol.asyncIterator]() {
        return this;
      },
    };
  };

  return streamResult;
}

function handleSuccessValue(
  state: InstrumentationState,
  type: WriterOperationType,
  body: Record<string, any>,
  startTime: [number, number],
  value: any,
): any {
  if (body.stream === true) {
    return instrumentAsyncIterableStream(value, type, body, startTime, state);
  }

  emitSuccessOnce(state, type, body, startTime, value);
  return value;
}

export function instrumentApiPromise(
  result: any,
  type: WriterOperationType,
  body: Record<string, any>,
  startTime: [number, number] = hrTime(),
  state: InstrumentationState = { handled: false },
): any {
  if (!result || typeof result !== "object") {
    return result;
  }
  if (result[PROMISE_PROXY]) {
    return result[PROMISE_PROXY];
  }

  const originalThen = typeof result.then === "function" ? result.then.bind(result) : null;
  const wrappedThen = originalThen
    ? function (onfulfilled?: any, onrejected?: any) {
        return originalThen(
          (value: any) => {
            const instrumentedValue = handleSuccessValue(state, type, body, startTime, value);
            return onfulfilled ? onfulfilled(instrumentedValue) : instrumentedValue;
          },
          (reason: any) => {
            emitErrorOnce(state, type, body, startTime, reason);
            if (onrejected) {
              return onrejected(reason);
            }
            throw reason;
          },
        );
      }
    : undefined;

  const originalCatch = typeof result.catch === "function" ? result.catch.bind(result) : null;
  const wrappedCatch = originalCatch
    ? function (onrejected?: any) {
        return originalCatch((reason: any) => {
          emitErrorOnce(state, type, body, startTime, reason);
          if (onrejected) {
            return onrejected(reason);
          }
          throw reason;
        });
      }
    : undefined;

  const originalWithResponse =
    typeof result.withResponse === "function" ? result.withResponse.bind(result) : null;
  const wrappedWithResponse = originalWithResponse
    ? async function () {
        try {
          const response = await originalWithResponse();
          return {
            ...response,
            data: handleSuccessValue(state, type, body, startTime, response.data),
          };
        } catch (err) {
          emitErrorOnce(state, type, body, startTime, err);
          throw err;
        }
      }
    : undefined;

  const originalThenUnwrap =
    typeof result._thenUnwrap === "function" ? result._thenUnwrap.bind(result) : null;
  const wrappedThenUnwrap = originalThenUnwrap
    ? function (...args: any[]) {
        return instrumentApiPromise(originalThenUnwrap(...args), type, body, startTime, state);
      }
    : undefined;

  const proxy = new Proxy(result, {
    get(target, prop, receiver) {
      if (prop === "then" && wrappedThen) return wrappedThen;
      if (prop === "catch" && wrappedCatch) return wrappedCatch;
      if (prop === "withResponse" && wrappedWithResponse) return wrappedWithResponse;
      if (prop === "_thenUnwrap" && wrappedThenUnwrap) return wrappedThenUnwrap;
      const value = Reflect.get(target, prop, receiver);
      return typeof value === "function" ? value.bind(target) : value;
    },
  });

  Object.defineProperty(result, PROMISE_PROXY, {
    value: proxy,
    configurable: true,
    enumerable: false,
  });

  return proxy;
}
export interface PatchedMethodTarget {
  target: any;
  methodName: string;
  originalMethod: any;
}

export function patchWriterMethod(
  target: any,
  methodName: string,
  type: WriterOperationType,
): PatchedMethodTarget | null {
  if (!target || typeof target[methodName] !== "function") {
    return null;
  }

  const patchedTarget: PatchedMethodTarget = {
    target,
    methodName,
    originalMethod: target[methodName],
  };

  target[methodName] = function (this: any, body: any, options?: any) {
    const startTime = hrTime();
    const normalizedBody = body && typeof body === "object" ? body : {};
    try {
      if (type === "chat") {
        emitToolSpansFromMessages(normalizedBody.messages);
      }
      return instrumentApiPromise(
        patchedTarget.originalMethod.call(this, body, options),
        type,
        normalizedBody,
        startTime,
      );
    } catch (err) {
      emitOperationError({ type, body: normalizedBody, startTime, error: err });
      throw err;
    }
  };

  return patchedTarget;
}
