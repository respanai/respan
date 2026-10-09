import { context } from "@opentelemetry/api";
import {
  data,
  internalWriterCall,
  refresh,
  snapshot,
  type CaptureOptions,
} from "./_privacy.js";
import {
  buildChatCompletionFromStreamState,
  buildCompletionFromStreamState,
  createChatStreamState,
  createTextStreamState,
  updateChatStreamState,
  updateTextStreamState,
} from "./_helpers.js";
import {
  finishOperation,
  startOperation,
  type Operation,
  type WriterOperationType,
} from "./_span_emitter.js";

function observeStream(stream: any, op: Operation): void {
  const nativeFactory = typeof data(stream, "iterator") === "function";
  const key = nativeFactory ? "iterator" : Symbol.asyncIterator;
  const factory = stream?.[key];
  if (typeof factory !== "function") {
    finishOperation(op, stream);
    return;
  }
  const state =
    op.type === "chat" ? createChatStreamState() : createTextStreamState();
  const final = () =>
    op.type === "chat"
      ? buildChatCompletionFromStreamState(state as any)
      : buildCompletionFromStreamState(state as any);
  stream[key] = function (this: any, ...args: any[]) {
    const iterator = factory.apply(this, args);
    // Keep the native iterator, iterator result, chunk and controller identities.
    for (const key of ["next", "return", "throw"] as const) {
      const original = iterator[key];
      if (typeof original !== "function") continue;
      iterator[key] = function (this: any, ...callArgs: any[]) {
        let result: any;
        try {
          result = original.apply(this, callArgs);
        } catch (error) {
          finishOperation(op, undefined, error);
          throw error;
        }
        result.then(
          (item: any) => {
            try {
              refresh(op.policy);
              if (key !== "next" || item.done) finishOperation(op, final());
              else if (op.policy.outputs) {
                const chunk = snapshot(item.value);
                if (op.type === "chat")
                  updateChatStreamState(state as any, chunk);
                else updateTextStreamState(state as any, chunk);
              } else {
                // Usage/model fields remain useful without retaining chunk content.
                const chunk = {
                  model: data(item.value, "model"),
                  usage: snapshot(data(item.value, "usage")),
                };
                if (op.type === "chat")
                  updateChatStreamState(state as any, chunk);
                else updateTextStreamState(state as any, chunk);
              }
            } catch {
              /* isolated observation */
            }
          },
          (error: unknown) => finishOperation(op, undefined, error),
        );
        return result;
      };
    }
    return iterator;
  };
}

/** Observe Stainless' native lazy parse boundary without a Proxy or eager parse. */
export function instrumentApiPromise(result: any, op: Operation): any {
  if (!result || typeof result !== "object") return result;
  const parseResponse = result.parseResponse;
  if (typeof parseResponse !== "function" || !result.responsePromise?.then)
    return result;
  result.responsePromise.then(
    (props: any) => {
      op.status = props.response.status;
    },
    (error: unknown) => finishOperation(op, undefined, error),
  );
  result.parseResponse = function (this: any, ...args: any[]) {
    let parsed: any;
    try {
      parsed = parseResponse.apply(this, args);
    } catch (error) {
      finishOperation(op, undefined, error);
      throw error;
    }
    parsed.then(
      (value: any) => {
        try {
          if (data(op.body, "stream") === true) observeStream(value, op);
          else finishOperation(op, value);
        } catch {
          /* isolated observation */
        }
      },
      (error: unknown) => finishOperation(op, undefined, error),
    );
    return parsed;
  };
  const asResponse = result.asResponse;
  if (typeof asResponse === "function")
    result.asResponse = function (this: any, ...args: any[]) {
      const promise = asResponse.apply(this, args);
      promise.then(
        () => {
          if (!this.parsedPromise) finishOperation(op);
        },
        (error: unknown) => finishOperation(op, undefined, error),
      );
      return promise;
    };
  return result;
}
export interface PatchedMethodTarget {
  target: any;
  methodName: string;
  originalMethod: any;
  wrappedMethod: any;
  owners: Set<CaptureOptions>;
}
export function patchWriterMethod(
  target: any,
  methodName: string,
  type: WriterOperationType,
  owners = new Set<CaptureOptions>([{}]),
): PatchedMethodTarget | null {
  const originalMethod = target?.[methodName];
  if (typeof originalMethod !== "function") return null;
  const patch: PatchedMethodTarget = {
    target,
    methodName,
    originalMethod,
    wrappedMethod: undefined,
    owners,
  };
  patch.wrappedMethod = function (this: any, ...args: any[]) {
    if (
      owners.size === 0 ||
      context.active().getValue(internalWriterCall) === true
    )
      return originalMethod.apply(this, args);
    const options: CaptureOptions = {
      traceContent: true,
      recordInputs: true,
      recordOutputs: true,
    };
    for (const owner of owners) {
      options.traceContent &&= owner.traceContent !== false;
      options.recordInputs &&= owner.recordInputs !== false;
      options.recordOutputs &&= owner.recordOutputs !== false;
    }
    let op: Operation | undefined;
    try {
      op = startOperation(type, args[0], options);
    } catch {
      /* fail open for native behavior */
    }
    let result: any;
    try {
      result = context.with(
        context.active().setValue(internalWriterCall, true),
        () => originalMethod.apply(this, args),
      );
    } catch (error) {
      if (op) finishOperation(op, undefined, error);
      throw error;
    }
    if (op)
      try {
        instrumentApiPromise(result, op);
      } catch {
        /* keep native result */
      }
    return result;
  };
  target[methodName] = patch.wrappedMethod;
  return patch;
}
