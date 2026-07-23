/**
 * Respan instrumentation plugin for the AWS Bedrock Runtime TypeScript SDK.
 *
 * The plugin patches `BedrockRuntimeClient.prototype.send` from
 * `@aws-sdk/client-bedrock-runtime` and emits canonical Respan chat spans for
 * InvokeModel, InvokeModelWithResponseStream, Converse, and ConverseStream.
 */

import { hrTime } from "@opentelemetry/core";
import {
  AWS_BEDROCK_INSTRUMENTATION_NAME,
  BODY_KEY,
  CONVERSE_OPERATION,
  CONVERSE_STREAM_OPERATION,
  INVOKE_MODEL_OPERATION,
  INVOKE_MODEL_STREAM_OPERATION,
  STREAMING_OPERATIONS,
  SUPPORTED_OPERATIONS,
} from "./_constants.js";
import {
  emitBedrockSpan,
  responsePayloadForInvoke,
} from "./_otel_emitter.js";

type HrTime = [number, number];
type AnyFunction = (...args: any[]) => any;
type PatchablePrototype = { send?: AnyFunction };
type InstrumentedPrototype = PatchablePrototype & Record<PropertyKey, unknown>;
type PatchableClientConstructor = {
  prototype?: PatchablePrototype;
};

export interface AWSBedrockRuntimeModule {
  BedrockRuntimeClient?: PatchableClientConstructor;
}

export interface AWSBedrockInstrumentorOptions {
  sdkModule?: AWSBedrockRuntimeModule;
  clientClass?: PatchableClientConstructor;
}

interface CommandLike {
  input?: Record<string, unknown>;
  constructor?: { name?: string };
}

const ORIGINAL_SEND = Symbol.for("respan.instrumentation.awsBedrock.originalSend");
const PATCHED_BY = Symbol.for("respan.instrumentation.awsBedrock.patchedBy");
const INSTRUMENTOR_LOG_PREFIX = "[respan] AWSBedrockInstrumentor";

function isPromiseLike(value: unknown): value is Promise<unknown> {
  return Boolean(
    value &&
      (typeof value === "object" || typeof value === "function") &&
      typeof (value as Promise<unknown>).then === "function",
  );
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function operationNameFromCommand(command: unknown): string | undefined {
  const commandName =
    command && typeof command === "object"
      ? (command as CommandLike).constructor?.name
      : undefined;
  switch (commandName) {
    case "InvokeModelCommand":
      return INVOKE_MODEL_OPERATION;
    case "InvokeModelWithResponseStreamCommand":
      return INVOKE_MODEL_STREAM_OPERATION;
    case "ConverseCommand":
      return CONVERSE_OPERATION;
    case "ConverseStreamCommand":
      return CONVERSE_STREAM_OPERATION;
    default:
      return undefined;
  }
}

function commandInput(command: unknown): Record<string, unknown> | undefined {
  if (!isRecord(command)) {
    return undefined;
  }
  return isRecord(command.input) ? command.input : undefined;
}

function statusCodeFromResponse(response: unknown): number {
  if (!isRecord(response) || !isRecord(response.$metadata)) {
    return 200;
  }
  const statusCode = response.$metadata.httpStatusCode;
  return typeof statusCode === "number" ? statusCode : 200;
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function statusCodeFromError(error: unknown): number {
  if (!isRecord(error)) {
    return 500;
  }
  const metadataStatus = isRecord(error.$metadata)
    ? error.$metadata.httpStatusCode
    : undefined;
  const value = metadataStatus ?? error.statusCode ?? error.status;
  return typeof value === "number" && value >= 400 ? value : 500;
}

function streamKeyForResponse(response: Record<string, unknown>): string | undefined {
  if (response.stream !== undefined) {
    return "stream";
  }
  if (response[BODY_KEY] !== undefined) {
    return BODY_KEY;
  }
  return undefined;
}

function isAsyncIterable(value: unknown): value is AsyncIterable<unknown> {
  return Boolean(
    value &&
      typeof value === "object" &&
      typeof (value as AsyncIterable<unknown>)[Symbol.asyncIterator] === "function",
  );
}

function isIterable(value: unknown): value is Iterable<unknown> {
  return Boolean(
    value &&
      typeof value === "object" &&
      typeof (value as Iterable<unknown>)[Symbol.iterator] === "function",
  );
}

function wrapStreamingResponse(params: {
  response: unknown;
  operationName: string;
  apiParams?: Record<string, unknown>;
  startTimeHr: HrTime;
}): unknown {
  if (!isRecord(params.response)) {
    return params.response;
  }

  const key = streamKeyForResponse(params.response);
  if (!key) {
    emitBedrockSpan({
      operationName: params.operationName,
      apiParams: params.apiParams,
      startTimeHr: params.startTimeHr,
      responsePayload: params.response,
      statusCode: statusCodeFromResponse(params.response),
    });
    return params.response;
  }

  const stream = params.response[key];
  if (!stream || (!isAsyncIterable(stream) && !isIterable(stream))) {
    emitBedrockSpan({
      operationName: params.operationName,
      apiParams: params.apiParams,
      startTimeHr: params.startTimeHr,
      responsePayload: params.response,
      statusCode: statusCodeFromResponse(params.response),
    });
    return params.response;
  }

  params.response[key] = instrumentStream({
    stream,
    operationName: params.operationName,
    apiParams: params.apiParams,
    startTimeHr: params.startTimeHr,
  });
  return params.response;
}

function instrumentStream(params: {
  stream: AsyncIterable<unknown> | Iterable<unknown>;
  operationName: string;
  apiParams?: Record<string, unknown>;
  startTimeHr: HrTime;
}): AsyncIterable<unknown> {
  let emitted = false;
  const events: unknown[] = [];

  const emit = (error?: unknown): void => {
    if (emitted) {
      return;
    }
    emitted = true;
    emitBedrockSpan({
      operationName: params.operationName,
      apiParams: params.apiParams,
      startTimeHr: params.startTimeHr,
      streamEvents: events,
      errorMessage: error === undefined ? undefined : errorMessage(error),
      statusCode: error === undefined ? 200 : statusCodeFromError(error),
    });
  };

  const iterable: AsyncIterable<unknown> = {
    [Symbol.asyncIterator]: async function*() {
      try {
        if (isAsyncIterable(params.stream)) {
          for await (const event of params.stream) {
            events.push(event);
            yield event;
          }
        } else {
          for (const event of params.stream) {
            events.push(event);
            yield event;
          }
        }
      } catch (error) {
        emit(error);
        throw error;
      } finally {
        emit();
      }
    },
  };

  return new Proxy(iterable, {
    get(target, property, receiver) {
      if (property in target) {
        return Reflect.get(target, property, receiver);
      }
      const value = (params.stream as unknown as Record<PropertyKey, unknown>)[property];
      return typeof value === "function" ? value.bind(params.stream) : value;
    },
  });
}

function handleSuccess(params: {
  response: unknown;
  operationName: string;
  apiParams?: Record<string, unknown>;
  startTimeHr: HrTime;
}): unknown {
  if (STREAMING_OPERATIONS.has(params.operationName)) {
    return wrapStreamingResponse(params);
  }

  const responsePayload =
    params.operationName === INVOKE_MODEL_OPERATION
      ? responsePayloadForInvoke(params.response)
      : params.response;

  emitBedrockSpan({
    operationName: params.operationName,
    apiParams: params.apiParams,
    startTimeHr: params.startTimeHr,
    responsePayload,
    statusCode: statusCodeFromResponse(params.response),
  });
  return params.response;
}

function instrumentedSend(
  originalSend: AnyFunction,
  instance: unknown,
  args: unknown[],
): unknown {
  const command = args[0];
  const operationName = operationNameFromCommand(command);
  if (!operationName || !SUPPORTED_OPERATIONS.has(operationName)) {
    return originalSend.apply(instance, args);
  }

  const apiParams = commandInput(command);
  const startTimeHr = hrTime();
  let result: unknown;
  try {
    result = originalSend.apply(instance, args);
  } catch (error) {
    emitBedrockSpan({
      operationName,
      apiParams,
      startTimeHr,
      errorMessage: errorMessage(error),
      statusCode: statusCodeFromError(error),
    });
    throw error;
  }

  if (isPromiseLike(result)) {
    return result.then(
      (response) =>
        handleSuccess({
          response,
          operationName,
          apiParams,
          startTimeHr,
        }),
      (error) => {
        emitBedrockSpan({
          operationName,
          apiParams,
          startTimeHr,
          errorMessage: errorMessage(error),
          statusCode: statusCodeFromError(error),
        });
        throw error;
      },
    );
  }

  return handleSuccess({
    response: result,
    operationName,
    apiParams,
    startTimeHr,
  });
}

export class AWSBedrockInstrumentor {
  public readonly name = AWS_BEDROCK_INSTRUMENTATION_NAME;

  private readonly _sdkModule?: AWSBedrockRuntimeModule;
  private readonly _clientClass?: PatchableClientConstructor;
  private readonly _patchedPrototypes: InstrumentedPrototype[] = [];
  private _isInstrumented = false;

  constructor(options: AWSBedrockInstrumentorOptions = {}) {
    this._sdkModule = options.sdkModule;
    this._clientClass = options.clientClass;
  }

  async activate(): Promise<void> {
    if (this._isInstrumented) {
      return;
    }

    const clientClass = await this._resolveClientClass();
    if (!clientClass?.prototype) {
      throw new Error(
        "AWSBedrockInstrumentor requires BedrockRuntimeClient from @aws-sdk/client-bedrock-runtime.",
      );
    }

    this._patchPrototype(clientClass.prototype);
    this._isInstrumented = true;
  }

  deactivate(): void {
    for (const prototype of this._patchedPrototypes) {
      const original = prototype[ORIGINAL_SEND];
      if (typeof original === "function") {
        prototype.send = original as AnyFunction;
      }
      delete prototype[ORIGINAL_SEND];
      delete prototype[PATCHED_BY];
    }
    this._patchedPrototypes.length = 0;
    this._isInstrumented = false;
  }

  private async _resolveClientClass(): Promise<PatchableClientConstructor | undefined> {
    if (this._clientClass) {
      return this._clientClass;
    }
    if (this._sdkModule?.BedrockRuntimeClient) {
      return this._sdkModule.BedrockRuntimeClient;
    }

    const sdkModule = await import("@aws-sdk/client-bedrock-runtime");
    return (sdkModule as unknown as AWSBedrockRuntimeModule).BedrockRuntimeClient;
  }

  private _patchPrototype(prototype: PatchablePrototype): void {
    const mutablePrototype = prototype as InstrumentedPrototype;
    const original = mutablePrototype.send;
    if (typeof original !== "function") {
      throw new Error(
        "AWSBedrockInstrumentor requires BedrockRuntimeClient.prototype.send to be a function.",
      );
    }

    if (mutablePrototype[ORIGINAL_SEND]) {
      return;
    }

    mutablePrototype[ORIGINAL_SEND] = original;
    mutablePrototype[PATCHED_BY] = this;
    mutablePrototype.send = function patchedBedrockSend(...args: unknown[]): unknown {
      try {
        return instrumentedSend(original, this, args);
      } catch (error) {
        console.warn(`${INSTRUMENTOR_LOG_PREFIX} send wrapper failed:`, error);
        return original.apply(this, args);
      }
    };
    this._patchedPrototypes.push(mutablePrototype);
  }
}

export {
  buildBedrockAttrs,
  emitBedrockSpan,
} from "./_otel_emitter.js";
export {
  parseBedrockRequest,
  parseBedrockResponse,
  parseBedrockStreamResponse,
} from "./_translator.js";
