import type { TogetherOperationSpec } from "./_types.js";
import { STREAM_INSTRUMENTED } from "./_constants.js";
import {
  normalizeChatToolCall,
  safeJson,
  toSerializableValue,
} from "./_helpers.js";
import { emitErrorSpan, emitSuccessSpan } from "./_span_emitter.js";

interface ToolCallAccumulator {
  id: string;
  type: string;
  index: number;
  function: {
    name: string;
    arguments: string;
  };
}

interface StreamState {
  model?: string;
  usage?: Record<string, any>;
  role: string;
  contentParts: string[];
  finishReason?: string | null;
  toolCalls: Map<number, ToolCallAccumulator>;
  speechChunkCount: number;
  speechB64Chars: number;
  rawChunks: any[];
}

function createStreamState(): StreamState {
  return {
    role: "assistant",
    contentParts: [],
    toolCalls: new Map(),
    speechChunkCount: 0,
    speechB64Chars: 0,
    rawChunks: [],
  };
}

function getToolCallAccumulator(
  state: StreamState,
  index: number,
): ToolCallAccumulator {
  let toolCall = state.toolCalls.get(index);
  if (!toolCall) {
    toolCall = {
      id: "",
      type: "function",
      index,
      function: {
        name: "",
        arguments: "",
      },
    };
    state.toolCalls.set(index, toolCall);
  }
  return toolCall;
}

function appendToolCallDelta(state: StreamState, delta: any): void {
  if (!Array.isArray(delta?.tool_calls)) return;

  for (const toolCallDelta of delta.tool_calls) {
    const index = typeof toolCallDelta?.index === "number" ? toolCallDelta.index : 0;
    const toolCall = getToolCallAccumulator(state, index);
    if (toolCallDelta.id) toolCall.id += String(toolCallDelta.id);
    if (toolCallDelta.type) toolCall.type = String(toolCallDelta.type);
    if (toolCallDelta.function?.name) {
      toolCall.function.name += String(toolCallDelta.function.name);
    }
    if (toolCallDelta.function?.arguments) {
      toolCall.function.arguments += String(toolCallDelta.function.arguments);
    }
  }
}

function updateChatStreamState(state: StreamState, chunk: any): void {
  if (chunk?.model) state.model = String(chunk.model);
  if (chunk?.usage) state.usage = { ...(state.usage ?? {}), ...chunk.usage };

  const choices = Array.isArray(chunk?.choices) ? chunk.choices : [];
  for (const choice of choices) {
    if (choice?.finish_reason !== undefined) {
      state.finishReason = choice.finish_reason;
    }
    const delta = choice?.delta ?? {};
    if (delta.role) state.role = String(delta.role);
    if (delta.content) state.contentParts.push(String(delta.content));
    if (delta.reasoning) state.contentParts.push(String(delta.reasoning));
    appendToolCallDelta(state, delta);
  }
}

function updateTextStreamState(state: StreamState, chunk: any): void {
  if (chunk?.model) state.model = String(chunk.model);
  if (chunk?.usage) state.usage = { ...(state.usage ?? {}), ...chunk.usage };
  if (chunk?.finish_reason !== undefined) state.finishReason = chunk.finish_reason;

  if (chunk?.token?.text) state.contentParts.push(String(chunk.token.text));
  const choices = Array.isArray(chunk?.choices) ? chunk.choices : [];
  for (const choice of choices) {
    if (choice?.finish_reason !== undefined) {
      state.finishReason = choice.finish_reason;
    }
    if (choice?.text) state.contentParts.push(String(choice.text));
    if (choice?.delta?.content) state.contentParts.push(String(choice.delta.content));
  }
}

function updateSpeechStreamState(state: StreamState, chunk: any): void {
  const payload = chunk?.data && chunk.data !== "[DONE]" ? chunk.data : chunk;
  if (payload?.model) state.model = String(payload.model);
  if (typeof payload?.b64 === "string") {
    state.speechChunkCount += 1;
    state.speechB64Chars += payload.b64.length;
  }
}

export function updateStreamState(
  state: StreamState,
  spec: TogetherOperationSpec,
  chunk: any,
): void {
  if (spec.kind === "chat") updateChatStreamState(state, chunk);
  else if (spec.kind === "text") updateTextStreamState(state, chunk);
  else if (spec.kind === "speech") updateSpeechStreamState(state, chunk);
  else state.rawChunks.push(toSerializableValue(chunk));
}

function buildChatResponse(state: StreamState, request: any): Record<string, any> {
  const message: Record<string, any> = {
    role: state.role || "assistant",
    content: state.contentParts.join(""),
  };
  const toolCalls = Array.from(state.toolCalls.entries())
    .sort((left, right) => left[0] - right[0])
    .map(([, toolCall]) => normalizeChatToolCall(toolCall))
    .filter(Boolean);
  if (toolCalls.length > 0) message.tool_calls = toolCalls;

  return {
    model: state.model ?? request?.model,
    choices: [
      {
        index: 0,
        finish_reason: state.finishReason ?? null,
        message,
      },
    ],
    usage: state.usage,
  };
}

function buildTextResponse(state: StreamState, request: any): Record<string, any> {
  return {
    model: state.model ?? request?.model,
    choices: [
      {
        finish_reason: state.finishReason ?? null,
        text: state.contentParts.join(""),
      },
    ],
    usage: state.usage,
  };
}

function buildSpeechResponse(state: StreamState, request: any): Record<string, any> {
  return {
    model: state.model ?? request?.model,
    object: "audio.tts.stream",
    chunks: state.speechChunkCount,
    total_b64_chars: state.speechB64Chars,
  };
}

export function buildStreamResponse(
  state: StreamState,
  spec: TogetherOperationSpec,
  request: any,
): Record<string, any> {
  if (spec.kind === "chat") return buildChatResponse(state, request);
  if (spec.kind === "text") return buildTextResponse(state, request);
  if (spec.kind === "speech") return buildSpeechResponse(state, request);
  return {
    chunks: state.rawChunks,
    serialized: safeJson(state.rawChunks),
  };
}

export function wrapStreamingResult(
  streamResult: any,
  spec: TogetherOperationSpec,
  request: any,
  startTime: [number, number],
): any {
  if (
    !streamResult ||
    typeof streamResult !== "object" ||
    streamResult[STREAM_INSTRUMENTED]
  ) {
    return streamResult;
  }

  Object.defineProperty(streamResult, STREAM_INSTRUMENTED, {
    value: true,
    configurable: true,
    enumerable: false,
  });

  const originalAsyncIterator = streamResult[Symbol.asyncIterator]?.bind(streamResult);
  if (typeof originalAsyncIterator !== "function") {
    emitSuccessSpan(spec, request, startTime, streamResult);
    return streamResult;
  }

  const state = createStreamState();
  let hasEmitted = false;

  const emitFinalSpan = (error?: unknown) => {
    if (hasEmitted) return;
    hasEmitted = true;
    if (error) {
      emitErrorSpan(spec, request, startTime, error);
      return;
    }
    emitSuccessSpan(spec, request, startTime, buildStreamResponse(state, spec, request));
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
            updateStreamState(state, spec, result.value);
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
