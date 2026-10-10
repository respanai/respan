import { EventStreamCodec } from "@smithy/eventstream-codec";
import { fromUtf8, toUtf8 } from "@smithy/util-utf8";
const encoder = new TextEncoder();
export const encode = (value) => encoder.encode(JSON.stringify(value));
export function eventBody(events, invoke = false, { failAfter } = {}) {
  const codec = new EventStreamCodec(toUtf8, fromUtf8);
  return (async function* () {
    let i = 0;
    for (const event of events) {
      if (failAfter !== undefined && i++ === failAfter)
        throw new Error("native stream transport failure");
      const type = invoke ? "chunk" : Object.keys(event)[0];
      const payload = invoke
        ? { bytes: Buffer.from(encode(event)).toString("base64") }
        : event[type];
      yield codec.encode({
        headers: {
          ":message-type": { type: "string", value: "event" },
          ":event-type": { type: "string", value: type },
          ":content-type": { type: "string", value: "application/json" },
        },
        body: encode(payload),
      });
    }
  })();
}
export const toolEvents = [
  { messageStart: { role: "assistant" } },
  {
    contentBlockStart: {
      contentBlockIndex: 1,
      start: { toolUse: { toolUseId: "native-tool", name: "lookup" } },
    },
  },
  {
    contentBlockDelta: {
      contentBlockIndex: 1,
      delta: { toolUse: { input: '{"city":' } },
    },
  },
  {
    contentBlockDelta: {
      contentBlockIndex: 1,
      delta: {
        toolUse: {
          input: '"Paris","enabled":false,"count":0,"empty":"","nil":null}',
        },
      },
    },
  },
  { contentBlockDelta: { contentBlockIndex: 0, delta: { text: "native " } } },
  { contentBlockDelta: { contentBlockIndex: 0, delta: { text: "stream" } } },
  { messageStop: { stopReason: "tool_use" } },
  {
    metadata: {
      usage: {
        inputTokens: 0,
        outputTokens: 0,
        totalTokens: 0,
        cacheReadInputTokens: 0,
        cacheWriteInputTokens: 0,
      },
    },
  },
];
export const invokeEvents = [
  {
    type: "message_start",
    message: { role: "assistant", usage: { input_tokens: 4 } },
  },
  {
    type: "content_block_start",
    index: 1,
    content_block: {
      type: "tool_use",
      id: "invoke-tool",
      name: "lookup",
      input: {},
    },
  },
  {
    type: "content_block_delta",
    index: 1,
    delta: { type: "input_json_delta", partial_json: '{"city":' },
  },
  {
    type: "content_block_delta",
    index: 1,
    delta: { type: "input_json_delta", partial_json: '"Paris"}' },
  },
  {
    type: "content_block_delta",
    index: 0,
    delta: { type: "text_delta", text: "invoke stream" },
  },
  {
    type: "message_delta",
    delta: { stop_reason: "tool_use" },
    usage: { output_tokens: 0 },
  },
];
export function handler({
  mode = "converse",
  events = toolEvents,
  response,
  delay = 0,
  status = 200,
  failAfter,
} = {}) {
  const calls = [];
  return {
    calls,
    async handle(req) {
      calls.push({ method: req.method, path: req.path, body: req.body });
      if (delay) await new Promise((r) => setTimeout(r, delay));
      const streaming =
        req.path.endsWith("converse-stream") ||
        req.path.endsWith("invoke-with-response-stream");
      const payload =
        response ??
        (mode === "invoke"
          ? {
              content: [{ type: "text", text: "native invoke" }],
              role: "assistant",
              usage: { input_tokens: 0, output_tokens: 0 },
            }
          : {
              output: {
                message: {
                  role: "assistant",
                  content: [
                    { text: "native response" },
                    {
                      toolUse: {
                        toolUseId: "response-tool",
                        name: "lookup",
                        input: {
                          enabled: false,
                          count: 0,
                          empty: "",
                          nil: null,
                        },
                      },
                    },
                  ],
                },
              },
              usage: {
                inputTokens: 0,
                outputTokens: 0,
                totalTokens: 0,
                cacheReadInputTokens: 0,
                cacheWriteInputTokens: 0,
              },
              stopReason: "tool_use",
            });
      return {
        response: {
          statusCode: status,
          headers: {
            "content-type": streaming
              ? "application/vnd.amazon.eventstream"
              : "application/json",
            "x-amzn-requestid": "fixture-native-request",
          },
          body: streaming
            ? eventBody(
                events,
                req.path.endsWith("invoke-with-response-stream"),
                { failAfter },
              )
            : encode(payload),
        },
      };
    },
  };
}
