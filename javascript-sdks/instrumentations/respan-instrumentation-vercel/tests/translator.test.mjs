import test from "node:test";
import assert from "node:assert/strict";

import { VercelAITranslator } from "../dist/_translator.js";

function runTranslator(name, attributes) {
  const span = {
    name,
    instrumentationLibrary: { name: "ai" },
    attributes: { ...attributes },
  };
  const writableSpan = {
    name,
    setAttribute(key, value) {
      span.attributes[key] = value;
    },
  };

  const translator = new VercelAITranslator();
  translator.onStart(writableSpan, undefined);
  translator.onEnd(span);

  return span.attributes;
}

const baseLLMSpan = {
  "ai.model.id": "gpt-4o-mini",
  "ai.prompt.messages": JSON.stringify([{ role: "user", content: "hi" }]),
  "ai.response.text": "hello",
  "gen_ai.usage.input_tokens": 5,
  "gen_ai.usage.output_tokens": 3,
  "traceloop.span.kind": "task",
};

test("ai.generateText.doGenerate is classified as LLM text, not task", () => {
  const attrs = runTranslator("ai.generateText.doGenerate", baseLLMSpan);

  assert.equal(attrs["respan.entity.log_type"], "text");
  assert.equal(attrs["llm.request.type"], "chat");
  assert.equal(attrs["gen_ai.request.model"], "gpt-4o-mini");
  assert.equal(attrs["traceloop.span.kind"], undefined);
});

test("ai.streamText.doStream is classified as LLM text, not task", () => {
  const attrs = runTranslator("ai.streamText.doStream", baseLLMSpan);

  assert.equal(attrs["respan.entity.log_type"], "text");
  assert.equal(attrs["llm.request.type"], "chat");
  assert.equal(attrs["gen_ai.request.model"], "gpt-4o-mini");
  assert.equal(attrs["traceloop.span.kind"], undefined);
});

test("ai.embed.doEmbed is classified as embedding without synthetic usage fields", () => {
  const attrs = runTranslator("ai.embed.doEmbed", {
    "ai.model.id": "text-embedding-3-small",
    "ai.values": [JSON.stringify("embed this")],
    "ai.embeddings": [JSON.stringify([0.1, 0.2, 0.3])],
    "ai.usage.tokens": 7,
    "traceloop.span.kind": "task",
  });

  assert.equal(attrs["respan.entity.log_type"], "embedding");
  assert.equal(attrs["llm.request.type"], "embedding");
  assert.equal(attrs["gen_ai.request.model"], "text-embedding-3-small");
  assert.equal(attrs["gen_ai.usage.input_tokens"], undefined);
  assert.equal(attrs["gen_ai.usage.prompt_tokens"], undefined);
  assert.equal(attrs["llm.model_name"], undefined);
  assert.equal(attrs.model, undefined);
  assert.equal(attrs["llm.token_count.prompt"], undefined);
  assert.equal(attrs.prompt_tokens, undefined);
  assert.equal(attrs.total_request_tokens, undefined);
  assert.equal(attrs["traceloop.span.kind"], undefined);
  assert.equal(attrs["ai.usage.tokens"], undefined);
  assert.equal(attrs["ai.embeddings"], undefined);
});

test("LLM spans emit tool definitions and tool calls in canonical fields only", () => {
  const tool = {
    type: "function",
    name: "weather",
    description: "Return weather.",
    inputSchema: {
      type: "object",
      properties: { city: { type: "string" } },
      required: ["city"],
    },
  };
  const toolCall = {
    id: "call_weather",
    type: "function",
    function: {
      name: "weather",
      arguments: JSON.stringify({ city: "Tokyo" }),
    },
  };

  const attrs = runTranslator("ai.generateText.doGenerate", {
    "ai.model.id": "gpt-4o",
    "ai.prompt.messages": JSON.stringify([{ role: "user", content: "weather in Tokyo" }]),
    "ai.prompt.tools": [JSON.stringify(tool)],
    "ai.response.toolCalls": JSON.stringify([toolCall]),
    "gen_ai.usage.input_tokens": 12,
    "gen_ai.usage.output_tokens": 4,
  });

  const expectedTools = [
    {
      type: "function",
      function: {
        name: "weather",
        description: "Return weather.",
        parameters: tool.inputSchema,
      },
    },
  ];

  // Canonical fields only (per contribution/span-contract.md):
  assert.equal(JSON.parse(attrs["llm.request.functions"]).length, 1);
  assert.deepEqual(JSON.parse(attrs["llm.request.functions"]), expectedTools);
  assert.deepEqual(JSON.parse(attrs["gen_ai.completion.0.tool_calls"]), [toolCall]);

  // Off-contract aliases must NOT be set:
  assert.equal(attrs.tools, undefined);
  assert.equal(attrs["respan.span.tools"], undefined);
  assert.equal(attrs.span_tools, undefined);
  assert.equal(attrs.tool_calls, undefined);
  assert.equal(attrs["respan.span.tool_calls"], undefined);
  assert.equal(attrs.has_tool_calls, undefined);
  assert.equal(attrs.parallel_tool_calls, undefined);

  // Vendor-specific raw attrs stripped:
  assert.equal(attrs["ai.prompt.tools"], undefined);
  assert.equal(attrs["ai.response.toolCalls"], undefined);
});

test("final text step does not echo prompt-history tool calls into completion", () => {
  const toolCall = {
    id: "call_weather",
    type: "function",
    function: {
      name: "weather",
      arguments: JSON.stringify({ city: "Tokyo" }),
    },
  };

  const attrs = runTranslator("ai.generateText.doGenerate", {
    "ai.model.id": "gpt-4o",
    "ai.prompt.messages": JSON.stringify([
      { role: "user", content: "weather in Tokyo" },
      { role: "assistant", content: "", tool_calls: [toolCall] },
      {
        role: "tool",
        tool_call_id: "call_weather",
        name: "weather",
        content: JSON.stringify({ city: "Tokyo", condition: "clear" }),
      },
    ]),
    "ai.response.text": "Tokyo is clear.",
  });

  // This turn emitted plain text, not a new tool call. The assistant's
  // earlier tool_calls remain in the prompt history (gen_ai.prompt.*),
  // not on this span's completion / top-level tool_calls fields.
  assert.equal(attrs.tool_calls, undefined);
  assert.equal(attrs["respan.span.tool_calls"], undefined);
  assert.equal(attrs["gen_ai.completion.0.tool_calls"], undefined);
  assert.equal(attrs.has_tool_calls, undefined);
  assert.deepEqual(JSON.parse(attrs["traceloop.entity.output"]), {
    role: "assistant",
    content: "Tokyo is clear.",
  });
});

test("ai.toolCall spans carry input/output only — no tool_calls aliases", () => {
  const attrs = runTranslator("ai.toolCall", {
    "ai.toolCall.id": "call_weather",
    "ai.toolCall.name": "weather",
    "ai.toolCall.args": JSON.stringify({ city: "Tokyo" }),
    "ai.toolCall.result": JSON.stringify({ city: "Tokyo", condition: "clear" }),
  });

  // The span's existence + log_type=tool IS the tool call.
  assert.equal(attrs["respan.entity.log_type"], "tool");
  assert.deepEqual(JSON.parse(attrs["traceloop.entity.input"]), {
    name: "weather",
    args: { city: "Tokyo" },
  });
  assert.deepEqual(JSON.parse(attrs["traceloop.entity.output"]), {
    city: "Tokyo",
    condition: "clear",
  });

  // Tool execution spans must NOT carry tool_calls aliases:
  assert.equal(attrs.tool_calls, undefined);
  assert.equal(attrs["respan.span.tool_calls"], undefined);
  assert.equal(attrs.span_tools, undefined);
});
