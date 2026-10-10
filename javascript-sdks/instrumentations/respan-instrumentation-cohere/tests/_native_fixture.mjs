import http from "node:http";
export const vector = Array.from({ length: 5001 }, (_, i) =>
  i === 0 ? 0 : i / 5001,
);
export const v1 = {
  text: "fixture-answer",
  generation_id: "fixture-generation",
  tool_calls: [{ name: "lookup", parameters: { value: 0, flag: false } }],
  meta: { tokens: { input_tokens: 0, output_tokens: 3 } },
};
export const v2 = {
  id: "fixture-id",
  finish_reason: "TOOL_CALL",
  message: {
    role: "assistant",
    content: [{ type: "text", text: "fixture-answer" }],
    tool_calls: [
      {
        id: "call_1",
        type: "function",
        function: { name: "lookup", arguments: '{"value":0,"flag":false}' },
      },
    ],
  },
  usage: {
    tokens: { input_tokens: 0, output_tokens: 3 },
    billed_units: { input_tokens: 9, output_tokens: 4 },
    cached_tokens: 0,
  },
};
export const generated = {
  id: "fixture-generated",
  generations: [
    { id: "one", text: "first", finish_reason: "COMPLETE" },
    { id: "two", text: "", finish_reason: "COMPLETE" },
  ],
  meta: { tokens: { input_tokens: 2, output_tokens: 0 } },
};
export const v2events = [
  {
    type: "message-start",
    id: "fixture-stream",
    delta: { message: { role: "assistant" } },
  },
  {
    type: "content-start",
    index: 0,
    delta: { message: { content: { type: "text", text: "" } } },
  },
  {
    type: "content-delta",
    index: 0,
    delta: { message: { content: { text: "fixture-" } } },
  },
  {
    type: "content-delta",
    index: 0,
    delta: { message: { content: { text: "answer" } } },
  },
  { type: "content-end", index: 0 },
  {
    type: "tool-call-start",
    index: 0,
    delta: {
      message: {
        tool_calls: {
          id: "call_1",
          type: "function",
          function: { name: "lookup", arguments: "" },
        },
      },
    },
  },
  {
    type: "tool-call-delta",
    index: 0,
    delta: {
      message: { tool_calls: { function: { arguments: '{"value":' } } },
    },
  },
  {
    type: "tool-call-delta",
    index: 0,
    delta: {
      message: { tool_calls: { function: { arguments: '0,"flag":false}' } } },
    },
  },
  { type: "tool-call-end", index: 0 },
  {
    type: "message-end",
    delta: {
      finish_reason: "TOOL_CALL",
      usage: { tokens: { input_tokens: 0, output_tokens: 3 } },
    },
  },
];
export async function fixture() {
  let received = [];
  const server = http.createServer(async (req, res) => {
    let text = "";
    for await (const chunk of req) text += chunk;
    const body = JSON.parse(text || "{}");
    received.push({ path: req.url, body });
    if (body.model === "fail") {
      res.writeHead(429, { "content-type": "application/json" });
      res.end(JSON.stringify({ message: "fixture-secret-error" }));
      return;
    }
    if (body.model === "delayed") await new Promise((r) => setTimeout(r, 20));
    const path = new URL(req.url, "http://local").pathname;
    if (body.stream) {
      const events = path.startsWith("/v2/")
        ? v2events
        : path.includes("generate")
          ? [
              { event_type: "text-generation", text: "first" },
              { event_type: "stream-end", response: generated },
            ]
          : [
              { event_type: "text-generation", text: "fixture-answer" },
              { event_type: "stream-end", response: v1 },
            ];
      res.writeHead(200, {
        "content-type": path.startsWith("/v2/")
          ? "text/event-stream"
          : "application/json",
      });
      for (const event of events)
        res.write(
          path.startsWith("/v2/")
            ? `event: ${event.type}\ndata: ${JSON.stringify(event)}\n\n`
            : JSON.stringify(event) + "\n",
        );
      if (path.startsWith("/v2/")) res.write("data: [DONE]\n\n");
      res.end();
      return;
    }
    let response = path.includes("parse")
      ? {
          id: "fixture-parse",
          pages: [
            {
              type: "markdown",
              index: 0,
              markdown: { content: "# parsed fixture", images: [] },
            },
          ],
          meta: { billed_units: { input_tokens: 99 } },
        }
      : path.includes("embed")
        ? {
            id: "fixture-embed",
            response_type: body.embedding_types?.length
              ? "embeddings_by_type"
              : "embeddings_floats",
            embeddings: body.embedding_types?.length
              ? {
                  float: [vector],
                  int8: [[0, -1, 127]],
                  uint8: [[0, 255]],
                  binary: [[0, -128]],
                  ubinary: [[0, 255]],
                }
              : [vector],
            texts: body.texts,
            meta: { billed_units: { input_tokens: 0 } },
          }
        : path.includes("rerank")
          ? {
              id: "fixture-rerank",
              results: [
                {
                  index: 1,
                  relevance_score: 0,
                  document: { text: "", flag: false, value: 0 },
                },
              ],
              meta: { billed_units: { search_units: 1 } },
            }
          : path.includes("generate")
            ? generated
            : path.startsWith("/v2/")
              ? structuredClone(v2)
              : structuredClone(v1);
    if (body.model === "no-usage") {
      delete response.usage;
      delete response.meta;
    }
    res.writeHead(200, { "content-type": "application/json" });
    res.end(JSON.stringify(response));
  });
  await new Promise((r) => server.listen(0, "127.0.0.1", r));
  return {
    url: `http://127.0.0.1:${server.address().port}`,
    received,
    close: () => {
      server.closeAllConnections();
      return new Promise((r) => server.close(r));
    },
  };
}
