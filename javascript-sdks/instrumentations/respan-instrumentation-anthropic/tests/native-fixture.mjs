export const MODEL = "claude-sonnet-5-5";
export const message = (extra = {}) => ({
  id: "msg_fixture",
  type: "message",
  role: "assistant",
  model: MODEL,
  content: [
    {
      type: "thinking",
      thinking: "fixture thought",
      signature: "fixture signature",
    },
    { type: "text", text: '{"enabled":false,"count":0,"empty":"","nil":null}' },
  ],
  stop_reason: "end_turn",
  stop_sequence: null,
  usage: {
    input_tokens: 0,
    output_tokens: 0,
    cache_read_input_tokens: 0,
    cache_creation_input_tokens: 0,
  },
  ...extra,
});
export const events = [
  {
    type: "message_start",
    message: message({
      content: [],
      stop_reason: null,
      usage: {
        input_tokens: 0,
        output_tokens: 0,
        cache_read_input_tokens: 0,
        cache_creation_input_tokens: 0,
      },
    }),
  },
  {
    type: "content_block_start",
    index: 0,
    content_block: { type: "thinking", thinking: "", signature: "" },
  },
  {
    type: "content_block_delta",
    index: 0,
    delta: { type: "thinking_delta", thinking: "fixture thought" },
  },
  {
    type: "content_block_delta",
    index: 0,
    delta: { type: "signature_delta", signature: "fixture signature" },
  },
  { type: "content_block_stop", index: 0 },
  {
    type: "content_block_start",
    index: 1,
    content_block: {
      type: "tool_use",
      id: "tool_fixture",
      name: "lookup",
      input: {},
    },
  },
  {
    type: "content_block_delta",
    index: 1,
    delta: {
      type: "input_json_delta",
      partial_json: '{"enabled":false,"count":0,',
    },
  },
  {
    type: "content_block_delta",
    index: 1,
    delta: { type: "input_json_delta", partial_json: '"empty":"","nil":null}' },
  },
  { type: "content_block_stop", index: 1 },
  {
    type: "content_block_start",
    index: 2,
    content_block: { type: "text", text: "" },
  },
  {
    type: "content_block_delta",
    index: 2,
    delta: { type: "text_delta", text: "fixture stream" },
  },
  { type: "content_block_stop", index: 2 },
  {
    type: "message_delta",
    delta: { stop_reason: "tool_use", stop_sequence: null },
    usage: { output_tokens: 0 },
  },
  { type: "message_stop" },
];
export function sse(values = events) {
  return values
    .map(
      (value) =>
        `event: ${value.type ?? "completion"}\ndata: ${JSON.stringify(value)}\n\n`,
    )
    .join("");
}
export function transport({
  response,
  streamEvents = events,
  status = 201,
  delay = 0,
  runner = false,
  batch = false,
  error = false,
} = {}) {
  const requests = [];
  const responses = [];
  const fetch = async (url, init) => {
    const body = init?.body ? JSON.parse(init.body) : {};
    requests.push({ url: String(url), body, init });
    if (delay) await new Promise((r) => setTimeout(r, delay));
    let result;
    if (error)
      result = Response.json(
        {
          type: "error",
          error: { type: "rate_limit_error", message: "private fixture error" },
        },
        { status: 429, headers: { "request-id": "fixture-error" } },
      );
    else if (String(url).includes("/count_tokens"))
      result = Response.json(
        { input_tokens: 0 },
        { status, headers: { "request-id": "fixture-count" } },
      );
    else if (String(url).includes("/results.jsonl"))
      result = new Response(
        [
          JSON.stringify({
            custom_id: "fixture-row",
            result: { type: "succeeded", message: message() },
          }),
          JSON.stringify({
            custom_id: "fixture-error-row",
            result: {
              type: "errored",
              error: {
                type: "invalid_request_error",
                message: "fixture row error",
              },
            },
          }),
        ].join("\n") + "\n",
        { status, headers: { "content-type": "application/binary" } },
      );
    else if (String(url).includes("/batches"))
      result = Response.json(
        {
          id: "msgbatch_fixture",
          type: "message_batch",
          processing_status: "ended",
          request_counts: {
            processing: 0,
            succeeded: 1,
            errored: 1,
            canceled: 0,
            expired: 0,
          },
          results_url: "https://fixture.invalid/results.jsonl",
          created_at: "2026-10-10T00:00:00Z",
          ended_at: "2026-10-10T00:00:01Z",
          expires_at: "2026-10-11T00:00:00Z",
        },
        { status, headers: { "request-id": "fixture-batch" } },
      );
    else if (body.stream)
      result = new Response(sse(streamEvents), {
        status,
        headers: {
          "content-type": "text/event-stream",
          "request-id": "fixture-stream",
        },
      });
    else if (String(url).includes("/complete"))
      result = Response.json(
        {
          id: "compl_fixture",
          type: "completion",
          model: MODEL,
          completion: "legacy fixture",
          stop_reason: "stop_sequence",
        },
        { status, headers: { "request-id": "fixture-completion" } },
      );
    else
      result = Response.json(
        response ??
          (runner &&
          requests.filter((r) => r.url.includes("/messages")).length === 1
            ? message({
                content: [
                  {
                    type: "tool_use",
                    id: "runner-tool",
                    name: "lookup",
                    input: { enabled: false, count: 0, empty: "", nil: null },
                  },
                ],
                stop_reason: "tool_use",
              })
            : message()),
        { status, headers: { "request-id": "fixture-message" } },
      );
    responses.push(result);
    return result;
  };
  return { fetch, requests, responses };
}
