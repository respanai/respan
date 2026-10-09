import { createServer } from "node:http";
import * as runtime from "@flue/runtime";
import * as internal from "@flue/runtime/internal";
import * as v from "valibot";

export { runtime };
export const current = typeof runtime.useModel === "function";
export const VECTOR = Array.from({ length: 5001 }, (_, index) => index / 5001);
export const BODY = `native-body:${"x".repeat(80000)}:native-tail`;

/** The real released Flue/Pi SDK calls this localhost OpenAI wire boundary. */
export async function nativeRuntime(options = {}) {
  const requests = [];
  const events = [];
  const calls = [];
  const toolOutput = options.toolOutput ?? {
    vector: VECTOR,
    flag: false,
    count: 0,
    empty: "",
  };
  const server = createServer(async (req, res) => {
    let raw = "";
    for await (const chunk of req) raw += chunk;
    const body = JSON.parse(raw);
    requests.push(body);
    options.onRequest?.(body);
    if (options.providerError) {
      res.writeHead(400, { "content-type": "application/json" });
      res.end(
        JSON.stringify({
          error: {
            message: "controlled native provider rejection",
            type: "invalid_request_error",
            code: "fixture_error",
          },
        }),
      );
      return;
    }
    const hasResult = body.messages.some((message) => message.role === "tool");
    const useTool =
      (options.tool && !hasResult) ||
      (options.delegate && requests.length === 1);
    res.writeHead(200, { "content-type": "text/event-stream" });
    const delta = useTool
      ? {
          role: "assistant",
          tool_calls: [
            {
              index: 0,
              id: "native-tool-call",
              type: "function",
              function: {
                name: options.delegate ? "task" : "native_lookup",
                arguments: JSON.stringify(
                  options.delegate
                    ? {
                        agent: "native_child",
                        prompt: "delegated native prompt",
                      }
                    : { flag: false, count: 0, empty: "" },
                ),
              },
            },
          ],
        }
      : { role: "assistant", content: options.large ? BODY : "native answer" };
    for (const chunk of [
      {
        id: "native-response",
        object: "chat.completion.chunk",
        created: 1,
        model: "gpt-4o-mini",
        choices: [{ index: 0, delta, finish_reason: null }],
      },
      {
        id: "native-response",
        object: "chat.completion.chunk",
        created: 1,
        model: "gpt-4o-mini",
        choices: [
          {
            index: 0,
            delta: {},
            finish_reason: useTool ? "tool_calls" : "stop",
          },
        ],
        usage: { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 },
      },
    ])
      res.write(`data: ${JSON.stringify(chunk)}\n\n`);
    res.end("data: [DONE]\n\n");
  });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const baseUrl = `http://127.0.0.1:${server.address().port}/v1`;
  if (current) {
    const { createProvider } = await import("@earendil-works/pi-ai");
    const { getBuiltinModel } =
      await import("@earendil-works/pi-ai/providers/all");
    const { openAICompletionsApi } =
      await import("@earendil-works/pi-ai/api/openai-completions.lazy");
    runtime.setProvider(
      createProvider({
        id: "openai",
        auth: {
          apiKey: {
            name: "local-test",
            resolve: async () => ({ auth: { apiKey: "controlled-local" } }),
          },
        },
        models: [
          {
            ...getBuiltinModel("openai", "gpt-4o-mini"),
            api: "openai-completions",
            baseUrl,
          },
        ],
        api: openAICompletionsApi(),
      }),
    );
  } else
    runtime.registerProvider("openai", {
      api: "openai-completions",
      baseUrl,
      apiKey: "controlled-local",
    });
  const schema = v.object({
    flag: v.boolean(),
    count: v.number(),
    empty: v.string(),
  });
  const execute = async (data, call) => {
    calls.push({ data, call });
    await options.onTool?.(call);
    if (options.toolError) throw options.toolError;
    return toolOutput;
  };
  const stopEvents = runtime.observe((event) => {
    events.push(event);
    options.onEvent?.(event);
  });
  let harness;
  let session;
  if (current) {
    function NativeAgent() {
      runtime.useModel(
        "openai/gpt-4o-mini",
        options.compaction
          ? { compaction: { keepRecentTokens: 1 } }
          : undefined,
      );
      if (options.tool)
        runtime.useTool(
          runtime.defineTool({
            name: "native_lookup",
            description: options.large ? BODY : "Controlled native lookup",
            input: schema,
            async run(call) {
              return { output: await execute(call.data, call) };
            },
          }),
        );
      if (options.delegate)
        runtime.useSubagent({
          name: "native_child",
          description: "A real delegated child",
          agent: () => "Complete the delegated native prompt.",
        });
      return "Exercise the genuine Flue model, tool and continuation APIs.";
    }
    const ctx = internal.createFlueContext({
      id: options.id ?? `native-${Date.now()}`,
      agentName: "native-agent",
      env: {},
      agentConfig: { resolveModel: internal.resolveModel },
    });
    harness = await ctx.initializeRootHarness(NativeAgent);
    session = harness;
  } else {
    const ctx = internal.createFlueContext({
      id: options.id ?? `native-${Date.now()}`,
      env: {},
      payload: {},
      agentConfig: { resolveModel: internal.resolveModel },
      createDefaultEnv: () =>
        internal.bashFactoryToSessionEnv(() => new internal.Bash()),
      defaultStore: new internal.InMemorySessionStore(),
    });
    const tool = runtime.defineTool({
      name: "native_lookup",
      description: options.large ? BODY : "Controlled native lookup",
      parameters: schema,
      async execute(data, signal) {
        return JSON.stringify(await execute(data, { signal }));
      },
    });
    harness = await ctx.init(
      runtime.createAgent(() => ({
        model: "openai/gpt-4o-mini",
        instructions: "Exercise genuine Flue APIs.",
        tools: options.tool ? [tool] : [],
        compaction: options.compaction ? { keepRecentTokens: 1 } : undefined,
        subagents: options.delegate
          ? [
              {
                name: "native_child",
                model: "openai/gpt-4o-mini",
                instructions: "Complete the delegated native prompt.",
              },
            ]
          : [],
      })),
    );
    session = await harness.session();
  }
  return {
    requests,
    events,
    calls,
    harness,
    session,
    toolOutput,
    prompt(text = options.large ? BODY : "native prompt") {
      return session.prompt(text);
    },
    async close() {
      stopEvents();
      await harness.close?.();
      await new Promise((resolve) => server.close(resolve));
    },
  };
}
