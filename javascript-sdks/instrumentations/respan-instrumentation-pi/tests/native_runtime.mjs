import http from "node:http";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import * as sdk from "@earendil-works/pi-coding-agent";

/** Native SDK + native provider parser; only the HTTP service is controlled. */
export async function nativeFixture({
  instrumentor,
  extension = false,
  responses = [{ text: "native final" }],
  history = [],
  tools = [],
  retry = false,
  systemPrompt = "native system instructions",
} = {}) {
  const directory = await mkdtemp(join(tmpdir(), "respan-pi-native-"));
  const requests = [];
  const sockets = new Set();
  const server = http.createServer(async (request, response) => {
    let body = "";
    for await (const chunk of request) body += chunk;
    const input = JSON.parse(body);
    requests.push(input);
    const plan = responses[Math.min(requests.length - 1, responses.length - 1)];
    if (plan.status) {
      response.writeHead(plan.status, { "content-type": "application/json" });
      response.end(
        JSON.stringify({
          error: {
            message: plan.error ?? "controlled provider error",
            type: "controlled_error",
          },
        }),
      );
      return;
    }
    response.writeHead(200, { "content-type": "text/event-stream" });
    const chunk = (delta, finish_reason = null, usage) =>
      response.write(
        `data: ${JSON.stringify({ id: `native-response-${requests.length}`, object: "chat.completion.chunk", created: 1, model: "native-audit", choices: [{ index: 0, delta, finish_reason }], ...(usage ? { usage } : {}) })}\n\n`,
      );
    chunk({ role: "assistant", content: "" });
    if (plan.tools)
      chunk({
        tool_calls: plan.tools.map((tool, index) => ({
          index,
          id: tool.id,
          type: "function",
          function: {
            name: tool.name,
            arguments: JSON.stringify(tool.arguments),
          },
        })),
      });
    if (plan.text !== undefined) {
      const midpoint = Math.floor(plan.text.length / 2);
      chunk({ content: plan.text.slice(0, midpoint) });
      chunk({ content: plan.text.slice(midpoint) });
    }
    if (plan.hold) return;
    chunk({}, plan.tools ? "tool_calls" : "stop", plan.usage);
    response.end("data: [DONE]\n\n");
  });
  server.on("connection", (socket) => {
    sockets.add(socket);
    socket.on("close", () => sockets.delete(socket));
  });
  await new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", resolve);
  });
  const config = {
    api: "openai-completions",
    baseUrl: `http://127.0.0.1:${server.address().port}/v1`,
    apiKey: "controlled-synthetic",
    models: [
      {
        id: "native-audit",
        name: "Native audit",
        reasoning: false,
        input: ["text"],
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
        contextWindow: 1000000,
        maxTokens: 4096,
      },
    ],
  };
  let models;
  let authStorage;
  let model;
  if (sdk.ModelRuntime) {
    models = await sdk.ModelRuntime.create({
      modelsPath: null,
      authPath: join(directory, "auth.json"),
      refreshOnCreate: false,
    });
    models.registerProvider("audit", config);
    model = models.getModel("audit", "native-audit");
  } else {
    authStorage = sdk.AuthStorage.inMemory();
    models = sdk.ModelRegistry.inMemory(authStorage);
    models.registerProvider("audit", config);
    model = models.find("audit", "native-audit");
  }
  const settingsManager = sdk.SettingsManager.inMemory({
    compaction: { enabled: false, keepRecentTokens: 32, reserveTokens: 16 },
    retry: { enabled: retry, maxRetries: 1, baseDelayMs: 1 },
  });
  const sessionManager = sdk.SessionManager.inMemory(directory);
  for (const message of history) sessionManager.appendMessage(message);
  const resourceLoader = new sdk.DefaultResourceLoader({
    cwd: directory,
    agentDir: directory,
    settingsManager,
    noSkills: true,
    noPromptTemplates: true,
    noThemes: true,
    noContextFiles: true,
    systemPrompt,
    extensionFactories: extension ? [instrumentor.extension] : [],
  });
  await resourceLoader.reload();
  const { session } = await sdk.createAgentSession({
    cwd: directory,
    agentDir: directory,
    ...(sdk.ModelRuntime
      ? { modelRuntime: models }
      : { modelRegistry: models, authStorage }),
    model,
    thinkingLevel: "off",
    resourceLoader,
    tools: tools.map((tool) => tool.name),
    customTools: tools,
    sessionManager,
    settingsManager,
  });
  if (extension) await session.bindExtensions({});
  const detach =
    instrumentor && !extension ? instrumentor.attach(session) : () => {};
  return {
    session,
    sessionManager,
    requests,
    model,
    async close() {
      detach();
      session.dispose();
      for (const socket of sockets) socket.destroy();
      await new Promise((resolve) => server.close(resolve));
      await rm(directory, { recursive: true, force: true });
    },
  };
}
