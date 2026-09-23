export type AutoInstrumentationCategory =
  | "direct-llm"
  | "agent-framework"
  | "app-framework"
  | "protocol-or-tooling"
  | "observability"
  | "vector-db";

export type InstrumentationStatus =
  | "enabled"
  | "disabled"
  | "missing"
  | "failed";

export interface AutoInstrumentationEntry {
  id: string;
  category: AutoInstrumentationCategory;
  provider?: string;
  sdkPackage: string;
  instrumentationPackage: string;
  instrumentorClass: string;
  enabledByDefault: boolean;
  priority: number;
  aliases?: string[];
  conflictsWith?: string[];
  genericTracingNames?: string[];
  autoDisabledReason?: string;
  docsUrl?: string;
}

export interface InstrumentationStatusEntry {
  id: string;
  category: AutoInstrumentationCategory;
  provider?: string;
  sdkPackage: string;
  instrumentationPackage: string;
  instrumentorClass: string;
  status: InstrumentationStatus;
  reason?: string;
}

const FRAMEWORK_DISABLED_REASON =
  "not auto-discovered; add explicitly when you want framework-level tracing to avoid duplicate LLM spans";
const OBSERVABILITY_DISABLED_REASON =
  "not auto-discovered; add explicitly because it translates an existing tracing/observability pipeline";
const TOOLING_DISABLED_REASON =
  "not auto-discovered; add explicitly because it is not a direct LLM SDK";

export const AUTO_INSTRUMENTATION_REGISTRY: AutoInstrumentationEntry[] = [
  {
    id: "openai",
    category: "direct-llm",
    provider: "openai",
    sdkPackage: "openai",
    instrumentationPackage: "@respan/instrumentation-openai",
    instrumentorClass: "OpenAIInstrumentor",
    enabledByDefault: true,
    priority: 100,
    aliases: ["openAI", "OpenAIInstrumentor"],
    conflictsWith: ["openai-agents", "vercel-ai", "langchain", "llama-index"],
    genericTracingNames: ["openAI"],
    docsUrl: "https://respan.ai/docs/integrations/openai-sdk",
  },
  {
    id: "anthropic",
    category: "direct-llm",
    provider: "anthropic",
    sdkPackage: "@anthropic-ai/sdk",
    instrumentationPackage: "@respan/instrumentation-anthropic",
    instrumentorClass: "AnthropicInstrumentor",
    enabledByDefault: true,
    priority: 100,
    aliases: ["AnthropicInstrumentor"],
    conflictsWith: ["claude-agent-sdk", "vercel-ai", "langchain", "llama-index"],
    genericTracingNames: ["anthropic"],
    docsUrl: "https://respan.ai/docs/integrations/anthropic",
  },
  {
    id: "azure-openai",
    category: "direct-llm",
    provider: "azure-openai",
    sdkPackage: "openai",
    instrumentationPackage: "@respan/instrumentation-azure-openai",
    instrumentorClass: "AzureOpenAIInstrumentor",
    enabledByDefault: true,
    priority: 100,
    aliases: ["azureOpenAI", "AzureOpenAIInstrumentor"],
    conflictsWith: ["vercel-ai", "langchain", "llama-index"],
    genericTracingNames: ["azureOpenAI"],
    docsUrl: "https://respan.ai/docs/integrations/providers/azure",
  },
  {
    id: "vertexai",
    category: "direct-llm",
    provider: "google",
    sdkPackage: "@google-cloud/vertexai",
    instrumentationPackage: "@respan/instrumentation-vertexai",
    instrumentorClass: "VertexAIInstrumentor",
    enabledByDefault: true,
    priority: 100,
    aliases: ["googleVertexAI", "vertex-ai", "VertexAIInstrumentor"],
    conflictsWith: ["google-adk", "vercel-ai", "langchain", "llama-index"],
    genericTracingNames: ["googleVertexAI"],
    docsUrl: "https://respan.ai/docs/integrations/vertex-ai",
  },
  {
    id: "openrouter",
    category: "direct-llm",
    provider: "openrouter",
    sdkPackage: "@openrouter/sdk",
    instrumentationPackage: "@respan/instrumentation-openrouter",
    instrumentorClass: "OpenRouterInstrumentor",
    enabledByDefault: true,
    priority: 100,
    aliases: ["OpenRouterInstrumentor"],
    conflictsWith: ["vercel-ai", "langchain", "llama-index"],
    docsUrl: "https://respan.ai/docs/integrations/openrouter",
  },
  {
    id: "aws-bedrock",
    category: "direct-llm",
    provider: "aws-bedrock",
    sdkPackage: "@aws-sdk/client-bedrock-runtime",
    instrumentationPackage: "@respan/instrumentation-aws-bedrock",
    instrumentorClass: "AWSBedrockInstrumentor",
    enabledByDefault: true,
    priority: 100,
    aliases: ["bedrock", "AWSBedrockInstrumentor"],
    conflictsWith: ["vercel-ai", "langchain", "llama-index"],
    genericTracingNames: ["bedrock"],
  },
  {
    id: "cohere",
    category: "direct-llm",
    provider: "cohere",
    sdkPackage: "cohere-ai",
    instrumentationPackage: "@respan/instrumentation-cohere",
    instrumentorClass: "CohereInstrumentor",
    enabledByDefault: true,
    priority: 100,
    aliases: ["CohereInstrumentor"],
    conflictsWith: ["vercel-ai", "langchain", "llama-index"],
    genericTracingNames: ["cohere"],
  },
  {
    id: "together-ai",
    category: "direct-llm",
    provider: "together-ai",
    sdkPackage: "together-ai",
    instrumentationPackage: "@respan/instrumentation-together-ai",
    instrumentorClass: "TogetherAIInstrumentor",
    enabledByDefault: true,
    priority: 100,
    aliases: [
      "together",
      "TogetherAIInstrumentor",
      "TogetherAIInstrumentation",
    ],
    conflictsWith: ["vercel-ai", "langchain", "llama-index"],
    genericTracingNames: ["together"],
  },
  {
    id: "writer",
    category: "direct-llm",
    provider: "writer",
    sdkPackage: "writer-sdk",
    instrumentationPackage: "@respan/instrumentation-writer",
    instrumentorClass: "WriterInstrumentor",
    enabledByDefault: true,
    priority: 100,
    aliases: ["WriterInstrumentor"],
    conflictsWith: ["vercel-ai", "langchain", "llama-index"],
  },
  {
    id: "vercel-ai",
    category: "app-framework",
    sdkPackage: "ai",
    instrumentationPackage: "@respan/instrumentation-vercel",
    instrumentorClass: "VercelAIInstrumentor",
    enabledByDefault: false,
    priority: 50,
    aliases: ["vercel", "VercelAIInstrumentor"],
    autoDisabledReason: FRAMEWORK_DISABLED_REASON,
  },
  {
    id: "langchain",
    category: "app-framework",
    sdkPackage: "@langchain/core",
    instrumentationPackage: "@respan/instrumentation-langchain",
    instrumentorClass: "LangChainInstrumentor",
    enabledByDefault: false,
    priority: 50,
    aliases: ["langChain", "LangChainInstrumentor"],
    genericTracingNames: ["langChain"],
    autoDisabledReason: FRAMEWORK_DISABLED_REASON,
  },
  {
    id: "llama-index",
    category: "app-framework",
    sdkPackage: "llamaindex",
    instrumentationPackage: "@respan/instrumentation-llama-index",
    instrumentorClass: "LlamaIndexInstrumentor",
    enabledByDefault: false,
    priority: 50,
    aliases: ["llamaIndex", "LlamaIndexInstrumentor"],
    genericTracingNames: ["llamaIndex"],
    autoDisabledReason: FRAMEWORK_DISABLED_REASON,
  },
  {
    id: "openai-agents",
    category: "agent-framework",
    sdkPackage: "@openai/agents",
    instrumentationPackage: "@respan/instrumentation-openai-agents",
    instrumentorClass: "OpenAIAgentsInstrumentor",
    enabledByDefault: false,
    priority: 40,
    aliases: ["OpenAIAgentsInstrumentor"],
    autoDisabledReason: FRAMEWORK_DISABLED_REASON,
  },
  {
    id: "claude-agent-sdk",
    category: "agent-framework",
    sdkPackage: "@anthropic-ai/claude-agent-sdk",
    instrumentationPackage: "@respan/instrumentation-claude-agent-sdk",
    instrumentorClass: "ClaudeAgentSDKInstrumentor",
    enabledByDefault: false,
    priority: 40,
    aliases: ["ClaudeAgentSDKInstrumentor"],
    autoDisabledReason: FRAMEWORK_DISABLED_REASON,
  },
  {
    id: "google-adk",
    category: "agent-framework",
    sdkPackage: "@google/adk",
    instrumentationPackage: "@respan/instrumentation-google-adk",
    instrumentorClass: "GoogleADKInstrumentor",
    enabledByDefault: false,
    priority: 40,
    aliases: ["GoogleADKInstrumentor"],
    autoDisabledReason: FRAMEWORK_DISABLED_REASON,
  },
  {
    id: "strands-agents",
    category: "agent-framework",
    sdkPackage: "@strands-agents/sdk",
    instrumentationPackage: "@respan/instrumentation-strands-agents",
    instrumentorClass: "StrandsAgentsInstrumentor",
    enabledByDefault: false,
    priority: 40,
    aliases: ["StrandsAgentsInstrumentor"],
    autoDisabledReason: FRAMEWORK_DISABLED_REASON,
  },
  {
    id: "beeai",
    category: "agent-framework",
    sdkPackage: "beeai-framework",
    instrumentationPackage: "@respan/instrumentation-beeai",
    instrumentorClass: "BeeAIInstrumentor",
    enabledByDefault: false,
    priority: 40,
    aliases: ["BeeAIInstrumentor"],
    autoDisabledReason: FRAMEWORK_DISABLED_REASON,
  },
  {
    id: "mastra",
    category: "agent-framework",
    sdkPackage: "@mastra/core",
    instrumentationPackage: "@respan/instrumentation-mastra",
    instrumentorClass: "MastraInstrumentor",
    enabledByDefault: false,
    priority: 40,
    aliases: ["MastraInstrumentor"],
    autoDisabledReason: FRAMEWORK_DISABLED_REASON,
  },
  {
    id: "codex-sdk",
    category: "agent-framework",
    sdkPackage: "@openai/codex-sdk",
    instrumentationPackage: "@respan/instrumentation-codex-sdk",
    instrumentorClass: "CodexSDKInstrumentor",
    enabledByDefault: false,
    priority: 40,
    aliases: ["codex", "CodexSDKInstrumentor"],
    autoDisabledReason: FRAMEWORK_DISABLED_REASON,
  },
  {
    id: "cursor-sdk",
    category: "agent-framework",
    sdkPackage: "@cursor/sdk",
    instrumentationPackage: "@respan/instrumentation-cursor",
    instrumentorClass: "CursorSDKInstrumentor",
    enabledByDefault: false,
    priority: 40,
    aliases: ["cursor", "CursorSDKInstrumentor"],
    autoDisabledReason: FRAMEWORK_DISABLED_REASON,
  },
  {
    id: "pi",
    category: "agent-framework",
    sdkPackage: "@earendil-works/pi-coding-agent",
    instrumentationPackage: "@respan/instrumentation-pi",
    instrumentorClass: "PiInstrumentor",
    enabledByDefault: false,
    priority: 40,
    aliases: ["pi-coding-agent", "PiInstrumentor"],
    autoDisabledReason: FRAMEWORK_DISABLED_REASON,
  },
  {
    id: "livekit",
    category: "agent-framework",
    sdkPackage: "@livekit/agents",
    instrumentationPackage: "@respan/instrumentation-livekit",
    instrumentorClass: "LiveKitInstrumentor",
    enabledByDefault: false,
    priority: 40,
    aliases: ["LiveKitInstrumentor"],
    autoDisabledReason: FRAMEWORK_DISABLED_REASON,
  },
  {
    id: "flue",
    category: "app-framework",
    sdkPackage: "@flue/runtime",
    instrumentationPackage: "@respan/instrumentation-flue",
    instrumentorClass: "FlueInstrumentor",
    enabledByDefault: false,
    priority: 40,
    aliases: ["FlueInstrumentor", "RespanFlueObserver"],
    autoDisabledReason: FRAMEWORK_DISABLED_REASON,
  },
  {
    id: "dify",
    category: "app-framework",
    provider: "dify",
    sdkPackage: "dify-client",
    instrumentationPackage: "@respan/instrumentation-dify",
    instrumentorClass: "DifyInstrumentor",
    enabledByDefault: false,
    priority: 40,
    aliases: ["DifyInstrumentor"],
    autoDisabledReason: FRAMEWORK_DISABLED_REASON,
  },
  {
    id: "mcp",
    category: "protocol-or-tooling",
    sdkPackage: "@modelcontextprotocol/sdk",
    instrumentationPackage: "@respan/instrumentation-mcp",
    instrumentorClass: "MCPInstrumentor",
    enabledByDefault: false,
    priority: 30,
    aliases: ["MCPInstrumentor"],
    autoDisabledReason: TOOLING_DISABLED_REASON,
  },
  {
    id: "superagent",
    category: "protocol-or-tooling",
    sdkPackage: "safety-agent",
    instrumentationPackage: "@respan/instrumentation-superagent",
    instrumentorClass: "SuperagentInstrumentor",
    enabledByDefault: false,
    priority: 30,
    aliases: ["SuperagentInstrumentor"],
    autoDisabledReason: TOOLING_DISABLED_REASON,
  },
  {
    id: "arize",
    category: "observability",
    sdkPackage: "@arizeai/phoenix-otel",
    instrumentationPackage: "@respan/instrumentation-arize",
    instrumentorClass: "ArizeInstrumentor",
    enabledByDefault: false,
    priority: 20,
    aliases: ["ArizeInstrumentor"],
    autoDisabledReason: OBSERVABILITY_DISABLED_REASON,
  },
  {
    id: "braintrust",
    category: "observability",
    sdkPackage: "braintrust",
    instrumentationPackage: "@respan/instrumentation-braintrust",
    instrumentorClass: "BraintrustInstrumentor",
    enabledByDefault: false,
    priority: 20,
    aliases: ["BraintrustInstrumentor"],
    autoDisabledReason: OBSERVABILITY_DISABLED_REASON,
  },
  {
    id: "openinference",
    category: "observability",
    sdkPackage: "@arizeai/openinference-instrumentation-*",
    instrumentationPackage: "@respan/instrumentation-openinference",
    instrumentorClass: "OpenInferenceInstrumentor",
    enabledByDefault: false,
    priority: 20,
    aliases: ["OpenInferenceInstrumentor"],
    autoDisabledReason: OBSERVABILITY_DISABLED_REASON,
  },
];

export const DIRECT_LLM_AUTO_INSTRUMENTATIONS: AutoInstrumentationEntry[] =
  AUTO_INSTRUMENTATION_REGISTRY.filter(
    (entry) => entry.category === "direct-llm" && entry.enabledByDefault,
  );

export function directLlmGenericTracingNames(): string[] {
  return DIRECT_LLM_AUTO_INSTRUMENTATIONS.flatMap(
    (entry) => entry.genericTracingNames ?? [],
  );
}

export function matchesAutoInstrumentationSelector(
  entry: AutoInstrumentationEntry,
  selector: string,
): boolean {
  const normalized = selector.toLowerCase();
  return [
    entry.id,
    entry.provider,
    entry.sdkPackage,
    entry.instrumentationPackage,
    entry.instrumentorClass,
    ...(entry.aliases ?? []),
  ]
    .filter((value): value is string => Boolean(value))
    .some((value) => value.toLowerCase() === normalized);
}

export function statusFromEntry(
  entry: AutoInstrumentationEntry,
  status: InstrumentationStatus,
  reason?: string,
): InstrumentationStatusEntry {
  return {
    id: entry.id,
    category: entry.category,
    provider: entry.provider,
    sdkPackage: entry.sdkPackage,
    instrumentationPackage: entry.instrumentationPackage,
    instrumentorClass: entry.instrumentorClass,
    status,
    reason,
  };
}
