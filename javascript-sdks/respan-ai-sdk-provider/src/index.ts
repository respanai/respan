import { createAnthropic } from "@ai-sdk/anthropic";
import { createGoogleGenerativeAI } from "@ai-sdk/google";
import { createOpenAI } from "@ai-sdk/openai";
import { createOpenAICompatible } from "@ai-sdk/openai-compatible";
import type {
  EmbeddingModelV4,
  ImageModelV4,
  LanguageModelV4,
  ProviderV4,
} from "@ai-sdk/provider";
import {
  loadApiKey,
  withoutTrailingSlash,
  type FetchFunction,
} from "@ai-sdk/provider-utils";
import { gatewayModelId, wireFormat } from "./_models.js";
import { linkingGatewayLog } from "./_tracing.js";

export { gatewayModelId, wireFormat, type WireFormat } from "./_models.js";

export const DEFAULT_BASE_URL = "https://gateway.respan.ai";

export interface RespanProviderSettings {
  /** Respan API key. Defaults to the `RESPAN_API_KEY` environment variable. */
  apiKey?: string;
  /** Gateway URL. Defaults to `RESPAN_GATEWAY_URL`, then `https://gateway.respan.ai`. */
  baseURL?: string;
  /** Extra headers sent with every request. */
  headers?: Record<string, string>;
  /** Custom fetch, for tests or proxies. */
  fetch?: FetchFunction;
}

export interface RespanProvider extends ProviderV4 {
  (modelId: string): LanguageModelV4;
  languageModel(modelId: string): LanguageModelV4;
  embeddingModel(modelId: string): EmbeddingModelV4;
  imageModel(modelId: string): ImageModelV4;
}

/**
 * Create a Respan gateway provider for the AI SDK.
 *
 * Each model is called in its own native format, so provider options such as
 * `providerOptions.anthropic` keep working: Claude models use Anthropic Messages,
 * Gemini models use Gemini generateContent, OpenAI models use the Responses API,
 * and everything else (including route names) uses Chat Completions.
 */
export function createRespan(settings: RespanProviderSettings = {}): RespanProvider {
  const envBaseURL =
    typeof process !== "undefined" ? process.env?.RESPAN_GATEWAY_URL : undefined;
  const baseURL = withoutTrailingSlash(
    settings.baseURL ?? envBaseURL ?? DEFAULT_BASE_URL,
  ) as string;
  const apiKey = () =>
    loadApiKey({
      apiKey: settings.apiKey,
      environmentVariableName: "RESPAN_API_KEY",
      description: "Respan",
    });
  const shared = { headers: settings.headers, fetch: linkingGatewayLog(settings.fetch) };

  // Each delegate is created on first use so a missing key fails at call time,
  // not at import time, and only the formats you use are instantiated.
  let anthropic: ReturnType<typeof createAnthropic> | undefined;
  let google: ReturnType<typeof createGoogleGenerativeAI> | undefined;
  let openai: ReturnType<typeof createOpenAI> | undefined;
  let compatible: ReturnType<typeof createOpenAICompatible> | undefined;

  const getAnthropic = () =>
    (anthropic ??= createAnthropic({ ...shared, baseURL: `${baseURL}/v1`, apiKey: apiKey() }));
  const getGoogle = () =>
    (google ??= createGoogleGenerativeAI({ ...shared, baseURL: `${baseURL}/v1beta`, apiKey: apiKey() }));
  const getOpenAI = () =>
    (openai ??= createOpenAI({ ...shared, baseURL: `${baseURL}/v1`, apiKey: apiKey() }));
  const getCompatible = () =>
    (compatible ??= createOpenAICompatible({
      ...shared,
      name: "respan",
      baseURL: `${baseURL}/v1`,
      apiKey: apiKey(),
    }));

  const languageModel = (modelId: string): LanguageModelV4 => {
    const id = gatewayModelId(modelId);
    switch (wireFormat(modelId)) {
      case "anthropic":
        return getAnthropic().languageModel(id);
      case "google":
        return getGoogle().languageModel(id);
      case "openai":
        return getOpenAI().responses(id);
      default:
        return getCompatible().chatModel(id);
    }
  };

  const embeddingModel = (modelId: string): EmbeddingModelV4 =>
    wireFormat(modelId) === "google"
      ? getGoogle().embeddingModel(gatewayModelId(modelId))
      : getCompatible().embeddingModel(gatewayModelId(modelId));

  const imageModel = (modelId: string): ImageModelV4 =>
    wireFormat(modelId) === "google"
      ? getGoogle().imageModel(gatewayModelId(modelId))
      : getOpenAI().imageModel(gatewayModelId(modelId));

  const provider = ((modelId: string) => languageModel(modelId)) as RespanProvider;
  Object.assign(provider, {
    specificationVersion: "v4" as const,
    languageModel,
    embeddingModel,
    imageModel,
  });
  return provider;
}

/** Default provider instance. Reads `RESPAN_API_KEY` when the first request is made. */
export const respan = createRespan();
