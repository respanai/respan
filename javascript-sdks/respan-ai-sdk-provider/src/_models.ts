/** The request format a model is called in. */
export type WireFormat = "anthropic" | "google" | "openai" | "chat";

const ANTHROPIC_PREFIXES = new Set(["anthropic"]);
const GOOGLE_PREFIXES = new Set(["google", "gemini"]);
const OPENAI_PREFIXES = new Set(["openai"]);

function split(modelId: string): { prefix?: string; name: string } {
  const slash = modelId.indexOf("/");
  return slash === -1
    ? { name: modelId }
    : { prefix: modelId.slice(0, slash).toLowerCase(), name: modelId.slice(slash + 1) };
}

/**
 * Pick the native format for a model ID such as `anthropic/claude-sonnet-4.5`,
 * `claude-sonnet-4-5-20250929`, `google/gemini-3.5-flash` or `gpt-5.5`.
 * IDs with another prefix (for example `azure/gpt-5.5` or `openrouter/...`)
 * and route names use Chat Completions, which the gateway accepts for every model.
 */
export function wireFormat(modelId: string): WireFormat {
  const { prefix, name } = split(modelId);
  if (prefix !== undefined) {
    if (ANTHROPIC_PREFIXES.has(prefix)) return "anthropic";
    if (GOOGLE_PREFIXES.has(prefix)) return "google";
    if (OPENAI_PREFIXES.has(prefix)) return "openai";
    return "chat";
  }
  const lower = name.toLowerCase();
  if (lower.startsWith("claude")) return "anthropic";
  if (lower.startsWith("gemini")) return "google";
  if (/^(gpt-|o\d|chatgpt-|codex-)/.test(lower)) return "openai";
  return "chat";
}

/**
 * The model ID sent to the gateway.
 *
 * In the AI SDK (and Vercel AI Gateway), `anthropic/claude-sonnet-4.5` names the
 * model's maker. In the Respan gateway a prefix pins one serving provider, and
 * `google` is not a provider name there (Gemini is `gemini`). So the maker
 * prefix of an OpenAI, Anthropic or Google model is dropped, and the gateway
 * routes the bare name across every provider that serves it, with fallback.
 * Any other prefix is a Respan provider pin (`azure/gpt-5.5`) and is kept.
 * The gateway treats `.` and `-` in versions alike, so `claude-sonnet-4.5`
 * and `claude-sonnet-4-5` name the same model.
 */
export function gatewayModelId(modelId: string): string {
  const { prefix, name } = split(modelId);
  if (prefix === undefined) return modelId;
  const maker =
    ANTHROPIC_PREFIXES.has(prefix) || GOOGLE_PREFIXES.has(prefix) || OPENAI_PREFIXES.has(prefix);
  return maker ? name : modelId;
}
