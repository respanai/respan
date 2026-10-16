/** Convert Stagehand values into the JSON shapes Respan renders. */

import { IMAGE_PLACEHOLDER, MAX_EXPORTED_IMAGE_BASE64 } from "./_constants.js";

type Json = null | boolean | number | string | Json[] | { [key: string]: Json };
const MAX_DEPTH = 32;

/** A JSON-safe copy: binary becomes its length, functions and classes their type. */
export function jsonable(value: unknown, depth = 0): Json {
  if (value === null || value === undefined) return null;
  if (typeof value === "string" || typeof value === "boolean") return value;
  if (typeof value === "number") return Number.isFinite(value) ? value : String(value);
  if (typeof value === "bigint") return value.toString();
  if (value instanceof Uint8Array) return { type: "bytes", length: value.byteLength };
  if (typeof value !== "object") return { type: typeof value };
  if (depth >= MAX_DEPTH) return { type: "object", truncated: true };
  if (Array.isArray(value)) return value.map((item) => jsonable(item, depth + 1));
  const proto = Object.getPrototypeOf(value);
  if (proto !== Object.prototype && proto !== null) {
    // zod schemas, Page / Locator instances, errors: never walk their internals.
    if (value instanceof Error) return { type: value.name, message: value.message };
    return { type: value.constructor?.name ?? "object" };
  }
  const out: Record<string, Json> = {};
  for (const [key, item] of Object.entries(value)) {
    if (item !== undefined) out[key] = jsonable(item, depth + 1);
  }
  return out;
}

export function json(value: unknown): string {
  return JSON.stringify(jsonable(value));
}

export function imagePart(base64: string, mime: string, keep: boolean): Record<string, unknown> {
  if (keep && base64.length <= MAX_EXPORTED_IMAGE_BASE64) {
    // A data-URI image part: Respan's ingest lifts it into media storage and
    // the log view renders it.
    return { type: "image_url", image_url: { url: `data:${mime};base64,${base64}` } };
  }
  return {
    type: "text",
    text: IMAGE_PLACEHOLDER.replace("{mime}", mime).replace("{size}", String(base64.length)),
  };
}

type Block = Record<string, any>;
export interface ChatMessage {
  role: string;
  content: string | Block[];
  tool_calls?: Block[];
  tool_call_id?: string;
}

function blocks(content: unknown): Block[] {
  if (Array.isArray(content)) return content as Block[];
  return content && typeof content === "object" ? [content as Block] : [];
}

function contentParts(items: Block[], keepImages: boolean): Block[] {
  const parts: Block[] = [];
  for (const block of items) {
    if (block.type === "text") parts.push({ type: "text", text: block.text });
    else if (block.type === "image") parts.push(imagePart(block.data, block.mimeType, keepImages));
  }
  return parts;
}

/** A plain string when the content is text only, else OpenAI-style parts. */
function flatten(parts: Block[]): string | Block[] {
  return parts.every((part) => part.type === "text") ? parts.map((part) => part.text).join("\n") : parts;
}

/**
 * Stagehand's `llm.generate` messages (MCP-style content blocks) as
 * OpenAI-shaped chat messages, the shape Respan's log view renders.
 */
export function chatMessages(params: Block, keepImages: boolean): ChatMessage[] {
  const messages: ChatMessage[] = [];
  if (typeof params.systemPrompt === "string" && params.systemPrompt) {
    messages.push({ role: "system", content: params.systemPrompt });
  }
  for (const message of params.messages ?? []) {
    const items = blocks(message.content);
    const results = items.filter((block) => block.type === "tool_result");
    for (const result of results) {
      messages.push({
        role: "tool",
        tool_call_id: result.toolUseId,
        content: result.structuredContent
          ? JSON.stringify(result.structuredContent)
          : flatten(contentParts(blocks(result.content), keepImages)),
      });
    }
    const rest = items.filter((block) => block.type !== "tool_result");
    if (!rest.length && results.length) continue;
    const chat: ChatMessage = { role: message.role, content: flatten(contentParts(rest, keepImages)) };
    const calls = toolCalls(rest);
    if (calls.length) chat.tool_calls = calls;
    messages.push(chat);
  }
  return messages;
}

export function toolCalls(items: Block[]): Block[] {
  return items
    .filter((block) => block.type === "tool_use")
    .map((block) => ({
      id: block.id,
      type: "function",
      function: { name: block.name, arguments: JSON.stringify(block.input ?? {}) },
    }));
}

/** The completion message of an `llm.generate` result. */
export function completion(result: Block): ChatMessage {
  const items = blocks(result.content);
  const text =
    result.outputFormat === "json_schema" && result.structuredContent !== undefined
      ? JSON.stringify(result.structuredContent)
      : items
          .filter((block) => block.type === "text")
          .map((block) => block.text)
          .join("");
  const message: ChatMessage = { role: "assistant", content: text };
  const calls = toolCalls(items);
  if (calls.length) message.tool_calls = calls;
  return message;
}

/** Stagehand's client tools as OpenAI function definitions. */
export function functions(tools: unknown): Block[] {
  if (!Array.isArray(tools)) return [];
  return tools.map((tool: Block) => ({
    name: tool.name,
    description: tool.description,
    parameters: tool.inputSchema,
  }));
}

export function contentText(content: string | Block[]): string {
  return typeof content === "string" ? content : JSON.stringify(content);
}
