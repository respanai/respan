import { data, snapshot } from "./_privacy.js";

export function json(value: unknown): string | undefined {
  try {
    return JSON.stringify(snapshot(value));
  } catch {
    return undefined;
  }
}
export function messages(value: unknown): any[] {
  if (!Array.isArray(value)) return [];
  return value.map((m) => {
    const copy = snapshot(m) ?? {};
    const options = data(copy, "options");
    const result = data(options, "toolResult");
    if (result)
      return {
        role: "tool",
        content: data(result, "result"),
        tool_call_id: data(result, "id"),
        name: data(result, "name"),
      };
    const out: any = {
      role: data(copy, "role"),
      content: data(copy, "content"),
    };
    const calls = data(options, "toolCall");
    if (Array.isArray(calls)) out.tool_calls = calls.map(toolCall);
    return out;
  });
}
export function toolCall(value: unknown): any {
  const name = data(value, "name");
  const input = data(value, "input");
  return {
    id: data(value, "id"),
    type: "function",
    function: {
      name,
      arguments: typeof input === "string" ? input : json(input),
    },
  };
}
export function tools(value: unknown): any[] | undefined {
  if (!Array.isArray(value)) return undefined;
  return value.map((t) => {
    // Native FunctionTool.metadata is a library getter; do not invoke customer getters.
    let owner: any = t;
    let metadata = data(t, "metadata") ?? data(t, "_metadata");
    while (metadata === undefined && owner && Object.getPrototypeOf(owner)) {
      owner = Object.getPrototypeOf(owner);
      const descriptor = Object.getOwnPropertyDescriptor(owner, "metadata");
      if (descriptor?.get && owner.constructor?.name === "FunctionTool") {
        metadata = descriptor.get.call(t);
        break;
      }
    }
    return { type: "function", function: snapshot(metadata) ?? {} };
  });
}
export function usage(response: unknown): Record<string, number> {
  const raw = data(response, "raw");
  const candidates = Array.isArray(raw) ? raw : [raw];
  for (let i = candidates.length - 1; i >= 0; i--) {
    const candidate = candidates[i];
    const value =
      data(candidate, "usage") ??
      data(data(candidate, "raw"), "usage") ??
      data(data(candidate, "response"), "usage");
    if (!value) continue;
    const result: Record<string, number> = {};
    for (const k of [
      "prompt_tokens",
      "completion_tokens",
      "total_tokens",
      "input_tokens",
      "output_tokens",
    ]) {
      const n = data(value, k);
      if (typeof n === "number" && Number.isFinite(n)) result[k] = n;
    }
    const cached = data(data(value, "prompt_tokens_details"), "cached_tokens");
    if (typeof cached === "number" && Number.isFinite(cached))
      result.cached_tokens = cached;
    if (Object.keys(result).length) return result;
  }
  return {};
}
export function responseMessage(response: unknown): any[] {
  const message = data(response, "message");
  if (message !== undefined) return messages([message]);
  const text = data(response, "text");
  return text === undefined
    ? []
    : [{ role: "assistant", content: snapshot(text) }];
}
