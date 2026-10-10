import { existsSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, join } from "node:path";
import { pathToFileURL } from "node:url";
import { snapshot } from "./_privacy.js";
export const PACKAGE_VERSION = createRequire(import.meta.url)("../package.json")
  .version as string;
export const INSTRUMENTATION_LIBRARY_NAME = "@respan/instrumentation-anthropic";
export function safeJson(value: unknown): string {
  try {
    return JSON.stringify(snapshot(value)) ?? "null";
  } catch {
    return "null";
  }
}
export function textOrJson(value: unknown): string {
  return typeof value === "string" ? value : safeJson(value);
}
export function toolCalls(content: unknown): Record<string, any>[] {
  if (!Array.isArray(content)) return [];
  return content
    .filter((b) => b && b.type === "tool_use")
    .map((b) => ({
      id: b.id,
      type: "function",
      function: { name: b.name, arguments: safeJson(b.input) },
    }));
}
export function messages(body: Record<string, any>): Record<string, any>[] {
  const out: Record<string, any>[] = [];
  if (body.system !== undefined)
    out.push({ role: "system", content: body.system });
  for (const message of Array.isArray(body.messages) ? body.messages : []) {
    const blocks = Array.isArray(message.content) ? message.content : undefined;
    if (!blocks) {
      out.push({ role: message.role, content: message.content });
      continue;
    }
    const ordinary = blocks.filter((b) => b?.type !== "tool_result");
    if (ordinary.length) {
      const calls = toolCalls(ordinary);
      out.push({
        role: message.role,
        content: ordinary,
        ...(calls.length ? { tool_calls: calls } : {}),
      });
    }
    for (const block of blocks)
      if (block?.type === "tool_result")
        out.push({
          role: "tool",
          content: block.content,
          tool_call_id: block.tool_use_id,
          is_error: block.is_error,
        });
  }
  return out;
}
export function tools(value: unknown): unknown[] {
  if (!Array.isArray(value)) return [];
  return value.map((tool) =>
    tool.type && tool.type !== "custom"
      ? tool
      : {
          type: "function",
          function: {
            ...tool,
            parameters: tool.input_schema,
            input_schema: undefined,
          },
        },
  );
}
function findPackageDirectory(resolvedEntry: string): string | null {
  let currentDir = dirname(resolvedEntry);

  while (true) {
    if (existsSync(join(currentDir, "package.json"))) {
      return currentDir;
    }

    const parentDir = dirname(currentDir);
    if (parentDir === currentDir) {
      return null;
    }
    currentDir = parentDir;
  }
}

function addAnthropicModuleCandidates(
  urls: Set<string>,
  resolverBase: string | URL,
): void {
  try {
    const require = createRequire(resolverBase);
    const resolvedEntry = require.resolve("@anthropic-ai/sdk");
    const packageDir = findPackageDirectory(resolvedEntry);

    if (!packageDir) return;

    for (const entryFile of ["index.mjs", "index.js"]) {
      const entryPath = join(packageDir, entryFile);
      if (existsSync(entryPath)) {
        urls.add(pathToFileURL(entryPath).href);
      }
    }
  } catch {
    // Ignore resolution failures for this candidate.
  }
}

export async function loadAnthropicConstructors(): Promise<any[]> {
  const candidateUrls = new Set<string>();
  const runtimeResolutionBases = [
    join(process.cwd(), "__respan_runtime__.js"),
    process.env.INIT_CWD
      ? join(process.env.INIT_CWD, "__respan_init__.js")
      : null,
    process.argv[1] ?? null,
    import.meta.url,
  ].filter(Boolean) as Array<string | URL>;

  for (const resolutionBase of runtimeResolutionBases) {
    addAnthropicModuleCandidates(candidateUrls, resolutionBase);
  }

  const constructors: any[] = [];
  for (const moduleUrl of candidateUrls) {
    try {
      const importedModule = await import(moduleUrl);
      const Anthropic = importedModule?.default ?? importedModule;
      if (
        typeof Anthropic === "function" &&
        !constructors.includes(Anthropic)
      ) {
        constructors.push(Anthropic);
      }
    } catch {
      // Ignore candidate import failures so we can keep trying others.
    }
  }

  return constructors;
}
