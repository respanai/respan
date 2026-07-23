/**
 * Respan instrumentation plugin for the OpenAI SDK.
 *
 * Wraps `@traceloop/instrumentation-openai` in the Respan plugin protocol.
 *
 * ```typescript
 * import { Respan } from "@respan/respan";
 * import { OpenAIInstrumentor } from "@respan/instrumentation-openai";
 *
 * const respan = new Respan({
 *   instrumentations: [new OpenAIInstrumentor()],
 * });
 * await respan.initialize();
 * ```
 */
import { existsSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, join } from "node:path";
import { pathToFileURL } from "node:url";

export class OpenAIInstrumentor {
  public readonly name = "openai";
  private static readonly _sharedState = {
    activeInstances: 0,
    instrumentor: null as any,
    openAI: null as any,
  };

  private _isInstrumented = false;

  async activate(): Promise<void> {
    if (this._isInstrumented) return;

    const sharedState = OpenAIInstrumentor._sharedState;

    if (sharedState.activeInstances === 0) {
      const { trace } = await import("@opentelemetry/api");
      const { OpenAIInstrumentation } = await import(
        "@traceloop/instrumentation-openai"
      );

      sharedState.instrumentor = new OpenAIInstrumentation();
      sharedState.instrumentor.setTracerProvider(trace.getTracerProvider());
      sharedState.openAI = (await importOpenAISdk()).default;
      sharedState.instrumentor.manuallyInstrument(sharedState.openAI);
      installAzureOpenAISkipGuard(sharedState.openAI);
    }

    sharedState.activeInstances += 1;
    this._isInstrumented = true;
  }

  deactivate(): void {
    if (!this._isInstrumented) return;

    const sharedState = OpenAIInstrumentor._sharedState;
    sharedState.activeInstances = Math.max(0, sharedState.activeInstances - 1);
    this._isInstrumented = false;

    if (sharedState.activeInstances > 0 || !sharedState.instrumentor) return;

    try {
      if (hasOpenAIUnwrapMarkers(sharedState.openAI)) {
        sharedState.instrumentor.unpatch({ OpenAI: sharedState.openAI });
      }
    } catch {
      /* ignore */
    }

    sharedState.instrumentor = null;
    sharedState.openAI = null;
  }
}

function hasOpenAIUnwrapMarkers(OpenAI: any): boolean {
  return [
    OpenAI?.Chat?.Completions?.prototype?.create,
    OpenAI?.Completions?.prototype?.create,
  ].some((method) => Boolean(method?.__original || method?.__wrapped));
}

const AZURE_OPENAI_SKIP_GUARD = Symbol.for("respan.instrumentation.openai.azureSkipGuard");

function installAzureOpenAISkipGuard(OpenAI: any): void {
  guardOpenAIMethod(OpenAI?.Chat?.Completions?.prototype, "create", OpenAI);
  guardOpenAIMethod(OpenAI?.Completions?.prototype, "create", OpenAI);
}

function guardOpenAIMethod(target: any, methodName: string, OpenAI: any): void {
  if (!target) return;

  const tracedMethod = target[methodName];
  if (typeof tracedMethod !== "function" || tracedMethod[AZURE_OPENAI_SKIP_GUARD]) {
    return;
  }

  const original = tracedMethod.__original;
  if (typeof original !== "function") {
    return;
  }

  const guardedMethod = function respanOpenAIAzureSkipGuard(this: any, ...args: any[]) {
    if (isAzureOpenAIResource(this, OpenAI)) {
      return original.apply(this, args);
    }
    return tracedMethod.apply(this, args);
  };

  Object.defineProperty(guardedMethod, "__original", {
    configurable: true,
    value: original,
  });
  Object.defineProperty(guardedMethod, "__wrapped", {
    configurable: true,
    value: tracedMethod.__wrapped ?? true,
  });
  Object.defineProperty(guardedMethod, AZURE_OPENAI_SKIP_GUARD, {
    configurable: true,
    value: true,
  });

  target[methodName] = guardedMethod;
}

function isAzureOpenAIResource(receiver: any, OpenAI: any): boolean {
  const client = receiver?._client ?? receiver;
  if (!client) return false;

  const AzureOpenAI = OpenAI?.AzureOpenAI;
  if (typeof AzureOpenAI === "function" && client instanceof AzureOpenAI) {
    return true;
  }

  const constructorName = client.constructor?.name;
  if (constructorName === "AzureOpenAI" || constructorName?.endsWith("AzureOpenAI")) {
    return true;
  }

  const baseURL = typeof client.baseURL === "string" ? client.baseURL.toLowerCase() : "";
  return typeof client.apiVersion === "string" && (baseURL.includes("azure") || Boolean(client.deploymentName));
}

async function importOpenAISdk(): Promise<any> {
  try {
    const hostRequire = createRequire(`${process.cwd()}/package.json`);
    const resolved = hostRequire.resolve("openai");
    const esmEntry = join(dirname(resolved), "index.mjs");
    const entry = existsSync(esmEntry) ? esmEntry : resolved;
    return await import(pathToFileURL(entry).href);
  } catch {
    return await import("openai");
  }
}
