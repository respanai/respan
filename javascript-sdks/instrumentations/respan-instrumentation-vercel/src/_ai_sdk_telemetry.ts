import { createRequire } from "node:module";
import { pathToFileURL } from "node:url";

type RuntimeModule = Record<string, any>;
type ResolveModule = (specifier: string) => string;

export type AISDKTelemetryRegistrationStatus =
  | "registered"
  | "already-registered"
  | "legacy"
  | "missing-ai"
  | "missing-adapter"
  | "incompatible-adapter";

export interface AISDKTelemetryRegistrationOptions {
  importModule?: (specifier: string) => Promise<RuntimeModule>;
  warn?: (message: string) => void;
}

export interface OwnedAISDKTelemetryLease {
  integration: unknown;
  leases: number;
}

export interface AISDKTelemetryRegistrationResult {
  status: AISDKTelemetryRegistrationStatus;
  lease?: OwnedAISDKTelemetryLease;
}

export interface RuntimeModuleResolutionOptions {
  hostResolve?: ResolveModule;
}

const AI_SDK_TELEMETRY_MARKER = Symbol.for(
  "@respan/instrumentation-vercel.ai-sdk-telemetry-registered",
);
/**
 * Resolve optional runtime peers from the host application. This mirrors the
 * host-first loader used by Respan core and the OpenRouter instrumentor. It
 * matters for linked/workspace installs, where a bare import from this package
 * can find a different hoisted AI SDK version than the application actually
 * calls.
 */
export function resolveRuntimeModuleURL(
  specifier: string,
  options: RuntimeModuleResolutionOptions = {},
): string {
  const hostRequire = createRequire(`${process.cwd()}/package.json`);
  const hostResolve = options.hostResolve ?? ((id: string) => hostRequire.resolve(id));

  return pathToFileURL(hostResolve(specifier)).href;
}

async function defaultImportModule(specifier: string): Promise<RuntimeModule> {
  try {
    return await import(/* webpackIgnore: true */ resolveRuntimeModuleURL(specifier));
  } catch {
    // Fall back to normal package-local resolution, matching the pattern used
    // by Respan core and other explicit instrumentors.
    return await import(/* webpackIgnore: true */ specifier);
  }
}

function telemetryIntegrations(): unknown[] {
  const integrations = (globalThis as any).AI_SDK_TELEMETRY_INTEGRATIONS;
  return Array.isArray(integrations) ? integrations : [];
}

function isOpenTelemetryIntegration(integration: unknown): boolean {
  const name = (integration as any)?.constructor?.name;
  return name === "OpenTelemetry" || name === "LegacyOpenTelemetry";
}

function ownedTelemetryLease(): OwnedAISDKTelemetryLease | undefined {
  const lease = (globalThis as any)[AI_SDK_TELEMETRY_MARKER] as
    | OwnedAISDKTelemetryLease
    | undefined;

  if (
    lease &&
    typeof lease === "object" &&
    lease.leases > 0 &&
    telemetryIntegrations().includes(lease.integration)
  ) {
    return lease;
  }

  // Discard a stale marker. A later release holding the old lease is then a
  // no-op because it no longer matches the global owner.
  delete (globalThis as any)[AI_SDK_TELEMETRY_MARKER];
  return undefined;
}

function acquireLease(
  lease: OwnedAISDKTelemetryLease,
): AISDKTelemetryRegistrationResult {
  lease.leases += 1;
  return { status: "already-registered", lease };
}

/**
 * AI SDK 7 moved OpenTelemetry emission out of ai and into the optional
 * @ai-sdk/otel adapter. Register it when available so activating the Respan
 * Vercel instrumentor remains sufficient to produce spans.
 *
 * AI SDK 4-6 do not export registerTelemetry; those versions keep using
 * their native experimental_telemetry path and are intentionally untouched.
 * Each Respan instrumentor that shares an owned adapter acquires a lease.
 */
export async function ensureAISDKTelemetry(
  options: AISDKTelemetryRegistrationOptions = {},
): Promise<AISDKTelemetryRegistrationResult> {
  const importModule = options.importModule ?? defaultImportModule;
  const warn = options.warn ?? console.warn;

  let aiModule: RuntimeModule;
  try {
    aiModule = await importModule("ai");
  } catch {
    return { status: "missing-ai" };
  }

  if (typeof aiModule.registerTelemetry !== "function") {
    return { status: "legacy" };
  }

  const currentLease = ownedTelemetryLease();
  if (currentLease) {
    return acquireLease(currentLease);
  }

  // A user-owned OpenTelemetry integration is never leased or removed by
  // Respan. Its presence makes automatic registration a no-op.
  if (telemetryIntegrations().some(isOpenTelemetryIntegration)) {
    return { status: "already-registered" };
  }

  let adapterModule: RuntimeModule;
  try {
    adapterModule = await importModule("@ai-sdk/otel");
  } catch {
    warn(
      '[Respan] AI SDK 7 detected, but "@ai-sdk/otel" is not installed. ' +
        "Install it to enable Vercel AI SDK tracing: npm install @ai-sdk/otel",
    );
    return { status: "missing-adapter" };
  }

  // Concurrent activations can both pass the pre-import checks. The first one
  // to resume registers the adapter; every later one must acquire its lease
  // before treating that adapter as a generic user-owned integration.
  const concurrentlyRegisteredLease = ownedTelemetryLease();
  if (concurrentlyRegisteredLease) {
    return acquireLease(concurrentlyRegisteredLease);
  }

  // A user integration may also have registered while the optional import was
  // resolving. It remains user-owned and is never leased by Respan.
  if (telemetryIntegrations().some(isOpenTelemetryIntegration)) {
    return { status: "already-registered" };
  }

  if (typeof adapterModule.OpenTelemetry !== "function") {
    warn(
      '[Respan] The installed "@ai-sdk/otel" package does not export OpenTelemetry. ' +
        "Install a compatible 1.x release to enable Vercel AI SDK tracing.",
    );
    return { status: "incompatible-adapter" };
  }

  const integration = new adapterModule.OpenTelemetry();
  aiModule.registerTelemetry(integration);

  const lease: OwnedAISDKTelemetryLease = { integration, leases: 1 };
  (globalThis as any)[AI_SDK_TELEMETRY_MARKER] = lease;
  return { status: "registered", lease };
}

/**
 * Release one instrumentor's claim on a Respan-owned AI SDK adapter. The
 * adapter is removed only after the final lease, and only by exact identity.
 * User-owned integrations are never represented by a lease and are untouched.
 */
export function releaseOwnedAISDKTelemetry(
  lease: OwnedAISDKTelemetryLease,
): boolean {
  const currentLease = (globalThis as any)[AI_SDK_TELEMETRY_MARKER] as
    | OwnedAISDKTelemetryLease
    | undefined;

  if (currentLease !== lease || lease.leases <= 0) {
    return false;
  }

  lease.leases -= 1;
  if (lease.leases > 0) {
    return false;
  }

  const integrations = (globalThis as any).AI_SDK_TELEMETRY_INTEGRATIONS;
  let removed = false;
  if (Array.isArray(integrations)) {
    for (let index = integrations.length - 1; index >= 0; index -= 1) {
      if (integrations[index] === lease.integration) {
        integrations.splice(index, 1);
        removed = true;
      }
    }
  }

  if ((globalThis as any)[AI_SDK_TELEMETRY_MARKER] === lease) {
    delete (globalThis as any)[AI_SDK_TELEMETRY_MARKER];
  }

  return removed;
}
