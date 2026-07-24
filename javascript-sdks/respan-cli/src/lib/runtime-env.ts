import { DEFAULT_BASE_URL, type AuthConfig } from './auth.js';
import { extractEnvVar, readTextFile } from './integrate.js';

export const DEFAULT_GATEWAY_BASE_URL = 'https://api.respan.ai/api';
export const DEFAULT_SMOKE_MODEL = 'gpt-4o-mini';
export const RESPAN_PLATFORM_URL = 'https://platform.respan.ai';

export interface RuntimeEnvironment {
  apiBaseUrl: string;
  gatewayBaseUrl: string;
  model: string;
  platformUrl: string;
  authenticated: boolean;
  gatewayReady: boolean;
  authSource: AuthConfig['source'] | 'none';
  apiKey?: string;
}

const RUNTIME_ENV_KEYS = [
  'RESPAN_API_BASE_URL',
  'RESPAN_GATEWAY_BASE_URL',
  'RESPAN_BASE_URL',
  'RESPAN_MODEL',
  'OPENAI_MODEL',
  'OPENAI_MODEL_NAME',
] as const;

export function readRuntimeEnvironmentFile(filePath?: string): NodeJS.ProcessEnv {
  if (!filePath) return {};
  const content = readTextFile(filePath);
  const values: NodeJS.ProcessEnv = {};
  for (const key of RUNTIME_ENV_KEYS) {
    const value = extractEnvVar(content, key);
    if (value) values[key] = value;
  }
  return values;
}

function stripTrailingSlashes(value: string): string {
  return value.replace(/\/+$/, '');
}

function normalizeGatewayBaseUrl(value: string): string {
  const normalized = stripTrailingSlashes(value.trim());
  if (normalized.endsWith('/chat/completions')) {
    return normalized.slice(0, -'/chat/completions'.length);
  }
  if (normalized.endsWith('/api')) return normalized;
  return `${normalized}/api`;
}

export function resolveGatewayBaseUrl(
  env: NodeJS.ProcessEnv = process.env,
): string {
  const configured = env.RESPAN_GATEWAY_BASE_URL || env.RESPAN_BASE_URL;
  return configured
    ? normalizeGatewayBaseUrl(configured)
    : DEFAULT_GATEWAY_BASE_URL;
}

export function resolveRuntimeEnvironment(
  auth?: AuthConfig,
  env: NodeJS.ProcessEnv = process.env,
): RuntimeEnvironment {
  const model = (
    env.RESPAN_MODEL ||
    env.OPENAI_MODEL ||
    env.OPENAI_MODEL_NAME ||
    DEFAULT_SMOKE_MODEL
  ).trim();

  return {
    apiBaseUrl: stripTrailingSlashes(auth?.baseUrl || env.RESPAN_API_BASE_URL || DEFAULT_BASE_URL),
    gatewayBaseUrl: resolveGatewayBaseUrl(env),
    model,
    platformUrl: RESPAN_PLATFORM_URL,
    authenticated: Boolean(auth?.apiKey || auth?.accessToken),
    gatewayReady: Boolean(auth?.apiKey),
    authSource: auth?.source || 'none',
    ...(auth?.apiKey ? { apiKey: auth.apiKey } : {}),
  };
}

function quoteShell(value: string): string {
  return `'${value.replace(/'/g, `'"'"'`)}'`;
}

export function formatRuntimeEnvironmentAsShell(
  runtime: RuntimeEnvironment,
  includeApiKey = false,
): string {
  const values: Record<string, string> = {
    RESPAN_API_BASE_URL: runtime.apiBaseUrl,
    RESPAN_GATEWAY_BASE_URL: runtime.gatewayBaseUrl,
    RESPAN_MODEL: runtime.model,
    RESPAN_PLATFORM_URL: runtime.platformUrl,
    RESPAN_AUTHENTICATED: String(runtime.authenticated),
    RESPAN_GATEWAY_READY: String(runtime.gatewayReady),
    RESPAN_AUTH_SOURCE: runtime.authSource,
  };

  if (includeApiKey && runtime.apiKey) {
    values.RESPAN_API_KEY = runtime.apiKey;
  }

  return Object.entries(values)
    .map(([key, value]) => `export ${key}=${quoteShell(value)}`)
    .join('\n');
}

export function runtimeEnvironmentJson(
  runtime: RuntimeEnvironment,
  includeApiKey = false,
): Record<string, unknown> {
  const { apiKey, ...safeRuntime } = runtime;
  return {
    ...safeRuntime,
    ...(includeApiKey && apiKey ? { apiKey } : {}),
  };
}
