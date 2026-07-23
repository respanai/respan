import * as path from 'node:path';
import { getCredential, Credential } from './config.js';
import { findProjectRoot, readTextFile, extractEnvVar } from './integrate.js';

export const DEFAULT_BASE_URL = 'https://api.respan.ai';
export const ENTERPRISE_BASE_URL = 'https://endpoint.respan.ai';

export interface AuthConfig {
  apiKey?: string;
  accessToken?: string;
  refreshToken?: string;
  baseUrl: string;
}

function normalizeBaseUrl(baseUrl?: string): string {
  return (baseUrl || DEFAULT_BASE_URL).replace(/\/+$/, '');
}

function resolveConfiguredBaseUrl(credential?: Credential, flagBaseUrl?: string): string {
  // Keep the legacy env override for scripted/CI flows while auth login remains
  // the main persistent configuration path for CLI users.
  return normalizeBaseUrl(
    flagBaseUrl || process.env.RESPAN_API_BASE_URL || credential?.baseUrl || DEFAULT_BASE_URL,
  );
}

/**
 * Read RESPAN_API_KEY from the project's `.env`. setup writes it there as the
 * project source of truth, so in-project commands (e.g. `respan logs list`)
 * work right after setup without a separate `respan auth login`.
 */
function readProjectEnvKey(): string | undefined {
  try {
    return extractEnvVar(readTextFile(path.join(findProjectRoot(), '.env')), 'RESPAN_API_KEY');
  } catch {
    return undefined;
  }
}

export function resolveAuth(flags: { 'api-key'?: string; 'base-url'?: string; profile?: string }): AuthConfig {
  const credential = getCredential(flags.profile);
  const baseUrl = resolveConfiguredBaseUrl(credential, flags['base-url']);

  if (flags['api-key']) {
    return { apiKey: flags['api-key'], baseUrl };
  }
  if (process.env.RESPAN_API_KEY) {
    return {
      apiKey: process.env.RESPAN_API_KEY,
      baseUrl,
    };
  }
  // The project .env key wins over a saved credential — but an explicit
  // --profile selects a specific credential (and its base URL), so don't let
  // an ambient .env key override that choice and break enterprise setups.
  if (!flags.profile) {
    const projectEnvKey = readProjectEnvKey();
    if (projectEnvKey) {
      return { apiKey: projectEnvKey, baseUrl };
    }
  }
  if (credential) {
    return credentialToAuth(credential, baseUrl);
  }
  throw new Error('Not authenticated. Run `respan auth login` or set RESPAN_API_KEY.');
}

function credentialToAuth(cred: Credential, baseUrl: string): AuthConfig {
  if (cred.type === 'api_key') {
    return { apiKey: cred.apiKey, baseUrl };
  }
  return {
    accessToken: cred.accessToken,
    refreshToken: cred.refreshToken,
    baseUrl,
  };
}

export async function refreshJwtToken(credential: Credential & { type: 'jwt' }): Promise<{ access: string }> {
  const origin = credential.baseUrl.replace(/\/api\/?$/, '');
  const response = await fetch(`${origin}/auth/jwt/refresh/`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ refresh: credential.refreshToken }),
  });
  if (!response.ok) {
    throw new Error('Token refresh failed. Please login again with `respan auth login`.');
  }
  return response.json() as Promise<{ access: string }>;
}
