import * as fs from 'node:fs';
import * as path from 'node:path';
import { getCredential, Credential } from './config.js';
import { findProjectRoot, readTextFile, extractEnvVar } from './integrate.js';

export const DEFAULT_BASE_URL = 'https://api.respan.ai';
export const ENTERPRISE_BASE_URL = 'https://endpoint.respan.ai';

export type AuthSource = 'flag' | 'environment' | 'env_file' | 'project_env' | 'profile';

export interface AuthConfig {
  apiKey?: string;
  accessToken?: string;
  refreshToken?: string;
  baseUrl: string;
  source: AuthSource;
}

export interface ResolveAuthOptions {
  envFile?: string;
  projectRoot?: string;
}

function normalizeBaseUrl(baseUrl?: string): string {
  return (baseUrl || DEFAULT_BASE_URL).replace(/\/+$/, '');
}

interface EnvAuthValues {
  apiKey?: string;
  apiBaseUrl?: string;
}

function readEnvAuthFile(filePath: string): EnvAuthValues {
  const content = readTextFile(filePath);
  return {
    apiKey: extractEnvVar(content, 'RESPAN_API_KEY'),
    apiBaseUrl: extractEnvVar(content, 'RESPAN_API_BASE_URL'),
  };
}

/**
 * Read RESPAN_API_KEY from the project's `.env`. setup writes it there as the
 * project source of truth, so in-project commands (e.g. `respan logs list`)
 * work right after setup without a separate `respan auth login`.
 */
function readProjectEnvAuth(projectRoot = findProjectRoot(), envFile?: string): EnvAuthValues | undefined {
  if (envFile) {
    return readEnvAuthFile(path.resolve(envFile));
  }

  let current = path.resolve(projectRoot);

  while (true) {
    const envPath = path.join(current, '.env');
    const values = readEnvAuthFile(envPath);
    if (values.apiKey) return values;

    const parent = path.dirname(current);
    if (parent === current || fs.existsSync(path.join(current, '.git'))) {
      return undefined;
    }
    current = parent;
  }
}

export function resolveAuth(
  flags: { 'api-key'?: string; 'base-url'?: string; profile?: string },
  options: ResolveAuthOptions = {},
): AuthConfig {
  const credential = getCredential(flags.profile);
  const directBaseUrl = normalizeBaseUrl(
    flags['base-url'] || process.env.RESPAN_API_BASE_URL || DEFAULT_BASE_URL,
  );

  if (flags['api-key']) {
    return { apiKey: flags['api-key'], baseUrl: directBaseUrl, source: 'flag' };
  }
  if (options.envFile) {
    const envFileAuth = readProjectEnvAuth(options.projectRoot, options.envFile);
    if (!envFileAuth?.apiKey) {
      throw new Error(`RESPAN_API_KEY was not found in ${path.resolve(options.envFile)}.`);
    }
    return {
      apiKey: envFileAuth.apiKey,
      baseUrl: normalizeBaseUrl(
        flags['base-url']
        || envFileAuth.apiBaseUrl
        || process.env.RESPAN_API_BASE_URL
        || DEFAULT_BASE_URL,
      ),
      source: 'env_file',
    };
  }
  if (flags.profile) {
    if (!credential) throw new Error(`Authentication profile "${flags.profile}" was not found.`);
    const profileBaseUrl = normalizeBaseUrl(
      flags['base-url'] || process.env.RESPAN_API_BASE_URL || credential.baseUrl,
    );
    return credentialToAuth(credential, profileBaseUrl, 'profile');
  }
  if (process.env.RESPAN_API_KEY) {
    return {
      apiKey: process.env.RESPAN_API_KEY,
      baseUrl: directBaseUrl,
      source: 'environment',
    };
  }
  const projectEnvAuth = readProjectEnvAuth(options.projectRoot);
  if (projectEnvAuth?.apiKey) {
    return {
      apiKey: projectEnvAuth.apiKey,
      baseUrl: normalizeBaseUrl(
        flags['base-url']
        || projectEnvAuth.apiBaseUrl
        || process.env.RESPAN_API_BASE_URL
        || DEFAULT_BASE_URL,
      ),
      source: 'project_env',
    };
  }
  if (credential) {
    const profileBaseUrl = normalizeBaseUrl(
      flags['base-url'] || process.env.RESPAN_API_BASE_URL || credential.baseUrl,
    );
    return credentialToAuth(credential, profileBaseUrl, 'profile');
  }
  throw new Error('Not authenticated. Run `respan auth login` or set RESPAN_API_KEY.');
}

function credentialToAuth(cred: Credential, baseUrl: string, source: AuthSource): AuthConfig {
  if (cred.type === 'api_key') {
    return { apiKey: cred.apiKey, baseUrl, source };
  }
  return {
    accessToken: cred.accessToken,
    refreshToken: cred.refreshToken,
    baseUrl,
    source,
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
