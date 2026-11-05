import * as fs from 'node:fs';
import * as path from 'node:path';
import * as os from 'node:os';

const CONFIG_DIR = path.join(os.homedir(), '.respan');
const CREDENTIALS_FILE = path.join(CONFIG_DIR, 'credentials.json');
const CONFIG_FILE = path.join(CONFIG_DIR, 'config.json');
const CONFIG_DIR_MODE = 0o700;
const CREDENTIALS_FILE_MODE = 0o600;

export interface ApiKeyCredential {
  type: 'api_key';
  apiKey: string;
  baseUrl: string;
}

export interface JwtCredential {
  type: 'jwt';
  accessToken: string;
  refreshToken: string;
  email: string;
  baseUrl: string;
}

export type Credential = ApiKeyCredential | JwtCredential;

export interface Config {
  activeProfile?: string;
  defaults?: Record<string, string>;
}

// Removes group and other access, keeping the owner's bits. Best-effort: a path
// we can't chmod (another owner, read-only mount) must stay usable, and Windows
// doesn't map these modes to ACLs.
function restrictToOwner(target: string): void {
  if (process.platform === 'win32') return;
  try {
    const mode = fs.statSync(target).mode & 0o777;
    if (mode & 0o077) fs.chmodSync(target, mode & 0o700);
  } catch {
    // Missing, or not ours to change
  }
}

function ensureConfigDir(): void {
  fs.mkdirSync(CONFIG_DIR, { recursive: true, mode: CONFIG_DIR_MODE });
  restrictToOwner(CONFIG_DIR);
}

function readJson<T>(filePath: string, fallback: T): T {
  try {
    if (!fs.existsSync(filePath)) return fallback;
    const raw = fs.readFileSync(filePath, 'utf-8');
    return JSON.parse(raw) as T;
  } catch {
    return fallback;
  }
}

function writeJson(filePath: string, data: unknown, mode: number = 0o666): void {
  ensureConfigDir();
  fs.writeFileSync(filePath, JSON.stringify(data, null, 2) + '\n', { encoding: 'utf-8', mode });
}

// --- Credentials ---

// Tightens files written by older CLI versions, which used the default modes.
export function tightenCredentialPermissions(): void {
  restrictToOwner(CONFIG_DIR);
  restrictToOwner(CREDENTIALS_FILE);
}

function writeCredentials(creds: Record<string, Credential>): void {
  // writeFileSync's mode only applies when it creates the file, so tighten an
  // existing one first: new secrets must not land in a world-readable file.
  restrictToOwner(CREDENTIALS_FILE);
  writeJson(CREDENTIALS_FILE, creds, CREDENTIALS_FILE_MODE);
}

export function getAllCredentials(): Record<string, Credential> {
  tightenCredentialPermissions();
  return readJson<Record<string, Credential>>(CREDENTIALS_FILE, {});
}

export function getCredential(profile?: string): Credential | undefined {
  const creds = getAllCredentials();
  const name = profile || getActiveProfile();
  return creds[name];
}

export function setCredential(profile: string, credential: Credential): void {
  const creds = getAllCredentials();
  creds[profile] = credential;
  writeCredentials(creds);
  // Set active profile if this is the first credential
  const config = getConfig();
  if (!config.activeProfile) {
    setActiveProfile(profile);
  }
}

export function deleteCredential(profile: string): void {
  const creds = getAllCredentials();
  delete creds[profile];
  writeCredentials(creds);
}

// --- Config ---

export function getConfig(): Config {
  return readJson<Config>(CONFIG_FILE, {});
}

export function getActiveProfile(): string {
  const config = getConfig();
  return config.activeProfile || 'default';
}

export function setActiveProfile(profile: string): void {
  const config = getConfig();
  config.activeProfile = profile;
  writeJson(CONFIG_FILE, config);
}

export function getConfigValue(key: string): string | undefined {
  const config = getConfig();
  if (key === 'activeProfile') return config.activeProfile;
  return config.defaults?.[key];
}

export function setConfigValue(key: string, value: string): void {
  const config = getConfig();
  if (key === 'activeProfile') {
    config.activeProfile = value;
  } else {
    if (!config.defaults) config.defaults = {};
    config.defaults[key] = value;
  }
  writeJson(CONFIG_FILE, config);
}
