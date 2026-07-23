import * as fs from 'node:fs';
import * as path from 'node:path';
import * as os from 'node:os';
import { execSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { Flags } from '@oclif/core';

// ── Defaults ──────────────────────────────────────────────────────────────

/** Default Respan API base (SaaS). Enterprise users override this. */
export const DEFAULT_BASE_URL = 'https://api.respan.ai/api';

// ── Shared flags ──────────────────────────────────────────────────────────

export const integrateFlags = {
  local: Flags.boolean({
    description: 'Write per-project config (default)',
    default: false,
    exclusive: ['global'],
  }),
  global: Flags.boolean({
    description: 'Write user-level global config',
    default: false,
    exclusive: ['local'],
  }),
  enable: Flags.boolean({
    description: 'Enable tracing (default)',
    default: false,
    exclusive: ['disable'],
  }),
  disable: Flags.boolean({
    description: 'Disable tracing',
    default: false,
    exclusive: ['enable'],
  }),
  'project-id': Flags.string({
    description: 'Respan project ID (added to metadata / resource attributes)',
    env: 'RESPAN_PROJECT_ID',
  }),
  'base-url': Flags.string({
    description: 'Respan API base URL (for enterprise deployments)',
    default: DEFAULT_BASE_URL,
  }),
  attrs: Flags.string({
    description: "Custom attributes JSON (e.g. '{\"env\":\"prod\"}')",
    default: '{}',
  }),
  'customer-id': Flags.string({
    description: 'Customer/user identifier for traces (e.g. your name or email)',
    env: 'RESPAN_CUSTOMER_ID',
  }),
  'span-name': Flags.string({
    description: 'Root span name for traces (default: claude-code)',
  }),
  'workflow-name': Flags.string({
    description: 'Workflow name for traces (default: claude-code)',
  }),
  'dry-run': Flags.boolean({
    description: 'Preview changes without writing files',
    default: false,
  }),
};

// ── Scope resolution ─────────────────────────────────────────────────────

export type Scope = 'local' | 'global' | 'both';

/**
 * Resolve whether to write local, global, or both configs.
 *
 * - `--global` → 'global'
 * - `--local`  → 'local'
 * - neither    → tool-specific default (passed by each command)
 */
export function resolveScope(
  flags: { local: boolean; global: boolean },
  defaultScope: Scope = 'local',
): Scope {
  if (flags.global) return 'global';
  if (flags.local) return 'local';
  return defaultScope;
}

/**
 * Find the project root (git root, or cwd if not in a repo).
 */
export function findProjectRoot(): string {
  try {
    return execSync('git rev-parse --show-toplevel', {
      encoding: 'utf-8',
      stdio: ['pipe', 'pipe', 'pipe'],
    }).trim();
  } catch {
    return process.cwd();
  }
}

// ── Hook script ───────────────────────────────────────────────────────────

/**
 * Return the bundled Claude Code hook script contents.
 *
 * The .py file lives in src/assets/ and is copied to dist/assets/ during
 * build.  Reading at runtime keeps the hook version-locked to the CLI
 * version — upgrading the CLI and re-running integrate updates the hook.
 */
/**
 * Return the bundled JS hook script contents for the given CLI tool.
 * These are standalone Node.js scripts bundled with esbuild.
 */
export function getJsHookScript(tool: 'claude-code' | 'gemini-cli' | 'codex-cli'): string {
  const dir = path.dirname(fileURLToPath(import.meta.url));
  const hookPath = path.join(dir, '..', 'hooks', `${tool}.cjs`);
  return fs.readFileSync(hookPath, 'utf-8');
}

// ── Utilities ─────────────────────────────────────────────────────────────

/**
 * Deep merge source into target.
 * Objects are recursively merged; arrays and primitives from source overwrite.
 */
export function deepMerge(
  target: Record<string, unknown>,
  source: Record<string, unknown>,
): Record<string, unknown> {
  const result = { ...target };
  for (const key of Object.keys(source)) {
    const sv = source[key];
    const tv = target[key];
    if (
      sv && typeof sv === 'object' && !Array.isArray(sv) &&
      tv && typeof tv === 'object' && !Array.isArray(tv)
    ) {
      result[key] = deepMerge(
        tv as Record<string, unknown>,
        sv as Record<string, unknown>,
      );
    } else {
      result[key] = sv;
    }
  }
  return result;
}

export function ensureDir(dirPath: string): void {
  fs.mkdirSync(dirPath, { recursive: true });
}

export function readJsonFile(filePath: string): Record<string, unknown> {
  try {
    return JSON.parse(fs.readFileSync(filePath, 'utf-8'));
  } catch {
    return {};
  }
}

export function writeJsonFile(filePath: string, data: Record<string, unknown>): void {
  ensureDir(path.dirname(filePath));
  fs.writeFileSync(filePath, JSON.stringify(data, null, 2) + '\n');
}

export function readTextFile(filePath: string): string {
  try {
    return fs.readFileSync(filePath, 'utf-8');
  } catch {
    return '';
  }
}

export function writeTextFile(filePath: string, content: string): void {
  ensureDir(path.dirname(filePath));
  fs.writeFileSync(filePath, content);
}

/**
 * Read a `KEY=value` line from .env-style content. Trims before stripping a
 * single layer of surrounding quotes — the closing quote can sit inside
 * trailing whitespace, so order matters. Returns undefined if absent or empty.
 */
export function extractEnvVar(content: string, key: string): string | undefined {
  const match = content.match(new RegExp(`^${key}=(.+)$`, 'm'));
  if (!match) return undefined;
  return match[1].trim().replace(/^["']|["']$/g, '').trim() || undefined;
}

export function expandHome(p: string): string {
  if (p.startsWith('~')) {
    return path.join(os.homedir(), p.slice(1));
  }
  return p;
}

export function parseAttrs(raw: string): Record<string, string> {
  try {
    const parsed = JSON.parse(raw);
    if (typeof parsed === 'object' && parsed !== null && !Array.isArray(parsed)) {
      return parsed as Record<string, string>;
    }
    return {};
  } catch {
    throw new Error(`Invalid JSON for --attrs: ${raw}`);
  }
}

/**
 * Convert a Record to OTel resource attributes string.
 * Format: key1=value1,key2=value2
 */
export function toOtelResourceAttrs(attrs: Record<string, string>): string {
  return Object.entries(attrs)
    .map(([k, v]) => `${k}=${v}`)
    .join(',');
}
