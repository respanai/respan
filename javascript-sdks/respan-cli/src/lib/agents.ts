// Coding-agent registry and detection primitives.
//
// This is the single source of truth for the agents Respan can configure.
// commands/setup/index.ts and commands/setup/doctor.ts previously maintained
// parallel copies of this metadata.

import * as fs from 'node:fs';
import * as path from 'node:path';
import { execSync } from 'node:child_process';

export type CliTool = 'claude-code' | 'cursor' | 'codex-cli' | 'gemini-cli' | 'opencode';

export interface ToolMeta {
  name: string;
  binary: string;
  description: string;
  configDirs: string[];
}

export const CLI_TOOLS: Record<CliTool, ToolMeta> = {
  'claude-code': {
    name: 'Claude Code',
    binary: 'claude',
    description: 'Anthropic\'s coding agent',
    configDirs: ['~/.claude', '.claude'],
  },
  'cursor': {
    name: 'Cursor',
    binary: 'cursor',
    description: 'AI-powered code editor',
    configDirs: ['.cursor', '.cursorrc'],
  },
  'codex-cli': {
    name: 'Codex CLI',
    binary: 'codex',
    description: 'OpenAI\'s coding agent',
    configDirs: ['~/.codex', '.codex'],
  },
  'gemini-cli': {
    name: 'Gemini CLI',
    binary: 'gemini',
    description: 'Google\'s coding agent',
    configDirs: ['~/.gemini', '.gemini'],
  },
  'opencode': {
    name: 'OpenCode',
    binary: 'opencode',
    description: 'Open-source coding agent',
    configDirs: ['.opencode'],
  },
};

export interface DetectionSignal {
  tool: CliTool;
  onPath: boolean;
  hasConfigDir: boolean;
  reason: string;
}

export function isBinaryInstalled(binary: string): boolean {
  try {
    execSync(`command -v ${binary}`, { stdio: 'pipe' });
    return true;
  } catch {
    return false;
  }
}

export function detectAgents(projectRoot: string, home: string): DetectionSignal[] {
  const signals: DetectionSignal[] = [];

  for (const [id, meta] of Object.entries(CLI_TOOLS)) {
    const onPath = isBinaryInstalled(meta.binary);
    const hasConfigDir = meta.configDirs.some((dir) => {
      const resolved = dir.startsWith('~')
        ? path.join(home, dir.slice(1))
        : dir.startsWith('.')
          ? path.join(projectRoot, dir)
          : dir;
      return fs.existsSync(resolved);
    });

    const reasons: string[] = [];
    if (onPath) reasons.push('binary on PATH');
    if (hasConfigDir) reasons.push('config directory found');

    signals.push({
      tool: id as CliTool,
      onPath,
      hasConfigDir,
      reason: reasons.length > 0 ? reasons.join(', ') : 'not detected',
    });
  }

  return signals;
}
