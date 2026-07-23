import * as fs from 'node:fs';
import * as path from 'node:path';
import * as os from 'node:os';
import { execSync } from 'node:child_process';
import { Flags } from '@oclif/core';
import { BaseCommand } from '../../lib/base-command.js';
import { findProjectRoot, expandHome, readTextFile } from '../../lib/integrate.js';
import { GREEN, RED, YELLOW, DIM, BOLD, RESET } from '../../lib/colors.js';
import { CliTool, isBinaryInstalled } from '../../lib/agents.js';

interface ToolCheck {
  tool: CliTool;
  name: string;
  binary: string;
  binaryFound: boolean;
  configPaths: { path: string; exists: boolean; scope: string }[];
  skillInstalled: boolean;
  skillPath?: string;
  notes: string[];
}

export default class SetupDoctor extends BaseCommand {
  static description = 'Diagnose coding-agent setup for Respan';

  static examples = [
    'respan setup doctor',
    'respan setup doctor --global',
  ];

  static flags = {
    ...BaseCommand.baseFlags,
    local: Flags.boolean({
      description: 'Check local project setup',
      default: false,
      exclusive: ['global'],
    }),
    global: Flags.boolean({
      description: 'Check global setup',
      default: false,
      exclusive: ['local'],
    }),
  };

  async run(): Promise<void> {
    const { flags } = await this.parse(SetupDoctor);
    this.globalFlags = flags;

    const projectRoot = findProjectRoot();
    const home = os.homedir();
    const checkGlobal = flags.global || !flags.local;
    const checkLocal = flags.local || !flags.global;

    this.log('');
    this.log(`  ${BOLD}Respan Setup Doctor${RESET}`);
    this.log('');

    // ── 1. API Key ─────────────────────────────────────────────────
    this.log(`  ${BOLD}API Key${RESET}`);
    const envPath = path.join(projectRoot, '.env');
    const envContent = readTextFile(envPath);
    const hasApiKey = /^RESPAN_API_KEY=.+$/m.test(envContent);
    const hasEnvVar = !!process.env.RESPAN_API_KEY;

    if (hasApiKey) {
      this.log(`    ${GREEN}\u2713${RESET} Found in ${DIM}${envPath}${RESET}`);
    } else {
      this.log(`    ${RED}\u2717${RESET} Not found in ${DIM}${envPath}${RESET}`);
    }
    if (hasEnvVar) {
      this.log(`    ${GREEN}\u2713${RESET} RESPAN_API_KEY environment variable is set`);
    }
    if (!hasApiKey && !hasEnvVar) {
      this.log(`    ${YELLOW}!${RESET} Run ${DIM}respan setup${RESET} to configure your API key`);
    }
    this.log('');

    // ── 2. Respan CLI ──────────────────────────────────────────────
    this.log(`  ${BOLD}Respan CLI${RESET}`);
    const respanInstalled = isBinaryInstalled('respan');
    if (respanInstalled) {
      try {
        const version = execSync('respan --version', { encoding: 'utf-8', stdio: ['pipe', 'pipe', 'pipe'] }).trim();
        this.log(`    ${GREEN}\u2713${RESET} Installed ${DIM}(${version})${RESET}`);
      } catch {
        this.log(`    ${GREEN}\u2713${RESET} Installed`);
      }
    } else {
      this.log(`    ${YELLOW}!${RESET} Not installed globally. Install with: ${DIM}npm install -g @respan/cli${RESET}`);
    }
    this.log('');

    // ── 3. Agent checks ────────────────────────────────────────────
    this.log(`  ${BOLD}Coding Agents${RESET}`);
    this.log('');

    const tools: Record<CliTool, { binary: string; name: string; globalConfigs: string[]; localConfigs: string[]; globalSkills: string[]; localSkills: string[] }> = {
      'claude-code': {
        binary: 'claude',
        name: 'Claude Code',
        globalConfigs: ['~/.claude/settings.json'],
        localConfigs: ['.claude/settings.local.json'],
        globalSkills: ['~/.claude/skills/respan/SKILL.md'],
        localSkills: ['.claude/skills/respan/SKILL.md'],
      },
      'cursor': {
        binary: 'cursor',
        name: 'Cursor',
        globalConfigs: ['~/.cursor/rules'],
        localConfigs: ['.cursor/rules'],
        globalSkills: ['~/.agents/skills/respan/SKILL.md', '~/.cursor/skills/respan/SKILL.md'],
        localSkills: ['.agents/skills/respan/SKILL.md', '.cursor/skills/respan/SKILL.md'],
      },
      'codex-cli': {
        binary: 'codex',
        name: 'Codex CLI',
        globalConfigs: ['~/.codex/config.toml'],
        localConfigs: ['.codex/respan.json'],
        globalSkills: ['~/.agents/skills/respan/SKILL.md', '~/.codex/skills/respan/SKILL.md'],
        localSkills: ['.agents/skills/respan/SKILL.md', '.codex/skills/respan/SKILL.md'],
      },
      'gemini-cli': {
        binary: 'gemini',
        name: 'Gemini CLI',
        globalConfigs: ['~/.gemini/settings.json'],
        localConfigs: ['.gemini/settings.json'],
        globalSkills: ['~/.agents/skills/respan/SKILL.md', '~/.gemini/skills/respan/SKILL.md'],
        localSkills: ['.agents/skills/respan/SKILL.md', '.gemini/skills/respan/SKILL.md'],
      },
      'opencode': {
        binary: 'opencode',
        name: 'OpenCode',
        globalConfigs: ['~/.config/opencode'],
        localConfigs: ['.opencode'],
        globalSkills: ['~/.agents/skills/respan/SKILL.md', '~/.config/opencode/skills/respan/SKILL.md'],
        localSkills: ['.agents/skills/respan/SKILL.md', '.opencode/skills/respan/SKILL.md'],
      },
    };

    for (const [id, tool] of Object.entries(tools)) {
      const binaryFound = isBinaryInstalled(tool.binary);

      this.log(`    ${BOLD}${tool.name}${RESET} ${DIM}(${tool.binary})${RESET}`);

      // Binary
      if (binaryFound) {
        this.log(`      ${GREEN}\u2713${RESET} Binary found on PATH`);
      } else {
        this.log(`      ${DIM}\u2013${RESET} Binary not found`);
      }

      // Config
      if (checkGlobal) {
        for (const cfg of tool.globalConfigs) {
          const resolved = expandHome(cfg);
          if (fs.existsSync(resolved)) {
            this.log(`      ${GREEN}\u2713${RESET} Global config: ${DIM}${cfg}${RESET}`);
          }
        }
      }
      if (checkLocal) {
        for (const cfg of tool.localConfigs) {
          const resolved = path.join(projectRoot, cfg);
          if (fs.existsSync(resolved)) {
            this.log(`      ${GREEN}\u2713${RESET} Local config: ${DIM}${cfg}${RESET}`);
          }
        }
      }

      // Skills \u2014 each tool only lists dirs it actually reads (the shared
      // ~/.agents, or ~/.claude for Claude Code) plus its own dir as a
      // fallback. Report the first match.
      let hasSkill = false;
      if (checkGlobal) {
        const found = tool.globalSkills.find((skill) => fs.existsSync(expandHome(skill)));
        if (found) {
          this.log(`      ${GREEN}\u2713${RESET} Global skill: ${DIM}${found}${RESET}`);
          hasSkill = true;
        }
      }
      if (checkLocal) {
        const found = tool.localSkills.find((skill) => fs.existsSync(path.join(projectRoot, skill)));
        if (found) {
          this.log(`      ${GREEN}\u2713${RESET} Local skill: ${DIM}${found}${RESET}`);
          hasSkill = true;
        }
      }

      if (binaryFound && !hasSkill) {
        this.log(`      ${YELLOW}!${RESET} No Respan setup skill installed. Run ${DIM}respan setup --agent ${id}${RESET}`);
      }

      this.log('');
    }

    // ── 4. SDK install docs ────────────────────────────────────────
    this.log(`  ${BOLD}SDK Install Docs${RESET}`);
    const localDocsPath = path.join(projectRoot, '.respan', 'skills', 'sdk-install');
    const globalDocsPath = expandHome('~/.respan/skills/sdk-install');

    if (checkLocal && fs.existsSync(localDocsPath)) {
      const files = fs.readdirSync(localDocsPath);
      this.log(`    ${GREEN}\u2713${RESET} Local: ${DIM}${localDocsPath}${RESET} (${files.length} files)`);
    } else if (checkLocal) {
      this.log(`    ${DIM}\u2013${RESET} No local SDK docs`);
    }
    if (checkGlobal && fs.existsSync(globalDocsPath)) {
      const files = fs.readdirSync(globalDocsPath);
      this.log(`    ${GREEN}\u2713${RESET} Global: ${DIM}${globalDocsPath}${RESET} (${files.length} files)`);
    } else if (checkGlobal) {
      this.log(`    ${DIM}\u2013${RESET} No global SDK docs`);
    }
    this.log('');

    // ── 5. Fetched docs ────────────────────────────────────────────
    this.log(`  ${BOLD}Documentation${RESET}`);
    const localFetchedDocs = path.join(projectRoot, '.respan', 'skills', 'docs');
    const globalFetchedDocs = expandHome('~/.respan/skills/docs');

    if (checkLocal && fs.existsSync(localFetchedDocs)) {
      const files = fs.readdirSync(localFetchedDocs);
      this.log(`    ${GREEN}\u2713${RESET} Local: ${DIM}${localFetchedDocs}${RESET} (${files.length} pages)`);
    }
    if (checkGlobal && fs.existsSync(globalFetchedDocs)) {
      const files = fs.readdirSync(globalFetchedDocs);
      this.log(`    ${GREEN}\u2713${RESET} Global: ${DIM}${globalFetchedDocs}${RESET} (${files.length} pages)`);
    }
    if (checkLocal && !fs.existsSync(localFetchedDocs)) {
      this.log(`    ${DIM}\u2013${RESET} No local docs`);
    }
    if (checkGlobal && !fs.existsSync(globalFetchedDocs)) {
      this.log(`    ${DIM}\u2013${RESET} No global docs`);
    }
    this.log('');
  }
}
