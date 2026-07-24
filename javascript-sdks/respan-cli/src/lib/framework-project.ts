import * as fs from 'node:fs';
import * as path from 'node:path';
import { spawnSync } from 'node:child_process';

export type PackageManager = 'npm' | 'pnpm' | 'yarn' | 'bun';

export interface PackageManifest {
  name?: string;
  packageManager?: string;
  scripts?: Record<string, string>;
  dependencies?: Record<string, string>;
  devDependencies?: Record<string, string>;
  optionalDependencies?: Record<string, string>;
  peerDependencies?: Record<string, string>;
  [key: string]: unknown;
}

export interface InstallCommand {
  command: string;
  args: string[];
  dependencyType: 'runtime' | 'development';
}

const LOCK_FILES: Array<{ file: string; packageManager: PackageManager }> = [
  { file: 'pnpm-lock.yaml', packageManager: 'pnpm' },
  { file: 'yarn.lock', packageManager: 'yarn' },
  { file: 'bun.lock', packageManager: 'bun' },
  { file: 'bun.lockb', packageManager: 'bun' },
  { file: 'package-lock.json', packageManager: 'npm' },
  { file: 'npm-shrinkwrap.json', packageManager: 'npm' },
];

export function findNearestPackageRoot(start = process.cwd()): string {
  let current = path.resolve(start);
  if (!fs.existsSync(current)) {
    throw new Error(`Project path does not exist: ${current}`);
  }
  if (fs.statSync(current).isFile()) {
    current = path.dirname(current);
  }

  while (true) {
    if (fs.existsSync(path.join(current, 'package.json'))) return current;
    const parent = path.dirname(current);
    if (parent === current) {
      throw new Error(`No package.json found from ${path.resolve(start)} upward.`);
    }
    current = parent;
  }
}

export function readPackageManifest(projectRoot: string): PackageManifest {
  const manifestPath = path.join(projectRoot, 'package.json');
  try {
    const parsed = JSON.parse(fs.readFileSync(manifestPath, 'utf8')) as unknown;
    if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
      throw new Error('root value must be a JSON object');
    }
    return parsed as PackageManifest;
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(`Could not read ${manifestPath}: ${message}`);
  }
}

export function getDependencySpec(
  manifest: PackageManifest,
  packageName: string,
): string | undefined {
  return manifest.dependencies?.[packageName]
    || manifest.devDependencies?.[packageName]
    || manifest.optionalDependencies?.[packageName]
    || manifest.peerDependencies?.[packageName];
}

function packageManagerFromManifest(manifest: PackageManifest): PackageManager | undefined {
  const configured = manifest.packageManager?.split('@')[0];
  return configured === 'npm' || configured === 'pnpm' || configured === 'yarn' || configured === 'bun'
    ? configured
    : undefined;
}

export function detectPackageManager(projectRoot: string): PackageManager {
  const configured = packageManagerFromManifest(readPackageManifest(projectRoot));
  if (configured) return configured;

  let current = path.resolve(projectRoot);
  while (true) {
    const lock = LOCK_FILES.find(({ file }) => fs.existsSync(path.join(current, file)));
    if (lock) return lock.packageManager;

    const parent = path.dirname(current);
    if (parent === current || fs.existsSync(path.join(current, '.git'))) break;
    current = parent;
  }

  return 'npm';
}

export function buildInstallCommands(
  packageManager: PackageManager,
  runtimePackages: string[],
  developmentPackages: string[],
): InstallCommand[] {
  const commands: InstallCommand[] = [];

  if (runtimePackages.length > 0) {
    const args = packageManager === 'npm'
      ? ['install', ...runtimePackages]
      : ['add', ...runtimePackages];
    commands.push({ command: packageManager, args, dependencyType: 'runtime' });
  }

  if (developmentPackages.length > 0) {
    const args = packageManager === 'npm'
      ? ['install', '--save-dev', ...developmentPackages]
      : packageManager === 'yarn' || packageManager === 'bun'
        ? ['add', '--dev', ...developmentPackages]
        : ['add', '--save-dev', ...developmentPackages];
    commands.push({ command: packageManager, args, dependencyType: 'development' });
  }

  return commands;
}

export function formatInstallCommand(command: InstallCommand): string {
  return [command.command, ...command.args].join(' ');
}

export function executeInstallCommands(
  commands: InstallCommand[],
  projectRoot: string,
  quiet = false,
): void {
  for (const installCommand of commands) {
    const result = spawnSync(installCommand.command, installCommand.args, {
      cwd: projectRoot,
      encoding: 'utf8',
      stdio: quiet ? 'pipe' : 'inherit',
    });

    if (result.error) {
      throw new Error(`Could not run ${formatInstallCommand(installCommand)}: ${result.error.message}`);
    }
    if (result.status !== 0) {
      const details = quiet ? (result.stderr || result.stdout || '').trim() : '';
      throw new Error(
        `Command failed (${result.status}): ${formatInstallCommand(installCommand)}`
        + (details ? `\n${details}` : ''),
      );
    }
  }
}
