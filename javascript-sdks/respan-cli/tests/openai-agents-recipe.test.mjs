import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';

import {
  applyOpenAIAgentsRecipePlan,
  createOpenAIAgentsRecipePlan,
  OPENAI_AGENTS_SMOKE_COMMAND,
  OPENAI_AGENTS_SMOKE_FILE,
  OPENAI_AGENTS_SMOKE_RESULT_PREFIX,
  OPENAI_AGENTS_SMOKE_SCRIPT,
  parseOpenAIAgentsSmokeResult,
} from '../dist/lib/openai-agents-recipe.js';
import {
  buildInstallCommands,
  findNearestPackageRoot,
} from '../dist/lib/framework-project.js';

function createProject(manifest = {}) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'respan-cli-recipe-'));
  fs.writeFileSync(path.join(root, 'package.json'), JSON.stringify({
    name: 'fixture',
    private: true,
    customField: { preserve: true },
    ...manifest,
  }, null, 2));
  return root;
}

function createLocalRespanRepo() {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'respan-local-'));
  const packages = [
    ['javascript-sdks/respan-sdk', '@respan/respan-sdk'],
    ['javascript-sdks/respan-tracing', '@respan/tracing'],
    ['javascript-sdks/respan', '@respan/respan'],
    ['javascript-sdks/instrumentations/respan-instrumentation-openai-agents', '@respan/instrumentation-openai-agents'],
  ];
  for (const [relativePath, name] of packages) {
    const packageRoot = path.join(root, relativePath);
    fs.mkdirSync(path.join(packageRoot, 'dist'), { recursive: true });
    fs.writeFileSync(path.join(packageRoot, 'package.json'), JSON.stringify({ name, main: 'dist/index.js' }));
    fs.writeFileSync(path.join(packageRoot, 'dist/index.js'), 'export {};\n');
  }
  return root;
}

test('recipe preserves package JSON and writes a deterministic smoke file', (t) => {
  const projectRoot = createProject({
    packageManager: 'npm@11.0.0',
    dependencies: { '@openai/agents': '^0.13.0' },
  });
  t.after(() => fs.rmSync(projectRoot, { recursive: true, force: true }));

  const plan = createOpenAIAgentsRecipePlan({ projectRoot });
  assert.equal(plan.packageManager, 'npm');
  assert.equal(plan.runtimePackages.some((value) => value.startsWith('@openai/agents@')), false);
  assert.match(plan.smokeSource, /setTracingDisabled\(false\)/);
  assert.match(plan.smokeSource, /setOpenAIAPI\("chat_completions"\)/);

  applyOpenAIAgentsRecipePlan(plan);
  const manifest = JSON.parse(fs.readFileSync(path.join(projectRoot, 'package.json'), 'utf8'));
  assert.deepEqual(manifest.customField, { preserve: true });
  assert.equal(manifest.scripts[OPENAI_AGENTS_SMOKE_SCRIPT], OPENAI_AGENTS_SMOKE_COMMAND);
  assert.equal(fs.existsSync(path.join(projectRoot, OPENAI_AGENTS_SMOKE_FILE)), true);
});

test('recipe resolves all Respan packages from a local checkout', (t) => {
  const projectRoot = createProject();
  const localRespanRepo = createLocalRespanRepo();
  t.after(() => fs.rmSync(projectRoot, { recursive: true, force: true }));
  t.after(() => fs.rmSync(localRespanRepo, { recursive: true, force: true }));

  const plan = createOpenAIAgentsRecipePlan({ projectRoot, localRespanRepo });
  const localPackages = plan.runtimePackages.filter((value) => value.startsWith('file:'));
  assert.equal(localPackages.length, 4);
  assert.equal(plan.runtimePackages.some((value) => value.startsWith('@respan/')), false);
});

test('recipe refuses to overwrite a conflicting package script', (t) => {
  const projectRoot = createProject({
    scripts: { [OPENAI_AGENTS_SMOKE_SCRIPT]: 'node custom-smoke.js' },
  });
  t.after(() => fs.rmSync(projectRoot, { recursive: true, force: true }));

  assert.throws(
    () => createOpenAIAgentsRecipePlan({ projectRoot }),
    /already has a different command/,
  );
});

test('recipe rejects unbounded or moving OpenAI Agents versions', (t) => {
  const unboundedRoot = createProject({ dependencies: { '@openai/agents': '>=0.12.0' } });
  const latestRoot = createProject({ dependencies: { '@openai/agents': 'latest' } });
  const prereleaseRoot = createProject({ dependencies: { '@openai/agents': '0.12.0-beta.1' } });
  t.after(() => fs.rmSync(unboundedRoot, { recursive: true, force: true }));
  t.after(() => fs.rmSync(latestRoot, { recursive: true, force: true }));
  t.after(() => fs.rmSync(prereleaseRoot, { recursive: true, force: true }));

  assert.throws(
    () => createOpenAIAgentsRecipePlan({ projectRoot: unboundedRoot }),
    /not provably inside/,
  );
  assert.throws(
    () => createOpenAIAgentsRecipePlan({ projectRoot: latestRoot }),
    /not provably inside/,
  );
  assert.throws(
    () => createOpenAIAgentsRecipePlan({ projectRoot: prereleaseRoot }),
    /not provably inside/,
  );
});

test('recipe accepts an explicitly bounded compatible OpenAI Agents range', (t) => {
  const projectRoot = createProject({ dependencies: { '@openai/agents': '>=0.12.0 <1' } });
  t.after(() => fs.rmSync(projectRoot, { recursive: true, force: true }));
  assert.doesNotThrow(() => createOpenAIAgentsRecipePlan({ projectRoot }));
});

test('local Respan recipes reject packages without build output', (t) => {
  const projectRoot = createProject();
  const localRespanRepo = createLocalRespanRepo();
  const missingEntrypoint = path.join(
    localRespanRepo,
    'javascript-sdks/instrumentations/respan-instrumentation-openai-agents/dist/index.js',
  );
  fs.rmSync(missingEntrypoint);
  t.after(() => fs.rmSync(projectRoot, { recursive: true, force: true }));
  t.after(() => fs.rmSync(localRespanRepo, { recursive: true, force: true }));

  assert.throws(
    () => createOpenAIAgentsRecipePlan({ projectRoot, localRespanRepo }),
    /is not built/,
  );
});

test('smoke result parser validates the trace marker', () => {
  const marker = `${OPENAI_AGENTS_SMOKE_RESULT_PREFIX}${JSON.stringify({
    traceId: '0123456789abcdef0123456789abcdef',
    workflowName: 'smoke',
    model: 'gpt-test',
    finalOutput: '42',
  })}`;
  assert.equal(parseOpenAIAgentsSmokeResult(`debug\n${marker}\n`).finalOutput, '42');
  assert.throws(() => parseOpenAIAgentsSmokeResult('no marker'), /trace result marker/);
});

test('project discovery rejects a missing explicit directory', () => {
  assert.throws(
    () => findNearestPackageRoot(path.join(os.tmpdir(), `respan-path-that-does-not-exist-${process.pid}`)),
    /Project path does not exist/,
  );
});

test('dependency install commands use each package manager supported flags', () => {
  assert.deepEqual(buildInstallCommands('npm', ['runtime-pkg'], ['dev-pkg']), [
    { command: 'npm', args: ['install', 'runtime-pkg'], dependencyType: 'runtime' },
    { command: 'npm', args: ['install', '--save-dev', 'dev-pkg'], dependencyType: 'development' },
  ]);
  assert.deepEqual(buildInstallCommands('pnpm', [], ['dev-pkg']), [
    { command: 'pnpm', args: ['add', '--save-dev', 'dev-pkg'], dependencyType: 'development' },
  ]);
  assert.deepEqual(buildInstallCommands('yarn', [], ['dev-pkg']), [
    { command: 'yarn', args: ['add', '--dev', 'dev-pkg'], dependencyType: 'development' },
  ]);
  assert.deepEqual(buildInstallCommands('bun', [], ['dev-pkg']), [
    { command: 'bun', args: ['add', '--dev', 'dev-pkg'], dependencyType: 'development' },
  ]);
});
