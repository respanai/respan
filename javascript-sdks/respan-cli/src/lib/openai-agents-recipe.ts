import * as fs from 'node:fs';
import * as path from 'node:path';
import {
  buildInstallCommands,
  detectPackageManager,
  getDependencySpec,
  readPackageManifest,
  type InstallCommand,
  type PackageManager,
  type PackageManifest,
} from './framework-project.js';

export const OPENAI_AGENTS_RECIPE_ID = 'openai-agents-ts';
export const OPENAI_AGENTS_SMOKE_RESULT_PREFIX = 'RESPAN_SMOKE_RESULT=';
export const OPENAI_AGENTS_SMOKE_FILE = '.respan/smoke/openai-agents-ts.mts';
export const OPENAI_AGENTS_SMOKE_SCRIPT = 'respan:smoke:openai-agents';
export const OPENAI_AGENTS_SMOKE_COMMAND = `node --import tsx ${OPENAI_AGENTS_SMOKE_FILE}`;
export const TESTED_OPENAI_AGENTS_VERSION = '0.13.0';

const REGISTRY_RUNTIME_PACKAGES: Record<string, string> = {
  '@openai/agents': `@openai/agents@${TESTED_OPENAI_AGENTS_VERSION}`,
  openai: 'openai@6.46.0',
  zod: 'zod@4.4.3',
  '@respan/respan': '@respan/respan@2.1.0',
  '@respan/instrumentation-openai-agents': '@respan/instrumentation-openai-agents@1.0.8',
};

const REGISTRY_DEVELOPMENT_PACKAGES: Record<string, string> = {
  tsx: 'tsx@4.23.0',
};

const LOCAL_RESPAN_PACKAGES = [
  ['javascript-sdks/respan-sdk', '@respan/respan-sdk'],
  ['javascript-sdks/respan-tracing', '@respan/tracing'],
  ['javascript-sdks/respan', '@respan/respan'],
  [
    'javascript-sdks/instrumentations/respan-instrumentation-openai-agents',
    '@respan/instrumentation-openai-agents',
  ],
] as const;

export interface OpenAIAgentsRecipeOptions {
  projectRoot: string;
  force?: boolean;
  localRespanRepo?: string;
}

export interface OpenAIAgentsRecipePlan {
  recipe: typeof OPENAI_AGENTS_RECIPE_ID;
  projectRoot: string;
  packageManager: PackageManager;
  smokeFile: string;
  smokeFileAction: 'create' | 'update' | 'unchanged';
  packageJsonAction: 'update' | 'unchanged';
  packageJsonIndent: string;
  runtimePackages: string[];
  developmentPackages: string[];
  installCommands: InstallCommand[];
  nextManifest: PackageManifest;
  smokeSource: string;
  testedOpenAIAgentsVersion: string;
  existingOpenAIAgentsSpec?: string;
  localRespanRepo?: string;
}

function isSupportedOpenAIAgentsSpec(spec: string): boolean {
  const normalized = spec.trim().replace(/^workspace:/, '');
  const boundedRange = normalized.match(/^>=\s*0\.(\d+)\.\d+\s+<\s*1(?:\.0\.0)?$/);
  if (boundedRange) return Number(boundedRange[1]) >= 12;

  const boundedZeroMajor = normalized.match(/^[~^]?v?0\.(\d+)(?:\.(?:\d+|x|\*))?$/);
  return Boolean(boundedZeroMajor && Number(boundedZeroMajor[1]) >= 12);
}

function resolveLocalRespanPackages(repoPath: string): string[] {
  const repoRoot = path.resolve(repoPath);
  const packageSpecs: string[] = [];

  for (const [relativePath, expectedName] of LOCAL_RESPAN_PACKAGES) {
    const packageRoot = path.join(repoRoot, relativePath);
    const manifestPath = path.join(packageRoot, 'package.json');
    if (!fs.existsSync(manifestPath)) {
      throw new Error(`Local Respan package is missing: ${manifestPath}`);
    }
    const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8')) as PackageManifest;
    if (manifest.name !== expectedName) {
      throw new Error(`Expected ${expectedName} at ${packageRoot}, found ${manifest.name || 'unnamed package'}.`);
    }
    const mainEntrypoint = typeof manifest.main === 'string'
      ? path.join(packageRoot, manifest.main)
      : undefined;
    if (!mainEntrypoint || !fs.existsSync(mainEntrypoint)) {
      throw new Error(
        `Local Respan package ${expectedName} is not built (missing ${mainEntrypoint || 'main entrypoint'}). `
        + `Build the JavaScript SDK workspaces under ${path.join(repoRoot, 'javascript-sdks')} first.`,
      );
    }
    packageSpecs.push(`file:${packageRoot}`);
  }

  return packageSpecs;
}

function packagesToInstall(
  manifest: PackageManifest,
  packages: Record<string, string>,
  force: boolean,
): string[] {
  return Object.entries(packages)
    .filter(([packageName]) => force || !getDependencySpec(manifest, packageName))
    .map(([, installSpec]) => installSpec);
}

function buildSmokeSource(): string {
  return `import { randomBytes } from "node:crypto";
import OpenAI from "openai";
import {
  Agent,
  Runner,
  setDefaultOpenAIClient,
  setOpenAIAPI,
  setTracingDisabled,
  tool,
} from "@openai/agents";
import { z } from "zod";
import { Respan } from "@respan/respan";
import { OpenAIAgentsInstrumentor } from "@respan/instrumentation-openai-agents";

const apiKey = process.env.RESPAN_API_KEY?.trim();
if (!apiKey) throw new Error("RESPAN_API_KEY is required.");

const model = process.env.RESPAN_MODEL?.trim() || "gpt-4o-mini";
const gatewayBaseUrl =
  process.env.RESPAN_GATEWAY_BASE_URL?.trim() || "https://api.respan.ai/api";
const traceId = randomBytes(16).toString("hex");
const workflowName = "Respan OpenAI Agents TypeScript smoke";

const openai = new OpenAI({ apiKey, baseURL: gatewayBaseUrl });
setDefaultOpenAIClient(openai);
setOpenAIAPI("chat_completions");
setTracingDisabled(false);

const respan = new Respan({
  apiKey,
  baseURL: process.env.RESPAN_API_BASE_URL?.trim(),
  instrumentations: [new OpenAIAgentsInstrumentor()],
});
await respan.initialize();

const additionInput = z.object({
  left: z.number(),
  right: z.number(),
});

const addNumbers = tool({
  name: "add_numbers",
  description: "Add two numbers. Always use this tool for arithmetic.",
  parameters: additionInput,
  execute: async ({ left, right }: z.infer<typeof additionInput>) => String(left + right),
});

const arithmeticAgent = new Agent({
  name: "Arithmetic Specialist",
  instructions:
    "Use add_numbers for every arithmetic request. After the tool returns, reply with only its result.",
  model,
  tools: [addNumbers],
});

const triageAgent = new Agent({
  name: "Triage Agent",
  instructions:
    "Immediately hand off every request to the Arithmetic Specialist. Do not answer directly.",
  model,
  handoffs: [arithmeticAgent],
});

const runner = new Runner({
  workflowName,
  traceId,
  groupId: \`respan-cli-smoke-\${traceId}\`,
  traceMetadata: {
    recipe: "${OPENAI_AGENTS_RECIPE_ID}",
    recipe_tested_sdk_version: "${TESTED_OPENAI_AGENTS_VERSION}",
  },
});

let finalOutput: unknown;
try {
  const result = await runner.run(
    triageAgent,
    "Use the specialist and its tool to add 20 and 22. Return only the result.",
    { maxTurns: 6 },
  );
  finalOutput = result.finalOutput;
} finally {
  await respan.flush();
}

console.log(
  "${OPENAI_AGENTS_SMOKE_RESULT_PREFIX}" +
    JSON.stringify({
      traceId,
      workflowName,
      model,
      finalOutput: String(finalOutput ?? ""),
    }),
);
`;
}

export function createOpenAIAgentsRecipePlan(
  options: OpenAIAgentsRecipeOptions,
): OpenAIAgentsRecipePlan {
  const projectRoot = path.resolve(options.projectRoot);
  const force = Boolean(options.force);
  const rawManifest = fs.readFileSync(path.join(projectRoot, 'package.json'), 'utf8');
  const packageJsonIndent = rawManifest.match(/\n([\t ]+)"/)?.[1] || '  ';
  const manifest = readPackageManifest(projectRoot);
  const existingAgentsSpec = getDependencySpec(manifest, '@openai/agents');

  if (existingAgentsSpec && !isSupportedOpenAIAgentsSpec(existingAgentsSpec) && !force) {
    throw new Error(
      `Existing @openai/agents version "${existingAgentsSpec}" is not provably inside the supported >=0.12.0 <1 range. `
      + 'Re-run with --force to install the tested version.',
    );
  }

  const smokeSource = buildSmokeSource();
  const smokeFile = path.join(projectRoot, OPENAI_AGENTS_SMOKE_FILE);
  const existingSmokeSource = fs.existsSync(smokeFile) ? fs.readFileSync(smokeFile, 'utf8') : undefined;
  if (existingSmokeSource !== undefined && existingSmokeSource !== smokeSource && !force) {
    throw new Error(`${smokeFile} already exists with different content. Re-run with --force to replace it.`);
  }

  const existingScript = manifest.scripts?.[OPENAI_AGENTS_SMOKE_SCRIPT];
  if (existingScript && existingScript !== OPENAI_AGENTS_SMOKE_COMMAND && !force) {
    throw new Error(
      `package.json script ${OPENAI_AGENTS_SMOKE_SCRIPT} already has a different command. `
      + 'Re-run with --force to replace it.',
    );
  }

  const nextManifest: PackageManifest = {
    ...manifest,
    scripts: {
      ...(manifest.scripts || {}),
      [OPENAI_AGENTS_SMOKE_SCRIPT]: OPENAI_AGENTS_SMOKE_COMMAND,
    },
  };

  let runtimePackages: string[];
  if (options.localRespanRepo) {
    const nonRespanPackages = Object.fromEntries(
      Object.entries(REGISTRY_RUNTIME_PACKAGES).filter(([name]) => !name.startsWith('@respan/')),
    );
    runtimePackages = [
      ...packagesToInstall(manifest, nonRespanPackages, force),
      ...resolveLocalRespanPackages(options.localRespanRepo),
    ];
  } else {
    runtimePackages = packagesToInstall(manifest, REGISTRY_RUNTIME_PACKAGES, force);
  }

  const developmentPackages = packagesToInstall(
    manifest,
    REGISTRY_DEVELOPMENT_PACKAGES,
    force,
  );
  const packageManager = detectPackageManager(projectRoot);

  return {
    recipe: OPENAI_AGENTS_RECIPE_ID,
    projectRoot,
    packageManager,
    smokeFile,
    smokeFileAction: existingSmokeSource === smokeSource
      ? 'unchanged'
      : existingSmokeSource === undefined ? 'create' : 'update',
    packageJsonAction: existingScript === OPENAI_AGENTS_SMOKE_COMMAND ? 'unchanged' : 'update',
    packageJsonIndent,
    runtimePackages,
    developmentPackages,
    installCommands: buildInstallCommands(packageManager, runtimePackages, developmentPackages),
    nextManifest,
    smokeSource,
    testedOpenAIAgentsVersion: TESTED_OPENAI_AGENTS_VERSION,
    ...(existingAgentsSpec ? { existingOpenAIAgentsSpec: existingAgentsSpec } : {}),
    ...(options.localRespanRepo ? { localRespanRepo: path.resolve(options.localRespanRepo) } : {}),
  };
}

export function applyOpenAIAgentsRecipePlan(plan: OpenAIAgentsRecipePlan): void {
  if (plan.smokeFileAction !== 'unchanged') {
    fs.mkdirSync(path.dirname(plan.smokeFile), { recursive: true });
    fs.writeFileSync(plan.smokeFile, plan.smokeSource, 'utf8');
  }

  if (plan.packageJsonAction !== 'unchanged') {
    fs.writeFileSync(
      path.join(plan.projectRoot, 'package.json'),
      `${JSON.stringify(plan.nextManifest, null, plan.packageJsonIndent)}\n`,
      'utf8',
    );
  }
}

export interface OpenAIAgentsSmokeResult {
  traceId: string;
  workflowName: string;
  model: string;
  finalOutput: string;
}

export function parseOpenAIAgentsSmokeResult(stdout: string): OpenAIAgentsSmokeResult {
  const marker = stdout
    .split(/\r?\n/)
    .reverse()
    .find((line) => line.startsWith(OPENAI_AGENTS_SMOKE_RESULT_PREFIX));
  if (!marker) {
    throw new Error('Smoke script completed without emitting a trace result marker.');
  }

  const parsed = JSON.parse(marker.slice(OPENAI_AGENTS_SMOKE_RESULT_PREFIX.length)) as Partial<OpenAIAgentsSmokeResult>;
  if (!parsed.traceId || !/^[0-9a-f]{32}$/.test(parsed.traceId)) {
    throw new Error('Smoke script emitted an invalid trace ID.');
  }
  if (!parsed.workflowName || !parsed.model || parsed.finalOutput === undefined) {
    throw new Error('Smoke script emitted an incomplete result.');
  }
  return parsed as OpenAIAgentsSmokeResult;
}
