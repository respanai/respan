import * as fs from 'node:fs';
import * as path from 'node:path';
import { spawnSync } from 'node:child_process';
import { Flags } from '@oclif/core';
import { BaseCommand, getErrorMessage } from '../../lib/base-command.js';
import {
  detectPackageManager,
  findNearestPackageRoot,
} from '../../lib/framework-project.js';
import {
  createOpenAIAgentsRecipePlan,
  OPENAI_AGENTS_SMOKE_FILE,
  OPENAI_AGENTS_SMOKE_SCRIPT,
  parseOpenAIAgentsSmokeResult,
} from '../../lib/openai-agents-recipe.js';
import {
  readRuntimeEnvironmentFile,
  resolveRuntimeEnvironment,
  RESPAN_PLATFORM_URL,
} from '../../lib/runtime-env.js';
import {
  evaluateTrace,
  TraceWaitTimeoutError,
  waitForTrace,
  type TraceAssertionResult,
  type TraceExpectations,
  type TraceRecord,
} from '../../lib/trace-validation.js';

export default class SmokeOpenAIAgentsTs extends BaseCommand {
  static description = 'Run the OpenAI Agents TypeScript probe and validate its exported Respan trace';

  static examples = [
    'respan smoke openai-agents-ts',
    'respan smoke openai-agents-ts --model gpt-4o-mini --timeout 60 --json',
    'respan smoke openai-agents-ts --no-validate',
  ];

  static flags = {
    ...BaseCommand.baseFlags,
    csv: Flags.boolean({ hidden: true, default: false }),
    dir: Flags.string({
      description: 'Project directory (defaults to the nearest package.json)',
    }),
    model: Flags.string({
      description: 'Gateway model for the probe (env: RESPAN_MODEL)',
      env: 'RESPAN_MODEL',
    }),
    'gateway-base-url': Flags.string({
      description: 'Respan gateway base URL (env: RESPAN_GATEWAY_BASE_URL)',
      env: 'RESPAN_GATEWAY_BASE_URL',
    }),
    timeout: Flags.integer({
      description: 'Maximum time to wait for trace assertions in seconds',
      default: 45,
    }),
    interval: Flags.integer({
      description: 'Trace polling interval in milliseconds',
      default: 1000,
    }),
    'run-timeout': Flags.integer({
      description: 'Maximum probe process runtime in seconds',
      default: 120,
    }),
    'no-validate': Flags.boolean({
      description: 'Run and flush the probe without querying Respan for assertions',
      default: false,
    }),
  };

  async run(): Promise<void> {
    const { flags } = await this.parse(SmokeOpenAIAgentsTs);
    this.globalFlags = flags;

    if (flags.timeout < 1 || flags.interval < 1 || flags['run-timeout'] < 1) {
      this.error('--timeout, --interval, and --run-timeout must be positive integers.', { exit: 1 });
    }

    let expectations: TraceExpectations | undefined;

    try {
      const projectRoot = findNearestPackageRoot(flags.dir || process.cwd());
      const smokeFile = path.join(projectRoot, OPENAI_AGENTS_SMOKE_FILE);
      if (!fs.existsSync(smokeFile)) {
        throw new Error(`Smoke recipe not found at ${smokeFile}. Run respan init openai-agents-ts first.`);
      }
      const recipePlan = createOpenAIAgentsRecipePlan({ projectRoot });
      if (recipePlan.smokeFileAction !== 'unchanged' || recipePlan.packageJsonAction !== 'unchanged') {
        throw new Error(
          'The OpenAI Agents smoke recipe is incomplete. Re-run respan init openai-agents-ts before executing it.',
        );
      }

      const auth = this.getAuth(projectRoot);
      if (!auth.apiKey) {
        throw new Error('The OpenAI Agents gateway smoke test requires a Respan API-key credential.');
      }

      const runtime = resolveRuntimeEnvironment(auth, {
        ...process.env,
        ...readRuntimeEnvironmentFile(flags['env-file']),
        ...(flags.model ? { RESPAN_MODEL: flags.model } : {}),
        ...(flags['gateway-base-url']
          ? { RESPAN_GATEWAY_BASE_URL: flags['gateway-base-url'] }
          : {}),
      });
      const packageManager = detectPackageManager(projectRoot);

      const child = spawnSync(
        packageManager,
        ['run', OPENAI_AGENTS_SMOKE_SCRIPT],
        {
          cwd: projectRoot,
          encoding: 'utf8',
          timeout: flags['run-timeout'] * 1000,
          env: {
            ...process.env,
            RESPAN_API_KEY: auth.apiKey,
            RESPAN_API_BASE_URL: runtime.apiBaseUrl,
            RESPAN_GATEWAY_BASE_URL: runtime.gatewayBaseUrl,
            RESPAN_MODEL: runtime.model,
            OPENAI_AGENTS_DISABLE_TRACING: '0',
          },
        },
      );

      if (child.error) {
        throw new Error(`Smoke process failed to start: ${child.error.message}`);
      }
      if (child.status !== 0) {
        const details = (child.stderr || child.stdout || '').trim();
        throw new Error(`Smoke process exited with status ${child.status}.${details ? `\n${details}` : ''}`);
      }

      const smokeResult = parseOpenAIAgentsSmokeResult(child.stdout || '');
      if (flags.verbose && child.stderr?.trim()) this.warn(child.stderr.trim());

      const smokeExpectations: TraceExpectations = {
        minSpans: 7,
        maxErrors: 0,
        types: {
          workflow: 1,
          agent: 2,
          chat: 2,
          tool: 1,
          task: 1,
        },
        names: { 'handoff.task': 1 },
        models: [smokeResult.model],
      };
      expectations = smokeExpectations;

      let assertion: TraceAssertionResult | undefined;
      if (!flags['no-validate']) {
        const client = this.getClient(projectRoot);
        const trace = await waitForTrace(
          async ({ abortSignal, timeoutMs }) => client.traces.retrieveTrace(
            {
              Authorization: this.getAuthHeader(projectRoot),
              trace_unique_id: smokeResult.traceId,
            },
            { abortSignal, timeoutInSeconds: timeoutMs / 1000 },
          ) as unknown as TraceRecord,
          {
            timeoutMs: flags.timeout * 1000,
            intervalMs: flags.interval,
            ready: (candidate) => evaluateTrace(candidate, smokeExpectations).passed,
          },
        );
        assertion = evaluateTrace(trace, smokeExpectations);
      }

      const output = {
        passed: assertion?.passed ?? true,
        validated: !flags['no-validate'],
        projectRoot,
        traceId: smokeResult.traceId,
        workflowName: smokeResult.workflowName,
        model: smokeResult.model,
        packageManager,
        finalOutput: smokeResult.finalOutput,
        platformUrl: `${RESPAN_PLATFORM_URL}/platform/traces`,
        ...(assertion ? { trace: assertion.inspection } : {}),
      };

      if (flags.json) {
        this.log(JSON.stringify(output, null, 2));
        return;
      }

      this.log(`OpenAI Agents TypeScript smoke ${flags['no-validate'] ? 'completed' : 'passed'}.`);
      this.log(`  trace: ${smokeResult.traceId}`);
      this.log(`  model: ${smokeResult.model}`);
      this.log(`  output: ${smokeResult.finalOutput}`);
      if (assertion) {
        this.log(`  spans: ${assertion.inspection.spanCount}`);
        this.log(`  types: ${JSON.stringify(assertion.inspection.typeCounts)}`);
      }
      this.log(`  view: ${RESPAN_PLATFORM_URL}/platform/traces`);
    } catch (error) {
      if (error instanceof TraceWaitTimeoutError && error.lastTrace && expectations) {
        const lastResult = evaluateTrace(error.lastTrace, expectations);
        const details = lastResult.issues
          .map((issue) => `${issue.field} expected ${issue.expected}, got ${issue.actual}`)
          .join('; ');
        if (flags.json) {
          this.log(JSON.stringify({
            ...lastResult,
            validated: true,
            error: 'Trace assertions did not pass before timeout.',
          }, null, 2));
        }
        this.handleError(new Error(`Smoke trace was exported but incomplete: ${details}`));
      }
      if (flags.json) {
        this.log(JSON.stringify({ passed: false, error: getErrorMessage(error) }, null, 2));
      }
      this.handleError(error);
    }
  }
}
