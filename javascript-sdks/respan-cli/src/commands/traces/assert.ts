import { Args, Flags } from '@oclif/core';
import { BaseCommand, getErrorMessage } from '../../lib/base-command.js';
import {
  evaluateTrace,
  parseCountExpectations,
  TraceWaitTimeoutError,
  waitForTrace,
  type TraceAssertionResult,
  type TraceExpectations,
  type TraceRecord,
} from '../../lib/trace-validation.js';

function formatFailure(result: TraceAssertionResult): string {
  return result.issues
    .map((issue) => `${issue.field}: expected ${issue.expected}, got ${issue.actual}`)
    .join('; ');
}

export default class TracesAssert extends BaseCommand {
  static description = 'Wait for a trace and assert its span types, names, models, and error count';

  static examples = [
    'respan traces assert TRACE_ID --expect-type workflow --expect-type agent:2 --expect-type tool',
    'respan traces assert TRACE_ID --expect-name handoff.task --expect-model gpt-4o-mini --json',
  ];

  static args = {
    id: Args.string({ description: 'Trace ID', required: true }),
  };

  static flags = {
    ...BaseCommand.baseFlags,
    csv: Flags.boolean({ hidden: true, default: false }),
    timeout: Flags.integer({
      description: 'Maximum time to wait for all assertions in seconds',
      default: 30,
    }),
    interval: Flags.integer({
      description: 'Polling interval in milliseconds',
      default: 1000,
    }),
    'min-spans': Flags.integer({
      description: 'Minimum total span count',
      default: 1,
    }),
    'max-errors': Flags.integer({
      description: 'Maximum allowed error span count',
      default: 0,
    }),
    'expect-type': Flags.string({
      description: 'Required log type as TYPE or TYPE:MIN_COUNT (repeatable)',
      multiple: true,
    }),
    'expect-name': Flags.string({
      description: 'Required span name as NAME or NAME:MIN_COUNT (repeatable)',
      multiple: true,
    }),
    'expect-model': Flags.string({
      description: 'Required model name (repeatable)',
      multiple: true,
    }),
  };

  async run(): Promise<void> {
    const { args, flags } = await this.parse(TracesAssert);
    this.globalFlags = flags;

    if (flags.timeout < 1 || flags.interval < 1 || flags['min-spans'] < 1 || flags['max-errors'] < 0) {
      this.error('--timeout, --interval, and --min-spans must be positive; --max-errors cannot be negative.', { exit: 1 });
    }

    try {
      const expectations: TraceExpectations = {
        minSpans: flags['min-spans'],
        maxErrors: flags['max-errors'],
        types: parseCountExpectations(flags['expect-type']),
        names: parseCountExpectations(flags['expect-name']),
        models: flags['expect-model'] || [],
      };
      const client = this.getClient();
      const trace = await waitForTrace(
        async ({ abortSignal, timeoutMs }) => client.traces.retrieveTrace(
          {
            Authorization: this.getAuthHeader(),
            trace_unique_id: args.id,
          },
          { abortSignal, timeoutInSeconds: timeoutMs / 1000 },
        ) as unknown as TraceRecord,
        {
          timeoutMs: flags.timeout * 1000,
          intervalMs: flags.interval,
          ready: (candidate) => evaluateTrace(candidate, expectations).passed,
        },
      );
      const result = evaluateTrace(trace, expectations);

      if (flags.json) {
        this.log(JSON.stringify(result, null, 2));
        return;
      }

      this.log(`Trace ${result.inspection.traceId || args.id} passed all assertions.`);
      this.log(`  spans: ${result.inspection.spanCount}`);
      this.log(`  errors: ${result.inspection.errorCount}`);
      this.log(`  types: ${JSON.stringify(result.inspection.typeCounts)}`);
      this.log(`  models: ${result.inspection.models.join(', ') || 'none'}`);
    } catch (error) {
      if (error instanceof TraceWaitTimeoutError && error.lastTrace) {
        const expectations: TraceExpectations = {
          minSpans: flags['min-spans'],
          maxErrors: flags['max-errors'],
          types: parseCountExpectations(flags['expect-type']),
          names: parseCountExpectations(flags['expect-name']),
          models: flags['expect-model'] || [],
        };
        const result = evaluateTrace(error.lastTrace, expectations);
        if (flags.json) {
          this.log(JSON.stringify({
            ...result,
            error: 'Trace assertions did not pass before timeout.',
          }, null, 2));
        }
        this.handleError(new Error(`Trace assertions did not pass before timeout: ${formatFailure(result)}`));
      }
      if (flags.json) {
        this.log(JSON.stringify({ passed: false, traceId: args.id, error: getErrorMessage(error) }, null, 2));
      }
      this.handleError(error);
    }
  }
}
