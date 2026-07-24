import { Args, Flags } from '@oclif/core';
import { BaseCommand, getErrorMessage } from '../../lib/base-command.js';
import {
  inspectTrace,
  TraceWaitTimeoutError,
  waitForTrace,
  type TraceRecord,
} from '../../lib/trace-validation.js';

export default class TracesWait extends BaseCommand {
  static description = 'Wait until an exported trace is queryable and has the requested span count';

  static examples = [
    'respan traces wait 0123456789abcdef0123456789abcdef',
    'respan traces wait 0123456789abcdef0123456789abcdef --min-spans 5 --timeout 60 --json',
  ];

  static args = {
    id: Args.string({ description: 'Trace ID', required: true }),
  };

  static flags = {
    ...BaseCommand.baseFlags,
    csv: Flags.boolean({ hidden: true, default: false }),
    timeout: Flags.integer({
      description: 'Maximum time to wait in seconds',
      default: 30,
    }),
    interval: Flags.integer({
      description: 'Polling interval in milliseconds',
      default: 1000,
    }),
    'min-spans': Flags.integer({
      description: 'Minimum span count before the trace is ready',
      default: 1,
    }),
  };

  async run(): Promise<void> {
    const { args, flags } = await this.parse(TracesWait);
    this.globalFlags = flags;

    if (flags.timeout < 1 || flags.interval < 1 || flags['min-spans'] < 1) {
      this.error('--timeout, --interval, and --min-spans must be positive integers.', { exit: 1 });
    }

    try {
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
          ready: (candidate) => inspectTrace(candidate).spanCount >= flags['min-spans'],
        },
      );
      const inspection = inspectTrace(trace);

      if (flags.json) {
        this.log(JSON.stringify({ ready: true, ...inspection }, null, 2));
        return;
      }

      this.log(`Trace ${inspection.traceId || args.id} is ready.`);
      this.log(`  spans: ${inspection.spanCount}`);
      this.log(`  errors: ${inspection.errorCount}`);
      this.log(`  types: ${JSON.stringify(inspection.typeCounts)}`);
      this.log(`  models: ${inspection.models.join(', ') || 'none'}`);
    } catch (error) {
      if (error instanceof TraceWaitTimeoutError && error.lastTrace) {
        const inspection = inspectTrace(error.lastTrace);
        if (flags.json) {
          this.log(JSON.stringify({
            ready: false,
            error: error.message,
            ...inspection,
          }, null, 2));
        }
        this.handleError(new Error(`${error.message} Last observed span count: ${inspection.spanCount}.`));
      }
      if (flags.json) {
        this.log(JSON.stringify({ ready: false, traceId: args.id, error: getErrorMessage(error) }, null, 2));
      }
      this.handleError(error);
    }
  }
}
