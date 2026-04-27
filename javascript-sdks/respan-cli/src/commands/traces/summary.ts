import { Flags } from '@oclif/core';
import { BaseCommand } from '../../lib/base-command.js';

export default class TracesSummary extends BaseCommand {
  static description = 'Get a summary of traces for a time range';
  static flags = {
    ...BaseCommand.baseFlags,
    'start-time': Flags.string({ description: 'Start time (ISO 8601)', required: true }),
    'end-time': Flags.string({ description: 'End time (ISO 8601)', required: true }),
  };

  async run(): Promise<void> {
    const { flags } = await this.parse(TracesSummary);
    this.globalFlags = flags;
    try {
      const client = this.getClient();
      const data = await this.spin('Fetching traces summary', () =>
        client.spans.getSpansSummary({
          Authorization: this.getAuthHeader(),
          start_time: flags['start-time'],
          end_time: flags['end-time'],
        } as any),
      );
      this.log(JSON.stringify(data, null, 2));
    } catch (error) {
      this.handleError(error);
    }
  }
}
