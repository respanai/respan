import { Args } from '@oclif/core';
import { BaseCommand } from '../../lib/base-command.js';

export default class TracesGet extends BaseCommand {
  static description = 'Get a specific trace';
  static args = { id: Args.string({ description: 'Trace ID', required: true }) };
  static flags = { ...BaseCommand.baseFlags };

  async run(): Promise<void> {
    const { args, flags } = await this.parse(TracesGet);
    this.globalFlags = flags;
    try {
      const client = this.getClient();
      const data = await this.spin('Fetching trace', () => client.traces.retrieveTrace({ Authorization: this.getAuthHeader(), trace_unique_id: args.id }));
      this.log(JSON.stringify(data, null, 2));
    } catch (error) {
      this.handleError(error);
    }
  }
}
