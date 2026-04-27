import { Args } from '@oclif/core';
import { BaseCommand } from '../../lib/base-command.js';

export default class LogsGet extends BaseCommand {
  static description = 'Get a specific log span';
  static args = { id: Args.string({ description: 'Span ID', required: true }) };
  static flags = { ...BaseCommand.baseFlags };

  async run(): Promise<void> {
    const { args, flags } = await this.parse(LogsGet);
    this.globalFlags = flags;
    try {
      const client = this.getClient();
      const data = await this.spin('Fetching span', () => client.spans.retrieveSpan({ Authorization: this.getAuthHeader(), unique_id: args.id }));
      this.log(JSON.stringify(data, null, 2));
    } catch (error) {
      this.handleError(error);
    }
  }
}
