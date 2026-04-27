import { Flags } from '@oclif/core';
import { BaseCommand } from '../../lib/base-command.js';

export default class ExperimentsList extends BaseCommand {
  static description = 'List experiments';
  static flags = {
    ...BaseCommand.baseFlags,
    limit: Flags.integer({ description: 'Number of results per page', default: 20 }),
    page: Flags.integer({ description: 'Page number', default: 1 }),
  };

  async run(): Promise<void> {
    const { flags } = await this.parse(ExperimentsList);
    this.globalFlags = flags;
    try {
      const client = this.getClient();
      const data = await this.spin('Fetching experiments', () =>
        client.experiments.listExperiments({ Authorization: this.getAuthHeader() }),
      );
      this.outputResult(data, ['id', 'name', 'dataset_id', 'status', 'created_at']);
    } catch (error) {
      this.handleError(error);
    }
  }
}
