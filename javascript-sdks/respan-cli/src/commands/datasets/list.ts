import { Flags } from '@oclif/core';
import { BaseCommand } from '../../lib/base-command.js';

export default class DatasetsList extends BaseCommand {
  static description = 'List datasets';
  static flags = {
    ...BaseCommand.baseFlags,
    limit: Flags.integer({ description: 'Number of results per page', default: 50 }),
    page: Flags.integer({ description: 'Page number', default: 1 }),
  };

  async run(): Promise<void> {
    const { flags } = await this.parse(DatasetsList);
    this.globalFlags = flags;
    try {
      const client = this.getClient();
      const data = await this.spin('Fetching datasets', () =>
        client.datasets.listDatasets({
          Authorization: this.getAuthHeader(),
          page: flags.page,
          page_size: flags.limit,
        }),
      );
      this.outputResult(data, ['id', 'name', 'description', 'created_at']);
    } catch (error) {
      this.handleError(error);
    }
  }
}
