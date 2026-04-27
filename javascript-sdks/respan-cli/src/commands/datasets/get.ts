import { Args } from '@oclif/core';
import { BaseCommand } from '../../lib/base-command.js';

export default class DatasetsGet extends BaseCommand {
  static description = 'Get a specific dataset';
  static args = { id: Args.string({ description: 'Dataset ID', required: true }) };
  static flags = { ...BaseCommand.baseFlags };

  async run(): Promise<void> {
    const { args, flags } = await this.parse(DatasetsGet);
    this.globalFlags = flags;
    try {
      const client = this.getClient();
      const data = await this.spin('Fetching dataset', () =>
        client.datasets.retrieveDataset({ Authorization: this.getAuthHeader(), dataset_id: args.id }),
      );
      this.log(JSON.stringify(data, null, 2));
    } catch (error) {
      this.handleError(error);
    }
  }
}
