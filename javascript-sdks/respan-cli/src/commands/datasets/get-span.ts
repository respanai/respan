import { Args } from '@oclif/core';
import { BaseCommand } from '../../lib/base-command.js';

export default class DatasetsGetSpan extends BaseCommand {
  static description = 'Get a specific span from a dataset';
  static args = {
    'dataset-id': Args.string({ description: 'Dataset ID', required: true }),
    'span-id': Args.string({ description: 'Span ID', required: true }),
  };
  static flags = { ...BaseCommand.baseFlags };

  async run(): Promise<void> {
    const { args, flags } = await this.parse(DatasetsGetSpan);
    this.globalFlags = flags;
    try {
      const client = this.getClient();
      const data = await this.spin('Fetching dataset span', () =>
        client.datasets.retrieveDatasetLog({
          Authorization: this.getAuthHeader(),
          dataset_id: args['dataset-id'],
          unique_id: args['span-id'],
        }),
      );
      this.log(JSON.stringify(data, null, 2));
    } catch (error) {
      this.handleError(error);
    }
  }
}
