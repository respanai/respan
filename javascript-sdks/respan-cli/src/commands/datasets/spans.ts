import { Args } from '@oclif/core';
import { BaseCommand } from '../../lib/base-command.js';

export default class DatasetsSpans extends BaseCommand {
  static description = 'List spans in a dataset';
  static args = { 'dataset-id': Args.string({ description: 'Dataset ID', required: true }) };
  static flags = { ...BaseCommand.baseFlags };

  async run(): Promise<void> {
    const { args, flags } = await this.parse(DatasetsSpans);
    this.globalFlags = flags;
    try {
      const client = this.getClient();
      const data = await this.spin('Fetching dataset spans', () =>
        client.datasets.listDatasetLogs({ Authorization: this.getAuthHeader(), dataset_id: args['dataset-id'] }),
      );
      this.outputResult(data, ['id', 'input', 'output', 'created_at']);
    } catch (error) {
      this.handleError(error);
    }
  }
}
