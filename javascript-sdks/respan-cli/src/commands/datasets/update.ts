import { Args, Flags } from '@oclif/core';
import { BaseCommand } from '../../lib/base-command.js';

export default class DatasetsUpdate extends BaseCommand {
  static description = 'Update a dataset';
  static args = { id: Args.string({ description: 'Dataset ID', required: true }) };
  static flags = {
    ...BaseCommand.baseFlags,
    name: Flags.string({ description: 'Dataset name' }),
    description: Flags.string({ description: 'Dataset description' }),
  };

  async run(): Promise<void> {
    const { args, flags } = await this.parse(DatasetsUpdate);
    this.globalFlags = flags;
    try {
      const client = this.getClient();
      const data = await this.spin('Updating dataset', () => client.datasets.updateDataset({
        Authorization: this.getAuthHeader(),
        dataset_id: args.id,
        ...(flags.name ? { name: flags.name } : {}),
        ...(flags.description ? { description: flags.description } : {}),
      }));
      this.log(JSON.stringify(data, null, 2));
    } catch (error) {
      this.handleError(error);
    }
  }
}
