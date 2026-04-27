import { Flags } from '@oclif/core';
import { BaseCommand } from '../../lib/base-command.js';

export default class ExperimentsCreate extends BaseCommand {
  static description = 'Create a new experiment';
  static flags = {
    ...BaseCommand.baseFlags,
    name: Flags.string({ description: 'Experiment name', required: true }),
    'dataset-id': Flags.string({ description: 'Dataset ID', required: true }),
    description: Flags.string({ description: 'Experiment description' }),
    workflows: Flags.string({ description: 'Workflows configuration as JSON string' }),
  };

  async run(): Promise<void> {
    const { flags } = await this.parse(ExperimentsCreate);
    this.globalFlags = flags;
    try {
      const client = this.getClient();
      let workflows: unknown = undefined;
      if (flags.workflows) {
        try {
          workflows = JSON.parse(flags.workflows);
        } catch {
          this.error('Invalid JSON for --workflows');
        }
      }
      const data = await this.spin('Creating experiment', () => client.experiments.createExperiment({
        Authorization: this.getAuthHeader(),
        name: flags.name,
        dataset_id: flags['dataset-id'],
        ...(flags.description ? { description: flags.description } : {}),
        ...(workflows ? { workflows: workflows as any } : {}),
      }));
      this.log(JSON.stringify(data, null, 2));
    } catch (error) {
      this.handleError(error);
    }
  }
}
