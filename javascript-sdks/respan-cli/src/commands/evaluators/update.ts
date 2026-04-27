import { Args, Flags } from '@oclif/core';
import { BaseCommand } from '../../lib/base-command.js';

export default class EvaluatorsUpdate extends BaseCommand {
  static description = 'Update an evaluator';
  static args = { id: Args.string({ description: 'Evaluator ID', required: true }) };
  static flags = {
    ...BaseCommand.baseFlags,
    name: Flags.string({ description: 'Evaluator name' }),
    description: Flags.string({ description: 'Evaluator description' }),
    config: Flags.string({ description: 'Evaluator config as JSON string' }),
  };

  async run(): Promise<void> {
    const { args, flags } = await this.parse(EvaluatorsUpdate);
    this.globalFlags = flags;
    try {
      const client = this.getClient();
      let extra: Record<string, unknown> = {};
      if (flags.config) {
        try {
          extra = JSON.parse(flags.config);
        } catch {
          this.error('Invalid JSON for --config');
        }
      }
      const data = await this.spin('Updating evaluator', () => client.evaluators.updateEvaluator({
        Authorization: this.getAuthHeader(),
        ...extra,
        evaluator_id: args.id,
        ...(flags.name ? { name: flags.name } : {}),
        ...(flags.description ? { description: flags.description } : {}),
      } as any));
      this.log(JSON.stringify(data, null, 2));
    } catch (error) {
      this.handleError(error);
    }
  }
}
