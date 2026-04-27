import { Args } from '@oclif/core';
import { BaseCommand } from '../../lib/base-command.js';

export default class EvaluatorsGet extends BaseCommand {
  static description = 'Get a specific evaluator';
  static args = { id: Args.string({ description: 'Evaluator ID', required: true }) };
  static flags = { ...BaseCommand.baseFlags };

  async run(): Promise<void> {
    const { args, flags } = await this.parse(EvaluatorsGet);
    this.globalFlags = flags;
    try {
      const client = this.getClient();
      const data = await this.spin('Fetching evaluator', () =>
        client.evaluators.retrieveEvaluator({ Authorization: this.getAuthHeader(), evaluator_id: args.id }),
      );
      this.log(JSON.stringify(data, null, 2));
    } catch (error) {
      this.handleError(error);
    }
  }
}
