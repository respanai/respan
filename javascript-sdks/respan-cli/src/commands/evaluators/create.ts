import { Flags } from '@oclif/core';
import { BaseCommand } from '../../lib/base-command.js';

export default class EvaluatorsCreate extends BaseCommand {
  static description = 'Create a new evaluator';
  static flags = {
    ...BaseCommand.baseFlags,
    name: Flags.string({ description: 'Evaluator name', required: true }),
    type: Flags.string({ description: 'Evaluator type' }),
    description: Flags.string({ description: 'Evaluator description' }),
    config: Flags.string({ description: 'Evaluator config as JSON string' }),
  };

  async run(): Promise<void> {
    const { flags } = await this.parse(EvaluatorsCreate);
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
      const data = await this.spin('Creating evaluator', () => client.evaluators.createEvaluator({
        Authorization: this.getAuthHeader(),
        name: flags.name,
        type: (flags.type ?? 'llm') as any,
        score_value_type: (extra.score_value_type ?? 'numerical') as any,
        ...(flags.description ? { description: flags.description } : {}),
        ...extra,
      } as any));
      this.log(JSON.stringify(data, null, 2));
    } catch (error) {
      this.handleError(error);
    }
  }
}
