import { Args, Flags } from '@oclif/core';
import { BaseCommand } from '../../lib/base-command.js';

export default class EvaluatorsRun extends BaseCommand {
  static description = 'Run an evaluator';
  static args = { id: Args.string({ description: 'Evaluator ID', required: true }) };
  static flags = {
    ...BaseCommand.baseFlags,
    'dataset-id': Flags.string({ description: 'Dataset ID to evaluate against' }),
    'log-ids': Flags.string({ description: 'Comma-separated log/span IDs to evaluate' }),
    params: Flags.string({ description: 'Additional parameters as JSON string' }),
  };

  async run(): Promise<void> {
    const { args, flags } = await this.parse(EvaluatorsRun);
    this.globalFlags = flags;
    try {
      const client = this.getClient();
      let extra: Record<string, unknown> = {};
      if (flags.params) {
        try {
          extra = JSON.parse(flags.params);
        } catch {
          this.error('Invalid JSON for --params');
        }
      }
      const data = await this.spin('Running evaluator', () => client.evaluators.runEvaluator({
        Authorization: this.getAuthHeader(),
        ...extra,
        evaluator_id: args.id,
        ...(flags['dataset-id'] ? { dataset_id: flags['dataset-id'] } : {}),
        ...(flags['log-ids']
          ? { log_ids: flags['log-ids'].split(',').map((s) => s.trim()) }
          : {}),
      }));
      this.log(JSON.stringify(data, null, 2));
    } catch (error) {
      this.handleError(error);
    }
  }
}
