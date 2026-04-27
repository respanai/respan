import { Args } from '@oclif/core';
import { resolve } from 'node:path';
import { BaseCommand } from '../../lib/base-command.js';
import { runEval } from '../../lib/eval/runner.js';

export default class EvalRun extends BaseCommand {
  static description = 'Run an eval defined in a JSON file (creates prompt + dataset on the platform)';
  static args = {
    file: Args.string({ description: 'Path to eval JSON file', required: true }),
  };
  static flags = { ...BaseCommand.baseFlags };

  async run(): Promise<void> {
    const { args, flags } = await this.parse(EvalRun);
    this.globalFlags = flags;
    try {
      const client = this.getClient();
      const filePath = resolve(args.file);
      const result = await runEval(filePath, client, this.getAuthHeader(), this.getBaseUrl(), (msg) => this.log(msg));
      this.log('');
      this.log(JSON.stringify(result, null, 2));
    } catch (error) {
      this.handleError(error);
    }
  }
}
