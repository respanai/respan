import { Flags } from '@oclif/core';
import { BaseCommand } from '../../lib/base-command.js';

export default class PromptsList extends BaseCommand {
  static description = 'List prompts';
  static flags = {
    ...BaseCommand.baseFlags,
    limit: Flags.integer({ description: 'Number of results per page', default: 50 }),
  };

  async run(): Promise<void> {
    const { flags } = await this.parse(PromptsList);
    this.globalFlags = flags;
    try {
      const client = this.getClient();
      const data = await this.spin('Fetching prompts', () =>
        client.prompts.listPrompts({ Authorization: this.getAuthHeader(), page_size: flags.limit }),
      );
      this.outputResult(data, ['id', 'name', 'description', 'is_active', 'current_version', 'updated_at']);
    } catch (error) {
      this.handleError(error);
    }
  }
}
