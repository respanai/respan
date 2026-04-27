import { Args } from '@oclif/core';
import { BaseCommand } from '../../lib/base-command.js';

export default class PromptsGet extends BaseCommand {
  static description = 'Get a specific prompt';
  static args = { id: Args.string({ description: 'Prompt ID', required: true }) };
  static flags = { ...BaseCommand.baseFlags };

  async run(): Promise<void> {
    const { args, flags } = await this.parse(PromptsGet);
    this.globalFlags = flags;
    try {
      const client = this.getClient();
      const data = await this.spin('Fetching prompt', () =>
        client.prompts.retrievePrompt({ Authorization: this.getAuthHeader(), prompt_id: args.id }),
      );
      this.log(JSON.stringify(data, null, 2));
    } catch (error) {
      this.handleError(error);
    }
  }
}
