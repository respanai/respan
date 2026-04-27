import { Args, Flags } from '@oclif/core';
import { BaseCommand } from '../../lib/base-command.js';

export default class PromptsUpdate extends BaseCommand {
  static description = 'Update a prompt';
  static args = { id: Args.string({ description: 'Prompt ID', required: true }) };
  static flags = {
    ...BaseCommand.baseFlags,
    name: Flags.string({ description: 'Prompt name' }),
    description: Flags.string({ description: 'Prompt description' }),
  };

  async run(): Promise<void> {
    const { args, flags } = await this.parse(PromptsUpdate);
    this.globalFlags = flags;
    try {
      const client = this.getClient();
      const data = await this.spin('Updating prompt', () => client.prompts.updatePrompt({
        Authorization: this.getAuthHeader(),
        prompt_id: args.id,
        ...(flags.name ? { name: flags.name } : {}),
        ...(flags.description ? { description: flags.description } : {}),
      }));
      this.log(JSON.stringify(data, null, 2));
    } catch (error) {
      this.handleError(error);
    }
  }
}
