import { Flags } from '@oclif/core';
import { BaseCommand } from '../../lib/base-command.js';

export default class PromptsCreate extends BaseCommand {
  static description = 'Create a new prompt';
  static flags = {
    ...BaseCommand.baseFlags,
    name: Flags.string({ description: 'Prompt name', required: true }),
    description: Flags.string({ description: 'Prompt description' }),
  };

  async run(): Promise<void> {
    const { flags } = await this.parse(PromptsCreate);
    this.globalFlags = flags;
    try {
      const client = this.getClient();
      const data = await this.spin('Creating prompt', () => client.prompts.createPrompt({
        Authorization: this.getAuthHeader(),
        name: flags.name,
        ...(flags.description ? { description: flags.description } : {}),
      }));
      this.log(JSON.stringify(data, null, 2));
    } catch (error) {
      this.handleError(error);
    }
  }
}
