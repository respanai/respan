import { Args } from '@oclif/core';
import { BaseCommand } from '../../lib/base-command.js';

export default class PromptsVersions extends BaseCommand {
  static description = 'List versions of a prompt';
  static args = { 'prompt-id': Args.string({ description: 'Prompt ID', required: true }) };
  static flags = { ...BaseCommand.baseFlags };

  async run(): Promise<void> {
    const { args, flags } = await this.parse(PromptsVersions);
    this.globalFlags = flags;
    try {
      const client = this.getClient();
      const data = await this.spin('Fetching prompt versions', () =>
        client.prompts.listPromptVersions({ Authorization: this.getAuthHeader(), prompt_id: args['prompt-id'] }),
      );
      this.outputResult(data, ['id', 'version', 'is_active', 'created_at']);
    } catch (error) {
      this.handleError(error);
    }
  }
}
