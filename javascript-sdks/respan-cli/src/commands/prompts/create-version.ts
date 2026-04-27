import { Args, Flags } from '@oclif/core';
import { BaseCommand } from '../../lib/base-command.js';

export default class PromptsCreateVersion extends BaseCommand {
  static description = 'Create a new version of a prompt';
  static args = { 'prompt-id': Args.string({ description: 'Prompt ID', required: true }) };
  static flags = {
    ...BaseCommand.baseFlags,
    messages: Flags.string({ description: 'Messages as JSON array string', required: true }),
    model: Flags.string({ description: 'Model name' }),
    temperature: Flags.string({ description: 'Temperature value' }),
    'max-tokens': Flags.integer({ description: 'Max tokens' }),
  };

  async run(): Promise<void> {
    const { args, flags } = await this.parse(PromptsCreateVersion);
    this.globalFlags = flags;
    try {
      const client = this.getClient();
      let messages: unknown;
      try {
        messages = JSON.parse(flags.messages);
      } catch {
        this.error('Invalid JSON for --messages');
      }
      // createVersion requires prompt_id, messages (string[]), and model (required)
      const messageList = Array.isArray(messages)
        ? (messages as Record<string, unknown>[])
        : [{ role: 'user', content: String(messages) }];

      const data = await this.spin('Creating prompt version', () => client.prompts.createPromptVersion({
        Authorization: this.getAuthHeader(),
        prompt_id: args['prompt-id'],
        messages: messageList,
        model: flags.model || 'gpt-4o',
        ...(flags.temperature ? { temperature: parseFloat(flags.temperature) } : {}),
        ...(flags['max-tokens'] !== undefined ? { max_tokens: flags['max-tokens'] } : {}),
      }));
      this.log(JSON.stringify(data, null, 2));
    } catch (error) {
      this.handleError(error);
    }
  }
}
