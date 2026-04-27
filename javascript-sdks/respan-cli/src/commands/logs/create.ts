import { Flags } from '@oclif/core';
import { BaseCommand } from '../../lib/base-command.js';

export default class LogsCreate extends BaseCommand {
  static description = 'Create a log span';
  static flags = {
    ...BaseCommand.baseFlags,
    input: Flags.string({ description: 'Input text or JSON', required: true }),
    output: Flags.string({ description: 'Output text or JSON' }),
    model: Flags.string({ description: 'Model name' }),
    metadata: Flags.string({ description: 'Metadata as JSON string' }),
  };

  async run(): Promise<void> {
    const { flags } = await this.parse(LogsCreate);
    this.globalFlags = flags;
    try {
      const client = this.getClient();
      let metadata: Record<string, unknown> | undefined;
      if (flags.metadata) {
        try {
          metadata = JSON.parse(flags.metadata);
        } catch {
          this.error('Invalid JSON for --metadata');
        }
      }
      const data = await this.spin('Creating span', () => client.spans.createSpan({
        Authorization: this.getAuthHeader(),
        prompt: flags.input,
        ...(flags.output ? { completion: flags.output } : {}),
        ...(flags.model ? { model: flags.model } : {}),
        ...(metadata ? { metadata } : {}),
      } as any));
      this.log(JSON.stringify(data, null, 2));
    } catch (error) {
      this.handleError(error);
    }
  }
}
