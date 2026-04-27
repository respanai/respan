import { Args, Flags } from '@oclif/core';
import { BaseCommand } from '../../lib/base-command.js';

export default class DatasetsCreateSpan extends BaseCommand {
  static description = 'Create a span in a dataset';
  static args = { 'dataset-id': Args.string({ description: 'Dataset ID', required: true }) };
  static flags = {
    ...BaseCommand.baseFlags,
    body: Flags.string({ description: 'Span body as JSON string', required: true }),
  };

  async run(): Promise<void> {
    const { args, flags } = await this.parse(DatasetsCreateSpan);
    this.globalFlags = flags;
    try {
      const client = this.getClient();
      let parsed: unknown;
      try {
        parsed = JSON.parse(flags.body);
      } catch {
        this.error('Invalid JSON for --body');
      }
      const spanData = parsed as Record<string, unknown>;
      const data = await this.spin('Creating dataset span', () =>
        client.datasets.createDatasetLog({
          Authorization: this.getAuthHeader(),
          dataset_id: args['dataset-id'],
          input: spanData.input,
          output: spanData.output,
          ...(spanData.expected_output !== undefined ? { expected_output: spanData.expected_output } : {}),
          ...(spanData.metadata ? { metadata: spanData.metadata as Record<string, unknown> } : {}),
        }),
      );
      this.log(JSON.stringify(data, null, 2));
    } catch (error) {
      this.handleError(error);
    }
  }
}
