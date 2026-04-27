import { Args, Flags } from '@oclif/core';
import { BaseCommand } from '../../lib/base-command.js';

export default class DatasetsAddSpans extends BaseCommand {
  static description = 'Add existing spans to a dataset';
  static args = { 'dataset-id': Args.string({ description: 'Dataset ID', required: true }) };
  static flags = {
    ...BaseCommand.baseFlags,
    'span-ids': Flags.string({ description: 'Comma-separated span IDs', required: true }),
  };

  async run(): Promise<void> {
    const { args, flags } = await this.parse(DatasetsAddSpans);
    this.globalFlags = flags;
    try {
      const client = this.getClient();
      const spanIds = flags['span-ids'].split(',').map((s) => s.trim());
      void client; void args; void spanIds;
      this.error('`datasets add-spans` is not supported by the current API SDK. Use `datasets create-span` to add individual logs.');
    } catch (error) {
      this.handleError(error);
    }
  }
}
