import { Flags } from '@oclif/core';
import { BaseCommand } from '../../lib/base-command.js';
import { extractPagination, formatPaginationInfo } from '../../lib/pagination.js';

export default class EvaluatorsList extends BaseCommand {
  static description = 'List evaluators';
  static flags = {
    ...BaseCommand.baseFlags,
    limit: Flags.integer({ description: 'Number of results per page', default: 20 }),
    page: Flags.integer({ description: 'Page number', default: 1 }),
  };

  async run(): Promise<void> {
    const { flags } = await this.parse(EvaluatorsList);
    this.globalFlags = flags;
    try {
      const client = this.getClient();
      const data = await this.spin('Fetching evaluators', () =>
        client.evaluators.listEvaluators({ Authorization: this.getAuthHeader() }),
      );
      this.outputResult(data, ['id', 'name', 'type', 'is_active', 'created_at']);
      const pagination = extractPagination(data, flags.page);
      this.log(formatPaginationInfo(pagination));
    } catch (error) {
      this.handleError(error);
    }
  }
}
