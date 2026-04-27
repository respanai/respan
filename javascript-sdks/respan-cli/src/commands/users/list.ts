import { Flags } from '@oclif/core';
import { BaseCommand } from '../../lib/base-command.js';
import { extractPagination, formatPaginationInfo } from '../../lib/pagination.js';

export default class UsersList extends BaseCommand {
  static description = 'List users (customers)';
  static flags = {
    ...BaseCommand.baseFlags,
    limit: Flags.integer({ description: 'Number of results per page', default: 20 }),
    page: Flags.integer({ description: 'Page number', default: 1 }),
    'sort-by': Flags.string({ description: 'Sort field' }),
    environment: Flags.string({ description: 'Environment filter' }),
  };

  async run(): Promise<void> {
    const { flags } = await this.parse(UsersList);
    this.globalFlags = flags;
    try {
      const client = this.getClient();
      const data = await this.spin('Fetching users', () => client.users.listCustomers({
        Authorization: this.getAuthHeader(),
        page_size: flags.limit,
        page: flags.page,
        ...(flags['sort-by'] ? { sort_by: flags['sort-by'] } : {}),
        ...(flags.environment ? { environment: flags.environment as any } : {}),
      }));
      this.outputResult(data, [
        'customer_identifier', 'name', 'email', 'number_of_requests', 'total_cost', 'last_active_timeframe',
      ]);
      const pagination = extractPagination(data, flags.page);
      this.log(formatPaginationInfo(pagination));
    } catch (error) {
      this.handleError(error);
    }
  }
}
