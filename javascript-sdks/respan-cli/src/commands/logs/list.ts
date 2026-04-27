import { Flags } from '@oclif/core';
import { BaseCommand } from '../../lib/base-command.js';
import { parseFilters, FILTER_SYNTAX_HELP, LOG_FIELDS_HELP, FILTER_EXAMPLES } from '../../lib/filters.js';
import { extractPagination, formatPaginationInfo } from '../../lib/pagination.js';

export default class LogsList extends BaseCommand {
  static description = `List and filter LLM request logs (spans).

Supports pagination, sorting, time range, and server-side filtering.

${FILTER_SYNTAX_HELP}

${LOG_FIELDS_HELP}

${FILTER_EXAMPLES}`;

  static flags = {
    ...BaseCommand.baseFlags,
    limit: Flags.integer({ description: 'Number of results per page (max 1000)', default: 50 }),
    page: Flags.integer({ description: 'Page number', default: 1 }),
    'sort-by': Flags.string({ description: 'Sort field (prefix with - for descending, e.g. -cost, -latency)' }),
    'start-time': Flags.string({ description: 'Start time filter (ISO 8601)' }),
    'end-time': Flags.string({ description: 'End time filter (ISO 8601)' }),
    filter: Flags.string({ description: 'Filter in field:operator:value format (repeatable)', multiple: true }),
    'all-envs': Flags.string({ description: 'Include all environments (true/false)' }),
    'is-test': Flags.string({ description: 'Filter by test (true) or production (false) environment' }),
    'include-fields': Flags.string({ description: 'Comma-separated fields to include in response' }),
  };

  async run(): Promise<void> {
    const { flags } = await this.parse(LogsList);
    this.globalFlags = flags;
    try {
      const client = this.getClient();

      let filters: Record<string, unknown> | undefined;
      if (flags.filter && flags.filter.length > 0) {
        filters = parseFilters(flags.filter);
      }

      const oneHourAgo = new Date(Date.now() - 60 * 60 * 1000).toISOString();
      const data = await this.spin('Fetching logs', () => client.spans.listSpans({
        Authorization: this.getAuthHeader(),
        start_time: flags['start-time'] || oneHourAgo,
        end_time: flags['end-time'] || new Date().toISOString(),
        sort_by: flags['sort-by'] || '-id',
        operator: 'AND',
        page_size: flags.limit,
        page: flags.page,
        is_test: flags['is-test'] as any,
        all_envs: flags['all-envs'] as any,
        fetch_filters: 'false' as any,
        include_fields: flags['include-fields'],
        filters: filters as any,
      }));
      this.outputResult(data, ['id', 'model', 'prompt_tokens', 'completion_tokens', 'cost', 'latency', 'timestamp']);
      const pagination = extractPagination(data, flags.page);
      this.log(formatPaginationInfo(pagination));
    } catch (error) {
      this.handleError(error);
    }
  }
}
