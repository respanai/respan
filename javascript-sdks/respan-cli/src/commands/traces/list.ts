import { Flags } from '@oclif/core';
import { BaseCommand } from '../../lib/base-command.js';
import { parseFilters, FILTER_SYNTAX_HELP, TRACE_FIELDS_HELP, FILTER_EXAMPLES } from '../../lib/filters.js';
import { extractPagination, formatPaginationInfo } from '../../lib/pagination.js';

export default class TracesList extends BaseCommand {
  static description = `List and filter traces.

A trace represents a complete workflow execution containing multiple spans.

${FILTER_SYNTAX_HELP}

${TRACE_FIELDS_HELP}

${FILTER_EXAMPLES}`;

  static flags = {
    ...BaseCommand.baseFlags,
    limit: Flags.integer({ description: 'Number of results per page', default: 10 }),
    page: Flags.integer({ description: 'Page number', default: 1 }),
    'sort-by': Flags.string({ description: 'Sort field (prefix with - for descending)', default: '-timestamp' }),
    'start-time': Flags.string({ description: 'Start time filter (ISO 8601)' }),
    'end-time': Flags.string({ description: 'End time filter (ISO 8601)' }),
    environment: Flags.string({ description: 'Environment filter' }),
    filter: Flags.string({ description: 'Filter in field:operator:value format (repeatable)', multiple: true }),
  };

  async run(): Promise<void> {
    const { flags } = await this.parse(TracesList);
    this.globalFlags = flags;
    try {
      const client = this.getClient();

      let bodyFilters: Record<string, unknown> | undefined;
      if (flags.filter && flags.filter.length > 0) {
        bodyFilters = parseFilters(flags.filter);
      }

      const data = await this.spin('Fetching traces', () =>
        client.traces.listTraces({
          Authorization: this.getAuthHeader(),
          page_size: flags.limit,
          page: flags.page,
          sort_by: flags['sort-by'],
          ...(flags['start-time'] ? { start_time: flags['start-time'] } : {}),
          ...(flags['end-time'] ? { end_time: flags['end-time'] } : {}),
          ...(flags.environment ? { environment: flags.environment } : {}),
          ...(bodyFilters ? { filters: bodyFilters as any } : {}),
        }),
      );
      this.outputResult(data, [
        'trace_unique_id', 'name', 'duration', 'span_count', 'total_cost', 'error_count', 'start_time',
      ]);
      const pagination = extractPagination(data, flags.page);
      this.log(formatPaginationInfo(pagination));
    } catch (error) {
      this.handleError(error);
    }
  }
}
