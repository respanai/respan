import { Flags } from '@oclif/core';
import { BaseCommand } from '../../lib/base-command.js';
import { parseFilters, FILTER_SYNTAX_HELP, LOG_FIELDS_HELP, FILTER_EXAMPLES } from '../../lib/filters.js';

export default class LogsSummary extends BaseCommand {
  static description = `Get aggregated summary statistics for log spans in a time range.

Returns total cost, total tokens, request count, and score summaries.

${FILTER_SYNTAX_HELP}

${LOG_FIELDS_HELP}

${FILTER_EXAMPLES}`;

  static flags = {
    ...BaseCommand.baseFlags,
    'start-time': Flags.string({ description: 'Start time (ISO 8601)', required: true }),
    'end-time': Flags.string({ description: 'End time (ISO 8601)', required: true }),
    filter: Flags.string({ description: 'Filter in field:operator:value format (repeatable)', multiple: true }),
    'all-envs': Flags.string({ description: 'Include all environments (true/false)' }),
    'is-test': Flags.string({ description: 'Filter by test (true) or production (false) environment' }),
  };

  async run(): Promise<void> {
    const { flags } = await this.parse(LogsSummary);
    this.globalFlags = flags;
    try {
      const client = this.getClient();

      let filters: Record<string, unknown> | undefined;
      if (flags.filter && flags.filter.length > 0) {
        filters = parseFilters(flags.filter);
      }

      const data = await this.spin('Fetching summary', () =>
        client.spans.getSpansSummary({
          Authorization: this.getAuthHeader(),
          start_time: flags['start-time'],
          end_time: flags['end-time'],
          is_test: flags['is-test'] as any,
          all_envs: flags['all-envs'] as any,
          filters: filters as any,
        } as any),
      );
      this.log(JSON.stringify(data, null, 2));
    } catch (error) {
      this.handleError(error);
    }
  }
}
