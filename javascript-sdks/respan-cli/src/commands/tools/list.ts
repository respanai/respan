import { Flags } from '@oclif/core';
import { BaseCommand } from '../../lib/base-command.js';
import { listMcpTools, withMcpClient } from '../../lib/mcp-client.js';

export default class ToolsList extends BaseCommand {
  static description = 'List the current Respan MCP tools, their input schemas, and annotations.';
  static flags = {
    ...BaseCommand.baseFlags,
    category: Flags.string({ description: 'Filter by a category returned by the MCP server' }),
    categories: Flags.boolean({ description: 'List available categories and tool counts instead of tools' }),
    'mcp-url': Flags.string({ description: 'Custom MCP server URL (defaults to the active Platform or Enterprise server)' }),
  };
  async run(): Promise<void> {
    const { flags } = await this.parse(ToolsList);
    this.globalFlags = flags;
    try {
      let tools = await withMcpClient(this.getAuth(), flags['mcp-url'], listMcpTools);
      const categoryOf = (tool: typeof tools[number]): string =>
        typeof tool._meta?.['respan/category'] === 'string' ? tool._meta['respan/category'] : 'uncategorized';
      const counts = new Map<string, number>();
      for (const tool of tools) {
        const category = categoryOf(tool);
        counts.set(category, (counts.get(category) ?? 0) + 1);
      }
      if (flags.category) {
        if (!counts.has(flags.category)) throw new Error(`Unknown category '${flags.category}'. Available: ${[...counts.keys()].sort().join(', ')}`);
        tools = tools.filter((tool) => categoryOf(tool) === flags.category);
      }
      if (flags.categories) {
        const rows = [...counts].filter(([category]) => !flags.category || category === flags.category)
          .sort(([a], [b]) => a.localeCompare(b)).map(([category, tool_count]) => ({ category, tool_count }));
        this.outputResult(rows, ['category', 'tool_count']);
      } else {
        this.outputResult(flags.json ? tools : tools.map((tool) => ({ ...tool, category: categoryOf(tool) })), ['category', 'name', 'description']);
      }
    } catch (error) {
      this.handleError(error);
    }
  }
}
