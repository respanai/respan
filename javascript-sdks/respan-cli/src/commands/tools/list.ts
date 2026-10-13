import { Flags } from '@oclif/core';
import { BaseCommand } from '../../lib/base-command.js';
import { listMcpTools, withMcpClient } from '../../lib/mcp-client.js';

export default class ToolsList extends BaseCommand {
  static description = 'List the current Respan MCP tools, their input schemas, and annotations.';
  static flags = {
    ...BaseCommand.baseFlags,
    'mcp-url': Flags.string({ description: 'Custom MCP server URL (defaults to the active Platform or Enterprise server)' }),
  };
  async run(): Promise<void> {
    const { flags } = await this.parse(ToolsList);
    this.globalFlags = flags;
    try {
      const tools = await withMcpClient(this.getAuth(), flags['mcp-url'], listMcpTools);
      this.outputResult(tools, ['name', 'description']);
    } catch (error) {
      this.handleError(error);
    }
  }
}
