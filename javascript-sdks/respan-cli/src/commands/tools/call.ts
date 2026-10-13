import { readFileSync } from 'node:fs';
import { Args, Flags } from '@oclif/core';
import { BaseCommand } from '../../lib/base-command.js';
import { withMcpClient } from '../../lib/mcp-client.js';

export default class ToolsCall extends BaseCommand {
  static description = 'Call a Respan MCP tool with its exact tool name and JSON arguments. Writes execute immediately; inspect tools list --json before calling.';
  static args = { name: Args.string({ description: 'MCP tool name, e.g. log_list or trace_get', required: true }) };
  static flags = {
    ...BaseCommand.baseFlags,
    input: Flags.string({ description: 'JSON object of MCP tool arguments', exclusive: ['input-file'] }),
    'input-file': Flags.string({ description: 'File containing a JSON object of MCP tool arguments', exclusive: ['input'] }),
    'mcp-url': Flags.string({ description: 'Custom MCP server URL (defaults to the active Platform or Enterprise server)' }),
  };
  async run(): Promise<void> {
    const { args, flags } = await this.parse(ToolsCall);
    this.globalFlags = flags;
    try {
      const input = JSON.parse(flags['input-file'] ? readFileSync(flags['input-file'], 'utf8') : flags.input ?? '{}');
      if (!input || typeof input !== 'object' || Array.isArray(input)) throw new Error('Tool input must be a JSON object.');
      const result = await withMcpClient(this.getAuth(), flags['mcp-url'], (client) => client.callTool({ name: args.name, arguments: input }));
      this.log(JSON.stringify(result, null, 2));
      if (result.isError) this.exit(1);
    } catch (error) {
      this.handleError(error);
    }
  }
}
