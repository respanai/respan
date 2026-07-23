import { Flags } from '@oclif/core';
import { SetupBaseCommand } from '../../lib/setup-base-command.js';

export default class Setup extends SetupBaseCommand {
  static description = `Interactive setup wizard for Respan.

Sets up your API key, installs skills and SDK docs for your preferred
coding agents, and optionally runs a setup agent. Asks whether you want
to set up tracing or gateway routing.

This is the recommended way to get started with Respan.`;

  static examples = [
    'npx @respan/cli setup',
    'respan setup',
    'respan setup tracing',
    'respan setup gateway',
    'respan setup --agent claude-code',
    'respan setup --no-instrument',
  ];

  static flags = {
    ...SetupBaseCommand.baseFlags,
    agent: Flags.string({
      description: 'Agent to configure (claude-code, cursor, codex-cli, gemini-cli, opencode)',
    }),
    'no-instrument': Flags.boolean({
      description: 'Skip opening the agent after setup',
      default: false,
    }),
  };

  async run(): Promise<void> {
    const { flags } = await this.parse(Setup);
    this.globalFlags = flags;

    // No mode → runSetup asks "Tracing or Gateway?"
    await this.runSetup(undefined, {
      agent: flags.agent,
      noInstrument: flags['no-instrument'],
    });
  }
}
