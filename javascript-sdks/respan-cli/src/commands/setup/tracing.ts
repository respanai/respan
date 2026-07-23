import { Flags } from '@oclif/core';
import { SetupBaseCommand } from '../../lib/setup-base-command.js';

export default class SetupTracing extends SetupBaseCommand {
  static description = `Set up Respan SDK tracing in this project.

Runs the tracing setup flow directly (skips the "Tracing or Gateway?"
question). Sets up your API key, installs the Respan skill, and optionally
opens your coding agent to instrument the project.`;

  static examples = [
    'respan setup tracing',
    'respan setup tracing --agent claude-code',
    'respan setup tracing --no-instrument',
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
    const { flags } = await this.parse(SetupTracing);
    this.globalFlags = flags;

    await this.runSetup('tracing', {
      agent: flags.agent,
      noInstrument: flags['no-instrument'],
    });
  }
}
