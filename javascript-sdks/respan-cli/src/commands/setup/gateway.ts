import { Flags } from '@oclif/core';
import { SetupBaseCommand } from '../../lib/setup-base-command.js';

export default class SetupGateway extends SetupBaseCommand {
  static description = `Set up Respan gateway routing in this project.

Runs the gateway setup flow directly (skips the "Tracing or Gateway?"
question). Sets up your API key, installs the Respan skill, and optionally
opens your coding agent to route the detected framework through the gateway.`;

  static examples = [
    'respan setup gateway',
    'respan setup gateway --agent claude-code',
    'respan setup gateway --no-instrument',
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
    const { flags } = await this.parse(SetupGateway);
    this.globalFlags = flags;

    await this.runSetup('gateway', {
      agent: flags.agent,
      noInstrument: flags['no-instrument'],
    });
  }
}
