import { Flags } from '@oclif/core';
import { SetupBaseCommand } from '../../lib/setup-base-command.js';

export default class SetupGateway extends SetupBaseCommand {
  static description = 'Deprecated gateway setup entry point. Prints the official gateway setup prompt.';

  static examples = [
    'respan setup gateway',
    'respan setup gateway --agent claude-code',
    'respan setup gateway --no-instrument',
  ];

  static flags = {
    ...SetupBaseCommand.baseFlags,
    agent: Flags.string({
      description: 'Deprecated compatibility flag; no agent is launched',
    }),
    'no-instrument': Flags.boolean({
      description: 'Deprecated compatibility flag; setup only prints a prompt',
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
