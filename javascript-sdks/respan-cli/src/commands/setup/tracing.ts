import { Flags } from '@oclif/core';
import { SetupBaseCommand } from '../../lib/setup-base-command.js';

export default class SetupTracing extends SetupBaseCommand {
  static description = 'Deprecated tracing setup entry point. Prints the official tracing setup prompt.';

  static examples = [
    'respan setup tracing',
    'respan setup tracing --agent claude-code',
    'respan setup tracing --no-instrument',
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
    const { flags } = await this.parse(SetupTracing);
    this.globalFlags = flags;

    await this.runSetup('tracing', {
      agent: flags.agent,
      noInstrument: flags['no-instrument'],
    });
  }
}
