import { Flags } from '@oclif/core';
import { SetupBaseCommand } from '../../lib/setup-base-command.js';

export default class Setup extends SetupBaseCommand {
  static description = 'Deprecated setup entry point. Prints tracing and gateway setup prompts.';

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
      description: 'Deprecated compatibility flag; no agent is launched',
    }),
    'no-instrument': Flags.boolean({
      description: 'Deprecated compatibility flag; setup only prints a prompt',
      default: false,
    }),
  };

  async run(): Promise<void> {
    const { flags } = await this.parse(Setup);
    this.globalFlags = flags;

    // No mode: print both prompts without an interactive question.
    await this.runSetup(undefined, {
      agent: flags.agent,
      noInstrument: flags['no-instrument'],
    });
  }
}
