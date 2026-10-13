import { BaseCommand } from './base-command.js';

export type SetupMode = 'tracing' | 'gateway';

interface RunSetupOptions {
  agent?: string;
  noInstrument?: boolean;
}

/** Compatibility entry point. Setup instructions are maintained in the docs. */
export abstract class SetupBaseCommand extends BaseCommand {
  protected async runSetup(mode?: SetupMode, _opts: RunSetupOptions = {}): Promise<void> {
    this.log('respan setup is deprecated. Paste a setup prompt into your coding agent instead.');
    this.log('The agent follows the official guide and uses the CLI to verify the result.');
    this.log('');
    if (mode !== 'gateway') {
      this.log('Help me set up Respan tracing: fetch and follow https://www.respan.ai/docs/documentation/agent-setup.md');
    }
    if (mode !== 'tracing') {
      this.log('Help me set up the Respan gateway: fetch and follow https://www.respan.ai/docs/documentation/agent-setup.md');
    }
  }
}
