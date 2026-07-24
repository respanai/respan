import { Flags } from '@oclif/core';
import { BaseCommand } from '../../lib/base-command.js';
import type { AuthConfig } from '../../lib/auth.js';
import { findProjectRoot } from '../../lib/integrate.js';
import {
  formatRuntimeEnvironmentAsShell,
  readRuntimeEnvironmentFile,
  resolveRuntimeEnvironment,
  runtimeEnvironmentJson,
} from '../../lib/runtime-env.js';

export default class Env extends BaseCommand {
  static description = 'Print resolved Respan runtime settings without exposing credentials by default';

  static examples = [
    'respan env',
    'respan env --format json',
    'eval "$(respan env --include-api-key)"',
  ];

  static flags = {
    ...BaseCommand.baseFlags,
    csv: Flags.boolean({ hidden: true, default: false }),
    format: Flags.string({
      description: 'Output format',
      options: ['shell', 'json'],
      default: 'shell',
    }),
    'include-api-key': Flags.boolean({
      description: 'Include the raw API key in output (explicitly exposes a credential)',
      default: false,
    }),
  };

  async run(): Promise<void> {
    const { flags } = await this.parse(Env);
    this.globalFlags = flags;

    let auth: AuthConfig | undefined;
    try {
      auth = this.getAuth(findProjectRoot());
    } catch (error) {
      if (flags['api-key'] || flags['env-file'] || flags.profile) {
        this.handleError(error);
      }
      auth = undefined;
    }

    const runtime = resolveRuntimeEnvironment(auth, {
      ...process.env,
      ...readRuntimeEnvironmentFile(flags['env-file']),
    });
    if (flags['include-api-key'] && !runtime.apiKey) {
      this.error('No API-key credential is available to include.', { exit: 1 });
    }

    if (flags.json || flags.format === 'json') {
      this.log(JSON.stringify(runtimeEnvironmentJson(runtime, flags['include-api-key']), null, 2));
      return;
    }

    this.log(formatRuntimeEnvironmentAsShell(runtime, flags['include-api-key']));
  }
}
