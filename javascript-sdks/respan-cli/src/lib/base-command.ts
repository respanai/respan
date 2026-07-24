import { Command, Flags, Interfaces } from '@oclif/core';
import { RespanClient } from '@respan/respan-api';
import { resolveAuth, AuthConfig } from './auth.js';
import { outputData } from './output.js';
import { createSpinner } from './spinner.js';

export type GlobalFlags = Interfaces.InferredFlags<typeof BaseCommand.baseFlags>;

export function getErrorMessage(error: unknown): string {
  return error instanceof Error ? error.message : 'An unexpected error occurred.';
}

export abstract class BaseCommand extends Command {
  static baseFlags = {
    'api-key': Flags.string({
      description: 'API key (env: RESPAN_API_KEY)',
      env: 'RESPAN_API_KEY',
    }),
    'base-url': Flags.string({
      description: 'Temporary API base URL override for this command',
    }),
    'env-file': Flags.string({
      description: 'Read RESPAN_API_KEY from an explicit .env file',
    }),
    profile: Flags.string({
      description: 'Named profile to use',
    }),
    json: Flags.boolean({
      description: 'Output as JSON',
      default: false,
    }),
    csv: Flags.boolean({
      description: 'Output as CSV',
      default: false,
    }),
    verbose: Flags.boolean({
      char: 'v',
      description: 'Show verbose output',
      default: false,
    }),
  };

  protected globalFlags!: GlobalFlags;

  protected getAuth(projectRoot?: string): AuthConfig {
    return resolveAuth({
      'api-key': this.globalFlags['api-key'],
      'base-url': this.globalFlags['base-url'],
      profile: this.globalFlags.profile,
    }, {
      projectRoot,
      envFile: this.globalFlags['env-file'],
    });
  }

  protected getClient(projectRoot?: string): RespanClient {
    const auth = this.getAuth(projectRoot);
    const token = auth.apiKey || auth.accessToken;
    if (!token) throw new Error('No API key or access token available.');
    return new RespanClient({
      headers: { Authorization: `Bearer ${token}` },
      ...(auth.baseUrl ? { environment: auth.baseUrl } : {}),
    });
  }

  protected getAuthHeader(projectRoot?: string): string {
    const auth = this.getAuth(projectRoot);
    const token = auth.apiKey || auth.accessToken;
    if (!token) throw new Error('No API key or access token available.');
    return `Bearer ${token}`;
  }

  protected getBaseUrl(projectRoot?: string): string {
    return this.getAuth(projectRoot).baseUrl || 'https://api.respan.ai';
  }

  protected getOutputFormat(): 'json' | 'csv' | 'table' {
    if (this.globalFlags.json) return 'json';
    if (this.globalFlags.csv) return 'csv';
    return 'table';
  }

  protected outputResult(data: unknown, columns?: string[]): void {
    this.log(outputData(data, this.getOutputFormat(), columns));
  }

  protected async spin<T>(label: string, fn: () => Promise<T>): Promise<T> {
    const spinner = createSpinner(label);
    spinner.start();
    try {
      const result = await fn();
      spinner.succeed();
      return result;
    } catch (error) {
      spinner.fail();
      throw error;
    }
  }

  protected handleError(error: unknown): never {
    this.error(getErrorMessage(error), { exit: 1 });
  }
}
