import { Flags } from '@oclif/core';
import { confirm } from '@inquirer/prompts';
import { BaseCommand } from '../../lib/base-command.js';
import { getActiveProfile, getCredential } from '../../lib/config.js';
import { GREEN, DIM, BOLD, RESET } from '../../lib/colors.js';
import { SEEDERS } from '../../lib/demo/seeders.js';
import { SeederContext, SeederResult, ExistingResource } from '../../lib/demo/types.js';

/**
 * Tear down what `respan demo` created. Each seeder owns how it identifies its own
 * demo resources — prompts by exact fixture name, traces by the `demo` workflow —
 * so a resource that was manually renamed/relabeled may not be matched, the same
 * identity contract used when seeding.
 */
export default class DemoClear extends BaseCommand {
  static description =
    'Delete the sample prompts, datasets, and evaluators created by `respan demo`. Demo traces are left in place (each run is timestamped history). Idempotent — safe to re-run.';

  static examples = ['<%= config.bin %> demo clear', '<%= config.bin %> demo clear --yes'];

  static flags = {
    ...BaseCommand.baseFlags,
    yes: Flags.boolean({ char: 'y', description: 'Skip the confirmation prompt', default: false }),
  };

  async run(): Promise<void> {
    const { flags } = await this.parse(DemoClear);
    this.globalFlags = flags;

    const profile = flags.profile || getActiveProfile();
    if (!getCredential(profile) && !flags['api-key']) {
      this.error(
        'Not authenticated. Run `respan auth login` (or set RESPAN_API_KEY) before running `respan demo clear`.',
        { exit: 1 },
      );
    }

    try {
      const client = this.getClient();
      const ctx: SeederContext = {
        client,
        authHeader: this.getAuthHeader(),
        baseUrl: this.getBaseUrl(),
      };

      // Preview what exists before deleting anything.
      const existing: { label: string; matches: ExistingResource[] }[] = [];
      for (const seeder of SEEDERS) {
        const matches = await this.spin(`Finding ${seeder.label}`, () => seeder.findExisting(ctx));
        existing.push({ label: seeder.label, matches });
      }

      const total = existing.reduce((n, e) => n + e.matches.length, 0);
      if (total === 0) {
        this.log('');
        this.log(`Nothing to clear in account ${DIM}(${profile})${RESET}.`);
        this.log('');
        return;
      }

      if (!flags.yes && !flags.json) {
        this.previewDeletions(profile, existing);
        if (!process.stdin.isTTY) {
          this.error('Refusing to delete without confirmation. Re-run with --yes.', { exit: 1 });
        }
        const proceed = await confirm({
          message: `Delete ${total} demo resource(s) from account "${profile}"?`,
          default: false,
        });
        if (!proceed) {
          this.log('Aborted. Nothing was deleted.');
          return;
        }
      }

      const results: SeederResult[] = [];
      for (const seeder of SEEDERS) {
        const result = await this.spin(`Clearing ${seeder.label}`, () => seeder.clear(ctx));
        results.push(result);
      }

      this.printSummary(profile, results);
    } catch (error) {
      this.handleError(error);
    }
  }

  private previewDeletions(
    profile: string,
    existing: { label: string; matches: ExistingResource[] }[],
  ): void {
    this.log('');
    this.log(`${BOLD}respan demo clear${RESET} will delete from account ${DIM}(${profile})${RESET}:`);
    for (const { label, matches } of existing) {
      if (matches.length === 0) continue;
      this.log('');
      this.log(`  ${BOLD}${label}${RESET}`);
      for (const m of matches) {
        this.log(`  ${DIM}- ${m.name}  (id: ${m.id})${RESET}`);
      }
    }
    this.log('');
  }

  private printSummary(profile: string, results: SeederResult[]): void {
    if (this.globalFlags.json) {
      this.outputResult(results);
      return;
    }

    this.log('');
    this.log(`${GREEN}✓${RESET} ${BOLD}respan demo clear${RESET} — account ${DIM}(${profile})${RESET}`);

    for (const result of results) {
      if (result.items.length === 0) continue; // e.g. traces, which are never deleted
      this.log('');
      this.log(`  ${BOLD}${result.label}${RESET}`);
      for (const item of result.items) {
        if (item.status === 'deleted') {
          this.log(`  ${GREEN}Deleted${RESET}   ${item.name}  ${DIM}(id: ${item.id})${RESET}`);
        } else {
          this.log(`  ${DIM}Skipped   ${item.name}  (${item.reason})${RESET}`);
        }
      }
    }

    const deletedCount = results.reduce(
      (n, r) => n + r.items.filter((i) => i.status === 'deleted').length,
      0,
    );
    const skippedCount = results.reduce(
      (n, r) => n + r.items.filter((i) => i.status === 'skipped').length,
      0,
    );

    this.log('');
    this.log(`  ${DIM}${deletedCount} deleted, ${skippedCount} not found${RESET}`);
    this.log('');
  }
}
