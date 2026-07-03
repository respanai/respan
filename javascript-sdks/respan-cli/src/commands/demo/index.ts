import { BaseCommand } from '../../lib/base-command.js';
import { getActiveProfile, getCredential } from '../../lib/config.js';
import { GREEN, DIM, BOLD, RESET } from '../../lib/colors.js';
import { SEEDERS } from '../../lib/demo/seeders.js';
import { SeederContext, SeederResult } from '../../lib/demo/types.js';

const PLATFORM_BASE = 'https://platform.respan.ai/platform';

/** Dashboard link for a seeded service, deep-linked/filtered where possible so the
 * user lands directly on what was just created. Traces are pre-filtered to the
 * `demo` workflow so they're isolated from any existing traffic. */
function viewLink(service: string): string | undefined {
  switch (service) {
    case 'prompts':
      return `${PLATFORM_BASE}/prompts`;
    case 'datasets':
      return `${PLATFORM_BASE}/datasets`;
    case 'evaluators':
      return `${PLATFORM_BASE}/evaluators`;
    case 'traces':
      return `${PLATFORM_BASE}/traces?span_workflow_name=demo`;
    default:
      return undefined;
  }
}

export default class Demo extends BaseCommand {
  static description =
    'Seed your account with sample data (prompts and traces) so you have something to explore. Idempotent — safe to re-run.';

  static examples = ['<%= config.bin %> demo'];

  static flags = { ...BaseCommand.baseFlags };

  async run(): Promise<void> {
    const { flags } = await this.parse(Demo);
    this.globalFlags = flags;

    const profile = flags.profile || getActiveProfile();
    if (!getCredential(profile) && !flags['api-key']) {
      this.error(
        'Not authenticated. Run `respan auth login` (or set RESPAN_API_KEY) before running `respan demo`.',
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

      const results: SeederResult[] = [];
      for (const seeder of SEEDERS) {
        const result = await this.spin(`Seeding ${seeder.label}`, () => seeder.run(ctx));
        results.push(result);
      }

      this.printSummary(profile, results);
    } catch (error) {
      this.handleError(error);
    }
  }

  private printSummary(profile: string, results: SeederResult[]): void {
    if (this.globalFlags.json) {
      this.outputResult(results);
      return;
    }

    this.log('');
    this.log(`${GREEN}✓${RESET} ${BOLD}respan demo${RESET} — seeded into account ${DIM}(${profile})${RESET}`);

    for (const result of results) {
      this.log('');
      this.log(`  ${BOLD}${result.label}${RESET}`);
      for (const item of result.items) {
        if (item.status === 'created') {
          this.log(`  ${GREEN}Created${RESET}   ${item.name}  ${DIM}(id: ${item.id})${RESET}`);
        } else {
          this.log(`  ${DIM}Skipped   ${item.name}  (${item.reason})${RESET}`);
        }
      }
      const link = viewLink(result.service);
      if (link) this.log(`  ${DIM}View:${RESET} ${link}`);
    }

    const createdCount = results.reduce(
      (n, r) => n + r.items.filter((i) => i.status === 'created').length,
      0,
    );
    const skippedCount = results.reduce(
      (n, r) => n + r.items.filter((i) => i.status === 'skipped').length,
      0,
    );

    this.log('');
    this.log(`  ${DIM}${createdCount} created, ${skippedCount} skipped${RESET}`);
    this.log('');
  }
}
