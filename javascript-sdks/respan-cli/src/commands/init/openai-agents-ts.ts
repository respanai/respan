import { Flags } from '@oclif/core';
import { BaseCommand, getErrorMessage } from '../../lib/base-command.js';
import {
  executeInstallCommands,
  findNearestPackageRoot,
  formatInstallCommand,
} from '../../lib/framework-project.js';
import {
  applyOpenAIAgentsRecipePlan,
  createOpenAIAgentsRecipePlan,
  OPENAI_AGENTS_SMOKE_SCRIPT,
} from '../../lib/openai-agents-recipe.js';

export default class InitOpenAIAgentsTs extends BaseCommand {
  static description = 'Install and scaffold the deterministic Respan probe for OpenAI Agents TypeScript';

  static examples = [
    'respan init openai-agents-ts',
    'respan init openai-agents-ts --dry-run --json',
    'respan init openai-agents-ts --local-respan ../respan',
  ];

  static flags = {
    ...BaseCommand.baseFlags,
    'api-key': Flags.string({ hidden: true }),
    'base-url': Flags.string({ hidden: true }),
    'env-file': Flags.string({ hidden: true }),
    profile: Flags.string({ hidden: true }),
    csv: Flags.boolean({ hidden: true, default: false }),
    dir: Flags.string({
      description: 'Project directory (defaults to the nearest package.json)',
    }),
    'local-respan': Flags.string({
      description: 'Use SDK packages from a local Respan repository checkout',
    }),
    'skip-install': Flags.boolean({
      description: 'Write the recipe without installing packages',
      default: false,
    }),
    'dry-run': Flags.boolean({
      description: 'Show planned files and commands without changing the project',
      default: false,
    }),
    force: Flags.boolean({
      description: 'Replace a conflicting smoke file/script and refresh tested package versions',
      default: false,
    }),
  };

  async run(): Promise<void> {
    const { flags } = await this.parse(InitOpenAIAgentsTs);
    this.globalFlags = flags;

    try {
      const projectRoot = findNearestPackageRoot(flags.dir || process.cwd());
      const plan = createOpenAIAgentsRecipePlan({
        projectRoot,
        force: flags.force,
        localRespanRepo: flags['local-respan'],
      });
      const summary = {
        recipe: plan.recipe,
        projectRoot: plan.projectRoot,
        packageManager: plan.packageManager,
        smokeFile: plan.smokeFile,
        smokeFileAction: plan.smokeFileAction,
        packageJsonAction: plan.packageJsonAction,
        script: OPENAI_AGENTS_SMOKE_SCRIPT,
        testedOpenAIAgentsVersion: plan.testedOpenAIAgentsVersion,
        existingOpenAIAgentsSpec: plan.existingOpenAIAgentsSpec,
        runtimePackages: plan.runtimePackages,
        developmentPackages: plan.developmentPackages,
        installCommands: plan.installCommands.map(formatInstallCommand),
        localRespanRepo: plan.localRespanRepo,
        dryRun: flags['dry-run'],
        installSkipped: flags['skip-install'],
      };

      if (!flags['dry-run']) {
        applyOpenAIAgentsRecipePlan(plan);
        if (!flags['skip-install'] && plan.installCommands.length > 0) {
          await this.spin('Installing recipe dependencies', async () => {
            executeInstallCommands(plan.installCommands, projectRoot, flags.json);
          });
        }
      }

      if (flags.json) {
        this.log(JSON.stringify(summary, null, 2));
        return;
      }

      this.log(`${flags['dry-run'] ? 'Planned' : 'Initialized'} OpenAI Agents TypeScript in ${projectRoot}`);
      this.log(`  smoke: ${plan.smokeFileAction} ${plan.smokeFile}`);
      this.log(`  package.json: ${plan.packageJsonAction}`);
      this.log(`  @openai/agents tested: ${plan.testedOpenAIAgentsVersion}`);
      if (plan.existingOpenAIAgentsSpec) {
        this.log(`  @openai/agents detected: ${plan.existingOpenAIAgentsSpec}`);
      }
      for (const command of plan.installCommands) {
        this.log(`  install: ${formatInstallCommand(command)}`);
      }
      if (flags['skip-install']) this.log('  dependencies: skipped');
      if (!flags['dry-run']) this.log(`Run: respan smoke openai-agents-ts --dir ${projectRoot}`);
    } catch (error) {
      if (flags.json) {
        this.log(JSON.stringify({ initialized: false, error: getErrorMessage(error) }, null, 2));
      }
      this.handleError(error);
    }
  }
}
