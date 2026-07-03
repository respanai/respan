/**
 * The registry of demo seeders, shared by `respan demo` and `respan demo clear`.
 *
 * To add a service later (traces, datasets, evaluators), implement a `Seeder` in
 * its own `seed-*.ts` and append it here — no other change is required.
 */

import { Seeder } from './types.js';
import { promptsSeeder } from './seed-prompts.js';
import { datasetsSeeder } from './seed-datasets.js';
import { evaluatorsSeeder } from './seed-evaluators.js';
import { tracesSeeder } from './seed-traces.js';

export const SEEDERS: Seeder[] = [promptsSeeder, datasetsSeeder, evaluatorsSeeder, tracesSeeder];
