/**
 * Shared types for the `respan demo` orchestrator and its per-service seeders.
 *
 * The orchestrator (`commands/demo/*`) is service-agnostic: it knows only the
 * {@link Seeder} contract and aggregates {@link SeederResult}s. Each service
 * (prompts today; traces/datasets/evaluators later) ships its own `seed-*.ts`
 * that implements {@link Seeder}, so adding a service is a drop-in.
 */

import { RespanClient } from '@respan/respan-api';

/** Authed handles every seeder needs to talk to the platform. */
export interface SeederContext {
  client: RespanClient;
  /** `"Bearer <token>"`, as the SDK request objects expect. */
  authHeader: string;
  /**
   * Resolved API base URL (e.g. `https://api.respan.ai`). Needed by seeders that
   * bypass the SDK — the traces seeder POSTs raw OTLP to `/v2/traces`, which the
   * SDK does not expose. Prompt seeding ignores this.
   */
  baseUrl: string;
}

/** A demo resource that already exists in the account (matched by name). */
export interface ExistingResource {
  name: string;
  id: string;
}

/** Outcome for a single resource. */
export interface SeedItemResult {
  name: string;
  status: 'created' | 'skipped' | 'deleted';
  /** Resource ID — the new ID when created, the removed ID when deleted. */
  id?: string;
  /** Why it was skipped (e.g. "already exists" / "not found"). */
  reason?: string;
}

/** Aggregated outcome for one service's seed or clear run. */
export interface SeederResult {
  /** Machine key, e.g. `"prompts"`. */
  service: string;
  /** Human label for summaries, e.g. `"Prompts"`. */
  label: string;
  items: SeedItemResult[];
}

/** A per-service seeder. Owns its fixtures and all create/find/delete logic for
 * one service, so the orchestrator stays service-agnostic. Both `run` and
 * `clear` must be safely re-runnable (idempotent). */
export interface Seeder {
  service: string;
  label: string;
  /** Create any missing demo resources, deploying them live. Skips existing. */
  run(ctx: SeederContext): Promise<SeederResult>;
  /** Delete the demo resources this seeder owns (matched by fixture name). */
  clear(ctx: SeederContext): Promise<SeederResult>;
  /** Find demo resources that currently exist in the account (by fixture name),
   * without modifying anything. Used for dedup and for clear previews. */
  findExisting(ctx: SeederContext): Promise<ExistingResource[]>;
}
