import { createRequire } from "node:module";
import { pathToFileURL } from "node:url";
import { RespanTelemetry, propagateAttributes, buildReadableSpan, injectSpan, ensureSpanId } from "@respan/tracing";
import { RespanSpanAttributes, RespanLogType } from "@respan/respan-sdk";
import type { RespanParams, RespanSpanNameStyle } from "@respan/respan-sdk";
import type { ProcessorConfig } from "@respan/tracing";
import type { RespanInstrumentation } from "./_types.js";
import {
  AUTO_INSTRUMENTATION_REGISTRY,
  directLlmGenericTracingNames,
  matchesAutoInstrumentationSelector,
  statusFromEntry,
  type AutoInstrumentationEntry,
  type InstrumentationStatusEntry,
} from "./_auto_instrumentation_registry.js";

export interface RespanOptions {
  apiKey?: string;
  baseURL?: string;
  appName?: string;
  instrumentations?: RespanInstrumentation[];
  disabledInstrumentations?: string[];
  traceContent?: boolean;
  logLevel?: "debug" | "info" | "warn" | "error";
  silenceInitializationMessage?: boolean;
  spanNameStyle?: RespanSpanNameStyle;
}

/**
 * Unified entry point for the Respan SDK.
 *
 * Creates a `RespanTelemetry` instance under the hood and activates
 * any instrumentation plugins passed in `options.instrumentations`.
 *
 * ```typescript
 * import { Respan, OTELInstrumentor } from "@respan/respan";
 * import { AnthropicInstrumentation } from "@traceloop/instrumentation-anthropic";
 *
 * const respan = new Respan({
 *   apiKey: "...",
 *   instrumentations: [new OTELInstrumentor(AnthropicInstrumentation)],
 * });
 * await respan.initialize();
 *
 * await respan.withWorkflow({ name: "my_flow" }, async () => { ... });
 * ```
 */
export class Respan {
  public readonly telemetry: RespanTelemetry;
  private _instrumentations: Map<string, RespanInstrumentation> = new Map();
  private _pendingInstrumentations: RespanInstrumentation[];
  private _hasExplicitInstrumentations: boolean;
  private _disabledInstrumentations: string[];
  private _instrumentationStatus: InstrumentationStatusEntry[] = [];
  private _initialized = false;

  constructor(options: RespanOptions = {}) {
    this._pendingInstrumentations = options.instrumentations ?? [];
    this._hasExplicitInstrumentations = options.instrumentations !== undefined;
    this._disabledInstrumentations = options.disabledInstrumentations ?? [];

    // When explicit instrumentations are provided, disable ALL Traceloop
    // auto-discovery to avoid duplicates (explicit means exclusive).
    // When no instrumentations provided, only disable the ones we replace
    // or don't want (frameworks, vector DBs, OpenAI/Anthropic).
    const allTraceloop = ["openAI", "anthropic", "azureOpenAI", "cohere", "bedrock",
      "googleVertexAI", "googleAIPlatform", "pinecone", "together",
      "langChain", "llamaIndex", "chromaDB", "qdrant"];
    const partialDisabled = [
      ...directLlmGenericTracingNames(), // covered by first-party direct LLM instrumentors
      "langChain",    // framework — would duplicate LLM spans
      "llamaIndex",   // framework — would duplicate LLM spans
      "pinecone",     // vector DB
      "chromaDB",     // vector DB
      "qdrant",       // vector DB
    ];
    const userDisabled = options.disabledInstrumentations ?? [];
    const baseDisabled = this._hasExplicitInstrumentations ? allTraceloop : partialDisabled;
    const disabledInstrumentations = [...new Set([...baseDisabled, ...userDisabled])] as any;

    // Create RespanTelemetry (the OTEL engine)
    this.telemetry = new RespanTelemetry({
      apiKey: options.apiKey,
      baseURL: options.baseURL,
      appName: options.appName,
      traceContent: options.traceContent,
      logLevel: options.logLevel,
      disabledInstrumentations,
      silenceInitializationMessage: options.silenceInitializationMessage,
      spanNameStyle: options.spanNameStyle,
    });
  }

  /**
   * Initialize tracing and activate all instrumentation plugins.
   * Must be called (and awaited) before tracing begins.
   */
  async initialize(): Promise<void> {
    if (this._initialized) return;
    this._initialized = true;

    await this.telemetry.initialize();

    // Activate explicit instrumentation plugins
    // (must happen after telemetry init so TracerProvider exists)
    for (const inst of this._pendingInstrumentations) {
      await this._activate(inst);
    }

    // Auto-discover Respan direct LLM instrumentation packages.
    // Only runs when user did NOT pass instrumentations option at all.
    // Passing `instrumentations: []` explicitly disables auto-discovery.
    if (!this._hasExplicitInstrumentations) {
      await this._autoDiscoverInstrumentations();
    }
    this._pendingInstrumentations = [];
  }

  private async _autoDiscoverInstrumentations(): Promise<void> {
    const discoveries = [...AUTO_INSTRUMENTATION_REGISTRY].sort(
      (a, b) => b.priority - a.priority,
    );

    for (const entry of discoveries) {
      if (entry.category !== "direct-llm") {
        this._recordInstrumentationStatus(
          statusFromEntry(
            entry,
            "disabled",
            entry.autoDisabledReason ?? "not a direct LLM SDK auto-instrumentation",
          ),
        );
        continue;
      }

      if (!entry.enabledByDefault) {
        this._recordInstrumentationStatus(
          statusFromEntry(
            entry,
            "disabled",
            entry.autoDisabledReason ?? "not enabled by default",
          ),
        );
        continue;
      }

      if (this._isDisabled(entry)) {
        this._recordInstrumentationStatus(
          statusFromEntry(entry, "disabled", "disabled by user configuration"),
        );
        continue;
      }

      try {
        const mod = await this._importInstrumentationPackage(entry.instrumentationPackage);
        const InstrumentorClass = mod[entry.instrumentorClass] ?? mod.default;
        if (InstrumentorClass) {
          await this._activate(new InstrumentorClass());
          this._recordInstrumentationStatus(statusFromEntry(entry, "enabled"));
        } else {
          this._recordInstrumentationStatus(
            statusFromEntry(
              entry,
              "failed",
              "instrumentor class " + entry.instrumentorClass + " was not exported",
            ),
          );
        }
      } catch (error) {
        this._recordInstrumentationStatus(
          statusFromEntry(
            entry,
            this._isLikelyMissingPackageError(error) ? "missing" : "failed",
            this._errorMessage(error),
          ),
        );
      }
    }
  }

  // ── Re-exported decorator methods from telemetry ──────────────────────

  public get withWorkflow() {
    return this.telemetry.withWorkflow;
  }

  public get withTask() {
    return this.telemetry.withTask;
  }

  public get withAgent() {
    return this.telemetry.withAgent;
  }

  public get withTool() {
    return this.telemetry.withTool;
  }

  public get withRespanSpanAttributes() {
    return this.telemetry.withRespanSpanAttributes;
  }

  // ── Proxy helpers ─────────────────────────────────────────────────────

  public getClient() {
    return this.telemetry.getClient();
  }

  public getSpanBufferManager() {
    return this.telemetry.getSpanBufferManager();
  }

  public addProcessor(config: ProcessorConfig): void {
    this.telemetry.addProcessor(config);
  }

  public getInstrumentationStatus(): InstrumentationStatusEntry[] {
    return [...this._instrumentationStatus];
  }

  // ── Context propagation ──────────────────────────────────────────────

  /**
   * Run a function within a context that attaches Respan attributes to all
   * spans created within its scope.
   *
   * Attributes are propagated via OpenTelemetry context — safe for concurrent
   * async tasks. Nested calls merge attributes (inner wins). Metadata dicts
   * are merged, not replaced.
   *
   * @param attrs - Attributes to propagate: customer_identifier, customer_email,
   *   customer_name, thread_identifier, custom_identifier, group_identifier,
   *   environment, metadata (dict), prompt (dict with prompt_id + variables).
   * @param fn - The function to execute within the propagation scope.
   * @returns The result of `fn`.
   *
   * @example
   * ```typescript
   * await respan.propagateAttributes(
   *   { customer_identifier: "user_123", thread_identifier: "conv_abc" },
   *   async () => {
   *     const result = await Runner.run(agent, "Hello");
   *     return result;
   *   }
   * );
   *
   * await respan.propagateAttributes(
   *   { prompt: { prompt_id: "abc123", variables: { x: "y" } } },
   *   async () => {
   *     const result = await Runner.run(agent, "Hello");
   *     return result;
   *   }
   * );
   * ```
   */
  propagateAttributes<T>(attrs: Partial<RespanParams>, fn: () => T): T {
    return propagateAttributes(attrs, fn);
  }

  // ── Batch API logging ──────────────────────────────────────────────────

  /**
   * Log OpenAI Batch API results as individual chat completion spans.
   *
   * Trace linking (in priority order):
   * 1. **OTEL context** — when called inside a `withTask` / `withWorkflow`,
   *    auto-links to the active trace and nests completions under the current span.
   * 2. **Explicit `traceId`** — for async batches where results arrive in a
   *    separate process (e.g. 24 hours later).
   * 3. **Auto-generated** — creates a new standalone trace.
   *
   * @param requests - Original batch request dicts (from the input JSONL).
   *   Each must have `custom_id` and `body.messages`.
   * @param results - Batch result dicts (from the output JSONL).
   *   Each must have `custom_id` and `response.body`.
   * @param traceId - Optional explicit trace ID to link results to.
   */
  logBatchResults(
    requests: Array<Record<string, any>>,
    results: Array<Record<string, any>>,
    traceId?: string
  ): void {
    const client = this.getClient();

    // Resolve trace context: OTEL > explicit > auto-generated.
    // OTEL returns all-zero IDs when no active span — treat as absent.
    let otelTraceId = client.getCurrentTraceId();
    let otelSpanId = client.getCurrentSpanId();
    if (otelTraceId && /^0+$/.test(otelTraceId)) otelTraceId = undefined;
    if (otelSpanId && /^0+$/.test(otelSpanId)) otelSpanId = undefined;
    const resolvedTraceId = otelTraceId ?? traceId ?? undefined;

    // Determine the parent for completion spans.
    // With OTEL context: nest under the active span directly.
    // Without: create a synthetic "batch_results" task span.
    const parentSpanId = otelSpanId ?? ensureSpanId();

    // Index original requests by custom_id
    const requestsById = new Map<string, Record<string, any>>();
    for (const r of requests) {
      requestsById.set(r.custom_id, r.body ?? {});
    }

    const completionTimestamps: Date[] = [];

    for (const result of results) {
      const customId = result.custom_id ?? "";
      const response = result.response ?? {};
      const body = response.body ?? {};
      const statusCode = response.status_code ?? 200;

      const original = requestsById.get(customId) ?? {};
      const messages = original.messages ?? [];

      const choices = body.choices ?? [{}];
      const output = choices[0]?.message ?? {};
      const usage = body.usage ?? {};

      // Extract timestamp from OpenAI response (unix epoch -> ISO 8601)
      const created: number | undefined = body.created;
      let endTimeIso: string | undefined;
      if (created) {
        const ts = new Date(created * 1000);
        endTimeIso = ts.toISOString();
        completionTimestamps.push(ts);
      }

      const model = body.model ?? original.model ?? "";

      const span = buildReadableSpan({
        name: `batch:${customId}`,
        traceId: resolvedTraceId,
        parentId: parentSpanId,
        endTimeIso,
        attributes: {
          [RespanSpanAttributes.LLM_REQUEST_TYPE]: RespanLogType.CHAT,
          [RespanSpanAttributes.GEN_AI_REQUEST_MODEL]: model,
          [RespanSpanAttributes.GEN_AI_USAGE_PROMPT_TOKENS]: usage.prompt_tokens ?? 0,
          [RespanSpanAttributes.GEN_AI_USAGE_COMPLETION_TOKENS]: usage.completion_tokens ?? 0,
          "traceloop.entity.input": JSON.stringify(messages),
          "traceloop.entity.output": JSON.stringify(output),
          "traceloop.entity.path": "batch_results",
          "traceloop.span.kind": RespanLogType.TASK,
          [RespanSpanAttributes.RESPAN_LOG_TYPE]: RespanLogType.CHAT,
        },
        statusCode,
      });
      injectSpan(span);
    }

    // Create the grouping "batch_results" task span (when no OTEL context)
    if (!otelSpanId) {
      let earliestIso: string | undefined;
      let latestIso: string | undefined;
      if (completionTimestamps.length > 0) {
        completionTimestamps.sort((a, b) => a.getTime() - b.getTime());
        earliestIso = completionTimestamps[0].toISOString();
        latestIso = completionTimestamps[completionTimestamps.length - 1].toISOString();
      }

      const parentSpan = buildReadableSpan({
        name: "batch_results.task",
        traceId: resolvedTraceId,
        spanId: parentSpanId,
        startTimeIso: earliestIso,
        endTimeIso: latestIso,
        attributes: {
          "traceloop.span.kind": RespanLogType.TASK,
          "traceloop.entity.name": "batch_results",
          "traceloop.entity.path": "",
          [RespanSpanAttributes.RESPAN_LOG_TYPE]: RespanLogType.TASK,
        },
      });
      injectSpan(parentSpan);
    }
  }

  // ── Lifecycle ─────────────────────────────────────────────────────────

  /**
   * Flush the OTEL pipeline.
   */
  async flush(): Promise<void> {
    await this.telemetry.flush();
  }

  /**
   * Deactivate plugins and shut down the OTEL pipeline.
   */
  async shutdown(): Promise<void> {
    // Deactivate plugins first
    for (const [, inst] of this._instrumentations) {
      try {
        await inst.deactivate();
      } catch {
        /* ignore */
      }
    }
    this._instrumentations.clear();

    await this.telemetry.shutdown();
  }

  // ── Private helpers ───────────────────────────────────────────────────

  private async _importInstrumentationPackage(packageName: string): Promise<any> {
    try {
      const hostRequire = createRequire(`${process.cwd()}/package.json`);
      return await import(pathToFileURL(hostRequire.resolve(packageName)).href);
    } catch {
      return await import(packageName);
    }
  }

  private _isDisabled(entry: AutoInstrumentationEntry): boolean {
    return this._disabledInstrumentations.some((disabled) =>
      matchesAutoInstrumentationSelector(entry, disabled),
    );
  }

  private _recordInstrumentationStatus(status: InstrumentationStatusEntry): void {
    const existingIndex = this._instrumentationStatus.findIndex(
      (entry) => entry.id === status.id,
    );
    if (existingIndex >= 0) {
      this._instrumentationStatus[existingIndex] = status;
    } else {
      this._instrumentationStatus.push(status);
    }
  }

  private _isLikelyMissingPackageError(error: unknown): boolean {
    const message = this._errorMessage(error).toLowerCase();
    return (
      message.includes("cannot find package") ||
      message.includes("cannot find module") ||
      message.includes("module not found") ||
      message.includes("not found")
    );
  }

  private _errorMessage(error: unknown): string {
    if (error instanceof Error) return error.message;
    if (typeof error === "string") return error;
    return "unknown error";
  }

  private async _activate(inst: RespanInstrumentation): Promise<void> {
    if (this._instrumentations.has(inst.name)) {
      console.warn(
        `[Respan] Instrumentation "${inst.name}" is already active — skipping.`
      );
      return;
    }
    await inst.activate();
    this._instrumentations.set(inst.name, inst);
  }
}
