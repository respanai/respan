/**
 * Respan instrumentation plugin for the Strands Agents TypeScript SDK.
 *
 * Strands already emits OpenTelemetry spans for agents, model calls, tools,
 * graph/swarm orchestration, and node execution. This plugin registers a
 * transformer so those spans use the canonical Respan span contract before
 * filtering and export.
 */

import {
  registerSpanTransformer,
  type RespanSpanTransformer,
  type SpanTransformerRegistration,
} from "@respan/tracing";
import {
  enrichStrandsAgentsSpan,
  StrandsAgentsSpanProcessor,
} from "./_processor.js";
import { STRANDS_SEMCONV_TOOL_DEFINITIONS_OPT_IN } from "./_constants.js";

export interface StrandsAgentsInstrumentorOptions {
  includeToolDefinitions?: boolean;
  traceContent?: boolean;
}

const TRANSFORMER_KEY = "@respan/instrumentation-strands-agents";
interface SharedOwners {
  contentDisabled: Set<object>;
  semconvOwners: Set<object>;
  originalSemconvOptIn?: string;
  enabledSemconvOptIn?: string;
}
const OWNER_STATE = Symbol.for(
  "respan.instrumentation.strands-agents.owners.v1",
);
const globalOwners = globalThis as typeof globalThis & {
  [OWNER_STATE]?: SharedOwners;
};
const owners = (globalOwners[OWNER_STATE] ??= {
  contentDisabled: new Set(),
  semconvOwners: new Set(),
});

export class StrandsAgentsInstrumentor {
  public readonly name = "strands-agents";

  private readonly _includeToolDefinitions: boolean;
  private readonly _traceContent: boolean;
  private _ownsSemconvOptIn = false;
  private _processor: StrandsAgentsSpanProcessor | null = null;
  private _transformer: RespanSpanTransformer | null = null;
  private _registration: SpanTransformerRegistration | null = null;
  private _isInstrumented = false;

  constructor(options: StrandsAgentsInstrumentorOptions = {}) {
    this._includeToolDefinitions = options.includeToolDefinitions ?? true;
    this._traceContent = options.traceContent !== false;
  }

  activate(): void {
    if (this._isInstrumented) {
      return;
    }

    if (!this._traceContent) owners.contentDisabled.add(this);
    this._enableSemconvOptIns();
    try {
      if (!this._processor) {
        this._processor = new StrandsAgentsSpanProcessor({
          traceContent: () => owners.contentDisabled.size === 0,
        });
      }
      const processor = this._processor;
      if (!this._transformer) {
        this._transformer = {
          onStart: (span, parentContext) =>
            processor.onStart(span, parentContext),
          onEnd: (span) => processor.onEnd(span),
          prepareForExport: (span) => processor.prepareForExport(span),
          dispose: () => {
            void processor.shutdown();
          },
        };
      }
      this._registration = registerSpanTransformer(
        TRANSFORMER_KEY,
        this._transformer,
      );
      this._isInstrumented = true;
    } catch (error) {
      owners.contentDisabled.delete(this);
      this._restoreSemconvOptIns();
      throw error;
    }
  }

  deactivate(): void {
    if (!this._isInstrumented) {
      return;
    }

    owners.contentDisabled.delete(this);
    this._registration?.unregister();
    this._registration = null;
    this._restoreSemconvOptIns();
    this._isInstrumented = false;
  }

  isActive(): boolean {
    return this._isInstrumented;
  }

  private _enableSemconvOptIns(): void {
    if (!this._includeToolDefinitions || this._ownsSemconvOptIn) {
      return;
    }
    acquireSemconvOptIn(this);
    this._ownsSemconvOptIn = true;
  }

  private _restoreSemconvOptIns(): void {
    if (!this._ownsSemconvOptIn) {
      return;
    }
    this._ownsSemconvOptIn = false;
    releaseSemconvOptIn(this);
  }
}

function acquireSemconvOptIn(owner: object): void {
  if (owners.semconvOwners.size === 0) {
    owners.originalSemconvOptIn = process.env.OTEL_SEMCONV_STABILITY_OPT_IN;
    const values = new Set(
      (owners.originalSemconvOptIn ?? "")
        .split(",")
        .map((value) => value.trim())
        .filter(Boolean),
    );
    values.add(STRANDS_SEMCONV_TOOL_DEFINITIONS_OPT_IN);
    owners.enabledSemconvOptIn = [...values].sort().join(",");
    process.env.OTEL_SEMCONV_STABILITY_OPT_IN = owners.enabledSemconvOptIn;
  }
  owners.semconvOwners.add(owner);
}

function releaseSemconvOptIn(owner: object): void {
  owners.semconvOwners.delete(owner);
  if (owners.semconvOwners.size) return;
  if (
    process.env.OTEL_SEMCONV_STABILITY_OPT_IN === owners.enabledSemconvOptIn
  ) {
    if (owners.originalSemconvOptIn === undefined)
      delete process.env.OTEL_SEMCONV_STABILITY_OPT_IN;
    else
      process.env.OTEL_SEMCONV_STABILITY_OPT_IN = owners.originalSemconvOptIn;
  }
  owners.originalSemconvOptIn = undefined;
  owners.enabledSemconvOptIn = undefined;
}

export { enrichStrandsAgentsSpan, StrandsAgentsSpanProcessor };
