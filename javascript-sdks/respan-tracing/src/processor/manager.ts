import { Context } from "@opentelemetry/api";
import {
  SpanProcessor,
  ReadableSpan,
  SpanExporter,
  BatchSpanProcessor,
  SimpleSpanProcessor,
} from "@opentelemetry/sdk-trace-base";
import {
  RespanSpanAttributes,
  type RespanSpanNameStyle,
} from "@respan/respan-sdk";
import {
  resolveSpanNameStyle,
  SpanNameTransformingExporter,
} from "./spanName.js";

/**
 * Configuration for a processor
 */
export interface ProcessorConfig {
  /** The span exporter to use */
  exporter: SpanExporter;
  /** Name identifier for this processor (used for routing) */
  name: string;
  /** Optional custom filter function for spans */
  filter?: (span: ReadableSpan) => boolean;
  /** Optional priority (higher = processed first) */
  priority?: number;
  /** Send spans immediately without batching. Useful for short-lived scripts. */
  disableBatch?: boolean;
}

export interface MultiProcessorManagerOptions {
  /** Send spans immediately without batching. Useful for short-lived scripts. */
  disableBatch?: boolean;
  /** Exported span.name style: "semantic" (default) or "legacy". */
  spanNameStyle?: RespanSpanNameStyle | string;
}

/**
 * Manager for multiple span processors with routing capabilities.
 * 
 * This allows you to route spans to different destinations based on:
 * - Processor name matching (from decorator `processors` parameter)
 * - Custom filter functions
 * - Priority ordering
 * 
 * @example
 * ```typescript
 * const manager = new MultiProcessorManager();
 * 
 * // Add debug processor (only receives spans with processors="debug")
 * manager.addProcessor({
 *   exporter: new FileExporter("./debug.json"),
 *   name: "debug"
 * });
 * 
 * // Add production processor with custom filter
 * manager.addProcessor({
 *   exporter: new RespanSpanExporter({...}),
 *   name: "production",
 *   filter: (span) => !span.name.includes("test")
 * });
 * ```
 */
export class MultiProcessorManager implements SpanProcessor {
  private processors: Array<{
    processor: SpanProcessor;
    config: ProcessorConfig;
  }> = [];
  private readonly spanNameStyle: RespanSpanNameStyle;

  constructor(private readonly options: MultiProcessorManagerOptions = {}) {
    this.spanNameStyle = resolveSpanNameStyle(options.spanNameStyle);
  }

  /**
   * Add a new processor with routing configuration
   * @param config - Processor configuration
   */
  addProcessor(config: ProcessorConfig): void {
    const exporter = new SpanNameTransformingExporter(
      config.exporter,
      this.spanNameStyle
    );
    const disableBatch = config.disableBatch ?? this.options.disableBatch ?? false;
    const processor = disableBatch
      ? new SimpleSpanProcessor(exporter)
      : new BatchSpanProcessor(exporter);

    // Insert in priority order (higher priority first)
    const priority = config.priority ?? 0;
    const insertIndex = this.processors.findIndex(
      (p) => (p.config.priority ?? 0) < priority
    );
    
    if (insertIndex === -1) {
      this.processors.push({ processor, config });
    } else {
      this.processors.splice(insertIndex, 0, { processor, config });
    }

    console.debug(
      `[Respan] Added processor "${config.name}" with priority ${priority} (${disableBatch ? "simple" : "batch"} mode)`
    );
  }

  /**
   * Get all configured processors
   */
  getProcessors(): ProcessorConfig[] {
    return this.processors.map((p) => p.config);
  }

  /**
   * Check if a span should be sent to a specific processor
   */
  private shouldSendToProcessor(span: ReadableSpan, config: ProcessorConfig): boolean {
    // Check if span has processors attribute
    const spanProcessors = span.attributes[RespanSpanAttributes.RESPAN_PROCESSORS];
    
    if (spanProcessors) {
      // Parse processors attribute (could be string or array)
      let processorsList: string[] = [];
      
      if (typeof spanProcessors === "string") {
        processorsList = [spanProcessors];
      } else if (Array.isArray(spanProcessors)) {
        // Filter out null/undefined values and ensure all are strings
        processorsList = spanProcessors.filter((p): p is string => typeof p === "string");
      }
      
      // Check if this processor's name is in the list
      const matchesName = processorsList.includes(config.name);
      
      // If there's a custom filter, both must match
      if (config.filter) {
        return matchesName && config.filter(span);
      }
      
      return matchesName;
    }
    
    // If no processors attribute, only send if there's a custom filter that matches
    if (config.filter) {
      return config.filter(span);
    }
    
    // If no processors attribute and no custom filter:
    // - Send to "default" processor (backward compatibility)
    // - Don't send to other named processors (they need explicit routing)
    return config.name === "default";
  }

  onStart(span: ReadableSpan, parentContext: Context): void {
    // Forward to all processors (they'll decide whether to process in onEnd)
    for (const { processor } of this.processors) {
      processor.onStart(span as any, parentContext);
    }
  }

  onEnd(span: ReadableSpan): void {
    // Check each processor to see if it should receive this span
    for (const { processor, config } of this.processors) {
      if (this.shouldSendToProcessor(span, config)) {
        console.debug(
          `[Respan] Sending span "${span.name}" to processor "${config.name}"`
        );
        processor.onEnd(span);
      } else {
        console.debug(
          `[Respan] Skipping span "${span.name}" for processor "${config.name}"`
        );
      }
    }
  }

  async shutdown(): Promise<void> {
    await Promise.all(
      this.processors.map(({ processor }) => processor.shutdown())
    );
  }

  async forceFlush(): Promise<void> {
    await Promise.all(
      this.processors.map(({ processor }) => processor.forceFlush())
    );
  }
}


