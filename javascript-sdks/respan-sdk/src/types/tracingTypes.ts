import type { RespanSpanNameStyle } from "./spanTypes.js";

export interface TracingDecoratorConfig {
  name: string;
  version?: number;
  associationProperties?: { [name: string]: string };
  traceContent?: boolean;
  inputParameters?: unknown[];
  suppressTracing?: boolean;
}

export interface TracingOptions {
  /**
   * The app name to be used when reporting traces. Optional.
   * Defaults to the package name.
   */
  appName?: string;

  /**
   * The API Key for sending traces data. Optional.
   * Defaults to the RESPAN_API_KEY environment variable.
   */
  apiKey?: string;

  /**
   * The OTLP endpoint for sending traces data. Optional.
   * Defaults to RESPAN_BASE_URL environment variable or https://api.respan.ai/
   */
  baseUrl?: string;

  /**
   * Sends traces and spans without batching, for local development. Optional.
   * Defaults to false.
   */
  disableBatch?: boolean;

  /**
   * Defines default log level for SDK and all instrumentations. Optional.
   * Defaults to error.
   */
  logLevel?: "debug" | "info" | "warn" | "error";

  /**
   * Whether to log prompts, completions and embeddings on traces. Optional.
   * Defaults to true.
   */
  traceContent?: boolean;

  /**
   * Whether to silence the initialization message. Optional.
   * Defaults to false.
   */
  silenceInitializationMessage?: boolean;

  /**
   * Whether to enable tracing. Optional.
   * Defaults to true.
   */
  tracingEnabled?: boolean;

  /**
   * Controls exported span.name formatting.
   * - semantic: use operation-prefixed names like agent.triage or llm.gpt-4o
   * - legacy: preserve instrumentation span names
   * Defaults to semantic. Can also be set with RESPAN_SPAN_NAME_STYLE=legacy.
   */
  spanNameStyle?: RespanSpanNameStyle;

  /**
   * Explicitly specify modules to instrument. Optional.
   * This allows for manual instrumentation of specific modules.
   */
  instrumentModules?: {
    [key: string]: any;
  };
} 