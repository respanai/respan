import { NodeSDK } from "@opentelemetry/sdk-node";
import { diag, DiagLogLevel } from "@opentelemetry/api";
import { OTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-http";
import type { ReadableSpan } from "@opentelemetry/sdk-trace-base";
import { Resource } from "@opentelemetry/resources";
import { ATTR_SERVICE_NAME } from "@opentelemetry/semantic-conventions";
import { RespanOptions, ProcessorConfig } from "../types/clientTypes.js";
import { MultiProcessorManager } from "../processor/manager.js";
import { RespanCompositeProcessor } from "../processor/composite.js";
import { 
  getInstrumentations, 
  initInstrumentations, 
  manuallyInitInstrumentations,
  configureTraceContent
} from "../instrumentation/index.js";
import { shouldSendTraces } from "./context.js";

// Global SDK instance (singleton)
let _sdk: NodeSDK | undefined;
let _initialized: boolean = false;
let _compositeProcessor: RespanCompositeProcessor | undefined;
let _flushPromise: Promise<void> | undefined;
let _shutdownPromise: Promise<void> | undefined;
let _beforeExitFlushRegistered = false;

const registerBeforeExitFlush = () => {
  if (_beforeExitFlushRegistered || typeof process === "undefined") {
    return;
  }

  _beforeExitFlushRegistered = true;
  process.once("beforeExit", () => {
    void flush();
  });
};

/**
 * Helper function to resolve and clean up the base URL
 */
export const _resolveBaseURL = (baseURL: string) => {
  const originalUrl = baseURL;

  // Remove trailing slash if it exists
  if (baseURL.endsWith("/")) {
    baseURL = baseURL.slice(0, -1);
  }
  // Remove trailing /api if it exists
  if (baseURL.endsWith("/api")) {
    baseURL = baseURL.slice(0, -4);
  }

  // Debug logging for URL resolution
  if (originalUrl !== baseURL) {
    console.debug(
      `[Respan Debug] URL resolved: "${originalUrl}" -> "${baseURL}"`
    );
  } else {
    console.debug(`[Respan Debug] URL used as-is: "${baseURL}"`);
  }

  return baseURL;
};

/**
 * Initializes the OpenTelemetry SDK with Respan-specific configuration.
 * This sets up the entire tracing pipeline: collection, processing, and export.
 *
 * @param options - Configuration options for the tracing setup
 */
export const startTracing = async (options: RespanOptions) => {
  // Prevent multiple initializations
  if (_initialized) {
    console.log("[Respan] Tracing already initialized, skipping...");
    return;
  }

  const {
    appName = "respan-app",
    apiKey = process.env.RESPAN_API_KEY || "",
    baseURL = process.env.RESPAN_BASE_URL || "https://api.respan.ai",
    logLevel = "error",
    exporter,
    headers = {},
    propagator,
    contextManager,
    silenceInitializationMessage = false,
    tracingEnabled = true,
    spanNameStyle = process.env.RESPAN_SPAN_NAME_STYLE,
    instrumentModules,
    traceContent = true,
    disabledInstrumentations = [],
    resourceAttributes = {},
    spanPostprocessCallback,
    disableBatch = false,
  } = options;

  // Debug logging for configuration
  console.debug("[Respan Debug] Tracing configuration:", {
    appName,
    baseURL,
    logLevel,
    tracingEnabled,
    spanNameStyle,
    traceContent,
    hasApiKey: !!apiKey,
    apiKeyLength: apiKey?.length || 0,
    hasInstrumentModules: !!(
      instrumentModules && Object.keys(instrumentModules).length > 0
    ),
    instrumentModulesKeys: instrumentModules
      ? Object.keys(instrumentModules)
      : [],
    customHeaders: Object.keys(headers),
    disabledInstrumentations: disabledInstrumentations || [],
    disableBatch,
  });

  if (!tracingEnabled) {
    if (!silenceInitializationMessage) {
      console.log("Respan tracing is disabled");
    }
    return;
  }

  // Debug API key validation
  if (!apiKey) {
    console.error(
      "[Respan Debug] WARNING: No API key provided. Traces may be rejected by the server."
    );
  } else if (apiKey.length < 10) {
    console.warn(
      "[Respan Debug] WARNING: API key seems unusually short. Please verify it's correct."
    );
  }

  // Initialize instrumentations with enhanced error logging
  try {
    if (instrumentModules && Object.keys(instrumentModules).length > 0) {
      console.debug(
        "[Respan Debug] Using manual instrumentation for modules:",
        Object.keys(instrumentModules)
      );
      await manuallyInitInstrumentations(
        instrumentModules,
        disabledInstrumentations
      );
    } else {
      console.debug(
        "[Respan Debug] Using automatic instrumentation discovery"
      );
      await initInstrumentations(disabledInstrumentations);
    }
    
    const instrumentationsList = getInstrumentations();
    console.debug(
      `[Respan Debug] Total instrumentations ready for SDK: ${instrumentationsList.length}`
    );
  } catch (error) {
    console.error(
      "[Respan Debug] Error during instrumentation initialization:",
      error
    );
    throw error;
  }

  // Configure trace content for instrumentations
  const shouldCapture = shouldSendTraces() && traceContent;
  configureTraceContent(shouldCapture);
  
  if (!shouldCapture) {
    console.debug(
      "[Respan Debug] Trace content disabled - sensitive data will not be captured"
    );
  } else {
    console.debug(
      "[Respan Debug] Trace content enabled - input/output data will be captured"
    );
  }

  // Only surface errors from OTEL internals — suppress info/debug
  // which includes noisy "exported XX spans" messages from exporters
  diag.setLogger(
    {
      error: (...args) => console.error("[Respan]", ...args),
      warn: () => {},
      info: () => {},
      debug: () => {},
      verbose: () => {},
    },
    DiagLogLevel.ERROR
  );

  // Create resource with custom attributes
  const resource = new Resource({
    [ATTR_SERVICE_NAME]: appName,
    ...resourceAttributes,
  });
  console.debug(
    "[Respan Debug] Created resource with service name:",
    appName,
    "and attributes:",
    resourceAttributes
  );

  // Prepare exporter URL and configuration
  // Use /v2/traces with OTLP JSON — same format as the Python RespanSpanExporter.
  const exporterUrl = `${_resolveBaseURL(baseURL)}/api/v2/traces`;
  const exporterHeaders = {
    Authorization: `Bearer ${apiKey}`,
    ...headers,
  };

  console.debug("[Respan Debug] Exporter configuration:", {
    url: exporterUrl,
    headersCount: Object.keys(exporterHeaders).length,
    hasAuth: exporterHeaders.Authorization ? "Yes" : "No",
    customHeaderKeys: Object.keys(headers),
  });

  // Create exporter with enhanced error handling
  const traceExporter =
    exporter ||
    new OTLPTraceExporter({
      url: exporterUrl,
      headers: exporterHeaders,
    });

  console.debug("[Respan Debug] Created OTLP trace exporter");

  // Initialize multi-processor manager
  const processorManager = new MultiProcessorManager({ disableBatch, spanNameStyle });
  
  // Add default Respan processor to the manager
  // This ensures backward compatibility: spans without a `processors` attribute
  // automatically go to the default Respan exporter
  processorManager.addProcessor({
    exporter: traceExporter,
    name: "default",
    priority: 0,
  });
  
  console.debug("[Respan Debug] Added default processor to multi-processor manager (backward compatibility)");

  // Create composite processor that does filtering + routing
  // Flow: SDK -> CompositeProcessor (filters) -> ProcessorManager (routes) -> Individual Processors
  _compositeProcessor = new RespanCompositeProcessor(
    processorManager,
    spanPostprocessCallback
  );
  
  console.debug("[Respan Debug] Created composite processor - filters spans and routes to multiple processors");

  // Get instrumentations for SDK
  const instrumentationsList = getInstrumentations();

  // Initialize SDK
  console.debug(
    `[Respan Debug] Initializing NodeSDK with ${instrumentationsList.length} successfully loaded instrumentations:`,
    instrumentationsList.map((inst: any) => inst.constructor.name)
  );

  _sdk = new NodeSDK({
    resource,
    // Disable default resource detectors (process.*, host.*, telemetry.sdk.*)
    // to keep span metadata clean. Only service.name is needed.
    resourceDetectors: [],
    spanProcessors: [_compositeProcessor],
    instrumentations: instrumentationsList,
    textMapPropagator: propagator,
    contextManager,
  });

  try {
    console.debug("[Respan Debug] Starting OpenTelemetry SDK...");
    _sdk.start();
    registerBeforeExitFlush();

    // Strip noisy resource attributes that NodeSDK adds automatically.
    // These pollute span metadata on the backend with no user value.
    try {
      const tp = (globalThis as any)[Symbol.for("opentelemetry.js.api.1")]?.trace?.getDelegate?.() ??
        (require("@opentelemetry/api") as any).trace.getTracerProvider();
      const provider = (tp as any)?._delegate ?? tp;
      if (provider?.resource?.attributes) {
        const resAttrs = provider.resource.attributes;
        const noise = Object.keys(resAttrs).filter(k =>
          k.startsWith("telemetry.sdk.") ||
          k.startsWith("process.") ||
          k.startsWith("host.") ||
          k === "os.type" || k === "os.version"
        );
        for (const k of noise) delete resAttrs[k];
      }
    } catch {
      // non-critical — if stripping fails, attributes just stay
    }

    _initialized = true;

    console.debug("[Respan Debug] SDK started successfully");

    if (!silenceInitializationMessage) {
      console.log("Respan tracing initialized successfully");
      console.log(`[Respan Debug] Traces will be sent to: ${exporterUrl}`);
    }
  } catch (error) {
    console.error(
      "[Respan Debug] Failed to start OpenTelemetry SDK:",
      error
    );
    console.error("[Respan Debug] Error details:", {
      message: (error as Error).message,
      stack: (error as Error).stack,
      exporterUrl,
      instrumentationCount: instrumentationsList.length,
    });
    throw error;
  }
};

/**
 * Flush pending spans without shutting down the SDK.
 */
export const flush = async (): Promise<void> => {
  if (!_compositeProcessor) {
    console.debug("[Respan Debug] No SDK to flush");
    return;
  }


  if (!_flushPromise) {
    _flushPromise = (async () => {
      try {
        console.debug("[Respan Debug] Flushing traces...");
        await _compositeProcessor?.forceFlush();
        console.debug("[Respan Debug] Trace flush completed");
      } catch (error) {
        console.error("[Respan Debug] Error during trace flush:", error);
        console.error("[Respan Debug] Flush error details:", {
          message: (error as Error).message,
          stack: (error as Error).stack,
        });
      }
    })().finally(() => {
      _flushPromise = undefined;
    });
  }

  await _flushPromise;
};

/**
 * Backward-compatible alias for flushing pending spans.
 */
export const forceFlush = flush;

/**
 * Flush pending spans and shut down the SDK.
 */
export const shutdownTracing = async (): Promise<void> => {
  if (!_sdk) {
    console.debug("[Respan Debug] No SDK to shutdown");
    return;
  }

  if (!_shutdownPromise) {
    _shutdownPromise = (async () => {
      try {
        console.debug("[Respan Debug] Shutting down SDK...");
        await flush();
        await _sdk?.shutdown();
        console.debug("[Respan Debug] SDK shutdown completed");
      } catch (error) {
        console.error("[Respan Debug] Error during SDK shutdown:", error);
        console.error("[Respan Debug] Shutdown error details:", {
          message: (error as Error).message,
          stack: (error as Error).stack,
        });
      } finally {
        _initialized = false;
        _compositeProcessor = undefined;
        _sdk = undefined;
      }
    })().finally(() => {
      _shutdownPromise = undefined;
    });
  }

  await _shutdownPromise;
};

/**
 * Gets the current SDK instance.
 * Useful for advanced configuration or checking if tracing is initialized.
 *
 * @returns The NodeSDK instance or undefined if not initialized
 */
export const getClient = (): NodeSDK | undefined => {
  return _sdk;
};

/**
 * Add a processor to the SDK for routing spans.
 * This allows routing spans to different destinations based on processor names.
 * 
 * @param config - Processor configuration
 */
/**
 * Inject a ReadableSpan into the OTEL pipeline.
 *
 * This is how plugin-constructed spans enter the pipeline without
 * needing a live tracer context. The span passes through the
 * RespanCompositeProcessor (filtering + routing) → exporter → /v2/traces.
 *
 * Equivalent to Python's ``inject_span()`` in ``span_factory.py``.
 *
 * @returns true if injected, false if SDK not initialized
 */
export const injectSpan = (span: ReadableSpan): boolean => {
  if (!_compositeProcessor) {
    console.warn("[Respan] Cannot inject span — SDK not initialized");
    return false;
  }
  _compositeProcessor.onEnd(span);
  return true;
};

export const addProcessorToSDK = (config: ProcessorConfig): void => {
  if (!_compositeProcessor) {
    console.error("[Respan] Cannot add processor - SDK not initialized");
    return;
  }
  
  const processorManager = _compositeProcessor.getProcessorManager();
  processorManager.addProcessor(config);
  console.log(`[Respan] Added processor: ${config.name}`);
};
