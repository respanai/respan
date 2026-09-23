import { diag } from "@opentelemetry/api";
import {
  InstrumentationBase,
  InstrumentationNodeModuleDefinition,
  type InstrumentationConfig,
} from "@opentelemetry/instrumentation";
import {
  BatchSpanProcessor,
  type SpanExporter,
  type SpanProcessor,
} from "@opentelemetry/sdk-trace-base";
import type { RespanSpanNameStyle } from "@respan/respan-sdk";
import { N8N_ATTRIBUTES } from "./_constants.js";
import { N8nTransformingExporter } from "./_exporter.js";
import { N8nSpanProcessor } from "./_processor.js";

const INSTRUMENTATION_NAME = "@respan/instrumentation-n8n";
const INSTRUMENTATION_VERSION = "0.1.0";
const SUPPORTED_SDK_NODE_RANGE = ">=0.213.0 <0.222.0";
const PATCH_MARK = Symbol.for("respan.instrumentation.n8n.nodeSdkPatch.v1");

interface ResourceLike {
  attributes?: Readonly<Record<string, unknown>>;
}

export interface NodeSdkConfigurationLike {
  resource?: ResourceLike;
  traceExporter?: SpanExporter;
  spanProcessor?: SpanProcessor;
  spanProcessors?: SpanProcessor[];
  [key: string]: unknown;
}

export interface N8nNativeInstrumentationOptions {
  spanNameStyle?: RespanSpanNameStyle | string;
}

export function isN8nNodeSdkConfiguration(
  configuration: NodeSdkConfigurationLike | undefined,
): boolean {
  const attrs = configuration?.resource?.attributes;
  return Boolean(
    attrs &&
      nonEmpty(attrs[N8N_ATTRIBUTES.instanceId]) &&
      nonEmpty(attrs[N8N_ATTRIBUTES.instanceRole]),
  );
}

/**
 * Replace n8n's single exporter path with an ordered translator + batch path.
 * The host NodeSDK remains the only provider and owns lifecycle/shutdown.
 */
export function buildN8nNodeSdkConfiguration(
  configuration: NodeSdkConfigurationLike,
  options: N8nNativeInstrumentationOptions = {},
): NodeSdkConfigurationLike {
  if (!isN8nNodeSdkConfiguration(configuration)) return configuration;

  if (configuration.spanProcessor || configuration.spanProcessors) {
    diag.warn(
      `${INSTRUMENTATION_NAME}: n8n supplied custom span processors; leaving its NodeSDK unchanged`,
    );
    return configuration;
  }

  const exporter = configuration.traceExporter;
  if (!exporter || typeof exporter.export !== "function") {
    diag.warn(
      `${INSTRUMENTATION_NAME}: n8n supplied no compatible trace exporter; leaving its NodeSDK unchanged`,
    );
    return configuration;
  }

  const processor = new N8nSpanProcessor();
  const transformingExporter = new N8nTransformingExporter(exporter, processor, options);
  const batchProcessor = new BatchSpanProcessor(transformingExporter);

  return {
    ...configuration,
    // NodeSDK chooses spanProcessors over traceExporter. Keep the original
    // exporter value intact because NodeSDK 0.213-0.221 constructs (but does
    // not use) its default processor before selecting the explicit array.
    spanProcessors: [processor, batchProcessor],
  };
}

type NodeSdkConstructor = new (configuration?: NodeSdkConfigurationLike) => object;
type SdkNodeModule = { NodeSDK?: NodeSdkConstructor };

interface PatchedConstructor extends NodeSdkConstructor {
  [PATCH_MARK]?: boolean;
}

export class N8nNativeSdkInstrumentation extends InstrumentationBase<InstrumentationConfig> {
  private _options: N8nNativeInstrumentationOptions;
  private readonly _patchedModules = new WeakMap<
    SdkNodeModule,
    { original: NodeSdkConstructor; replacement: NodeSdkConstructor }
  >();

  constructor(options: N8nNativeInstrumentationOptions = {}) {
    super(INSTRUMENTATION_NAME, INSTRUMENTATION_VERSION, { enabled: false });
    this._options = options;
  }

  setOptions(options: N8nNativeInstrumentationOptions): void {
    this._options = options;
  }

  protected init(): InstrumentationNodeModuleDefinition {
    return new InstrumentationNodeModuleDefinition(
      "@opentelemetry/sdk-node",
      [SUPPORTED_SDK_NODE_RANGE],
      (moduleExports: SdkNodeModule) => this._patchNodeSdk(moduleExports),
      (moduleExports: SdkNodeModule) => this._unpatchNodeSdk(moduleExports),
    );
  }

  private _patchNodeSdk(moduleExports: SdkNodeModule): SdkNodeModule {
    const OriginalNodeSDK = moduleExports.NodeSDK as PatchedConstructor | undefined;
    if (!OriginalNodeSDK || OriginalNodeSDK[PATCH_MARK]) return moduleExports;

    const options = this._options;
    const WrappedNodeSDK = class extends OriginalNodeSDK {
      constructor(configuration: NodeSdkConfigurationLike = {}) {
        super(buildN8nNodeSdkConfiguration(configuration, options));
      }
    } as PatchedConstructor;

    Object.defineProperty(WrappedNodeSDK, PATCH_MARK, { value: true });
    try {
      Object.defineProperty(WrappedNodeSDK, "name", {
        value: OriginalNodeSDK.name,
        configurable: true,
      });
    } catch {
      // Class names are cosmetic; do not fail instrumentation over one.
    }

    Object.defineProperty(moduleExports, "NodeSDK", {
      value: WrappedNodeSDK,
      writable: true,
      enumerable: true,
      configurable: true,
    });
    this._patchedModules.set(moduleExports, {
      original: OriginalNodeSDK,
      replacement: WrappedNodeSDK,
    });
    return moduleExports;
  }

  private _unpatchNodeSdk(moduleExports: SdkNodeModule): void {
    const patch = this._patchedModules.get(moduleExports);
    if (!patch) return;
    this._patchedModules.delete(moduleExports);
    if (moduleExports.NodeSDK !== patch.replacement) return;

    Object.defineProperty(moduleExports, "NodeSDK", {
      value: patch.original,
      writable: true,
      enumerable: true,
      configurable: true,
    });
  }
}

function nonEmpty(value: unknown): boolean {
  return value !== undefined && value !== null && String(value).trim() !== "";
}

export const N8N_SUPPORTED_SDK_NODE_RANGE = SUPPORTED_SDK_NODE_RANGE;
