import { withTask, withWorkflow, withAgent, withTool } from "./decorators/index.js";
import { WithFunctionType } from "./types/decoratorTypes.js";
import { RespanOptions, ProcessorConfig } from "./types/clientTypes.js";
import { withRespanSpanAttributes } from "./contexts/span.js";
import { startTracing, flush, shutdownTracing, addProcessorToSDK, injectSpan } from "./utils/tracing.js";
import { enableInstrumentation } from "./instrumentation/index.js";
import { getClient as getClientAPI } from "./utils/client.js";
import { getSpanBufferManager } from "./utils/spanBuffer.js";

/**
 * Respan client for trace management and instrumentation.
 * This class provides an interface for initializing and managing OpenTelemetry-based tracing
 * for various AI/ML services and frameworks.
 * 
 * ## Instrumentation Management
 * 
 * ### Automatic Discovery (Default)
 * By default, Respan will attempt to load all available instrumentations automatically:
 * ```typescript
 * const respan = new RespanTelemetry({
 *   apiKey: 'your-api-key',
 *   logLevel: 'info' // Shows what gets loaded successfully
 * });
 * ```
 * 
 * ### Manual Instrumentation (Next.js/Webpack environments)
 * For environments where dynamic imports don't work properly:
 * ```typescript
 * import OpenAI from 'openai';
 * import Anthropic from '@anthropic-ai/sdk';
 * 
 * const respan = new RespanTelemetry({
 *   apiKey: 'your-api-key',
 *   instrumentModules: {
 *     openAI: OpenAI,
 *     anthropic: Anthropic
 *   }
 * });
 * ```
 * 
 * ### Disable Specific Instrumentations
 * Block instrumentations you don't want to use:
 * ```typescript
 * const respan = new RespanTelemetry({
 *   apiKey: 'your-api-key',
 *   disabledInstrumentations: ['bedrock', 'chromaDB', 'qdrant']
 * });
 * ```
 * 
 * ### Available Instrumentations (consistent camelCase naming)
 * - `openAI` - OpenAI API instrumentation
 * - `anthropic` - Anthropic API instrumentation  
 * - `azureOpenAI` - Azure OpenAI instrumentation
 * - `cohere` - Cohere API instrumentation
 * - `bedrock` - AWS Bedrock instrumentation
 * - `googleVertexAI` - Google Vertex AI instrumentation
 * - `googleAIPlatform` - Google AI Platform instrumentation
 * - `pinecone` - Pinecone vector database instrumentation
 * - `together` - Together AI instrumentation
 * - `langChain` - LangChain framework instrumentation
 * - `llamaIndex` - LlamaIndex framework instrumentation
 * - `chromaDB` - ChromaDB vector database instrumentation
 * - `qdrant` - Qdrant vector database instrumentation
 * 
 * ### Debugging Instrumentation Loading
 * Set `logLevel: 'info'` or `logLevel: 'debug'` to see:
 * - Which instrumentations loaded successfully
 * - Which ones failed and installation instructions
 * - Which ones were disabled by configuration
 */
export class RespanTelemetry {
    private options: RespanOptions;
    private initialized: boolean = false;
    private initializing: boolean = false;

    constructor(options: RespanOptions) {
        this.options = {
            ...options,
            appName: options.appName || process.env.RESPAN_APP_NAME || "default",
            disableBatch: options.disableBatch || false,
            baseURL: options.baseURL || process.env.RESPAN_BASE_URL || "https://api.respan.ai",
            apiKey: options.apiKey || process.env.RESPAN_API_KEY || "",
            instrumentModules: options.instrumentModules || {},
            disabledInstrumentations: options.disabledInstrumentations || [],
            tracingEnabled: options.tracingEnabled !== false,
            spanNameStyle: (options.spanNameStyle ||
                process.env.RESPAN_SPAN_NAME_STYLE ||
                "semantic") as RespanOptions["spanNameStyle"],
            traceContent: options.traceContent !== false,
            logLevel: options.logLevel || "error",
            silenceInitializationMessage: options.silenceInitializationMessage || false,
        };
        
        // Don't auto-initialize - let user call initialize() explicitly
        // This prevents timing issues and double initialization
    }

    private async _initialize() {
        if (this.initialized || this.initializing) {
            return;
        }
        
        this.initializing = true;
        try {
            await startTracing(this.options);
            this.initialized = true;
        } catch (error) {
            console.error("Failed to initialize Respan tracing:", error);
        } finally {
            this.initializing = false;
        }
    }

    /**
     * Manually initialize tracing. This is useful if you want to ensure
     * tracing is fully initialized before proceeding.
     */
    public async initialize(): Promise<void> {
        await this._initialize();
    }

    /**
     * Check if tracing has been initialized
     */
    public isInitialized(): boolean {
        return this.initialized;
    }

    public withTask: WithFunctionType = withTask;

    public withWorkflow: WithFunctionType = withWorkflow;

    public withAgent: WithFunctionType = withAgent;

    public withTool: WithFunctionType = withTool;

    public withRespanSpanAttributes = withRespanSpanAttributes;

    /**
     * Enable instrumentation for a specific provider
     * @param name - The name of the instrumentation (e.g., 'openai', 'anthropic')
     */
    public async enableInstrumentation(name: string): Promise<void> {
        await enableInstrumentation(name);
    }

    /**
     * Enable multiple instrumentations
     * @param names - Array of instrumentation names
     */
    public async enableInstrumentations(names: string[]): Promise<void> {
        await Promise.all(names.map(name => enableInstrumentation(name)));
    }

    /**
     * Flush pending spans without shutting down tracing.
     */
    public async flush(): Promise<void> {
        await flush();
    }

    /**
     * Flush pending spans and shutdown tracing.
     */
    public async shutdown(): Promise<void> {
        await shutdownTracing();
        this.initialized = false;
    }

    /**
     * Add a processor for routing spans to different destinations.
     * 
     * Note: A default processor is automatically configured to send spans to Respan.
     * You only need to call this method if you want to route spans to additional destinations.
     * 
     * @param config - Processor configuration
     * 
     * @example
     * ```typescript
     * // Add debug processor (in addition to default Respan processor)
     * respan.addProcessor({
     *   exporter: new FileExporter("./debug.json"),
     *   name: "debug"
     * });
     * 
     * // Route specific spans to debug processor
     * await respan.withTask(
     *   { name: "my_task", processors: "debug" },
     *   async () => { ... }
     * );
     * 
     * // Spans without processors attribute go to default Respan processor
     * await respan.withTask(
     *   { name: "normal_task" },
     *   async () => { ... }  // <- Goes to default processor
     * );
     * ```
     */
    public addProcessor(config: ProcessorConfig): void {
        addProcessorToSDK(config);
    }

    /**
     * Get the client API for span management.
     * Provides methods to update spans, add events, record exceptions, etc.
     * 
     * @returns The Respan client instance
     * 
     * @example
     * ```typescript
     * const client = respan.getClient();
     * const traceId = client.getCurrentTraceId();
     * client.updateCurrentSpan({
     *   respanParams: {
     *     customerIdentifier: "user-123"
     *   }
     * });
     * ```
     */
    public getClient() {
        return getClientAPI();
    }

    /**
     * Get the span buffer manager for manual span control.
     * 
     * @returns The span buffer manager
     * 
     * @example
     * ```typescript
     * const manager = respan.getSpanBufferManager();
     * const buffer = manager.createBuffer("trace-123");
     * buffer.createSpan("step1", { status: "completed" });
     * const spans = buffer.getAllSpans();
     * await manager.processSpans(spans);
     * ```
     */
    public getSpanBufferManager() {
        return getSpanBufferManager();
    }
}
