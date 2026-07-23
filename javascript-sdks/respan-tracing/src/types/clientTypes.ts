import { SpanExporter, SpanProcessor } from "@opentelemetry/sdk-trace-base";
import { TextMapPropagator, ContextManager } from "@opentelemetry/api";
import type { RespanSpanNameStyle } from "@respan/respan-sdk";

/**
 * Available instrumentation names for consistent camelCase naming.
 * Abbreviations like AI, API, DB are kept uppercase.
 */
export type InstrumentationName = 
    | 'http'
    | 'openAI'
    | 'anthropic'
    | 'azureOpenAI'
    | 'cohere'
    | 'bedrock'
    | 'googleVertexAI'
    | 'googleAIPlatform'
    | 'pinecone'
    | 'together'
    | 'langChain'
    | 'llamaIndex'
    | 'chromaDB'
    | 'qdrant';

/**
 * Configuration for a span processor with routing capabilities
 */
export interface ProcessorConfig {
    /** The span exporter to use */
    exporter: SpanExporter;
    /** Name identifier for this processor (used for routing) */
    name: string;
    /** Optional custom filter function for spans */
    filter?: (span: any) => boolean;
    /** Optional priority (higher = processed first) */
    priority?: number;
    /** Send spans immediately without batching. Useful for short-lived scripts. */
    disableBatch?: boolean;
}

/**
 * Options for initializing the Respan SDK.
 */
export interface RespanOptions {
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
     * Defaults to RESPAN_BASE_URL environment variable or https://api.respan.co/
     */
    baseURL?: string;

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
     * The OpenTelemetry SpanExporter to be used for sending traces data. Optional.
     * Defaults to the OTLP exporter.
     */
    exporter?: SpanExporter;

    /**
     * The headers to be sent with the traces data. Optional.
     */
    headers?: Record<string, string>;

    /**
     * The OpenTelemetry SpanProcessor to be used for processing traces data. Optional.
     * Defaults to the BatchSpanProcessor.
     */
    processor?: SpanProcessor;

    /**
     * The OpenTelemetry Propagator to use. Optional.
     * Defaults to OpenTelemetry SDK defaults.
     */
    propagator?: TextMapPropagator;

    /**
     * The OpenTelemetry ContextManager to use. Optional.
     * Defaults to OpenTelemetry SDK defaults.
     */
    contextManager?: ContextManager;

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
     * Additional resource attributes to attach to all spans. Optional.
     * Useful for adding environment, version, region, etc.
     * 
     * @example
     * ```typescript
     * resourceAttributes: {
     *   environment: 'production',
     *   version: '1.0.0',
     *   region: 'us-east-1'
     * }
     * ```
     */
    resourceAttributes?: Record<string, string>;

    /**
     * Optional callback function to process spans before export.
     * This is called for each span before it's sent to the exporter.
     * 
     * @example
     * ```typescript
     * spanPostprocessCallback: (span) => {
     *   // Add custom processing
     *   console.log('Processing span:', span.name);
     * }
     * ```
     */
    spanPostprocessCallback?: (span: any) => void;

    /**
     * Explicitly specify modules to instrument. Optional.
     * This is a workaround specific to Next.js and other environments where
     * dynamic imports might not work properly.
     * 
     * Usage example:
     * ```typescript
     * import OpenAI from 'openai';
     * import Anthropic from '@anthropic-ai/sdk';
     * 
     * const respan = new RespanTelemetry({
     *   instrumentModules: {
     *     openAI: OpenAI,
     *     anthropic: Anthropic,
     *   }
     * });
     * ```
     */
    instrumentModules?: {
        openAI?: any; // typeof import('openai').OpenAI
        anthropic?: any; // typeof import('@anthropic-ai/sdk')
        azureOpenAI?: any; // typeof import('@azure/openai')
        cohere?: any; // typeof import('cohere-ai')
        bedrock?: any; // typeof import('@aws-sdk/client-bedrock-runtime')
        googleVertexAI?: any; // typeof import('@google-cloud/vertexai')
        googleAIPlatform?: any; // typeof import('@google-cloud/aiplatform')
        pinecone?: any; // typeof import('@pinecone-database/pinecone')
        together?: any; // typeof import('together-ai').Together
        langChain?: {
            chainsModule?: any; // typeof import('langchain/chains')
            agentsModule?: any; // typeof import('langchain/agents')
            toolsModule?: any; // typeof import('langchain/tools')
            runnablesModule?: any; // typeof import('@langchain/core/runnables')
            vectorStoreModule?: any; // typeof import('@langchain/core/vectorstores')
        };
        llamaIndex?: any; // typeof import('llamaindex')
        chromaDB?: any; // typeof import('chromadb')
        qdrant?: any; // typeof import('@qdrant/js-client-rest')
    };

    /**
     * List of instrumentation modules to disable. Optional.
     * Useful for preventing certain instrumentations from being loaded.
     * Uses consistent camelCase naming with abbreviations kept uppercase.
     * 
     * Usage example:
     * ```typescript
     * const respan = new RespanTelemetry({
     *   disabledInstrumentations: ['bedrock', 'chromaDB', 'qdrant']
     * });
     * ```
     */
    disabledInstrumentations?: InstrumentationName[];
}

/**
 * The main client options for Respan.
 */
export interface RespanClientOptions {
    apiKey: string;
    appName: string;
    baseURL?: string;
}

/**
 * Base class for all Respan errors.
 */
export class RespanError extends Error {
    constructor(message: string) {
        super(message);
        this.name = 'RespanError';
    }
}

export class InitializationError extends RespanError {
    constructor(message?: string) {
        super(message || 'Failed to initialize Respan');
        this.name = 'InitializationError';
    }
}

export class NotInitializedError extends RespanError {
    constructor() {
        super('Respan has not been initialized');
        this.name = 'NotInitializedError';
    }
}
