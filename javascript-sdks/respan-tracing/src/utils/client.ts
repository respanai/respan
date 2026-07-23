import { trace, context, SpanStatusCode, Tracer } from "@opentelemetry/api";
import { ReadableSpan } from "@opentelemetry/sdk-trace-base";
import { RespanSpanAttributes } from "@respan/respan-sdk";
import { metadataAttributeKey, LOG_PREFIX, LOG_PREFIX_WARN } from "../constants/index.js";

/**
 * Options for updating a span
 */
export interface UpdateSpanOptions {
  /** New name for the span */
  name?: string;
  /** Custom attributes to add to the span */
  attributes?: Record<string, any>;
  /** Status to set on the span */
  status?: {
    code: SpanStatusCode;
    message?: string;
  };
  /** Respan-specific parameters */
  respanParams?: {
    /** Customer identifier for grouping traces by user */
    customerIdentifier?: string;
    /** Trace group identifier for organizing traces */
    traceGroupIdentifier?: string;
    /** Additional metadata */
    metadata?: Record<string, any>;
  };
}

/**
 * Respan client interface for span management and tracing operations.
 * 
 * This client provides methods to:
 * - Get current trace and span IDs
 * - Update span attributes and Respan parameters
 * - Add events and record exceptions
 * - Create manual spans
 * - Control span buffering
 * 
 * @example
 * ```typescript
 * import { getClient } from '@respan/tracing';
 * 
 * const client = getClient();
 * const traceId = client.getCurrentTraceId();
 * 
 * client.updateCurrentSpan({
 *   respanParams: {
 *     customerIdentifier: 'user-123',
 *     traceGroupIdentifier: 'experiment-456'
 *   }
 * });
 * ```
 */
export interface RespanClient {
  /**
   * Get the current trace ID
   * @returns The current trace ID or undefined if no active span
   */
  getCurrentTraceId(): string | undefined;

  /**
   * Get the current span ID
   * @returns The current span ID or undefined if no active span
   */
  getCurrentSpanId(): string | undefined;

  /**
   * Get the OpenTelemetry tracer for manual span creation
   * @returns The tracer instance
   */
  getTracer(): Tracer;

  /**
   * Update the current span with new attributes, name, status, or Respan parameters
   * @param options - Options for updating the span
   * 
   * @example
   * ```typescript
   * client.updateCurrentSpan({
   *   name: 'updated_name',
   *   attributes: { 'custom.field': 'value' },
   *   respanParams: {
   *     customerIdentifier: 'user-123',
   *     traceGroupIdentifier: 'experiment-456',
   *     metadata: { version: '1.0' }
   *   }
   * });
   * ```
   */
  updateCurrentSpan(options: UpdateSpanOptions): void;

  /**
   * Add an event to the current span
   * @param name - Event name
   * @param attributes - Optional event attributes
   * 
   * @example
   * ```typescript
   * client.addEvent('validation_started', { record_count: 100 });
   * ```
   */
  addEvent(name: string, attributes?: Record<string, any>): void;

  /**
   * Record an exception on the current span
   * @param exception - The exception to record
   * 
   * @example
   * ```typescript
   * try {
   *   // ... some code
   * } catch (error) {
   *   client.recordException(error as Error);
   *   throw error;
   * }
   * ```
   */
  recordException(exception: Error): void;

  /**
   * Check if the current span is recording
   * @returns True if the span is recording, false otherwise
   */
  isRecording(): boolean;

  /**
   * Force flush all pending spans
   * @returns Promise that resolves when flush is complete
   */
  flush(): Promise<void>;
}

/**
 * Implementation of the Respan client
 */
class RespanClientImpl implements RespanClient {
  private readonly tracerName = "@respan/tracing";

  getCurrentTraceId(): string | undefined {
    const currentSpan = trace.getActiveSpan();
    if (!currentSpan) {
      return undefined;
    }
    const spanContext = currentSpan.spanContext();
    return spanContext.traceId;
  }

  getCurrentSpanId(): string | undefined {
    const currentSpan = trace.getActiveSpan();
    if (!currentSpan) {
      return undefined;
    }
    const spanContext = currentSpan.spanContext();
    return spanContext.spanId;
  }

  getTracer(): Tracer {
    return trace.getTracer(this.tracerName);
  }

  updateCurrentSpan(options: UpdateSpanOptions): void {
    const currentSpan = trace.getActiveSpan();
    if (!currentSpan) {
      console.warn("[Respan] No active span to update");
      return;
    }

    // Update span name
    if (options.name) {
      currentSpan.updateName(options.name);
    }

    // Update attributes
    if (options.attributes) {
      for (const [key, value] of Object.entries(options.attributes)) {
        currentSpan.setAttribute(key, value);
      }
    }

    // Update status
    if (options.status) {
      currentSpan.setStatus(options.status);
    }

    // Update Respan-specific parameters
    if (options.respanParams) {
      const { customerIdentifier, traceGroupIdentifier, metadata } = options.respanParams;

      if (customerIdentifier) {
        currentSpan.setAttribute(RespanSpanAttributes.RESPAN_CUSTOMER_PARAMS_ID, customerIdentifier);
      }

      if (traceGroupIdentifier) {
        currentSpan.setAttribute(RespanSpanAttributes.RESPAN_TRACE_GROUP_ID, traceGroupIdentifier);
      }

      if (metadata) {
        // Flatten metadata into attributes with respan.metadata prefix
        for (const [key, value] of Object.entries(metadata)) {
          currentSpan.setAttribute(metadataAttributeKey(key), value);
        }
      }
    }
  }

  addEvent(name: string, attributes?: Record<string, any>): void {
    const currentSpan = trace.getActiveSpan();
    if (!currentSpan) {
      console.warn("[Respan] No active span to add event to");
      return;
    }

    currentSpan.addEvent(name, attributes);
  }

  recordException(exception: Error): void {
    const currentSpan = trace.getActiveSpan();
    if (!currentSpan) {
      console.warn("[Respan] No active span to record exception on");
      return;
    }

    currentSpan.recordException(exception);
    currentSpan.setStatus({
      code: SpanStatusCode.ERROR,
      message: exception.message,
    });
  }

  isRecording(): boolean {
    const currentSpan = trace.getActiveSpan();
    if (!currentSpan) {
      return false;
    }

    return currentSpan.isRecording();
  }

  async flush(): Promise<void> {
    const { flush } = await import("./tracing.js");
    await flush();
  }
}

// Singleton instance
let _clientInstance: RespanClient | undefined;

/**
 * Get the Respan client instance for span management.
 * 
 * This function returns a singleton client that provides methods to:
 * - Get current trace and span IDs
 * - Update spans with custom attributes and Respan parameters
 * - Add events and record exceptions
 * - Access the tracer for manual span creation
 * 
 * @returns The Respan client instance
 * 
 * @example
 * ```typescript
 * import { getClient } from '@respan/tracing';
 * 
 * const client = getClient();
 * 
 * // Get current trace information
 * const traceId = client.getCurrentTraceId();
 * const spanId = client.getCurrentSpanId();
 * 
 * // Update span with Respan parameters
 * client.updateCurrentSpan({
 *   respanParams: {
 *     customerIdentifier: 'user-123',
 *     traceGroupIdentifier: 'data-processing-pipeline',
 *     metadata: {
 *       version: '1.0',
 *       environment: 'production'
 *     }
 *   }
 * });
 * 
 * // Add event to track progress
 * client.addEvent('validation_started', {
 *   record_count: 100
 * });
 * 
 * // Record exception
 * try {
 *   // ... some code
 * } catch (error) {
 *   client.recordException(error as Error);
 *   throw error;
 * }
 * ```
 */
export function getClient(): RespanClient {
  if (!_clientInstance) {
    _clientInstance = new RespanClientImpl();
  }
  return _clientInstance;
}


