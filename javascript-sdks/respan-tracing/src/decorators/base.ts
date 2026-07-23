import { Span, context, SpanStatusCode } from "@opentelemetry/api";
import { suppressTracing } from "@opentelemetry/core";
import { RespanSpanAttributes } from "@respan/respan-sdk";
import {
  ASSOCIATION_PROPERTIES_KEY,
  ENTITY_NAME_KEY,
  getEntityPath,
  WORKFLOW_NAME_KEY,
  shouldSendTraces,
} from "../utils/context.js";
import { getTracer } from "../utils/span.js";
import {
  CONTEXT_KEY_ALLOW_TRACE_CONTENT,
  SpanAttributes,
  TraceloopSpanKindValues,
} from "@traceloop/ai-semantic-conventions";

export type DecoratorConfig = {
  name: string;
  version?: number;
  associationProperties?: { [name: string]: string };
  traceContent?: boolean;
  inputParameters?: unknown[];
  suppressTracing?: boolean;
  /**
   * Route this span to specific processors by name.
   * Can be a single processor name or an array of names.
   * 
   * @example
   * ```typescript
   * // Single processor
   * { processors: "debug" }
   * 
   * // Multiple processors
   * { processors: ["debug", "analytics"] }
   * ```
   */
  processors?: string | string[];
};

/**
 * Core function that wraps any function with OpenTelemetry tracing capabilities.
 * This creates a span around the function execution and manages the trace context.
 *
 * @param type - The type of entity being traced (WORKFLOW, TASK, AGENT, TOOL)
 * @param config - Configuration for the tracing decorator
 * @param fn - The function to be wrapped with tracing
 * @param thisArg - The 'this' context for the function
 * @param args - Arguments to pass to the function
 * @returns The result of the wrapped function
 */
function withEntity<
  A extends unknown[],
  F extends (...args: A) => ReturnType<F>
>(
  type: TraceloopSpanKindValues,
  {
    name,
    version,
    associationProperties,
    traceContent: overrideTraceContent,
    inputParameters,
    suppressTracing: shouldSuppressTracing,
    processors,
  }: DecoratorConfig,
  fn: F,
  thisArg?: ThisParameterType<F>,
  ...args: A
): ReturnType<F> {
  // STEP 1: Get the current active context (inherits from parent spans)
  // This context contains all the trace information from parent operations
  let entityContext = context.active();

  // STEP 2: Handle workflow/agent context setup
  // Workflows and agents are top-level entities that can contain other entities
  if (
    type === TraceloopSpanKindValues.WORKFLOW ||
    type === TraceloopSpanKindValues.AGENT
  ) {
    // Store the workflow name in context so child spans can access it
    entityContext = entityContext.setValue(WORKFLOW_NAME_KEY, name);
  }

  // STEP 3: Build hierarchical entity paths for tools and tasks
  // This creates a dot-separated path like "workflow.task.subtask"
  const entityPath = getEntityPath(entityContext);
  if (
    type === TraceloopSpanKindValues.TOOL ||
    type === TraceloopSpanKindValues.TASK
  ) {
    // Create full path: if we're in "myWorkflow" and this is "myTask",
    // the full name becomes "myWorkflow.myTask"
    const fullEntityName = entityPath ? `${entityPath}.${name}` : name;
    entityContext = entityContext.setValue(ENTITY_NAME_KEY, fullEntityName);
  }

  // STEP 4: Configure trace content settings
  // This controls whether input/output data should be captured in traces
  if (overrideTraceContent != undefined) {
    entityContext = entityContext.setValue(
      CONTEXT_KEY_ALLOW_TRACE_CONTENT,
      overrideTraceContent
    );
  }

  // STEP 5: Add association properties for linking related spans
  // These are custom key-value pairs that help correlate spans
  if (associationProperties) {
    entityContext = entityContext.setValue(
      ASSOCIATION_PROPERTIES_KEY,
      associationProperties
    );
  }

  // STEP 6: Handle tracing suppression
  // This allows disabling tracing for specific operations
  if (shouldSuppressTracing) {
    entityContext = suppressTracing(entityContext);
  }

  // STEP 7: Execute the function within the trace context
  // context.with() ensures all operations inside use the entityContext
  return context.with(entityContext, () =>
    // STEP 8: Create and start a new span
    getTracer().startActiveSpan(
      `${name}.${type}`, // Span name format: "functionName.WORKFLOW"
      {}, // Span options (empty for now)
      entityContext, // The context to use for this span
      (span: Span) => {
        // STEP 9: Set span attributes for metadata
        // These attributes help identify and categorize the span

        // Special handling for workflow/agent spans
        if (
          type === TraceloopSpanKindValues.WORKFLOW ||
          type === TraceloopSpanKindValues.AGENT
        ) {
          span.setAttribute(SpanAttributes.TRACELOOP_WORKFLOW_NAME, name);
        }

        // Standard attributes for all spans
        span.setAttribute(SpanAttributes.TRACELOOP_ENTITY_NAME, name);
        span.setAttribute(
          SpanAttributes.TRACELOOP_ENTITY_PATH,
          entityPath || ""
        );
        span.setAttribute(SpanAttributes.TRACELOOP_SPAN_KIND, type);

        // Optional version information
        if (version) {
          span.setAttribute(SpanAttributes.TRACELOOP_ENTITY_VERSION, version);
        }

        // Set processor routing if specified
        if (processors) {
          span.setAttribute(RespanSpanAttributes.RESPAN_PROCESSORS, processors);
        }

        // STEP 10: Capture input parameters if tracing is enabled
        if (shouldSendTraces()) {
          try {
            const hasDisplayInput = inputParameters !== undefined;
            const input = inputParameters ?? args;

            // Explicit display inputs should be recorded as the caller-provided
            // payload instead of leaking the wrapper's args/kwargs shape.
            if (hasDisplayInput && input.length === 1) {
              span.setAttribute(
                SpanAttributes.TRACELOOP_ENTITY_INPUT,
                serialize(input[0])
              );
            } else if (
              input.length === 1 &&
              typeof input[0] === "object" &&
              !(input[0] instanceof Map)
            ) {
              // Handle single object parameter (common pattern)
              span.setAttribute(
                SpanAttributes.TRACELOOP_ENTITY_INPUT,
                serialize({ args: [], kwargs: input[0] })
              );
            } else {
              // Handle multiple parameters
              span.setAttribute(
                SpanAttributes.TRACELOOP_ENTITY_INPUT,
                serialize({
                  args: input,
                  kwargs: {},
                })
              );
            }
          } catch (error) {
            console.error("Error serializing input:", error);
          }
        }

        // STEP 11: Execute the wrapped function with error handling
        try {
          const res = fn.apply(thisArg, args);

          // STEP 12: Handle async functions (Promises)
          if (res instanceof Promise) {
            return res
              .then((resolvedRes) => {
                // Capture successful async result
                try {
                  if (shouldSendTraces()) {
                    span.setAttribute(
                      SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
                      serialize(resolvedRes)
                    );
                  }
                } catch (error) {
                  console.error("Error serializing output:", error);
                } finally {
                  // Always end the span when async operation completes
                  span.end();
                }

                return resolvedRes;
              })
              .catch((error) => {
                // Handle async errors
                if (error instanceof Error) {
                  span.recordException(error);
                  span.setStatus({
                    code: SpanStatusCode.ERROR,
                    message: error.message,
                  });
                } else {
                  span.setStatus({
                    code: SpanStatusCode.ERROR,
                    message: String(error),
                  });
                }
                span.end();
                throw error; // Re-throw to maintain error propagation
              }) as ReturnType<F>;
          }

          // STEP 13: Handle synchronous functions
          try {
            if (shouldSendTraces()) {
              span.setAttribute(
                SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
                serialize(res)
              );
            }
          } catch (error) {
            console.error("Error serializing output:", error);
          } finally {
            // End span for synchronous operations
            span.end();
          }

          return res;
        } catch (error) {
          // STEP 14: Handle synchronous errors
          if (error instanceof Error) {
            span.recordException(error);
            span.setStatus({
              code: SpanStatusCode.ERROR,
              message: error.message,
            });
          } else {
            span.setStatus({
              code: SpanStatusCode.ERROR,
              message: String(error),
            });
          }
          span.end();
          throw error; // Re-throw to maintain error propagation
        }
      }
    )
  ) as ReturnType<F>;
}

export function withWorkflow<
  A extends unknown[],
  F extends (...args: A) => ReturnType<F>
>(config: DecoratorConfig, fn: F, ...args: A): ReturnType<F> {
  return withEntity(
    TraceloopSpanKindValues.WORKFLOW,
    config,
    fn,
    undefined,
    ...args
  );
}

export function withTask<
  A extends unknown[],
  F extends (...args: A) => ReturnType<F>
>(config: DecoratorConfig, fn: F, ...args: A): ReturnType<F> {
  return withEntity(
    TraceloopSpanKindValues.TASK,
    config,
    fn,
    undefined,
    ...args
  );
}

export function withAgent<
  A extends unknown[],
  F extends (...args: A) => ReturnType<F>
>(config: DecoratorConfig, fn: F, ...args: A): ReturnType<F> {
  return withEntity(
    TraceloopSpanKindValues.AGENT,
    config,
    fn,
    undefined,
    ...args
  );
}

export function withTool<
  A extends unknown[],
  F extends (...args: A) => ReturnType<F>
>(config: DecoratorConfig, fn: F, ...args: A): ReturnType<F> {
  return withEntity(
    TraceloopSpanKindValues.TOOL,
    config,
    fn,
    undefined,
    ...args
  );
}

function cleanInput(input: unknown): unknown {
  if (Array.isArray(input)) {
    return input.map((value) => cleanInput(value));
  } else if (!input) {
    return input;
  } else if (typeof input === "object" && input !== null) {
    const cleaned: any = {};
    for (const [key, value] of Object.entries(input)) {
      cleaned[key] = cleanInput(value);
    }
    return cleaned;
  }
  return input;
}

function serialize(input: unknown): string {
  try {
    return JSON.stringify(cleanInput(input));
  } catch (error) {
    return String(input);
  }
}

function entity(
  type: TraceloopSpanKindValues,
  config:
    | Partial<DecoratorConfig>
    | ((thisArg: unknown, ...funcArgs: unknown[]) => Partial<DecoratorConfig>)
) {
  return function (
    target: unknown,
    propertyKey: string,
    descriptor: PropertyDescriptor
  ) {
    const originalMethod = descriptor.value;

    descriptor.value = function (...args: unknown[]) {
      let actualConfig;

      if (typeof config === "function") {
        actualConfig = config(this, ...args);
      } else {
        actualConfig = config;
      }

      const entityName = actualConfig.name ?? originalMethod.name;

      return withEntity(
        type,
        { ...actualConfig, name: entityName },
        originalMethod,
        this,
        ...args
      );
    };
  };
}

export function workflow(
  config:
    | Partial<DecoratorConfig>
    | ((thisArg: unknown, ...funcArgs: unknown[]) => Partial<DecoratorConfig>)
) {
  return entity(TraceloopSpanKindValues.WORKFLOW, config ?? {});
}

export function task(
  config:
    | Partial<DecoratorConfig>
    | ((thisArg: unknown, ...funcArgs: unknown[]) => Partial<DecoratorConfig>)
) {
  return entity(TraceloopSpanKindValues.TASK, config ?? {});
}

export function agent(
  config:
    | Partial<DecoratorConfig>
    | ((thisArg: unknown, ...funcArgs: unknown[]) => Partial<DecoratorConfig>)
) {
  return entity(TraceloopSpanKindValues.AGENT, config ?? {});
}

export function tool(
  config:
    | Partial<DecoratorConfig>
    | ((thisArg: unknown, ...funcArgs: unknown[]) => Partial<DecoratorConfig>)
) {
  return entity(TraceloopSpanKindValues.TOOL, config ?? {});
}
