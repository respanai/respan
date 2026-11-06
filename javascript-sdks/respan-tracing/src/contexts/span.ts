import { trace, context, AttributeValue } from '@opentelemetry/api';
import { RespanParams, RESPAN_SPAN_ATTRIBUTES_MAP } from '@respan/respan-sdk';
import {
    setTimeToFirstTokenAttribute,
    timeToFirstTokenParam,
} from '../utils/timeToFirstToken.js';

/**
 * A context manager for setting Respan-specific attributes on the current span
 * 
 * @param fn - The function to execute within the context
 * @param params - Dictionary of parameters to set as span attributes
 * @returns The result of the function execution
 */
export async function withRespanSpanAttributes<T>(
    fn: () => Promise<T> | T,
    params: Partial<RespanParams>
): Promise<T> {
    const currentSpan = trace.getSpan(context.active());
    
    if (!currentSpan) {
        console.warn("No active span found. Attributes will not be set.");
        return fn();
    }

    // Set attributes first, but don't let errors block execution
    try {
        setTimeToFirstTokenAttribute(currentSpan, timeToFirstTokenParam(params));

        Object.entries(params).forEach(([key, value]) => {
            if (key in RESPAN_SPAN_ATTRIBUTES_MAP && key !== "metadata") {
                try {
                    currentSpan.setAttribute(RESPAN_SPAN_ATTRIBUTES_MAP[key], value as AttributeValue);
                } catch (e) {
                    console.warn(`Failed to set span attribute ${RESPAN_SPAN_ATTRIBUTES_MAP[key]}=${value}: ${e}`);
                }
            }
        });

        if (params.metadata) {
            Object.entries(params.metadata).forEach(([key, value]) => {
                currentSpan.setAttribute(
                    `${RESPAN_SPAN_ATTRIBUTES_MAP.metadata}.${key}`,
                    JSON.stringify(value)
                );
            });
        }
    } catch (e) {
        console.warn(`Error setting span attributes: ${e}`);
    }

    return fn();
}
