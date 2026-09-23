import type { RespanSpanNameStyle } from "@respan/respan-sdk";
import {
  N8nNativeSdkInstrumentation,
  type N8nNativeInstrumentationOptions,
} from "./_native_instrumentation.js";

const RUNTIME_STATE_SYMBOL = Symbol.for("respan.instrumentation.n8n.runtime.v1");

interface RuntimeState {
  ownerCount: number;
  instrumentation?: N8nNativeSdkInstrumentation;
  spanNameStyle?: RespanSpanNameStyle | string;
}

interface N8nInstrumentorOptions extends N8nNativeInstrumentationOptions {}

/**
 * Installs a preload-safe module hook for n8n's native OpenTelemetry NodeSDK.
 * It never creates or registers another tracer provider.
 */
export class N8nInstrumentor {
  public readonly name = "n8n";
  private readonly _options: N8nInstrumentorOptions;
  private _active = false;

  constructor(options: N8nInstrumentorOptions = {}) {
    this._options = options;
  }

  activate(): void {
    if (this._active) return;

    const state = getRuntimeState();
    if (state.ownerCount === 0) {
      const instrumentation =
        state.instrumentation ?? new N8nNativeSdkInstrumentation(this._options);
      instrumentation.setOptions(this._options);
      try {
        instrumentation.enable();
      } catch (error) {
        try {
          instrumentation.disable();
        } catch {
          // Preserve the activation error.
        }
        throw error;
      }
      state.instrumentation = instrumentation;
      state.spanNameStyle = this._options.spanNameStyle;
    } else if (
      this._options.spanNameStyle !== undefined &&
      state.spanNameStyle !== this._options.spanNameStyle
    ) {
      console.warn(
        "[respan] n8n instrumentation is already active; the first spanNameStyle remains authoritative",
      );
    }

    state.ownerCount += 1;
    this._active = true;
  }

  deactivate(): void {
    if (!this._active) return;

    const state = getRuntimeState();
    state.ownerCount = Math.max(0, state.ownerCount - 1);
    this._active = false;

    if (state.ownerCount !== 0) return;

    state.spanNameStyle = undefined;
    state.instrumentation?.disable();
  }

  isActive(): boolean {
    return this._active;
  }
}

function getRuntimeState(): RuntimeState {
  const globalState = globalThis as typeof globalThis & {
    [RUNTIME_STATE_SYMBOL]?: RuntimeState;
  };
  if (!globalState[RUNTIME_STATE_SYMBOL]) {
    globalState[RUNTIME_STATE_SYMBOL] = { ownerCount: 0 };
  }
  return globalState[RUNTIME_STATE_SYMBOL];
}
