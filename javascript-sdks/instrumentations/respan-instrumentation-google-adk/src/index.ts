import {
  registerSpanTransformer,
  type SpanTransformerRegistration,
} from "@respan/tracing";
import {
  ACTIVE_CONTENT_POLICIES,
  type GoogleADKInstrumentationOptions,
} from "./_config.js";
import { GoogleADKTranslator } from "./_processor.js";

export type { GoogleADKInstrumentationOptions } from "./_config.js";
export {
  GoogleADKTranslator,
  isGoogleADKSpan,
  translateGoogleADKSpan,
} from "./_processor.js";

export class GoogleADKInstrumentor {
  public readonly name = "google-adk";
  private readonly _options: Readonly<GoogleADKInstrumentationOptions>;
  private _registration?: SpanTransformerRegistration;

  constructor(options: GoogleADKInstrumentationOptions = {}) {
    this._options = Object.freeze({
      traceContent: options.traceContent !== false,
    });
  }

  activate(): void {
    if (this._registration) return;
    ACTIVE_CONTENT_POLICIES.add(this._options);
    try {
      this._registration = registerSpanTransformer(
        "@respan/instrumentation-google-adk",
        new GoogleADKTranslator(this._options, true),
      );
    } catch (error) {
      ACTIVE_CONTENT_POLICIES.delete(this._options);
      throw error;
    }
  }

  deactivate(): void {
    this._registration?.unregister();
    this._registration = undefined;
    ACTIVE_CONTENT_POLICIES.delete(this._options);
  }

  isActive(): boolean {
    return this._registration !== undefined;
  }
}
