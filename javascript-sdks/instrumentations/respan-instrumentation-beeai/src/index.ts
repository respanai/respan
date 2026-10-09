import { NativeBeeAIInstrumentor } from "./_native.js";

export type BeeAIInstrumentationClass = new (...args: any[]) => any;
interface InstrumentationDelegate {
  activate(): void;
  deactivate(): void;
}
type DelegateFactory = (
  instrumentationClass: BeeAIInstrumentationClass,
  sdkModule: Record<string, unknown>,
) => InstrumentationDelegate | Promise<InstrumentationDelegate>;

export interface BeeAIInstrumentorOptions {
  /** Pass the application-resolved module in ESM, linked or bundled installs. */
  sdkModule?: Record<string, unknown>;
  /** Content capture also obeys environment, context and ancestor vetoes. */
  traceContent?: boolean;
  /** Retained for custom delegateFactory compatibility; native mode ignores it. */
  instrumentationClass?: BeeAIInstrumentationClass;
  /**
   * Explicit custom delegate compatibility. The caller owns its span mapping,
   * privacy, suppression and serialization behavior; native guarantees do not
   * apply to custom delegates. Provide instrumentationClass with this option.
   */
  delegateFactory?: DelegateFactory;
}

/** Native instrumentation for released BeeAI Framework 0.1.9 through 0.1.x. */
export class BeeAIInstrumentor {
  public readonly name = "beeai";
  private delegate: InstrumentationDelegate | null = null;
  private activation: Promise<void> | null = null;
  private generation = 0;
  private requested = false;

  constructor(private readonly options: BeeAIInstrumentorOptions = {}) {}

  async activate(): Promise<void> {
    this.requested = true;
    if (this.delegate) return;
    if (this.activation) {
      await this.activation;
      if (this.requested && !this.delegate) await this.activate();
      return;
    }
    const pending = this.activateGeneration(this.generation);
    this.activation = pending;
    try {
      await pending;
    } finally {
      if (this.activation === pending) this.activation = null;
    }
  }

  deactivate(): void {
    this.requested = false;
    this.generation++;
    const delegate = this.delegate;
    this.delegate = null;
    delegate?.deactivate();
  }

  private async activateGeneration(generation: number): Promise<void> {
    let delegate: InstrumentationDelegate | undefined;
    let activationAttempted = false;
    const current = () => this.requested && this.generation === generation;
    try {
      const sdk = this.options.sdkModule ?? (await import("beeai-framework"));
      if (!current()) return;
      if (this.options.delegateFactory) {
        if (!this.options.instrumentationClass)
          throw new Error(
            "BeeAI custom delegateFactory requires instrumentationClass",
          );
        delegate = await this.options.delegateFactory(
          this.options.instrumentationClass,
          sdk,
        );
      } else {
        const version =
          typeof sdk.Version === "string"
            ? /^0\.1\.(\d+)$/.exec(sdk.Version)
            : null;
        if (!version || Number(version[1]) < 9)
          throw new Error(
            "BeeAIInstrumentor supports released beeai-framework >=0.1.9 <0.2.0",
          );
        delegate = new NativeBeeAIInstrumentor(
          sdk,
          this.options.traceContent !== false,
        );
      }
      if (!current()) return;
      activationAttempted = true;
      delegate.activate();
      if (!current()) {
        delegate.deactivate();
        return;
      }
      this.delegate = delegate;
    } catch (error) {
      if (delegate && activationAttempted) {
        try {
          delegate.deactivate();
        } catch {
          /* Preserve the activation failure. */
        }
      }
      throw error;
    }
  }
}
