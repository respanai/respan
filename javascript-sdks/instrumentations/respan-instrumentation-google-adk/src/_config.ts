export interface GoogleADKInstrumentationOptions {
  /** Disable message and tool payloads while retaining operation metadata. */
  traceContent?: boolean;
}

const CONTENT_POLICIES_SYMBOL = Symbol.for(
  "respan.instrumentation.google-adk.contentPolicies.v1",
);
const policyGlobals = globalThis as typeof globalThis & {
  [CONTENT_POLICIES_SYMBOL]?: Set<Readonly<GoogleADKInstrumentationOptions>>;
};
export const ACTIVE_CONTENT_POLICIES = (policyGlobals[
  CONTENT_POLICIES_SYMBOL
] ??= new Set<Readonly<GoogleADKInstrumentationOptions>>());
