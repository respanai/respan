import { RespanSpanAttributes } from "@respan/respan-sdk";

export const SUPERAGENT_INSTRUMENTATION_NAME = "superagent";
export const SAFETY_AGENT_MODULE_NAME = "safety-agent";

export const GUARD_METHOD = "guard";
export const REDACT_METHOD = "redact";
export const SCAN_METHOD = "scan";

export const SUPPORTED_METHODS = [
  GUARD_METHOD,
  REDACT_METHOD,
  SCAN_METHOD,
] as const;

export type SuperagentMethodName = (typeof SUPPORTED_METHODS)[number];

export const INPUT_KEY = "input";
export const MODEL_KEY = "model";
export const REPO_KEY = "repo";

export const SUPERAGENT_METADATA_INTEGRATION = `${RespanSpanAttributes.RESPAN_METADATA}.integration`;
export const SUPERAGENT_METADATA_METHOD = `${RespanSpanAttributes.RESPAN_METADATA}.superagent_method`;
export const SUPERAGENT_METADATA_MODEL = `${RespanSpanAttributes.RESPAN_METADATA}.superagent_model`;
export const SUPERAGENT_METADATA_CLASSIFICATION = `${RespanSpanAttributes.RESPAN_METADATA}.superagent_classification`;
export const SUPERAGENT_METADATA_REDACT_FINDINGS = `${RespanSpanAttributes.RESPAN_METADATA}.superagent_redact_findings`;
