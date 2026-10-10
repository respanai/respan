export const SUPERAGENT_INSTRUMENTATION_NAME = "superagent";
export const SAFETY_AGENT_MODULE_NAME = "safety-agent";
export const SUPPORTED_METHODS = ["guard", "redact", "scan"] as const;
export type SuperagentMethodName = (typeof SUPPORTED_METHODS)[number];
