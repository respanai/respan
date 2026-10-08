"""Superagent instrumentation constants."""

from respan_sdk.constants.span_attributes import RESPAN_METADATA

SUPERAGENT_INSTRUMENTATION_NAME = "superagent"
SAFETY_AGENT_CLIENT_MODULE = "safety_agent.client"
SAFETY_CLIENT_CLASS_NAME = "SafetyClient"

GUARD_METHOD = "guard"
REDACT_METHOD = "redact"
SCAN_METHOD = "scan"
TEST_METHOD = "test"
SUPPORTED_METHODS = (
    GUARD_METHOD,
    REDACT_METHOD,
    SCAN_METHOD,
)

INPUT_KEY = "input"
MODEL_KEY = "model"
REPO_KEY = "repo"

SUPERAGENT_METADATA_INTEGRATION = f"{RESPAN_METADATA}.integration"
SUPERAGENT_METADATA_METHOD = f"{RESPAN_METADATA}.superagent_method"
SUPERAGENT_METADATA_MODEL = f"{RESPAN_METADATA}.superagent_model"
SUPERAGENT_METADATA_CLASSIFICATION = f"{RESPAN_METADATA}.superagent_classification"
SUPERAGENT_METADATA_REDACT_FINDINGS = f"{RESPAN_METADATA}.superagent_redact_findings"
SUPERAGENT_METADATA_USAGE = f"{RESPAN_METADATA}.superagent_usage"
