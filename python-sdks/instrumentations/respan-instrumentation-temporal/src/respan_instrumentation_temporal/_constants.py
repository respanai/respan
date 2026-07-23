"""Temporal instrumentation-local constants."""

from respan_sdk.constants.llm_logging import LOG_TYPE_TASK, LOG_TYPE_WORKFLOW


TEMPORAL_INSTRUMENTATION_NAME = "temporal"
TEMPORAL_CLIENT_MODULE = "temporalio.client"
TEMPORAL_CLIENT_CONNECT_TARGET = "Client.connect"
TEMPORAL_OTEL_MODULE = "temporalio.contrib.opentelemetry"

# This key only moves a Python object between our interceptor subclass and tracer
# proxy. It is removed before OpenTelemetry sees the attributes.
TEMPORAL_CAPTURED_INPUT = "__respan_temporal_captured_input__"

TEMPORAL_RAW_ATTRIBUTE_KEYS = frozenset(
    {
        "temporalWorkflowID",
        "temporalRunID",
        "temporalActivityID",
        "temporalActivityType",
        "temporalUpdateID",
    }
)

WORKFLOW_OPERATION_PREFIXES = frozenset(
    {
        "StartWorkflow",
        "SignalWithStartWorkflow",
        "RunWorkflow",
        "CompleteWorkflow",
        "StartChildWorkflow",
    }
)

TASK_LOG_TYPE = LOG_TYPE_TASK
WORKFLOW_LOG_TYPE = LOG_TYPE_WORKFLOW
MAX_ATTRIBUTE_CHARS = 16_000
MAX_COLLECTION_ITEMS = 30
MAX_SERIALIZATION_DEPTH = 8
