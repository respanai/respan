"""Restate integration constants."""

RESTATE_INSTRUMENTATION_NAME = "restate"
RESTATE_CONTEXT_MANAGER_MARKER = "_respan_restate_invocation_context"

RESTATE_REGISTRATION_TARGETS = (
    ("restate.service", "Service.handler"),
    ("restate.object", "VirtualObject.handler"),
    ("restate.workflow", "Workflow._add_handler"),
)
