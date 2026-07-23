#!/usr/bin/env python3
"""
Example showing how to use the log_level parameter in RespanTelemetry
to enable debug logging for troubleshooting.
"""

import os
from respan_tracing import RespanTelemetry
from respan_tracing.decorators import workflow, task
from openai import OpenAI

# Example 1: Enable debug logging
print("=== Example 1: Debug logging enabled ===")
telemetry_debug = RespanTelemetry(
    log_level="DEBUG",  # Enable debug logging
    app_name="debug_example"
)

# Example 2: Default INFO logging
print("\n=== Example 2: Default INFO logging ===")
telemetry_info = RespanTelemetry(
    log_level="INFO",  # Default level
    app_name="info_example"
)

# Example 3: Warning level only
print("\n=== Example 3: Warning level only ===")
telemetry_warning = RespanTelemetry(
    log_level="WARNING",  # Only warnings and errors
    app_name="warning_example"
)

# Example function to demonstrate logging
@task(name="test_task")
def simple_task():
    """A simple task for demonstration"""
    return "Task completed"

@workflow(name="test_workflow")
def simple_workflow():
    """A simple workflow for demonstration"""
    result = simple_task()
    return result

# Test the workflow with debug logging
print("\n=== Running workflow with debug logging ===")
result = simple_workflow()
print(f"Result: {result}")

# Force flush to ensure all spans are sent

print("\n=== Available log levels ===")
print("- DEBUG: Shows all debug messages including trace endpoints")
print("- INFO: Shows informational messages (default)")
print("- WARNING: Shows only warnings and errors")
print("- ERROR: Shows only errors")
print("- CRITICAL: Shows only critical errors")

print("\n=== Environment variable alternative ===")
print("You can also set RESPAN_LOG_LEVEL=DEBUG as an environment variable")

print("\n=== What debug logging shows ===")
print("With DEBUG level, you'll see:")
print("- Traces endpoint configuration")
print("- Internal Respan tracing debug messages")
print("- All child module debug messages (exporter, tracer, etc.)")

print("\n=== Using the constant ===")
print("The logger name is now defined as a constant in respan_tracing.constants.generic.LOGGER_NAME") 