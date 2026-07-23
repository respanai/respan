#!/usr/bin/env python3
"""
Complete example showing the debug logging features in Respan Tracing.

This example demonstrates:
1. Using the log_level parameter
2. Environment variable support
3. Using constants for logger name
4. Child logger inheritance (exporter debug messages)
"""

import os
import logging
from respan_tracing import RespanTelemetry
from respan_tracing.decorators import workflow, task
from respan_tracing.constants import LOGGER_NAME, SDK_PREFIX

print("=== Respan Tracing Debug Logging Example ===")
print(f"Logger name constant: {LOGGER_NAME}")
print(f"SDK prefix constant: {SDK_PREFIX}")

# Example 1: Direct debug logging
print("\n1. Direct debug logging:")
telemetry1 = RespanTelemetry(log_level="DEBUG", app_name="debug_example")

@task(name="debug_task")
def debug_task():
    return "Debug task completed"

@workflow(name="debug_workflow") 
def debug_workflow():
    return debug_task()

result1 = debug_workflow()
print(f"Result 1: {result1}")

# Example 2: Environment variable (comment out to avoid conflicts)
print("\n2. Environment variable example:")
print("   Run: RESPAN_LOG_LEVEL=DEBUG python examples/debug_logging_complete_example.py")
print("   This will enable debug logging via environment variable")

# Example 3: Custom logger usage
print("\n3. Custom logger usage:")
# You can also create your own logger that inherits from the Respan logger
custom_logger = logging.getLogger(f"{LOGGER_NAME}.custom")
custom_logger.info("This is a custom message that inherits Respan logging config")

print("\n=== Summary ===")
print("✅ Debug logging shows:")
print("   - Traces endpoint configuration")
print("   - Internal Respan tracing debug messages")
print("   - All child module debug messages (exporter, tracer, etc.)")
print("✅ Constants defined:")
print(f"   - LOGGER_NAME = '{LOGGER_NAME}'")
print(f"   - SDK_PREFIX = '{SDK_PREFIX}'")
print("✅ Environment variable support:")
print("   - Set RESPAN_LOG_LEVEL=DEBUG")
print("✅ Child logger inheritance:")
print("   - Exporter and other components inherit the log level")
