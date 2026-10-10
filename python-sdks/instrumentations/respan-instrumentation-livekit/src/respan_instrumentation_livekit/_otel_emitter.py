"""Tool spans are live OpenTelemetry scopes around native execute_function_call.

The implementation lives with the owned runtime in _instrumentation; no
post-hoc synthetic builder or independent export path is used.
"""
