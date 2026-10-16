SDK_PREFIX = "respan"
LOGGER_NAME = "respan_tracing"

# Logger names for different modules (child loggers under LOGGER_NAME)
LOGGER_NAME_TRACER = "core.tracer"
LOGGER_NAME_CLIENT = "core.client"
LOGGER_NAME_EXPORTER = "core.exporter"
LOGGER_NAME_PROCESSOR = "core.processor"
LOGGER_NAME_SPAN = "contexts.span"
LOGGER_NAME_DECORATORS = "decorators"

# Environment switches for capturing prompts/completions/inputs/outputs.
TRACE_CONTENT_ENV_VARS = ("TRACELOOP_TRACE_CONTENT", "RESPAN_TRACE_CONTENT")
TRACE_CONTENT_OFF_VALUES = frozenset({"0", "false", "no", "off"})
