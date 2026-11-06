TRACER_NAME = "respan.tracer"
SPAN_BUFFER_TRACER_NAME = "respan.span_buffer"
EXPORT_FILTER_ATTR = "respan.export_filter"
PROCESSORS_ATTR = "respan.processors"
SAMPLE_RATE_ATTR = "respan.sample_rate"

# Bare backend column key for time to first token, in seconds. Mirrors
# respan_sdk RespanSpanAttributes.TIME_TO_FIRST_TOKEN; restated here so this
# package works on its released respan-sdk floor.
TIME_TO_FIRST_TOKEN_ATTR = "time_to_first_token"
