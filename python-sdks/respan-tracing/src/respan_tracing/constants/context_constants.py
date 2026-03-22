# Context keys for Respan tracing
WORKFLOW_NAME_KEY = "respan_workflow_name"
ENTITY_PATH_KEY = "respan_entity_path"
TRACE_GROUP_ID_KEY = "respan_trace_group_identifier"
PARAMS_KEY = "respan_params"
ENABLE_CONTENT_TRACING_KEY = "respan_enable_content_tracing"
PENDING_SPAN_LINKS_KEY = "respan_pending_span_links"

# Propagation attribute keys (used in propagate_attributes() kwargs and
# read_propagated_attributes() dispatch)
PROP_CUSTOMER_IDENTIFIER = "customer_identifier"
PROP_CUSTOMER_EMAIL = "customer_email"
PROP_CUSTOMER_NAME = "customer_name"
PROP_THREAD_IDENTIFIER = "thread_identifier"
PROP_CUSTOM_IDENTIFIER = "custom_identifier"
PROP_GROUP_IDENTIFIER = "group_identifier"
PROP_ENVIRONMENT = "environment"
PROP_METADATA = "metadata"
PROP_PROMPT = "prompt"
