"""Elasticsearch instrumentation-local constants."""

from respan_sdk.constants.llm_logging import LOG_TYPE_TASK


ELASTICSEARCH_INSTRUMENTATION_NAME = "elasticsearch"
ELASTIC_TRANSPORT_MODULE = "elastic_transport"
ELASTIC_TRANSPORT_TARGETS = (
    ("Transport.perform_request", False),
    ("AsyncTransport.perform_request", True),
)

ELASTICSEARCH_METHOD = "elasticsearch.method"
ELASTICSEARCH_TARGET = "elasticsearch.target"
ELASTICSEARCH_STATUS_CODE = "elasticsearch.status_code"

MAX_ATTRIBUTE_CHARS = 16_000
MAX_COLLECTION_ITEMS = 50
MAX_SERIALIZATION_DEPTH = 8

TASK_LOG_TYPE = LOG_TYPE_TASK
