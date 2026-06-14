"""Chroma instrumentation constants."""

from typing import NamedTuple

from respan_sdk.constants.llm_logging import LOG_TYPE_TASK


CHROMA_INSTRUMENTATION_NAME = "chroma"
CHROMA_CLIENT_MODULE = "chromadb.api.client"
CHROMA_CLIENT_CLASS_NAME = "Client"
CHROMA_COLLECTION_MODULE = "chromadb.api.models.Collection"
CHROMA_COLLECTION_CLASS_NAME = "Collection"

MAX_ATTRIBUTE_CHARS = 16000
MAX_PREVIEW_CHARS = 500
MAX_PREVIEW_ITEMS = 5

EMBEDDING_KEYS = frozenset(
    {
        "embedding",
        "embeddings",
        "query_embedding",
        "query_embeddings",
    }
)
IMAGE_KEYS = frozenset(
    {
        "image",
        "images",
        "query_image",
        "query_images",
    }
)
DATA_KEYS = frozenset({"data"})


class OperationConfig(NamedTuple):
    log_type: str
    span_name: str
    entity_name: str


CLIENT_METHODS: dict[str, OperationConfig] = {
    "count_collections": OperationConfig(
        LOG_TYPE_TASK,
        "chroma.client.count_collections",
        "chroma.count_collections",
    ),
    "create_collection": OperationConfig(
        LOG_TYPE_TASK,
        "chroma.client.create_collection",
        "chroma.create_collection",
    ),
    "delete_collection": OperationConfig(
        LOG_TYPE_TASK,
        "chroma.client.delete_collection",
        "chroma.delete_collection",
    ),
    "get_collection": OperationConfig(
        LOG_TYPE_TASK,
        "chroma.client.get_collection",
        "chroma.get_collection",
    ),
    "get_or_create_collection": OperationConfig(
        LOG_TYPE_TASK,
        "chroma.client.get_or_create_collection",
        "chroma.get_or_create_collection",
    ),
    "get_version": OperationConfig(
        LOG_TYPE_TASK,
        "chroma.client.get_version",
        "chroma.get_version",
    ),
    "heartbeat": OperationConfig(
        LOG_TYPE_TASK,
        "chroma.client.heartbeat",
        "chroma.heartbeat",
    ),
    "list_collections": OperationConfig(
        LOG_TYPE_TASK,
        "chroma.client.list_collections",
        "chroma.list_collections",
    ),
}

COLLECTION_METHODS: dict[str, OperationConfig] = {
    "add": OperationConfig(LOG_TYPE_TASK, "chroma.collection.add", "chroma.add"),
    "count": OperationConfig(LOG_TYPE_TASK, "chroma.collection.count", "chroma.count"),
    "delete": OperationConfig(
        LOG_TYPE_TASK,
        "chroma.collection.delete",
        "chroma.delete",
    ),
    "fork": OperationConfig(LOG_TYPE_TASK, "chroma.collection.fork", "chroma.fork"),
    "get": OperationConfig(LOG_TYPE_TASK, "chroma.collection.get", "chroma.get"),
    "modify": OperationConfig(
        LOG_TYPE_TASK,
        "chroma.collection.modify",
        "chroma.modify",
    ),
    "peek": OperationConfig(LOG_TYPE_TASK, "chroma.collection.peek", "chroma.peek"),
    "query": OperationConfig(LOG_TYPE_TASK, "chroma.collection.query", "chroma.query"),
    "search": OperationConfig(
        LOG_TYPE_TASK,
        "chroma.collection.search",
        "chroma.search",
    ),
    "update": OperationConfig(
        LOG_TYPE_TASK,
        "chroma.collection.update",
        "chroma.update",
    ),
    "upsert": OperationConfig(
        LOG_TYPE_TASK,
        "chroma.collection.upsert",
        "chroma.upsert",
    ),
}
