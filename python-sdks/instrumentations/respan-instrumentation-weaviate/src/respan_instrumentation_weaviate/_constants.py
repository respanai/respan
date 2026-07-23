"""Weaviate SDK-specific instrumentation constants."""

from typing import NamedTuple


class PatchSpec(NamedTuple):
    module: str
    class_name: str
    label: str
    methods: tuple[str, ...]
    is_async: bool = False


WEAVIATE_INSTRUMENTATION_NAME = "weaviate"

COLLECTION_OPERATIONS = (
    "create",
    "create_from_config",
    "create_from_dict",
    "delete",
    "delete_all",
    "exists",
    "export_config",
    "list_all",
)
DATA_OPERATIONS = (
    "delete_by_id",
    "delete_many",
    "exists",
    "ingest",
    "insert",
    "insert_many",
    "reference_add",
    "reference_add_many",
    "reference_delete",
    "reference_replace",
    "replace",
    "update",
)
QUERY_OPERATIONS = (
    "bm25",
    "fetch_object_by_id",
    "fetch_objects",
    "fetch_objects_by_ids",
    "hybrid",
    "near_image",
    "near_media",
    "near_object",
    "near_text",
    "near_vector",
)
AGGREGATE_OPERATIONS = (
    "hybrid",
    "near_image",
    "near_object",
    "near_text",
    "near_vector",
    "over_all",
)
CONFIG_OPERATIONS = (
    "add_property",
    "add_reference",
    "add_vector",
    "delete_property_index",
    "get",
    "get_shards",
    "update",
    "update_shards",
)
BATCH_OPERATIONS = ("add_object", "add_reference", "flush")
TENANT_OPERATIONS = (
    "create",
    "exists",
    "get",
    "get_by_name",
    "get_by_names",
    "remove",
    "update",
)

WEAVIATE_PATCH_SPECS = (
    PatchSpec(
        "weaviate.collections.collections.sync",
        "_Collections",
        "collections",
        COLLECTION_OPERATIONS,
    ),
    PatchSpec(
        "weaviate.collections.collections.async_",
        "_CollectionsAsync",
        "collections",
        COLLECTION_OPERATIONS,
        True,
    ),
    PatchSpec(
        "weaviate.collections.data.sync",
        "_DataCollection",
        "data",
        DATA_OPERATIONS,
    ),
    PatchSpec(
        "weaviate.collections.data.async_",
        "_DataCollectionAsync",
        "data",
        DATA_OPERATIONS,
        True,
    ),
    PatchSpec(
        "weaviate.collections.query",
        "_QueryCollection",
        "query",
        QUERY_OPERATIONS,
    ),
    PatchSpec(
        "weaviate.collections.query",
        "_QueryCollectionAsync",
        "query",
        QUERY_OPERATIONS,
        True,
    ),
    PatchSpec(
        "weaviate.collections.aggregate",
        "_AggregateCollection",
        "aggregate",
        AGGREGATE_OPERATIONS,
    ),
    PatchSpec(
        "weaviate.collections.aggregate",
        "_AggregateCollectionAsync",
        "aggregate",
        AGGREGATE_OPERATIONS,
        True,
    ),
    PatchSpec(
        "weaviate.collections.config.sync",
        "_ConfigCollection",
        "config",
        CONFIG_OPERATIONS,
    ),
    PatchSpec(
        "weaviate.collections.config.async_",
        "_ConfigCollectionAsync",
        "config",
        CONFIG_OPERATIONS,
        True,
    ),
    PatchSpec(
        "weaviate.collections.batch.collection",
        "_BatchCollection",
        "batch",
        BATCH_OPERATIONS,
    ),
    PatchSpec(
        "weaviate.collections.batch.collection",
        "_BatchCollectionSync",
        "batch",
        BATCH_OPERATIONS,
    ),
    PatchSpec(
        "weaviate.collections.batch.collection",
        "_BatchCollectionAsync",
        "batch",
        BATCH_OPERATIONS,
        True,
    ),
    PatchSpec(
        "weaviate.collections.tenants.sync",
        "_Tenants",
        "tenants",
        TENANT_OPERATIONS,
    ),
    PatchSpec(
        "weaviate.collections.tenants.async_",
        "_TenantsAsync",
        "tenants",
        TENANT_OPERATIONS,
        True,
    ),
)

MAX_ATTRIBUTE_CHARS = 16_000
MAX_PREVIEW_ITEMS = 64
SENSITIVE_KEY_PARTS = (
    "api_key",
    "authorization",
    "password",
    "secret",
    "token",
)
