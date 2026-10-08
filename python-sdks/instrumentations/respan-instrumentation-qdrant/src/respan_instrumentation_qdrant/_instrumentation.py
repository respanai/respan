"""Native Qdrant boundaries retain actual SDK engine and client semantics."""

from ._constants import QDRANT_OPERATIONS
from ._native_instrumentation import NativeClientInstrumentor, PatchSpec


class QdrantInstrumentor(NativeClientInstrumentor):
    name = "qdrant"
    vendor = "qdrant"
    patches = (
        PatchSpec("qdrant_client", "QdrantClient", QDRANT_OPERATIONS),
        PatchSpec("qdrant_client", "AsyncQdrantClient", QDRANT_OPERATIONS),
    )
