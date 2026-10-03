"""Capture vectors on PydanticAI's native embedding span before it ends."""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from functools import wraps

from opentelemetry import trace
from opentelemetry.semconv_ai import SpanAttributes

logger = logging.getLogger(__name__)


def load_embedder_class():
    try:
        from pydantic_ai.embeddings import Embedder
    except ImportError:
        return None
    return Embedder


def install_embedding_capture():
    try:
        from pydantic_ai.embeddings.instrumented import InstrumentedEmbeddingModel
    except ImportError:
        return lambda: None

    original = InstrumentedEmbeddingModel._instrument

    @contextmanager
    @wraps(original)
    def instrument(self, *args, **kwargs):
        with original(self, *args, **kwargs) as finish:

            def capture(result):
                finish(result)
                if self.instrumentation_settings.include_content:
                    try:
                        # The SDK updates its initial attribute dict after span
                        # creation. Write vectors onto the live span explicitly.
                        # Do not use bounded message serialization for vectors.
                        trace.get_current_span().set_attribute(
                            SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
                            json.dumps([list(vector) for vector in result.embeddings]),
                        )
                    except Exception:
                        logger.debug(
                            "Failed to capture embedding vectors", exc_info=True
                        )

            yield capture

    InstrumentedEmbeddingModel._instrument = instrument

    def restore():
        if InstrumentedEmbeddingModel._instrument is instrument:
            InstrumentedEmbeddingModel._instrument = original

    return restore
