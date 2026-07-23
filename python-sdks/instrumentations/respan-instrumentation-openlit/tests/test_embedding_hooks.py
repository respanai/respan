from __future__ import annotations

import json
from types import ModuleType, SimpleNamespace

import respan_instrumentation_openlit._embeddings as embeddings
from opentelemetry.semconv_ai import SpanAttributes


class RecordingSpan:
    def __init__(self) -> None:
        self.attributes: dict[str, object] = {}
        self.set_calls = 0

    def set_attribute(self, key: str, value: object) -> None:
        self.set_calls += 1
        self.attributes[key] = value


def _module(name: str) -> tuple[ModuleType, object]:
    module = ModuleType(name)

    def process_embedding_response(*args, **kwargs):
        del args
        return kwargs["response"]

    module.process_embedding_response = process_embedding_response
    return module, process_embedding_response


def test_sync_and_async_openai_hooks_capture_every_vector_dimension(
    monkeypatch,
) -> None:
    sync_module, _ = _module("sync_openai")
    async_module, _ = _module("async_openai")
    modules = {"sync_openai": sync_module, "async_openai": async_module}
    monkeypatch.setattr(embeddings, "_OPENAI_EMBEDDING_MODULES", tuple(modules))
    monkeypatch.setattr(
        embeddings.importlib, "import_module", lambda name: modules[name]
    )
    hooks = embeddings.install_openai_embedding_hooks(capture_content=True)
    vector = [index / 1000 for index in range(513)]

    sync_span = RecordingSpan()
    sync_response = {"data": [{"index": 0, "embedding": vector}]}
    result = sync_module.process_embedding_response(
        response=sync_response, span=sync_span
    )
    assert result is sync_response
    assert json.loads(sync_span.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]) == [
        vector
    ]

    async_span = RecordingSpan()
    async_response = SimpleNamespace(
        data=[
            SimpleNamespace(index=0, embedding=vector),
            SimpleNamespace(index=1, embedding=list(reversed(vector))),
        ]
    )
    result = async_module.process_embedding_response(
        response=async_response, span=async_span
    )
    assert result is async_response
    async_vectors = json.loads(
        async_span.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]
    )
    assert len(async_vectors) == 2
    assert len(async_vectors[0]) == 513
    assert async_vectors[0][-1] == vector[-1]
    assert async_vectors[1][0] == vector[-1]
    assert sync_span.set_calls == async_span.set_calls == 1

    embeddings.remove_openai_embedding_hooks(hooks)


def test_embedding_hook_honors_capture_content_false(monkeypatch) -> None:
    module, _ = _module("sync_openai")
    monkeypatch.setattr(embeddings, "_OPENAI_EMBEDDING_MODULES", ("sync_openai",))
    monkeypatch.setattr(embeddings.importlib, "import_module", lambda name: module)
    hooks = embeddings.install_openai_embedding_hooks(capture_content=False)
    span = RecordingSpan()

    module.process_embedding_response(
        response={"data": [{"embedding": [0.5] * 384}]}, span=span
    )

    assert span.attributes == {}
    assert span.set_calls == 0
    embeddings.remove_openai_embedding_hooks(hooks)


def test_embedding_hooks_restore_only_functions_the_adapter_owns(
    monkeypatch,
) -> None:
    sync_module, sync_original = _module("sync_openai")
    async_module, async_original = _module("async_openai")
    modules = {"sync_openai": sync_module, "async_openai": async_module}
    monkeypatch.setattr(embeddings, "_OPENAI_EMBEDDING_MODULES", tuple(modules))
    monkeypatch.setattr(
        embeddings.importlib, "import_module", lambda name: modules[name]
    )

    hooks = embeddings.install_openai_embedding_hooks(capture_content=True)
    assert sync_module.process_embedding_response is not sync_original
    assert async_module.process_embedding_response is not async_original

    def external_replacement(*args, **kwargs):
        del args, kwargs

    async_module.process_embedding_response = external_replacement
    embeddings.remove_openai_embedding_hooks(hooks)

    assert sync_module.process_embedding_response is sync_original
    assert async_module.process_embedding_response is external_replacement
