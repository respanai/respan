from types import SimpleNamespace

from respan_instrumentation_openlit import OpenLITInstrumentor


def test_embedding_hooks_follow_adapter_reference_count(monkeypatch) -> None:
    import respan_instrumentation_openlit._instrumentation as lifecycle

    class Active:
        _span_processors: tuple[object, ...] = ()

    class Provider:
        _active_span_processor = Active()

    provider = Provider()
    init_calls = 0
    install_calls: list[bool] = []
    removed_hooks: list[list[object]] = []
    hook = object()

    def init(**kwargs) -> None:
        nonlocal init_calls
        del kwargs
        init_calls += 1

    original_import_module = lifecycle.importlib.import_module

    def import_module(name: str):
        if name == "openlit":
            return SimpleNamespace(init=init)
        return original_import_module(name)

    monkeypatch.setattr(lifecycle.importlib, "import_module", import_module)
    monkeypatch.setattr(lifecycle, "_instrumentors", lambda: {})
    monkeypatch.setattr(lifecycle.trace, "get_tracer_provider", lambda: provider)
    monkeypatch.setattr(
        lifecycle,
        "install_openai_embedding_hooks",
        lambda *, capture_content: install_calls.append(capture_content) or [hook],
    )
    monkeypatch.setattr(
        lifecycle,
        "remove_openai_embedding_hooks",
        lambda hooks: removed_hooks.append(list(hooks)),
    )
    monkeypatch.setattr(lifecycle, "_REFCOUNT", 0)
    monkeypatch.setattr(lifecycle, "_PROCESSOR", None)
    monkeypatch.setattr(lifecycle, "_PROVIDER", None)
    monkeypatch.setattr(lifecycle, "_OWNED_INSTRUMENTORS", [])
    monkeypatch.setattr(lifecycle, "_EMBEDDING_HOOKS", [])

    first = OpenLITInstrumentor(capture_content=True)
    second = OpenLITInstrumentor(capture_content=False)
    first.activate()
    second.activate()

    assert init_calls == 1
    assert install_calls == [True]
    assert lifecycle._REFCOUNT == 2
    assert len(provider._active_span_processor._span_processors) == 1

    first.deactivate()
    assert lifecycle._REFCOUNT == 1
    assert removed_hooks == []
    assert len(provider._active_span_processor._span_processors) == 1

    second.deactivate()
    assert lifecycle._REFCOUNT == 0
    assert removed_hooks == [[hook]]
    assert provider._active_span_processor._span_processors == ()
