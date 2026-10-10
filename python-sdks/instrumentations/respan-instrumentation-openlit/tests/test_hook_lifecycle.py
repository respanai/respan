from concurrent.futures import ThreadPoolExecutor
from types import ModuleType, SimpleNamespace

import pytest
from respan_instrumentation_openlit import OpenLITInstrumentor


def test_embedding_hooks_follow_adapter_reference_count(monkeypatch) -> None:
    import respan_instrumentation_openlit._instrumentation as lifecycle

    foreign_processor = object()

    class Active:
        _span_processors: tuple[object, ...] = (foreign_processor,)

    class Provider:
        _active_span_processor = Active()

    provider = Provider()
    init_calls = 0
    install_calls: list[bool] = []
    removed_hooks: list[list[object]] = []
    hook = object()

    init_kwargs: dict[str, object] = {}

    def init(**kwargs) -> None:
        nonlocal init_calls
        init_kwargs.update(kwargs)
        init_calls += 1

    original_import_module = lifecycle.importlib.import_module

    def import_module(name: str):
        if name == "openlit":
            return SimpleNamespace(init=init)
        return original_import_module(name)

    monkeypatch.setattr(lifecycle.importlib, "import_module", import_module)
    monkeypatch.setattr(lifecycle, "_instrumentors", dict)
    monkeypatch.setattr(lifecycle, "snapshot_openai_resource_methods", dict)
    monkeypatch.setattr(lifecycle, "capture_openai_patches", lambda before: [])
    monkeypatch.setattr(lifecycle, "restore_openai_patches", lambda patches: None)
    monkeypatch.setattr(lifecycle, "install_openai_request_hooks", lambda **kwargs: [])
    monkeypatch.setattr(lifecycle, "remove_openai_request_hooks", lambda hooks: None)
    monkeypatch.setattr(
        lifecycle, "install_openai_stream_factory_hooks", lambda **kwargs: []
    )
    monkeypatch.setattr(
        lifecycle, "remove_openai_stream_factory_hooks", lambda hooks: None
    )
    monkeypatch.setattr(lifecycle, "install_openai_stream_usage_hooks", list)
    monkeypatch.setattr(
        lifecycle, "remove_openai_stream_usage_hooks", lambda hooks: None
    )
    monkeypatch.setattr(lifecycle.trace, "get_tracer_provider", lambda: provider)
    monkeypatch.setattr(
        lifecycle,
        "install_openai_embedding_hooks",
        lambda *, capture_content, max_content_length: (
            install_calls.append(capture_content) or [hook]
        ),
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
    monkeypatch.setattr(lifecycle, "_REQUEST_HOOKS", [])
    monkeypatch.setattr(lifecycle, "_STREAM_USAGE_HOOKS", [])
    monkeypatch.setattr(lifecycle, "_OPENAI_PATCHES", [])
    monkeypatch.setattr(lifecycle, "_CONFIG", None)

    first = OpenLITInstrumentor(capture_content=True)
    second = OpenLITInstrumentor(capture_content=True)
    first.activate()
    second.activate()

    assert init_calls == 1
    assert init_kwargs["max_content_length"] == 16_000
    assert set(init_kwargs["disabled_instrumentors"]) >= {
        "httpx",
        "requests",
        "urllib",
        "urllib3",
    }
    assert install_calls == [True]
    assert lifecycle._REFCOUNT == 2
    assert provider._active_span_processor._span_processors[1:] == (foreign_processor,)

    first.deactivate()
    assert lifecycle._REFCOUNT == 1
    assert removed_hooks == []
    assert provider._active_span_processor._span_processors[1:] == (foreign_processor,)

    second.deactivate()
    assert lifecycle._REFCOUNT == 0
    assert removed_hooks == [[hook]]
    assert provider._active_span_processor._span_processors == (foreign_processor,)


def test_concurrent_same_instance_activation_is_idempotent(monkeypatch) -> None:
    import respan_instrumentation_openlit._instrumentation as lifecycle

    class Active:
        _span_processors: tuple[object, ...] = ()

    class Provider:
        _active_span_processor = Active()

    stable_provider = Provider()

    init_calls = 0

    def init(**kwargs) -> None:
        nonlocal init_calls
        del kwargs
        init_calls += 1

    original_import_module = lifecycle.importlib.import_module
    monkeypatch.setattr(
        lifecycle.importlib,
        "import_module",
        lambda name: (
            SimpleNamespace(init=init)
            if name == "openlit"
            else original_import_module(name)
        ),
    )
    monkeypatch.setattr(lifecycle, "_instrumentors", dict)
    monkeypatch.setattr(lifecycle, "snapshot_openai_resource_methods", dict)
    monkeypatch.setattr(lifecycle, "capture_openai_patches", lambda before: [])
    monkeypatch.setattr(lifecycle, "restore_openai_patches", lambda patches: None)
    monkeypatch.setattr(lifecycle, "install_openai_request_hooks", lambda **kwargs: [])
    monkeypatch.setattr(lifecycle, "remove_openai_request_hooks", lambda hooks: None)
    monkeypatch.setattr(
        lifecycle, "install_openai_stream_factory_hooks", lambda **kwargs: []
    )
    monkeypatch.setattr(
        lifecycle, "remove_openai_stream_factory_hooks", lambda hooks: None
    )
    monkeypatch.setattr(lifecycle, "install_openai_stream_usage_hooks", list)
    monkeypatch.setattr(
        lifecycle, "remove_openai_stream_usage_hooks", lambda hooks: None
    )
    monkeypatch.setattr(
        lifecycle, "install_openai_embedding_hooks", lambda **kwargs: []
    )
    monkeypatch.setattr(lifecycle, "remove_openai_embedding_hooks", lambda hooks: None)
    monkeypatch.setattr(lifecycle.trace, "get_tracer_provider", lambda: stable_provider)
    for name, value in (
        ("_REFCOUNT", 0),
        ("_PROCESSOR", None),
        ("_PROVIDER", None),
        ("_OWNED_INSTRUMENTORS", []),
        ("_EMBEDDING_HOOKS", []),
        ("_REQUEST_HOOKS", []),
        ("_STREAM_USAGE_HOOKS", []),
        ("_OPENAI_PATCHES", []),
        ("_CONFIG", None),
    ):
        monkeypatch.setattr(lifecycle, name, value)

    instrumentor = OpenLITInstrumentor()
    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(lambda _: instrumentor.activate(), range(32)))
    assert init_calls == 1
    assert lifecycle._REFCOUNT == 1
    instrumentor.deactivate()
    assert lifecycle._REFCOUNT == 0

    instrumentors = [OpenLITInstrumentor() for _ in range(8)]
    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(lambda item: item.activate(), instrumentors))
    assert init_calls == 2
    assert lifecycle._REFCOUNT == len(instrumentors)
    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(lambda item: item.deactivate(), instrumentors))
    assert lifecycle._REFCOUNT == 0


def test_active_configuration_mismatch_is_rejected(monkeypatch) -> None:
    import respan_instrumentation_openlit._instrumentation as lifecycle

    class Active:
        _span_processors: tuple[object, ...] = ()

    class Provider:
        _active_span_processor = Active()

    stable_provider = Provider()

    original_import_module = lifecycle.importlib.import_module
    monkeypatch.setattr(
        lifecycle.importlib,
        "import_module",
        lambda name: (
            SimpleNamespace(init=lambda **kwargs: None)
            if name == "openlit"
            else original_import_module(name)
        ),
    )
    monkeypatch.setattr(lifecycle, "_instrumentors", dict)
    monkeypatch.setattr(lifecycle, "snapshot_openai_resource_methods", dict)
    monkeypatch.setattr(lifecycle, "capture_openai_patches", lambda before: [])
    monkeypatch.setattr(lifecycle, "restore_openai_patches", lambda patches: None)
    monkeypatch.setattr(lifecycle, "install_openai_request_hooks", lambda **kwargs: [])
    monkeypatch.setattr(lifecycle, "remove_openai_request_hooks", lambda hooks: None)
    monkeypatch.setattr(
        lifecycle, "install_openai_stream_factory_hooks", lambda **kwargs: []
    )
    monkeypatch.setattr(
        lifecycle, "remove_openai_stream_factory_hooks", lambda hooks: None
    )
    monkeypatch.setattr(lifecycle, "install_openai_stream_usage_hooks", list)
    monkeypatch.setattr(
        lifecycle, "remove_openai_stream_usage_hooks", lambda hooks: None
    )
    monkeypatch.setattr(
        lifecycle, "install_openai_embedding_hooks", lambda **kwargs: []
    )
    monkeypatch.setattr(lifecycle, "remove_openai_embedding_hooks", lambda hooks: None)
    monkeypatch.setattr(lifecycle.trace, "get_tracer_provider", lambda: stable_provider)
    for name, value in (
        ("_REFCOUNT", 0),
        ("_PROCESSOR", None),
        ("_PROVIDER", None),
        ("_OWNED_INSTRUMENTORS", []),
        ("_EMBEDDING_HOOKS", []),
        ("_REQUEST_HOOKS", []),
        ("_STREAM_USAGE_HOOKS", []),
        ("_OPENAI_PATCHES", []),
        ("_CONFIG", None),
    ):
        monkeypatch.setattr(lifecycle, name, value)

    first = OpenLITInstrumentor(capture_content=True)
    first.activate()
    second = OpenLITInstrumentor(capture_content=False)
    with pytest.raises(RuntimeError, match="different adapter configuration"):
        second.activate()
    assert lifecycle._REFCOUNT == 1
    assert not second._is_instrumented
    first.deactivate()
    assert lifecycle._REFCOUNT == 0


def test_partial_activation_rolls_back_and_raises(monkeypatch) -> None:
    import respan_instrumentation_openlit._instrumentation as lifecycle

    class Upstream:
        _is_instrumented_by_opentelemetry = False
        uninstrument_calls = 0

        def uninstrument(self) -> None:
            self._is_instrumented_by_opentelemetry = False
            self.uninstrument_calls += 1

    upstream = Upstream()
    request_hook = object()
    factory_hook = object()
    restored: list[str] = []

    def init(**kwargs) -> None:
        del kwargs
        upstream._is_instrumented_by_opentelemetry = True
        raise ValueError("partial OpenLIT install")

    original_import_module = lifecycle.importlib.import_module
    monkeypatch.setattr(
        lifecycle.importlib,
        "import_module",
        lambda name: (
            SimpleNamespace(init=init)
            if name == "openlit"
            else original_import_module(name)
        ),
    )
    monkeypatch.setattr(lifecycle, "_instrumentors", lambda: {"openai": upstream})
    monkeypatch.setattr(lifecycle, "snapshot_openai_resource_methods", dict)
    monkeypatch.setattr(lifecycle, "capture_openai_patches", lambda before: [])
    monkeypatch.setattr(
        lifecycle,
        "restore_openai_patches",
        lambda patches: restored.append("patches"),
    )
    monkeypatch.setattr(
        lifecycle, "install_openai_request_hooks", lambda **kwargs: [request_hook]
    )
    monkeypatch.setattr(
        lifecycle,
        "remove_openai_request_hooks",
        lambda hooks: restored.append("request") if hooks == [request_hook] else None,
    )
    monkeypatch.setattr(
        lifecycle,
        "install_openai_stream_factory_hooks",
        lambda **kwargs: [factory_hook],
    )
    monkeypatch.setattr(
        lifecycle,
        "remove_openai_stream_factory_hooks",
        lambda hooks: restored.append("factory") if hooks == [factory_hook] else None,
    )
    monkeypatch.setattr(lifecycle, "install_openai_stream_usage_hooks", list)
    monkeypatch.setattr(
        lifecycle, "remove_openai_stream_usage_hooks", lambda hooks: None
    )
    monkeypatch.setattr(
        lifecycle, "install_openai_embedding_hooks", lambda **kwargs: []
    )
    monkeypatch.setattr(lifecycle, "remove_openai_embedding_hooks", lambda hooks: None)
    for name, value in (
        ("_REFCOUNT", 0),
        ("_PROCESSOR", None),
        ("_PROVIDER", None),
        ("_OWNED_INSTRUMENTORS", []),
        ("_EMBEDDING_HOOKS", []),
        ("_REQUEST_HOOKS", []),
        ("_STREAM_USAGE_HOOKS", []),
        ("_OPENAI_PATCHES", []),
        ("_CONFIG", None),
    ):
        monkeypatch.setattr(lifecycle, name, value)

    instrumentor = OpenLITInstrumentor()
    with pytest.raises(RuntimeError, match="Failed to activate OpenLIT"):
        instrumentor.activate()
    assert upstream.uninstrument_calls == 1
    assert "factory" in restored
    assert "patches" in restored
    assert "request" in restored
    assert lifecycle._REFCOUNT == 0
    assert lifecycle._CONFIG is None
    assert not instrumentor._is_instrumented


def test_restore_openai_patches_preserves_later_foreign_patch() -> None:
    from respan_instrumentation_openlit._openai_hooks import (
        remove_openai_request_hooks,
        restore_openai_patches,
    )
    from wrapt import FunctionWrapper

    def original(self):
        return self

    def installed(self):
        return self

    def foreign(self):
        return self

    class Resource:
        method = installed

    foreign_wrapper = FunctionWrapper(
        installed, lambda wrapped, _, args, kwargs: wrapped(*args, **kwargs)
    )
    Resource.method = foreign_wrapper
    restore_openai_patches([(Resource, "method", original, installed)])
    assert vars(Resource)["method"] is foreign_wrapper
    assert vars(Resource)["method"].__wrapped__ is original

    Resource.method = FunctionWrapper(
        installed,
        lambda wrapped, _, args, kwargs: wrapped(*args, **kwargs),
    )
    request_hook = installed
    remove_openai_request_hooks([(Resource, "method", original, request_hook)])
    assert isinstance(vars(Resource)["method"], FunctionWrapper)
    assert vars(Resource)["method"].__wrapped__ is original


def test_request_hook_partial_install_is_transactional(monkeypatch) -> None:
    import respan_instrumentation_openlit._openai_hooks as hooks

    def first_create(self):
        return self

    def second_create(self):
        return self

    class First:
        create = first_create

    class RejectCreate(type):
        def __setattr__(cls, name, value):
            if name == "create":
                raise RuntimeError("reject partial hook")
            return super().__setattr__(name, value)

    class Second(metaclass=RejectCreate):
        create = second_create

    first_module = ModuleType("first_resource")
    first_module.First = First
    second_module = ModuleType("second_resource")
    second_module.Second = Second
    modules = {"first_resource": first_module, "second_resource": second_module}
    monkeypatch.setattr(
        hooks,
        "_REQUEST_TARGETS",
        (
            ("first_resource", "First", "create", False, False),
            ("second_resource", "Second", "create", False, False),
        ),
    )
    monkeypatch.setattr(hooks.importlib, "import_module", lambda name: modules[name])

    with pytest.raises(RuntimeError, match="reject partial hook"):
        hooks.install_openai_request_hooks(
            capture_content=True,
            max_content_length=1_024,
        )
    assert vars(First)["create"] is first_create
