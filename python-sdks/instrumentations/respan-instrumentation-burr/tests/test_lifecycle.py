from types import SimpleNamespace

from respan_instrumentation_burr import BurrInstrumentor
import respan_instrumentation_burr._instrumentation as instrumentation


def test_builder_receives_only_one_respan_adapter(monkeypatch) -> None:
    adapter = SimpleNamespace(
        capture_content=True,
        enabled=True,
        _respan_burr_lifecycle_adapter=True,
    )
    monkeypatch.setattr(instrumentation, "_ADAPTER", adapter)
    builder = SimpleNamespace(lifecycle_adapters=[])
    instrumentation._ensure_adapter(builder)
    instrumentation._ensure_adapter(builder)
    assert builder.lifecycle_adapters == [adapter]


def test_activate_and_deactivate_patch_builder(monkeypatch) -> None:
    wrapped = []
    unwrapped = []
    monkeypatch.setattr(
        instrumentation.importlib,
        "import_module",
        lambda name: SimpleNamespace(),
    )
    monkeypatch.setattr(
        instrumentation,
        "wrap_function_wrapper",
        lambda module, target, wrapper: wrapped.append((module, target)),
    )
    monkeypatch.setattr(
        instrumentation,
        "unwrap",
        lambda module, target: unwrapped.append((module, target)),
    )
    monkeypatch.setattr(
        instrumentation,
        "BurrLifecycleAdapter",
        lambda capture_content: SimpleNamespace(
            capture_content=capture_content,
            enabled=True,
            _respan_burr_lifecycle_adapter=True,
        ),
    )
    monkeypatch.setattr(instrumentation, "_ACTIVATION_COUNT", 0)
    monkeypatch.setattr(instrumentation, "_PATCHED", False)
    monkeypatch.setattr(instrumentation, "_ADAPTER", None)

    first = BurrInstrumentor()
    second = BurrInstrumentor(capture_content=False)
    first.activate()
    second.activate()
    assert len(wrapped) == 1
    assert instrumentation._ACTIVATION_COUNT == 2

    first.deactivate()
    assert unwrapped == []
    second.deactivate()
    assert unwrapped == [
        (
            instrumentation.BURR_APPLICATION_MODULE,
            instrumentation.BURR_APPLICATION_BUILD_TARGET,
        )
    ]
