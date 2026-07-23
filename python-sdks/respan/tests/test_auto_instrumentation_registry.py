import contextlib
import importlib.util
import sys
import types
import unittest
from unittest.mock import patch


def _install_respan_tracing_stub():
    if "respan_tracing" in sys.modules:
        return

    class _PropagatedAttributes:
        def set(self, _value):
            return None

    class _FakeTelemetry:
        def __init__(self, **_kwargs):
            pass

        def flush(self):
            pass

    @contextlib.contextmanager
    def _propagate_attributes(**_kwargs):
        yield

    def _decorator(*args, **_kwargs):
        if args and callable(args[0]):
            return args[0]

        def _wrap(fn):
            return fn

        return _wrap

    respan_tracing = types.ModuleType("respan_tracing")
    respan_tracing.RespanTelemetry = _FakeTelemetry
    respan_tracing.workflow = _decorator
    respan_tracing.task = _decorator
    respan_tracing.agent = _decorator
    respan_tracing.tool = _decorator
    respan_tracing.RespanClient = object
    respan_tracing.get_client = lambda: None
    respan_tracing.respan_span_attributes = {}

    span_factory = types.ModuleType("respan_tracing.utils.span_factory")
    span_factory._PROPAGATED_ATTRIBUTES = _PropagatedAttributes()
    span_factory.build_readable_span = lambda *args, **kwargs: {
        "args": args,
        "kwargs": kwargs,
    }
    span_factory.inject_span = lambda _span: None
    span_factory.propagate_attributes = _propagate_attributes

    exporters = types.ModuleType("respan_tracing.exporters")
    exporters.propagate_attributes = _propagate_attributes

    opentelemetry = types.ModuleType("opentelemetry")
    trace = types.ModuleType("opentelemetry.trace")
    trace.get_tracer_provider = lambda: None
    opentelemetry.trace = trace

    sys.modules["opentelemetry"] = opentelemetry
    sys.modules["opentelemetry.trace"] = trace
    sys.modules["respan_tracing"] = respan_tracing
    sys.modules["respan_tracing.utils"] = types.ModuleType(
        "respan_tracing.utils"
    )
    sys.modules["respan_tracing.utils.span_factory"] = span_factory
    sys.modules["respan_tracing.exporters"] = exporters


try:
    _HAS_OTEL = importlib.util.find_spec("opentelemetry.sdk.trace") is not None
except ModuleNotFoundError:
    _HAS_OTEL = False

if not _HAS_OTEL:
    _install_respan_tracing_stub()

from respan import _core
from respan import _auto_instrumentation_registry as registry
from respan._auto_instrumentation_registry import (
    AutoInstrumentationActivation,
    AutoInstrumentationSpec,
    AutoInstrumentationStatus,
    activate_auto_instrumentations,
    list_auto_instrumentation_specs,
)


def _spec(**overrides):
    values = {
        "id": "fake",
        "category": "direct_llm",
        "provider": "Fake",
        "sdk_package": "fake-sdk",
        "instrumentation_package": "respan-instrumentation-fake",
        "entry_point": "fake",
        "import_path": "missing_fake:FakeInstrumentor",
    }
    values.update(overrides)
    return AutoInstrumentationSpec(**values)


class FakeEntryPoint:
    def __init__(self, instrumentor_class):
        self.name = "fake"
        self._instrumentor_class = instrumentor_class
        self.loaded = False

    def load(self):
        self.loaded = True
        return self._instrumentor_class


class FakeInstrumentor:
    name = "fake"
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self._is_instrumented = False
        FakeInstrumentor.instances.append(self)

    def activate(self):
        self._is_instrumented = True

    def deactivate(self):
        self._is_instrumented = False


class TestAutoInstrumentationRegistry(unittest.TestCase):
    def setUp(self):
        FakeInstrumentor.instances.clear()

    def test_auto_registry_activates_loaded_entry_point(self):
        entry_point = FakeEntryPoint(FakeInstrumentor)

        with patch.object(
            registry,
            "_discover_entry_points",
            return_value={"fake": entry_point},
        ):
            activations = activate_auto_instrumentations(
                registry=(
                    _spec(constructor_kwargs=(("marker", "value"),)),
                )
            )

        self.assertTrue(entry_point.loaded)
        self.assertEqual(len(activations), 1)
        self.assertEqual(activations[0].status.status, "enabled")
        self.assertIs(activations[0].instrumentor, FakeInstrumentor.instances[0])
        self.assertEqual(FakeInstrumentor.instances[0].kwargs, {"marker": "value"})

    def test_auto_registry_reports_disabled_without_loading(self):
        entry_point = FakeEntryPoint(FakeInstrumentor)

        with patch.object(
            registry,
            "_discover_entry_points",
            return_value={"fake": entry_point},
        ):
            activations = activate_auto_instrumentations(
                registry=(
                    _spec(
                        enabled_by_default=False,
                        auto_disabled_reason="manual only",
                    ),
                )
            )

        self.assertFalse(entry_point.loaded)
        self.assertEqual(activations[0].status.status, "disabled")
        self.assertEqual(activations[0].status.reason, "manual only")

    def test_auto_registry_reports_missing_package(self):
        with patch.object(registry, "_discover_entry_points", return_value={}):
            activations = activate_auto_instrumentations(registry=(_spec(),))

        self.assertEqual(activations[0].status.status, "missing")
        self.assertIn(
            "respan-instrumentation-fake is not installed",
            activations[0].status.reason,
        )

    def test_auto_registry_skips_already_activated_names(self):
        entry_point = FakeEntryPoint(FakeInstrumentor)

        with patch.object(
            registry,
            "_discover_entry_points",
            return_value={"fake": entry_point},
        ):
            activations = activate_auto_instrumentations(
                already_activated=("fake",),
                registry=(_spec(),),
            )

        self.assertFalse(entry_point.loaded)
        self.assertEqual(activations[0].status.status, "disabled")
        self.assertEqual(activations[0].status.reason, "already activated explicitly")

    def test_only_verified_gateway_sdks_are_enabled_by_default(self):
        enabled_ids = [
            spec.id for spec in list_auto_instrumentation_specs(include_disabled=False)
        ]

        self.assertEqual(
            enabled_ids,
            [
                "openai",
                "anthropic",
                "google-genai",
                "together",
                "mistralai",
                "litellm",
            ],
        )

    def test_openrouter_is_opt_in_but_keeps_url_marker_only_normalization(self):
        openrouter = next(
            spec for spec in list_auto_instrumentation_specs() if spec.id == "openrouter"
        )

        self.assertFalse(openrouter.enabled_by_default)
        self.assertEqual(
            dict(openrouter.constructor_kwargs),
            {"normalize_all_openai_spans": False},
        )

    def test_non_gateway_native_sdks_are_not_enabled_by_default(self):
        specs = {spec.id: spec for spec in list_auto_instrumentation_specs()}

        for spec_id in (
            "vertexai",
            "aws-bedrock",
            "cohere",
            "groq",
            "ollama",
            "aleph-alpha",
            "huggingface",
            "replicate",
            "sagemaker",
            "watsonx",
            "writer",
            "portkey",
        ):
            self.assertFalse(specs[spec_id].enabled_by_default, spec_id)
            self.assertTrue(specs[spec_id].auto_disabled_reason, spec_id)

    def test_framework_and_agent_instrumentations_are_not_enabled_by_default(self):
        specs = {spec.id: spec for spec in list_auto_instrumentation_specs()}

        self.assertFalse(specs["langchain"].enabled_by_default)
        self.assertFalse(specs["openai-agents"].enabled_by_default)
        self.assertFalse(specs["pydantic-ai"].enabled_by_default)
        self.assertFalse(specs["mcp"].enabled_by_default)

    def test_respan_auto_mode_does_not_forward_broad_otel_auto(self):
        captured = {}
        auto_called = {}

        class FakeTelemetry:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            def flush(self):
                pass

        fake_instrumentor = FakeInstrumentor()

        def fake_activate_auto_instrumentations(*, already_activated):
            auto_called["already_activated"] = tuple(already_activated)
            return (
                AutoInstrumentationActivation(
                    status=AutoInstrumentationStatus(
                        id="fake",
                        name="fake",
                        status="enabled",
                        provider="Fake",
                        sdk_package="fake-sdk",
                        instrumentation_package="respan-instrumentation-fake",
                    ),
                    instrumentor=fake_instrumentor,
                ),
            )

        with patch.object(_core, "RespanTelemetry", FakeTelemetry), patch.object(
            _core,
            "activate_auto_instrumentations",
            fake_activate_auto_instrumentations,
        ):
            respan = _core.Respan(api_key="test", is_auto_instrument=True)

        self.assertFalse(captured["is_auto_instrument"])
        self.assertEqual(auto_called["already_activated"], ())
        self.assertEqual(respan.auto_instrumentation_status[0].status, "enabled")
        self.assertIs(respan._instrumentations["fake"], fake_instrumentor)

    def test_respan_explicit_instrumentations_disable_auto_by_default(self):
        captured = {}

        class FakeTelemetry:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        class ExplicitInstrumentor:
            name = "explicit"

            def __init__(self):
                self._is_instrumented = False

            def activate(self):
                self._is_instrumented = True

        def fail_auto(**_kwargs):
            raise AssertionError("auto-instrumentation should not run")

        with patch.object(_core, "RespanTelemetry", FakeTelemetry), patch.object(
            _core,
            "activate_auto_instrumentations",
            fail_auto,
        ):
            respan = _core.Respan(
                api_key="test",
                instrumentations=[ExplicitInstrumentor()],
            )

        self.assertFalse(captured["is_auto_instrument"])
        self.assertEqual(respan.auto_instrumentation_status, ())
        self.assertIn("explicit", respan._instrumentations)

    def test_shutdown_deactivates_instrumentations_and_flushes_telemetry(self):
        events = []

        class FakeTelemetry:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def flush(self):
                events.append("flush")

        class ExplicitInstrumentor:
            name = "explicit"

            def __init__(self):
                self._is_instrumented = False

            def activate(self):
                self._is_instrumented = True

            def deactivate(self):
                events.append("deactivate")
                self._is_instrumented = False

        with patch.object(_core, "RespanTelemetry", FakeTelemetry):
            respan = _core.Respan(
                api_key="test",
                instrumentations=[ExplicitInstrumentor()],
            )
            respan.shutdown()

        self.assertEqual(events, ["deactivate", "flush"])
        self.assertEqual(respan._instrumentations, {})


if __name__ == "__main__":
    unittest.main()
