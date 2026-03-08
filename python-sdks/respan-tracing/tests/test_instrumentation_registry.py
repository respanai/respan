"""Tests for the registry-based instrumentation system.

Validates that the registry pattern correctly replaces the old
800-line boilerplate with a data-driven approach.
"""
import pytest
from unittest.mock import patch, MagicMock
from respan_tracing.instruments import Instruments
from respan_tracing.utils.instrumentation import (
    INSTRUMENT_REGISTRY,
    InstrumentConfig,
    _POST_INIT_HOOKS,
    _init_single_instrument,
    init_instrumentations,
    is_package_installed,
)

_MOD = "respan_tracing.utils.instrumentation"
_MOCK_IS_INSTALLED = f"{_MOD}.is_package_installed"
_MOCK_IMPORT_MODULE = f"{_MOD}.importlib.import_module"
_MOCK_INIT_SINGLE = f"{_MOD}._init_single_instrument"


class TestInstrumentRegistry:
    """Registry completeness and correctness."""

    def test_every_enum_has_registry_entry(self):
        """Every Instruments enum value must have a registry entry."""
        missing = []
        for instrument in Instruments:
            if instrument not in INSTRUMENT_REGISTRY:
                missing.append(instrument)
        assert missing == [], f"Missing registry entries: {missing}"

    def test_registry_has_no_extra_keys(self):
        """Registry should not contain keys that aren't in the enum."""
        for key in INSTRUMENT_REGISTRY:
            assert isinstance(key, Instruments), f"Extra key: {key}"

    def test_all_configs_are_instrumentconfig(self):
        """Every registry value must be an InstrumentConfig."""
        for instrument, config in INSTRUMENT_REGISTRY.items():
            assert isinstance(config, InstrumentConfig), (
                f"{instrument} has wrong type: {type(config)}"
            )

    def test_all_configs_have_required_fields(self):
        """Every config must have module and class_name."""
        for instrument, config in INSTRUMENT_REGISTRY.items():
            assert config.module, f"{instrument} missing module"
            assert config.class_name, f"{instrument} missing class_name"

    def test_threading_has_no_package_check(self):
        """Threading is stdlib — package should be None."""
        config = INSTRUMENT_REGISTRY[Instruments.THREADING]
        assert config.package is None

    def test_openai_has_post_init_hook(self):
        """OpenAI should have the chat prompt capture patch."""
        config = INSTRUMENT_REGISTRY[Instruments.OPENAI]
        assert "_patch_chat_prompt_capture" in config.post_init_hooks

    def test_non_openai_have_no_hooks(self):
        """Most instruments should have no post-init hooks."""
        for instrument, config in INSTRUMENT_REGISTRY.items():
            if instrument != Instruments.OPENAI:
                assert config.post_init_hooks == (), (
                    f"{instrument} has unexpected hooks: {config.post_init_hooks}"
                )

    def test_celery_in_registry(self):
        """Celery instrumentation must be registered."""
        config = INSTRUMENT_REGISTRY[Instruments.CELERY]
        assert config.package == "celery"
        assert config.module == "opentelemetry.instrumentation.celery"
        assert config.class_name == "CeleryInstrumentor"

    def test_new_infra_instruments_in_registry(self):
        """New infrastructure instrumentations must be registered."""
        new_instruments = [
            Instruments.CELERY,
            Instruments.DJANGO,
            Instruments.FASTAPI,
            Instruments.FLASK,
            Instruments.SQLALCHEMY,
            Instruments.PSYCOPG2,
            Instruments.AIOHTTP_CLIENT,
            Instruments.GRPC,
        ]
        for instrument in new_instruments:
            assert instrument in INSTRUMENT_REGISTRY, f"{instrument} not in registry"


class TestPostInitHooks:
    """Post-init hook registration."""

    def test_chat_prompt_hook_registered(self):
        """_patch_chat_prompt_capture must be in the hook registry."""
        assert "_patch_chat_prompt_capture" in _POST_INIT_HOOKS

    def test_hook_is_callable(self):
        """All registered hooks must be callable."""
        for name, hook in _POST_INIT_HOOKS.items():
            assert callable(hook), f"Hook '{name}' is not callable"


class TestInitSingleInstrument:
    """_init_single_instrument behavior."""

    def test_unknown_instrument_returns_false(self):
        """An instrument not in the registry returns False."""
        fake_instrument = MagicMock()
        fake_instrument.value = "nonexistent"
        # Use a real call but with a mock that won't be in the dict
        with patch.dict(INSTRUMENT_REGISTRY, clear=False):
            result = _init_single_instrument(fake_instrument)
        assert result is False

    def test_missing_package_returns_false(self):
        """If the package isn't installed, returns False."""
        with patch(
            _MOCK_IS_INSTALLED,
            return_value=False,
        ):
            result = _init_single_instrument(Instruments.CELERY)
        assert result is False

    def test_successful_init(self):
        """Successful instrumentation returns True."""
        mock_instrumentor = MagicMock()
        mock_instrumentor.is_instrumented_by_opentelemetry = False
        mock_module = MagicMock()
        mock_module.CeleryInstrumentor.return_value = mock_instrumentor

        with patch(
            _MOCK_IS_INSTALLED,
            return_value=True,
        ), patch(
            _MOCK_IMPORT_MODULE,
            return_value=mock_module,
        ):
            result = _init_single_instrument(Instruments.CELERY)

        assert result is True
        mock_instrumentor.instrument.assert_called_once()

    def test_already_instrumented_skips(self):
        """If already instrumented, don't call instrument() again."""
        mock_instrumentor = MagicMock()
        mock_instrumentor.is_instrumented_by_opentelemetry = True
        mock_module = MagicMock()
        mock_module.CeleryInstrumentor.return_value = mock_instrumentor

        with patch(
            _MOCK_IS_INSTALLED,
            return_value=True,
        ), patch(
            _MOCK_IMPORT_MODULE,
            return_value=mock_module,
        ):
            result = _init_single_instrument(Instruments.CELERY)

        assert result is True
        mock_instrumentor.instrument.assert_not_called()

    def test_import_error_returns_false(self):
        """If the OTEL instrumentation package isn't installed, returns False."""
        with patch(
            _MOCK_IS_INSTALLED,
            return_value=True,
        ), patch(
            _MOCK_IMPORT_MODULE,
            side_effect=ImportError("No module"),
        ):
            result = _init_single_instrument(Instruments.CELERY)

        assert result is False

    def test_post_init_hook_called(self):
        """Post-init hooks are called after instrument()."""
        mock_instrumentor = MagicMock()
        mock_instrumentor.is_instrumented_by_opentelemetry = False
        mock_module = MagicMock()
        mock_module.OpenAIInstrumentor.return_value = mock_instrumentor
        mock_hook = MagicMock()

        with patch(
            _MOCK_IS_INSTALLED,
            return_value=True,
        ), patch(
            _MOCK_IMPORT_MODULE,
            return_value=mock_module,
        ), patch.dict(_POST_INIT_HOOKS, {"_patch_chat_prompt_capture": mock_hook}):
            result = _init_single_instrument(Instruments.OPENAI)

        assert result is True
        mock_hook.assert_called_once()

    def test_threading_no_package_check(self):
        """Threading (package=None) skips package check."""
        mock_instrumentor = MagicMock()
        mock_instrumentor.is_instrumented_by_opentelemetry = False
        mock_module = MagicMock()
        mock_module.ThreadingInstrumentor.return_value = mock_instrumentor

        with patch(
            _MOCK_IMPORT_MODULE,
            return_value=mock_module,
        ):
            result = _init_single_instrument(Instruments.THREADING)

        assert result is True
        mock_instrumentor.instrument.assert_called_once()


class TestInitInstrumentations:
    """init_instrumentations public API."""

    def test_threading_auto_included(self):
        """Threading is auto-included when user specifies instruments."""
        initialized = set()

        def track_init(instrument):
            initialized.add(instrument)
            return False  # pretend nothing installed

        with patch(
            _MOCK_INIT_SINGLE,
            side_effect=track_init,
        ):
            init_instrumentations(instruments={Instruments.OPENAI})

        assert Instruments.THREADING in initialized
        assert Instruments.OPENAI in initialized

    def test_threading_blockable(self):
        """Threading can be explicitly blocked."""
        initialized = set()

        def track_init(instrument):
            initialized.add(instrument)
            return False

        with patch(
            _MOCK_INIT_SINGLE,
            side_effect=track_init,
        ):
            init_instrumentations(
                instruments={Instruments.OPENAI},
                block_instruments={Instruments.THREADING},
            )

        assert Instruments.THREADING not in initialized

    def test_block_removes_instruments(self):
        """Blocked instruments are excluded."""
        initialized = set()

        def track_init(instrument):
            initialized.add(instrument)
            return False

        with patch(
            _MOCK_INIT_SINGLE,
            side_effect=track_init,
        ):
            init_instrumentations(
                instruments={Instruments.OPENAI, Instruments.REDIS},
                block_instruments={Instruments.REDIS},
            )

        assert Instruments.REDIS not in initialized

    def test_none_instruments_enables_all(self):
        """instruments=None enables all instruments."""
        initialized = set()

        def track_init(instrument):
            initialized.add(instrument)
            return True

        with patch(
            _MOCK_INIT_SINGLE,
            side_effect=track_init,
        ):
            result = init_instrumentations(instruments=None)

        assert result is True
        assert initialized == set(Instruments)

    def test_returns_false_when_none_initialized(self):
        """Returns False if no instruments succeeded."""
        with patch(
            _MOCK_INIT_SINGLE,
            return_value=False,
        ):
            result = init_instrumentations(instruments={Instruments.OPENAI})

        assert result is False

    def test_exception_in_init_doesnt_crash(self):
        """Exception in one instrument doesn't crash the loop."""
        call_count = 0

        def failing_then_success(instrument):
            nonlocal call_count
            call_count += 1
            if instrument == Instruments.OPENAI:
                raise RuntimeError("boom")
            return True

        with patch(
            _MOCK_INIT_SINGLE,
            side_effect=failing_then_success,
        ):
            result = init_instrumentations(
                instruments={Instruments.OPENAI, Instruments.REDIS}
            )

        assert result is True  # REDIS succeeded


class TestIsPackageInstalled:
    """is_package_installed edge cases."""

    def test_none_package_always_available(self):
        """package=None (stdlib) returns True."""
        assert is_package_installed(None) is True

    def test_installed_package(self):
        """An installed package returns True."""
        assert is_package_installed("json") is True

    def test_missing_package(self):
        """A missing package returns False."""
        assert is_package_installed("definitely_not_a_real_package_xyz") is False


class TestConfigImmutability:
    """InstrumentConfig is frozen (immutable)."""

    def test_cannot_mutate_config(self):
        """Config fields cannot be changed after creation."""
        config = INSTRUMENT_REGISTRY[Instruments.OPENAI]
        with pytest.raises(AttributeError):
            config.package = "changed"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
