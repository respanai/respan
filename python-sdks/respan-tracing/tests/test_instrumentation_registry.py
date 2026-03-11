"""Tests for the auto-discovery instrumentation system.

Validates that the entry-point-based auto-discovery correctly replaces
manual registration — pip install a package, it auto-instruments.
"""
import pytest
from unittest.mock import patch, MagicMock, PropertyMock
from respan_tracing.instruments import Instruments
from respan_tracing.utils.instrumentation import (
    ENTRY_POINT_GROUP,
    _POST_INIT_HOOKS,
    _discover_instrumentors,
    _enum_to_entry_point_name,
    _instrument_entry_point,
    init_instrumentations,
)

_MOD = "respan_tracing.utils.instrumentation"


def _make_entry_point(name, instrumentor_cls=None):
    """Create a mock entry point."""
    ep = MagicMock()
    ep.name = name
    if instrumentor_cls is None:
        inst = MagicMock()
        inst.is_instrumented_by_opentelemetry = False
        instrumentor_cls = MagicMock(return_value=inst)
    ep.load.return_value = instrumentor_cls
    return ep


class TestAutoDiscovery:
    """Entry point discovery."""

    def test_discover_returns_dict(self):
        """_discover_instrumentors returns a dict of name → entry_point."""
        result = _discover_instrumentors()
        assert isinstance(result, dict)
        # Should find at least threading (always installed)
        assert "threading" in result

    def test_discover_finds_openai(self):
        """OpenAI instrumentor should be discoverable (installed in dev)."""
        result = _discover_instrumentors()
        assert "openai" in result

    def test_discover_handles_exception(self):
        """Exception during discovery returns empty dict."""
        with patch(
            f"{_MOD}.importlib.metadata.entry_points",
            side_effect=RuntimeError("boom"),
        ):
            result = _discover_instrumentors()
        assert result == {}


class TestEnumToEntryPoint:
    """Instruments enum → entry point name mapping."""

    def test_direct_match(self):
        """Most enums map directly to their value."""
        assert _enum_to_entry_point_name(Instruments.OPENAI) == "openai"
        assert _enum_to_entry_point_name(Instruments.THREADING) == "threading"
        assert _enum_to_entry_point_name(Instruments.CELERY) == "celery"

    def test_grpc_override(self):
        """GRPC maps to grpc_client entry point."""
        assert _enum_to_entry_point_name(Instruments.GRPC) == "grpc_client"

    def test_aiohttp_direct_match(self):
        """AIOHTTP_CLIENT maps directly — enum value equals entry point name."""
        assert _enum_to_entry_point_name(Instruments.AIOHTTP_CLIENT) == "aiohttp_client"


class TestInstrumentEntryPoint:
    """_instrument_entry_point behavior."""

    def test_successful_init(self):
        """Successfully loads and instruments."""
        ep = _make_entry_point("celery")
        assert _instrument_entry_point(ep, "celery") is True
        ep.load.assert_called_once()

    def test_already_instrumented_skips_instrument(self):
        """Already instrumented → don't call instrument() again."""
        inst = MagicMock()
        inst.is_instrumented_by_opentelemetry = True
        cls = MagicMock(return_value=inst)
        ep = _make_entry_point("celery", cls)

        assert _instrument_entry_point(ep, "celery") is True
        inst.instrument.assert_not_called()

    def test_load_failure_returns_false(self):
        """If load() raises, returns False."""
        ep = MagicMock()
        ep.name = "broken"
        ep.load.side_effect = ImportError("missing dep")

        assert _instrument_entry_point(ep, "broken") is False

    def test_post_init_hook_called(self):
        """Post-init hooks fire after instrument()."""
        mock_hook = MagicMock()
        ep = _make_entry_point("openai")

        with patch.dict(_POST_INIT_HOOKS, {"openai": mock_hook}):
            result = _instrument_entry_point(ep, "openai")

        assert result is True
        mock_hook.assert_called_once()

    def test_no_hook_for_most_instruments(self):
        """Instruments without hooks don't crash."""
        ep = _make_entry_point("celery")
        # No hook registered for "celery" — should still succeed
        assert _instrument_entry_point(ep, "celery") is True


class TestInitInstrumentations:
    """init_instrumentations public API."""

    def _mock_discover(self, names):
        """Create a mock discovery result."""
        return {name: _make_entry_point(name) for name in names}

    def test_auto_discover_all(self):
        """instruments=None discovers and instruments everything."""
        discovered = self._mock_discover(["openai", "threading", "celery"])

        with patch(f"{_MOD}._discover_instrumentors", return_value=discovered):
            result = init_instrumentations(instruments=None)

        assert result is True
        for ep in discovered.values():
            ep.load.assert_called_once()

    def test_explicit_instruments_filters(self):
        """Only specified instruments are initialized."""
        discovered = self._mock_discover(["openai", "threading", "celery"])

        with patch(f"{_MOD}._discover_instrumentors", return_value=discovered):
            result = init_instrumentations(instruments={Instruments.OPENAI})

        assert result is True
        discovered["openai"].load.assert_called_once()
        # Threading auto-included
        discovered["threading"].load.assert_called_once()
        # Celery NOT included (not in instruments set)
        discovered["celery"].load.assert_not_called()

    def test_threading_auto_included(self):
        """Threading is auto-included when user specifies instruments."""
        discovered = self._mock_discover(["openai", "threading"])

        with patch(f"{_MOD}._discover_instrumentors", return_value=discovered):
            init_instrumentations(instruments={Instruments.OPENAI})

        discovered["threading"].load.assert_called_once()

    def test_threading_blockable(self):
        """Threading can be explicitly blocked."""
        discovered = self._mock_discover(["openai", "threading"])

        with patch(f"{_MOD}._discover_instrumentors", return_value=discovered):
            init_instrumentations(
                instruments={Instruments.OPENAI},
                block_instruments={Instruments.THREADING},
            )

        discovered["threading"].load.assert_not_called()

    def test_block_removes_instruments(self):
        """Blocked instruments are excluded."""
        discovered = self._mock_discover(["openai", "redis", "threading"])

        with patch(f"{_MOD}._discover_instrumentors", return_value=discovered):
            init_instrumentations(
                instruments={Instruments.OPENAI, Instruments.REDIS},
                block_instruments={Instruments.REDIS},
            )

        discovered["redis"].load.assert_not_called()

    def test_missing_entry_point_skipped(self):
        """Instruments not installed are silently skipped."""
        # Only threading installed, user asks for celery too
        discovered = self._mock_discover(["threading"])

        with patch(f"{_MOD}._discover_instrumentors", return_value=discovered):
            result = init_instrumentations(
                instruments={Instruments.CELERY}
            )

        # Only threading succeeded (auto-included), celery not found
        assert result is True

    def test_returns_false_when_none_initialized(self):
        """Returns False if no instruments succeeded."""
        with patch(f"{_MOD}._discover_instrumentors", return_value={}):
            result = init_instrumentations(instruments={Instruments.OPENAI})

        assert result is False

    def test_exception_in_one_doesnt_crash(self):
        """Exception in one instrumentor doesn't stop others."""
        good_ep = _make_entry_point("threading")
        bad_ep = MagicMock()
        bad_ep.name = "openai"
        bad_ep.load.side_effect = RuntimeError("boom")
        discovered = {"openai": bad_ep, "threading": good_ep}

        with patch(f"{_MOD}._discover_instrumentors", return_value=discovered):
            result = init_instrumentations(instruments=None)

        assert result is True
        good_ep.load.assert_called_once()

    def test_undiscovered_instruments_pip_install_works(self):
        """Simulates: user pip installs a new package, it auto-discovers."""
        # First call: only openai
        discovered_v1 = self._mock_discover(["openai", "threading"])

        with patch(f"{_MOD}._discover_instrumentors", return_value=discovered_v1):
            init_instrumentations(instruments=None)

        discovered_v1["openai"].load.assert_called_once()

        # Second call: celery now installed (simulated by adding to discovery)
        discovered_v2 = self._mock_discover(["openai", "threading", "celery"])

        with patch(f"{_MOD}._discover_instrumentors", return_value=discovered_v2):
            init_instrumentations(instruments=None)

        # celery is now discovered and instrumented
        discovered_v2["celery"].load.assert_called_once()


class TestPostInitHooks:
    """Post-init hook registration."""

    def test_openai_hook_registered(self):
        """OpenAI chat prompt patch must be in the hook registry."""
        assert "openai" in _POST_INIT_HOOKS

    def test_hook_is_callable(self):
        """All registered hooks must be callable."""
        for name, hook in _POST_INIT_HOOKS.items():
            assert callable(hook), f"Hook '{name}' is not callable"


class TestBackwardCompat:
    """Backward compatibility with the Instruments enum."""

    def test_all_enum_values_are_strings(self):
        """Every Instruments value is a string (used for entry point matching)."""
        for instrument in Instruments:
            assert isinstance(instrument.value, str)

    def test_enum_has_infra_instruments(self):
        """Infrastructure instruments exist in enum for block_instruments usage."""
        assert hasattr(Instruments, "CELERY")
        assert hasattr(Instruments, "DJANGO")
        assert hasattr(Instruments, "FASTAPI")
        assert hasattr(Instruments, "REDIS")
        assert hasattr(Instruments, "THREADING")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
