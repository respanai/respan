"""Tests for the GoogleAdkInstrumentor class."""
from unittest.mock import patch

from respan_instrumentation_google_adk import GoogleAdkInstrumentor


class TestInstrumentationProtocol:
    """Verify GoogleAdkInstrumentor satisfies the Instrumentation protocol."""

    def test_has_name_attribute(self):
        instrumentor = GoogleAdkInstrumentor()
        assert instrumentor.name == "google-adk"

    def test_satisfies_instrumentation_protocol(self):
        from respan._types import Instrumentation
        inst = GoogleAdkInstrumentor()
        assert isinstance(inst, Instrumentation)

    def test_activate_takes_no_arguments(self):
        """activate() matches the Instrumentation protocol (no args)."""
        instrumentor = GoogleAdkInstrumentor()
        with patch("respan_instrumentation_google_adk.instrumentor.patch_span_processors"):
            instrumentor.activate()

    def test_activate_calls_patch_span_processors(self):
        instrumentor = GoogleAdkInstrumentor()
        with patch("respan_instrumentation_google_adk.instrumentor.patch_span_processors") as mock_patch:
            instrumentor.activate()
            mock_patch.assert_called_once()

    def test_deactivate_does_not_raise(self):
        instrumentor = GoogleAdkInstrumentor()
        instrumentor.deactivate()

    def test_constructor_accepts_kwargs(self):
        instrumentor = GoogleAdkInstrumentor(
            environment="staging",
            customer_identifier="user-42",
            some_future_param="value",
        )
        assert instrumentor.name == "google-adk"

    def test_stores_environment(self):
        instrumentor = GoogleAdkInstrumentor(environment="development")
        assert instrumentor.environment == "development"

    def test_stores_customer_identifier(self):
        instrumentor = GoogleAdkInstrumentor(customer_identifier="user-42")
        assert instrumentor.customer_identifier == "user-42"

    def test_defaults_to_none(self):
        instrumentor = GoogleAdkInstrumentor()
        assert instrumentor.environment is None
        assert instrumentor.customer_identifier is None
